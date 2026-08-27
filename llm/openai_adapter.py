"""OpenAI cloud adapter for the deliberation harness (hexagonal port impl).

`OpenAICloudClient` subclasses `LLMClient` and overrides only `chat()`, so the
engine/harness never learn which backend answered. Design + provenance:
`docs/design/openai-gpt5-mini-adapter-design.md` (research cycle 2026-08-26
+ the BPEL Study-3 addendum).

Deliberate choices, each traceable to that doc:

* **Chat Completions, not Responses** — one wire format for both backends
  keeps Arm-B traces structurally comparable with the 390-run local
  collection. (Responses translation is a contained follow-up if a pilot
  demands it.)
* **Raw aiohttp, not the SDK** — the runner drives per-thread event loops;
  Study 3 hit the SDK's "Future attached to a different loop" failure in the
  same situation. Per-call ClientSession mirrors the parent class exactly.
* **GPT-5 reasoning-model param discipline** — sampling params the model
  rejects (`temperature`, `top_p`, `seed`, penalties, `max_tokens`) are
  stripped here, not pushed onto the harness; `max_completion_tokens`
  replaces `max_tokens`; `reasoning_effort`/`verbosity` are adapter knobs.
* **Strict tools + forced choice** — `strict: true` is injected on every
  function schema (constrained decoding of args); when the harness passes
  exactly ONE tool the adapter forces it by name (Study-3 pattern), else
  `tool_choice: "required"`.
* **429 discipline** — Retry-After honored as a floor, exponential backoff
  with jitter; quota/spend-limit errors abort immediately (retrying money
  errors is never right).
* **Budget guard + cost ledger** — every call's usage (incl. cached and
  reasoning tokens) is priced and appended to a JSONL ledger; when the
  cumulative spend crosses `budget_usd`, the client raises `BudgetExceeded`
  so the run dies cleanly instead of running up a bill overnight.

Environment (all optional except the key):
    OPENAI_API_KEY          required at construction
    OPENAI_MODEL            default "gpt-5-mini-2025-08-07" (pinned snapshot)
    OPENAI_REASONING_EFFORT default "minimal"
    OPENAI_VERBOSITY        default "low"
    OPENAI_SERVICE_TIER     default unset (omit field); "flex" for 50% pricing
    OPENAI_BUDGET_USD       default 25.0 — hard abort ceiling for this process
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import aiohttp

from llm.client import LLMClient, _build_chat_trace_record, _as_int, _as_object_map
from llm.types import ChatMessage, ObjectMap, ToolCall, ToolSchema

logger = logging.getLogger(__name__)

DEFAULT_CLOUD_MODEL = "gpt-5-mini-2025-08-07"

# List prices (USD per 1M tokens) — gpt-5-mini, 2026-08; flex halves these.
# Used for the ledger/budget guard only; not a billing source of truth.
PRICE_PER_M = {"input": 0.25, "cached_input": 0.025, "output": 2.00}

# Sampling params GPT-5 reasoning models reject with 400s — silently stripped
# so the harness does not need backend-specific knowledge.
UNSUPPORTED_PARAMS = (
    "temperature", "top_p", "presence_penalty", "frequency_penalty",
    "logprobs", "logit_bias", "n", "seed", "max_tokens",
)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
FATAL_429_CODES = {"insufficient_quota", "project_spend_limit_exceeded",
                   "billing_hard_limit_reached"}


class BudgetExceeded(RuntimeError):
    """Cumulative estimated spend crossed the configured ceiling."""


@dataclass
class OpenAICloudClient(LLMClient):
    """LLMClient port implementation for api.openai.com Chat Completions."""

    base_url: str = "https://api.openai.com"
    model: str = ""
    api_flavor: str = "openai-cloud"
    reasoning_effort: str = ""
    verbosity: str = ""
    service_tier: str = ""
    prompt_cache_key: str = ""
    budget_usd: float = 0.0
    api_key: str = field(default="", repr=False)
    _spend_usd: float = field(init=False, default=0.0)
    _ledger_path: Path | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. The cloud adapter fails hard rather "
                "than degrading (same posture as Study 3)."
            )
        self.model = self.model or os.environ.get("OPENAI_MODEL", DEFAULT_CLOUD_MODEL)
        self.reasoning_effort = self.reasoning_effort or os.environ.get(
            "OPENAI_REASONING_EFFORT", "minimal")
        self.verbosity = self.verbosity or os.environ.get("OPENAI_VERBOSITY", "low")
        self.service_tier = self.service_tier or os.environ.get("OPENAI_SERVICE_TIER", "")
        if not self.budget_usd:
            self.budget_usd = float(os.environ.get("OPENAI_BUDGET_USD", "25.0"))
        if self.trace_path is not None:
            self._ledger_path = Path(str(self.trace_path)).with_suffix(".cost.jsonl")

    # -- request shaping -------------------------------------------------------

    def _shape_payload(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSchema] | None,
        max_completion_tokens: int,
    ) -> ObjectMap:
        payload: ObjectMap = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_completion_tokens": max_completion_tokens,
            "reasoning_effort": self.reasoning_effort,
            "verbosity": self.verbosity,
            "parallel_tool_calls": False,
        }
        if self.service_tier:
            payload["service_tier"] = self.service_tier
        if self.prompt_cache_key:
            payload["prompt_cache_key"] = self.prompt_cache_key
        if tools:
            payload["tools"] = [self._strictify(t) for t in tools]
            if len(tools) == 1:
                name = cast(ObjectMap, tools[0].get("function", {})).get("name", "")
                payload["tool_choice"] = {"type": "function",
                                          "function": {"name": name}}
            else:
                payload["tool_choice"] = "required"
        for k in UNSUPPORTED_PARAMS:
            payload.pop(k, None)
        return payload

    @staticmethod
    def _strictify(tool: ToolSchema) -> ToolSchema:
        """Inject strict:true + additionalProperties:false without mutating input."""
        t = json.loads(json.dumps(tool))  # deep copy
        fn = t.get("function")
        if isinstance(fn, dict):
            fn["strict"] = True
            params = fn.get("parameters")
            if isinstance(params, dict):
                params.setdefault("additionalProperties", False)
        return cast(ToolSchema, t)

    # -- cost ledger -----------------------------------------------------------

    def _record_cost(self, usage: ObjectMap, hook: str) -> None:
        pt = _as_int(usage.get("prompt_tokens", 0))
        ct = _as_int(usage.get("completion_tokens", 0))
        ptd = _as_object_map(usage.get("prompt_tokens_details", {}))
        ctd = _as_object_map(usage.get("completion_tokens_details", {}))
        cached = _as_int(ptd.get("cached_tokens", 0))
        reasoning = _as_int(ctd.get("reasoning_tokens", 0))
        tier_scale = 0.5 if self.service_tier == "flex" else 1.0
        cost = tier_scale * (
            (pt - cached) * PRICE_PER_M["input"]
            + cached * PRICE_PER_M["cached_input"]
            + ct * PRICE_PER_M["output"]
        ) / 1_000_000
        self._spend_usd += cost
        entry = {
            "ts": time.time(), "hook": hook, "model": self.model,
            "prompt_tokens": pt, "cached_tokens": cached,
            "completion_tokens": ct, "reasoning_tokens": reasoning,
            "cost_usd": round(cost, 6), "cum_usd": round(self._spend_usd, 4),
        }
        if self._ledger_path is not None:
            try:
                with open(self._ledger_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except OSError:  # ledger is telemetry; never kill a run over it
                logger.warning("cost ledger write failed", exc_info=True)
        if self._spend_usd > self.budget_usd:
            raise BudgetExceeded(
                f"estimated spend ${self._spend_usd:.2f} exceeded budget "
                f"${self.budget_usd:.2f} — aborting before more calls"
            )

    # -- the port --------------------------------------------------------------

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSchema] | None = None,
        *,
        hook: str = "",
        agent_id: str = "",
        max_tokens_override: int | None = None,
        think: bool | None = None,  # accepted for port parity; unused on cloud
    ) -> ChatMessage:
        # Generous cap: reasoning tokens spend from this budget BEFORE visible
        # output; a tight cap yields finish_reason=length with an empty tool
        # call that is still billed (design doc, risk #1).
        cap = max_tokens_override if max_tokens_override is not None else max(
            self.max_tokens, 2000)
        payload = self._shape_payload(messages, tools, cap)
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}

        retry_nudge: ChatMessage = {
            "role": "user",
            "content": ("Your previous reply did not include a valid tool call. "
                        "Call the required tool with short, complete JSON "
                        "arguments."),
        }
        retries_left = 3
        backoff = 2.0
        t0 = time.monotonic()

        while True:
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        url, json=payload, headers=headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        if resp.status in RETRYABLE_STATUS:
                            err_body = await resp.text()
                            err_code = ""
                            try:
                                err_code = str(_as_object_map(
                                    _as_object_map(json.loads(err_body)).get("error", {})
                                ).get("code", ""))
                            except (json.JSONDecodeError, TypeError):
                                pass
                            if resp.status == 429 and err_code in FATAL_429_CODES:
                                logger.error("fatal quota/spend 429 (%s) — aborting",
                                             err_code)
                                resp.raise_for_status()
                            if retries_left > 0:
                                retry_after = float(resp.headers.get("Retry-After", 0) or 0)
                                delay = max(retry_after,
                                            backoff * (1 + random.random() * 0.25))
                                logger.warning(
                                    "openai %d (%s), retrying in %.1fs "
                                    "(%d retries left)",
                                    resp.status, err_code or "no-code",
                                    delay, retries_left)
                                await asyncio.sleep(delay)
                                retries_left -= 1
                                backoff *= 2
                                continue
                        if resp.status >= 400:
                            err_body = await resp.text()
                            logger.error("openai chat %s returned %d: %s",
                                         hook or "?", resp.status, err_body[:500])
                        resp.raise_for_status()
                        data = await resp.json()
                except aiohttp.ClientResponseError:
                    raise
                except aiohttp.ClientError as e:
                    if retries_left > 0:  # network blips are retryable
                        logger.warning("openai transport error %s, retrying", e)
                        await asyncio.sleep(backoff)
                        retries_left -= 1
                        backoff *= 2
                        continue
                    raise

            choices = cast(list[ObjectMap], data.get("choices") or [{}])
            msg = cast(ChatMessage, choices[0].get("message", {}) if choices else {})
            tool_calls = cast(list[ToolCall], msg.get("tool_calls", []) or [])
            finish = str(choices[0].get("finish_reason", "")) if choices else ""

            # Empty-args / missing-tool-call retry (Study-3 pattern: corrective
            # user turn on a COPY; the harness's message list stays clean).
            bad_tool_call = bool(tools) and (
                not tool_calls
                or any(not cast(ObjectMap, cast(ObjectMap, tc).get("function", {}))
                       .get("arguments") for tc in tool_calls)
            )
            if bad_tool_call and finish != "length" and retries_left > 0:
                logger.warning("openai chat %s: no/empty tool call, corrective retry",
                               hook or "?")
                payload["messages"] = [*messages, retry_nudge]
                retries_left -= 1
                continue
            break

        elapsed = time.monotonic() - t0
        usage = _as_object_map(data.get("usage", {}))
        self._record_cost(usage, hook)  # raises BudgetExceeded past the ceiling

        fingerprint = data.get("system_fingerprint")
        if fingerprint:
            logger.debug("system_fingerprint=%s", fingerprint)

        msg_map = cast(ObjectMap, msg)
        for k in ("refusal", "annotations", "audio", "function_call", "reasoning"):
            msg_map.pop(k, None)

        self._seq += 1
        record = _build_chat_trace_record(
            seq=self._seq, hook=hook, agent_id=agent_id, messages=messages,
            msg=msg, tool_calls=tool_calls,
            prompt_tokens=_as_int(usage.get("prompt_tokens", 0)),
            gen_tokens=_as_int(usage.get("completion_tokens", 0)),
            elapsed_s=elapsed, model=self.model,
        )
        self._trace.append(record)
        self.flush_trace()
        return msg
