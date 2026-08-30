"""OpenRouter / Z.ai client for Arm C (GLM-5.3-Flash).

Arm C is a **model-generality robustness check**, not a substitute for Arm B:
Arm B is defined as `gemma-4-E2B-it-Q8_0` and every Sprint-15 ablation
(reasoning b256, Q8 vs Q4) is gemma-specific. This runs the same engine,
scenario, compositions and seeds against a different model so the deliberation
findings can be tested for model dependence.

Subclasses `OpenAICloudClient` for its retry/backoff, budget guard and cost
ledger, and overrides everything GPT-5-specific:

  * `max_tokens`, not `max_completion_tokens`
  * **keeps `temperature`** — the parent strips it because GPT-5 reasoning
    models 400 on it, but temperature is a controlled variable here
  * drops `reasoning_effort` / `verbosity` / `service_tier` / `prompt_cache_key`
  * GLM list prices, so the budget guard and ledger are not ~8x wrong on output
  * `strict:true` tool schemas off by default — that is an OpenAI
    structured-outputs feature and is not guaranteed on this route

Env:
    OPENROUTER_API_KEY      required (falls back to a key file, see below)
    OPENROUTER_MODEL        default z-ai/glm-5.3-flash
    OPENROUTER_BASE_URL     default https://openrouter.ai/api
    OPENROUTER_BUDGET_USD   default 30.0 — hard abort ceiling for this process
    OPENROUTER_STRICT_TOOLS set to 1 to re-enable strict tool schemas
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from llm.client import LLMClient
from llm.openai_adapter import (
    BudgetExceeded,
    OpenAICloudClient,
    _as_int,
    _as_object_map,
)
from llm.types import ChatMessage, ObjectMap, ToolSchema

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_BASE_URL = "https://openrouter.ai/api"

# List prices (USD per 1M tokens) — GLM-5.3-Flash launch promo, 2026-08.
# Ledger/budget-guard only; not a billing source of truth. Override with
# OPENROUTER_PRICE_IN / OPENROUTER_PRICE_OUT if the promo lapses.
GLM_PRICE_PER_M = {"input": 0.075, "cached_input": 0.075, "output": 0.25}

# Params the parent strips for GPT-5 that this route DOES accept.
_KEEP = ("temperature",)


@dataclass
class OpenRouterClient(OpenAICloudClient):
    """LLMClient port for OpenRouter / Z.ai Chat Completions."""

    base_url: str = DEFAULT_BASE_URL
    model: str = ""
    api_flavor: str = "openrouter"

    def __post_init__(self) -> None:  # noqa: D105
        # Deliberately does NOT call OpenAICloudClient.__post_init__: that
        # hard-fails on a missing OPENAI_API_KEY and applies GPT-5 defaults.
        LLMClient.__post_init__(self)
        self.api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            key_file = os.environ.get("OPENROUTER_KEY_FILE",
                                      os.path.expanduser("~/.openrouter_key"))
            try:
                self.api_key = Path(key_file).read_text().strip()
            except OSError:
                pass
        if not self.api_key:
            raise RuntimeError(
                "No OpenRouter credential. Set OPENROUTER_API_KEY or place the "
                f"key in ~/.openrouter_key (chmod 600). Fails hard rather than "
                "degrading, matching the cloud adapter's posture."
            )
        self.model = self.model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.base_url = os.environ.get("OPENROUTER_BASE_URL", self.base_url)
        # GPT-5-only knobs: keep empty so _shape_payload never emits them.
        self.reasoning_effort = ""
        self.verbosity = ""
        self.service_tier = ""
        self.prompt_cache_key = ""
        if not self.budget_usd:
            self.budget_usd = float(os.environ.get("OPENROUTER_BUDGET_USD", "30.0"))
        self._price = dict(GLM_PRICE_PER_M)
        if os.environ.get("OPENROUTER_PRICE_IN"):
            self._price["input"] = float(os.environ["OPENROUTER_PRICE_IN"])
            self._price["cached_input"] = self._price["input"]
        if os.environ.get("OPENROUTER_PRICE_OUT"):
            self._price["output"] = float(os.environ["OPENROUTER_PRICE_OUT"])
        if self.trace_path is not None:
            self._ledger_path = Path(str(self.trace_path)).with_suffix(".cost.jsonl")

    # -- request shaping -------------------------------------------------------

    def _shape_payload(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSchema] | None,
        max_completion_tokens: int,
    ) -> ObjectMap:
        strict = os.environ.get("OPENROUTER_STRICT_TOOLS", "") == "1"
        payload: ObjectMap = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": max_completion_tokens,   # not max_completion_tokens
            "temperature": self.temperature,       # controlled variable here
            "parallel_tool_calls": False,
        }
        if tools:
            payload["tools"] = [self._strictify(t) for t in tools] if strict else list(tools)
            if len(tools) == 1:
                name = cast(ObjectMap, tools[0].get("function", {})).get("name", "")
                payload["tool_choice"] = {"type": "function",
                                          "function": {"name": name}}
            else:
                payload["tool_choice"] = "required"
        return payload

    # -- cost ledger (GLM prices) ---------------------------------------------

    def _record_cost(self, usage: ObjectMap, hook: str) -> None:
        pt = _as_int(usage.get("prompt_tokens", 0))
        ct = _as_int(usage.get("completion_tokens", 0))
        ptd = _as_object_map(usage.get("prompt_tokens_details", {}))
        ctd = _as_object_map(usage.get("completion_tokens_details", {}))
        cached = _as_int(ptd.get("cached_tokens", 0))
        reasoning = _as_int(ctd.get("reasoning_tokens", 0))
        cost = (
            (pt - cached) * self._price["input"]
            + cached * self._price["cached_input"]
            + ct * self._price["output"]
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
