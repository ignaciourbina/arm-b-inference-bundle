"""Async client for local LLM inference via llama-server (llama.cpp).

Uses the OpenAI-compatible /v1/chat/completions endpoint.
llama-server also exposes /api/chat (Ollama-compatible) but that path
has a null-content bug on re-submitted assistant messages; use openai only.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import aiohttp

from .types import ChatMessage, ObjectMap, ToolCall, ToolSchema, TraceRecordDict

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:20434"
DEFAULT_MODEL = "gemma-4-E2B-it-Q8_0.gguf"
# "openai" -> /v1/chat/completions (llama-server, vLLM)
# "ollama" -> /api/chat (legacy, has null-content bug — do not use)
DEFAULT_API_FLAVOR = "openai"


def _as_int(value: object, default: int = 0) -> int:
    return value if isinstance(value, int) else default


def _as_str(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _as_object_map(value: object) -> ObjectMap:
    return cast(ObjectMap, value) if isinstance(value, dict) else {}


def _snapshot_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Freeze chat history for tracing so later loop mutations do not rewrite it."""
    return cast(list[ChatMessage], deepcopy(messages))


def _build_chat_trace_record(
    *,
    seq: int,
    hook: str,
    agent_id: str,
    messages: list[ChatMessage],
    msg: ChatMessage,
    tool_calls: list[ToolCall],
    prompt_tokens: int,
    gen_tokens: int,
    elapsed_s: float,
    model: str,
) -> TraceRecord:
    frozen_messages = _snapshot_messages(messages)
    return TraceRecord(
        seq=seq,
        hook=hook,
        agent_id=agent_id,
        prompt=json.dumps(frozen_messages, ensure_ascii=False),
        response_raw=json.dumps(msg, ensure_ascii=False),
        response_parsed={
            "tool_calls": tool_calls,
            "content": _as_str(msg.get("content", "")),
        },
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        elapsed_s=elapsed_s,
        model=model,
        conversation=frozen_messages,
    )


@dataclass
class TraceRecord:
    """Single LLM call record for the trace log."""

    seq: int
    hook: str  # "voice", "evaluate", "reflect", or ""
    agent_id: str
    prompt: str
    response_raw: str
    response_parsed: ObjectMap
    prompt_tokens: int
    gen_tokens: int
    elapsed_s: float
    model: str
    error: str | None = None
    conversation: list[ChatMessage] | None = None
    tool_calls_made: list[ToolCall] | None = None
    loop_rounds: int = 1

    def to_dict(self) -> TraceRecordDict:
        d: TraceRecordDict = {
            "seq": self.seq,
            "hook": self.hook,
            "agent_id": self.agent_id,
            "prompt": self.prompt,
            "response_raw": self.response_raw,
            "response_parsed": self.response_parsed,
            "prompt_tokens": self.prompt_tokens,
            "gen_tokens": self.gen_tokens,
            "elapsed_s": round(self.elapsed_s, 3),
            "model": self.model,
            "error": self.error,
        }
        if self.conversation is not None:
            d["conversation"] = self.conversation
        if self.tool_calls_made is not None:
            d["tool_calls_made"] = self.tool_calls_made
        if self.loop_rounds > 1:
            d["loop_rounds"] = self.loop_rounds
        return d


@dataclass
class LLMClient:
    """Async client for local LLM inference."""

    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    max_concurrent: int = 4
    timeout: float = 120.0
    max_tokens: int = 64
    temperature: float = 1.0
    trace_path: Path | None = None
    api_flavor: str = DEFAULT_API_FLAVOR
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)
    _trace: list[TraceRecord] = field(init=False, repr=False, default_factory=list)
    _seq: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    async def generate(
        self,
        prompt: str,
        json_mode: bool = True,
        *,
        session: aiohttp.ClientSession | None = None,
        hook: str = "",
        agent_id: str = "",
    ) -> ObjectMap:
        """Send a single completion request. Returns parsed JSON or raw text."""
        payload: ObjectMap = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        }
        if json_mode:
            payload["format"] = "json"

        url = f"{self.base_url}/api/generate"
        own_session = session is None
        if own_session:
            session = aiohttp.ClientSession()
        assert session is not None

        error_msg = None
        try:
            async with self._semaphore:
                t0 = time.monotonic()
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                elapsed = time.monotonic() - t0
        finally:
            if own_session:
                await session.close()

        response_text = data.get("response", "")
        prompt_tokens = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)

        logger.debug(
            "generate: %d prompt_tokens, %d gen_tokens, %.1fs (%.1f tok/s)",
            prompt_tokens,
            eval_count,
            elapsed,
            eval_count / elapsed if elapsed > 0 else 0,
        )

        parsed: ObjectMap
        if json_mode:
            try:
                parsed = _as_object_map(json.loads(response_text))
            except json.JSONDecodeError:
                logger.warning("JSON parse failed, raw response: %s", response_text)
                parsed = {"_raw": response_text, "_error": "json_parse_failed"}
                error_msg = "json_parse_failed"
        else:
            parsed = {"_raw": response_text}

        # Record trace
        self._seq += 1
        record = TraceRecord(
            seq=self._seq,
            hook=hook,
            agent_id=agent_id,
            prompt=prompt,
            response_raw=response_text,
            response_parsed=parsed,
            prompt_tokens=prompt_tokens,
            gen_tokens=eval_count,
            elapsed_s=elapsed,
            model=self.model,
            error=error_msg,
        )
        self._trace.append(record)

        return parsed

    async def generate_batch(
        self, prompts: list[str], json_mode: bool = True
    ) -> list[ObjectMap]:
        """Send multiple prompts concurrently. Returns results in order."""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.generate(p, json_mode=json_mode, session=session)
                for p in prompts
            ]
            return await asyncio.gather(*tasks)

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSchema] | None = None,
        *,
        hook: str = "",
        agent_id: str = "",
        max_tokens_override: int | None = None,
        think: bool | None = None,
    ) -> ChatMessage:
        """Send a chat completion with optional tool definitions.

        Uses /api/chat (Ollama chat API). Returns the response message dict,
        which may contain tool_calls.
        """
        # Tool-call generation budget. 2048, not 1024: every observed
        # parse-500 (2026-07-20 gauntlet + pilot) was a generation cut at
        # exactly the 1024 cap mid-tool-JSON, and those errors are
        # unrecoverable at retry time (see below) — so prevent the
        # truncation instead. The model stops at EOS, so the higher cap
        # only costs latency on the rare rambling call (~+50s at 20 t/s)
        # vs. a crashed run attempt (~90s round redo + restart).
        # Per-slot ctx is 4096; the server clamps over-budget requests
        # gracefully (verified: 2.5k-token prompt + max_tokens=4096).
        num_predict = max_tokens_override if max_tokens_override is not None else (
            max(self.max_tokens, 2048) if tools else self.max_tokens
        )

        if self.api_flavor == "openai":
            # llama-server / vLLM OpenAI-compatible schema.
            payload: ObjectMap = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "temperature": self.temperature,
                "max_tokens": num_predict,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            url = f"{self.base_url}/v1/chat/completions"
        else:
            # Ollama native schema.
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": num_predict,
                    "temperature": self.temperature,
                },
            }
            if tools:
                payload["tools"] = tools
                if think is None:
                    think = True
            if think is not None:
                payload["think"] = think
            url = f"{self.base_url}/api/chat"
        # Note: no asyncio.Semaphore here — concurrency is bounded by the
        # ThreadPoolExecutor in the runner and by Ollama's NUM_PARALLEL slots.
        # Using self._semaphore (init'd on another event loop) across threads
        # raises "Semaphore is bound to a different event loop".
        retries_left = 3
        backoff = 2.0
        t0 = time.monotonic()

        # Retry nudge (2026-07-20): a malformed tool-call generation 500s at
        # response-parse time, and the server then replays the identical error
        # instantly for byte-different retries (fresh seed + larger max_tokens
        # did not trigger re-generation — zero slot activity server-side).
        # Only changing the PROMPT reliably forces a fresh generation, so
        # retries append a corrective user turn. It never enters the harness's
        # conversation (`messages` itself is not mutated), so later tool-loop
        # rounds and the saved trace stay marker-free.
        retry_nudge: ChatMessage = {
            "role": "user",
            "content": ("Your previous reply failed to parse — the tool call's "
                        "JSON arguments were cut off. Call the tool again with "
                        "short, complete JSON arguments."),
        }

        def _perturb_retry() -> None:
            # (1) fresh explicit sampling seed (llama-server samples
            # deterministically by default); (2) doubled token budget in case
            # of max_tokens truncation mid-JSON; (3) the retry nudge above,
            # sent on a copy of the message list. First attempts stay
            # seed-free and nudge-free (deterministic, reproducible).
            nonlocal num_predict
            num_predict = min(num_predict * 2, 4096)
            fresh_seed = random.getrandbits(31)
            payload["messages"] = [*messages, retry_nudge]
            if self.api_flavor == "openai":
                payload["max_tokens"] = num_predict
                payload["seed"] = fresh_seed
            else:
                opts = cast(ObjectMap, payload["options"])
                opts["num_predict"] = num_predict
                opts["seed"] = fresh_seed

        while True:
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as resp:
                        if resp.status == 500:
                            err_body = await resp.text()
                            # llama-server (v10050) replays a tool-call
                            # parse error verbatim for ANY retry — fresh
                            # seed, larger budget, even a changed prompt
                            # trigger no re-generation (zero slot launches
                            # server-side; verified 2026-07-20). Retrying
                            # is dead time: fail fast so the supervisor's
                            # restart+resume recovers the round sooner.
                            if "Failed to parse tool call" in err_body:
                                logger.error(
                                    "chat %s got unrecoverable parse-500, "
                                    "failing fast: %s",
                                    hook or "?", err_body[:300])
                                # Zero the budget so the ClientResponseError
                                # handler below re-raises instead of retrying.
                                retries_left = 0
                                resp.raise_for_status()
                            if retries_left > 0:
                                logger.warning(
                                    "chat got 500, retrying in %.1fs "
                                    "(%d retries left, max_tokens -> %d)",
                                    backoff, retries_left,
                                    min(num_predict * 2, 4096))
                                _perturb_retry()
                                await asyncio.sleep(backoff)
                                retries_left -= 1
                                backoff *= 2
                                continue
                        if resp.status >= 400:
                            err_body = await resp.text()
                            logger.error("chat %s returned %d: %s",
                                         hook or "?", resp.status, err_body[:500])
                        resp.raise_for_status()
                        data = await resp.json()
                        break
                except aiohttp.ClientResponseError as e:
                    if e.status == 500 and retries_left > 0:
                        logger.warning("chat got 500 (exception), retrying in %.1fs",
                                       backoff)
                        _perturb_retry()
                        await asyncio.sleep(backoff)
                        retries_left -= 1
                        backoff *= 2
                        continue
                    raise
        elapsed = time.monotonic() - t0

        if self.api_flavor == "openai":
            # OpenAI: {choices: [{message: {role, content, tool_calls}}], usage: {...}}
            # Leave tool_calls[].function.arguments as a string (per OpenAI spec).
            # vLLM's /v1 validator rejects dicts when the message is re-submitted in
            # a follow-up turn. Downstream consumers parse arguments at dispatch.
            choices = cast(list[ObjectMap], data.get("choices") or [{}])
            msg = cast(ChatMessage, choices[0].get("message", {}) if choices else {})
            tool_calls = cast(list[ToolCall], msg.get("tool_calls", []) or [])
            # Strip nullable OpenAI fields that vLLM rejects on re-submit.
            msg_map = cast(ObjectMap, msg)
            for k in ("refusal", "annotations", "audio", "function_call", "reasoning"):
                msg_map.pop(k, None)
            usage = _as_object_map(data.get("usage", {}))
            prompt_tokens = _as_int(usage.get("prompt_tokens", 0))
            eval_count = _as_int(usage.get("completion_tokens", 0))
        else:
            # Ollama schema
            msg = cast(ChatMessage, data.get("message", {}))
            tool_calls = cast(list[ToolCall], msg.get("tool_calls", []))
            prompt_tokens = _as_int(data.get("prompt_eval_count", 0))
            eval_count = _as_int(data.get("eval_count", 0))

        logger.debug(
            "chat: %d prompt_tokens, %d gen_tokens, %.1fs, %d tool_calls",
            prompt_tokens, eval_count, elapsed, len(tool_calls),
        )

        self._seq += 1
        record = _build_chat_trace_record(
            seq=self._seq,
            hook=hook,
            agent_id=agent_id,
            messages=messages,
            msg=msg,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            gen_tokens=eval_count,
            elapsed_s=elapsed,
            model=self.model,
        )
        self._trace.append(record)
        self.flush_trace()

        return msg

    def generate_sync(self, prompt: str, json_mode: bool = True) -> ObjectMap:
        """Blocking wrapper for synchronous code paths."""
        return asyncio.run(self.generate(prompt, json_mode=json_mode))

    async def health(self) -> bool:
        """Check if the server is reachable."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    def get_trace(self) -> list[TraceRecordDict]:
        """Return all trace records as dicts."""
        return [r.to_dict() for r in self._trace]

    def save_trace(self, path: Path | str | None = None) -> Path:
        """Write trace to JSON file. Returns the path used."""
        path = Path(path) if path else self.trace_path
        if path is None:
            path = Path(f"llm_trace_{int(time.time())}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "model": self.model,
                    "n_calls": len(self._trace),
                    "total_elapsed_s": round(
                        sum(r.elapsed_s for r in self._trace), 3
                    ),
                    "calls": self.get_trace(),
                },
                f,
                indent=2,
            )
        return path

    def flush_trace(self) -> Path | None:
        """Incrementally save trace to disk. Call after each LLM call."""
        if self.trace_path is None:
            return None
        return self.save_trace(self.trace_path)

    def clear_trace(self) -> None:
        """Reset trace records and sequence counter."""
        self._trace.clear()
        self._seq = 0
