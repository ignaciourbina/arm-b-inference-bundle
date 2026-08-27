"""Offline tests for the OpenAI cloud adapter.

No network, no API key: aiohttp.ClientSession is monkeypatched with a fake
that replays scripted responses. These guard the behaviors the design doc
declares load-bearing: param discipline, strict/forced tools, 429 backoff,
fatal-quota abort, the budget guard, and cost-ledger capture.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import llm.openai_adapter as oa
from llm.openai_adapter import BudgetExceeded, OpenAICloudClient

TOOL = {
    "type": "function",
    "function": {
        "name": "submit_voice",
        "description": "Voice one consideration.",
        "parameters": {
            "type": "object",
            "properties": {"cid": {"type": "string"}},
            "required": ["cid"],
        },
    },
}
TOOL2 = {
    "type": "function",
    "function": {"name": "update_weight", "parameters": {"type": "object", "properties": {}}},
}


def ok_body(cached: int = 0, reasoning: int = 0, prompt: int = 1000, comp: int = 100):
    return {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "submit_voice",
                                 "arguments": json.dumps({"cid": "c3"})},
                }],
            },
        }],
        "usage": {
            "prompt_tokens": prompt, "completion_tokens": comp,
            "prompt_tokens_details": {"cached_tokens": cached},
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
        "system_fingerprint": "fp_test",
    }


class FakeResponse:
    def __init__(self, status: int, body: dict, headers: dict | None = None):
        self.status = status
        self._body = body
        self.headers = headers or {}

    async def text(self):
        return json.dumps(self._body)

    async def json(self):
        return self._body

    def raise_for_status(self):
        if self.status >= 400:
            import aiohttp
            raise aiohttp.ClientResponseError(None, (), status=self.status,
                                              message=str(self._body))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """Replays a scripted list of (status, body, headers); records requests."""

    script: list[tuple[int, dict, dict]] = []
    requests: list[dict] = []

    def __init__(self, *a, **k):
        pass

    def post(self, url, json=None, headers=None, timeout=None):
        FakeSession.requests.append(
            {"url": url, "payload": json, "headers": headers})
        status, body, hdrs = (FakeSession.script.pop(0)
                              if FakeSession.script else (200, ok_body(), {}))
        return FakeResponse(status, body, hdrs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def fake_http(monkeypatch):
    FakeSession.script = []
    FakeSession.requests = []
    monkeypatch.setattr(oa.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(oa.asyncio, "sleep", _instant_sleep)  # no real backoff waits
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    yield


async def _instant_sleep(_secs):
    return None


def make_client(**kw) -> OpenAICloudClient:
    kw.setdefault("budget_usd", 5.0)
    return OpenAICloudClient(**kw)


def run(coro):
    return asyncio.run(coro)


def last_payload() -> dict:
    return FakeSession.requests[-1]["payload"]


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAICloudClient()


def test_param_discipline_and_auth():
    c = make_client(reasoning_effort="minimal", verbosity="low")
    run(c.chat([{"role": "user", "content": "hi"}], tools=[TOOL]))
    req = FakeSession.requests[-1]
    p = req["payload"]
    assert req["headers"]["Authorization"] == "Bearer sk-test-not-real"
    assert p["model"] == oa.DEFAULT_CLOUD_MODEL
    assert "max_completion_tokens" in p and "max_tokens" not in p
    for banned in oa.UNSUPPORTED_PARAMS:
        assert banned not in p, banned
    assert p["reasoning_effort"] == "minimal"
    assert p["verbosity"] == "low"
    assert p["parallel_tool_calls"] is False


def test_single_tool_forced_by_name_and_strict():
    c = make_client()
    run(c.chat([{"role": "user", "content": "x"}], tools=[TOOL]))
    p = last_payload()
    assert p["tool_choice"] == {"type": "function",
                                "function": {"name": "submit_voice"}}
    fn = p["tools"][0]["function"]
    assert fn["strict"] is True
    assert fn["parameters"]["additionalProperties"] is False
    # the caller's schema object was not mutated
    assert "strict" not in TOOL["function"]


def test_multi_tool_uses_required():
    c = make_client()
    run(c.chat([{"role": "user", "content": "x"}], tools=[TOOL, TOOL2]))
    assert last_payload()["tool_choice"] == "required"


def test_429_retries_honoring_retry_after_then_succeeds():
    FakeSession.script = [
        (429, {"error": {"code": "rate_limit_exceeded"}}, {"Retry-After": "3"}),
        (200, ok_body(), {}),
    ]
    c = make_client()
    msg = run(c.chat([{"role": "user", "content": "x"}], tools=[TOOL]))
    assert msg["tool_calls"][0]["function"]["name"] == "submit_voice"
    assert len(FakeSession.requests) == 2


def test_fatal_quota_429_aborts_without_retry():
    import aiohttp
    FakeSession.script = [
        (429, {"error": {"code": "insufficient_quota"}}, {}),
    ]
    c = make_client()
    with pytest.raises(aiohttp.ClientResponseError):
        run(c.chat([{"role": "user", "content": "x"}], tools=[TOOL]))
    assert len(FakeSession.requests) == 1  # no retry on money errors


def test_empty_tool_args_triggers_corrective_retry():
    empty = ok_body()
    empty["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = ""
    FakeSession.script = [(200, empty, {}), (200, ok_body(), {})]
    c = make_client()
    msgs = [{"role": "user", "content": "x"}]
    run(c.chat(msgs, tools=[TOOL]))
    assert len(FakeSession.requests) == 2
    retry_msgs = FakeSession.requests[-1]["payload"]["messages"]
    assert retry_msgs[-1]["role"] == "user" and "tool call" in retry_msgs[-1]["content"]
    assert len(msgs) == 1  # harness's message list untouched


def test_budget_guard_aborts(tmp_path: Path):
    # 2M output tokens at $2/M = $4 > $3 budget on the second call.
    FakeSession.script = [
        (200, ok_body(comp=1_000_000), {}),
        (200, ok_body(comp=1_000_000), {}),
    ]
    c = make_client(budget_usd=3.0, trace_path=tmp_path / "t.json")
    run(c.chat([{"role": "user", "content": "x"}], tools=[TOOL]))
    with pytest.raises(BudgetExceeded):
        run(c.chat([{"role": "user", "content": "x"}], tools=[TOOL]))


def test_cost_ledger_captures_cached_and_reasoning(tmp_path: Path):
    FakeSession.script = [(200, ok_body(cached=800, reasoning=150), {})]
    c = make_client(trace_path=tmp_path / "t.json", service_tier="flex")
    run(c.chat([{"role": "user", "content": "x"}], tools=[TOOL]))
    ledger = (tmp_path / "t.cost.jsonl").read_text().strip().splitlines()
    entry = json.loads(ledger[-1])
    assert entry["cached_tokens"] == 800
    assert entry["reasoning_tokens"] == 150
    # flex halves list price: (200*0.25 + 800*0.025 + 100*2.0)/1e6 * 0.5
    expected = 0.5 * (200 * 0.25 + 800 * 0.025 + 100 * 2.0) / 1e6
    assert abs(entry["cost_usd"] - expected) < 1e-9
