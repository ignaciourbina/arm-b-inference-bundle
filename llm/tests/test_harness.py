import asyncio

import pytest

from agora.agents import Agent, AgentParams
from agora.considerations import ArgumentPool, Consideration

from llm.harness import (
    HookLoopError,
    ToolCallHarness,
    _coerce_text_tool_call,
    _dispatch_tool,
    _voice_user_msg,
)


def _build_pool() -> ArgumentPool:
    pool = ArgumentPool()
    pool.add(Consideration(id="C_01", label="Pro statement", direction=1.0, persuasiveness=0.8))
    pool.add(Consideration(id="C_02", label="Con statement", direction=-1.0, persuasiveness=0.6))
    return pool


def test_update_weight_uses_weight_and_stance_contract() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={})
    updates: list[dict[str, object]] = []

    result, is_terminal = _dispatch_tool(
        "update_weight",
        {"cid": "C_01", "weight": 0.6, "stance": "reject"},
        agent,
        pool,
        updates,
    )

    assert not is_terminal
    assert result.startswith("Adopted C_01 to -0.600 (weight=0.600, stance=reject).")
    assert "done_reflecting({\"done\":true})" in result
    assert agent.weights["C_01"] == -0.6
    assert updates == [
        {
            "cid": "C_01",
            "weight": 0.6,
            "stance": "reject",
            "new_weight": -0.6,
            "_adopted": True,
        }
    ]


def test_update_weight_rejects_x_contract() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={})
    updates: list[dict[str, object]] = []

    result, is_terminal = _dispatch_tool(
        "update_weight",
        {"cid": "C_01", "weight": 0.6, "x": -1},
        agent,
        pool,
        updates,
    )

    assert not is_terminal
    assert result == "Invalid: update_weight requires numeric weight in [0, 1] and stance in {'endorse', 'reject'}"
    assert updates == []
    assert agent.weights == {}


def test_update_weight_clips_weight_above_one() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={})
    updates: list[dict[str, object]] = []

    result, is_terminal = _dispatch_tool(
        "update_weight",
        {"cid": "C_01", "weight": 1.4, "stance": "endorse"},
        agent,
        pool,
        updates,
    )

    assert not is_terminal
    assert result.startswith("Adopted C_01 to +1.000 (weight=1.000, stance=endorse).")
    assert "done_reflecting({\"done\":true})" in result
    assert agent.weights["C_01"] == 1.0
    assert updates == [
        {
            "cid": "C_01",
            "weight": 1.0,
            "stance": "endorse",
            "new_weight": 1.0,
            "submitted_weight": 1.4,
            "_adopted": True,
        }
    ]


def test_update_weight_zero_removes_existing_consideration() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={"C_01": 0.8})
    updates: list[dict[str, object]] = []

    result, is_terminal = _dispatch_tool(
        "update_weight",
        {"cid": "C_01", "weight": 0, "stance": "endorse"},
        agent,
        pool,
        updates,
    )

    assert not is_terminal
    assert result.startswith("Removed C_01 from repertoire.")
    assert "done_reflecting({\"done\":true})" in result
    assert agent.weights == {}
    assert updates == [
        {
            "cid": "C_01",
            "weight": 0.0,
            "stance": "endorse",
            "new_weight": 0.0,
            "_removed": True,
        }
    ]


def test_update_weight_clips_negative_weight_to_removal() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={"C_01": 0.8})
    updates: list[dict[str, object]] = []

    result, is_terminal = _dispatch_tool(
        "update_weight",
        {"cid": "C_01", "weight": -0.2, "stance": "reject"},
        agent,
        pool,
        updates,
    )

    assert not is_terminal
    assert result.startswith("Removed C_01 from repertoire.")
    assert "done_reflecting({\"done\":true})" in result
    assert agent.weights == {}
    assert updates == [
        {
            "cid": "C_01",
            "weight": 0.0,
            "stance": "reject",
            "new_weight": 0.0,
            "_removed": True,
            "submitted_weight": -0.2,
        }
    ]


def test_update_weight_rejects_invalid_new_weight_payload() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={})
    updates: list[dict[str, object]] = []

    result, is_terminal = _dispatch_tool(
        "update_weight",
        {"cid": "C_01", "new_weight": -0.6},
        agent,
        pool,
        updates,
    )

    assert not is_terminal
    assert result == "Invalid: update_weight requires numeric weight in [0, 1] and stance in {'endorse', 'reject'}"
    assert updates == []
    assert agent.weights == {}


def test_submit_voice_rejects_disagreed_statement() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={"C_01": -0.4, "C_02": 0.5})
    updates: list[dict[str, object]] = []

    result, is_terminal = _dispatch_tool(
        "submit_voice",
        {"cid": "C_01"},
        agent,
        pool,
        updates,
    )

    assert not is_terminal
    assert result == (
        "Invalid: submit_voice must reference a currently supported statement "
        "or a listed attack fallback option"
    )


def test_submit_voice_accepts_listed_attack_fallback_option() -> None:
    pool = _build_pool()
    pool.add(Consideration(id="C_03", label="Counter statement", direction=1.0, persuasiveness=0.5))
    agent = Agent(id="A1", params=AgentParams(), weights={"C_01": -0.4, "C_02": -0.8})
    updates: list[dict[str, object]] = []

    result, is_terminal = _dispatch_tool(
        "submit_voice",
        {"cid": "C_03"},
        agent,
        pool,
        updates,
        {"C_03"},
    )

    assert is_terminal
    assert result == ""


def test_submit_influence_accepts_scale_integer() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={"C_01": 0.4})
    updates: list[dict[str, object]] = []

    result, is_terminal = _dispatch_tool(
        "submit_influence",
        {"score": 100},
        agent,
        pool,
        updates,
    )

    assert is_terminal
    assert result == ""
    assert updates == []


def test_submit_influence_normalizes_integer_valued_float() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={"C_01": 0.4})
    updates: list[dict[str, object]] = []
    args: dict[str, object] = {"score": 75.0}

    result, is_terminal = _dispatch_tool(
        "submit_influence",
        args,
        agent,
        pool,
        updates,
    )

    assert is_terminal
    assert result == ""
    assert args["score"] == 75


def test_submit_influence_rejects_fractional_score() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={"C_01": 0.4})
    updates: list[dict[str, object]] = []

    result, is_terminal = _dispatch_tool(
        "submit_influence",
        {"score": 0.5},
        agent,
        pool,
        updates,
    )

    assert not is_terminal
    assert result == "Invalid: score must be an integer persuasiveness rating from 0 to 100"


def test_submit_influence_rejects_out_of_range_score() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={"C_01": 0.4})
    updates: list[dict[str, object]] = []

    result, is_terminal = _dispatch_tool(
        "submit_influence",
        {"score": 101},
        agent,
        pool,
        updates,
    )

    assert not is_terminal
    assert result == "Invalid: score must be an integer persuasiveness rating from 0 to 100"


def test_no_update_requires_unchanged_true() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={"C_01": 0.4})
    updates: list[dict[str, object]] = []

    result, is_terminal = _dispatch_tool(
        "no_update",
        {},
        agent,
        pool,
        updates,
    )

    assert not is_terminal
    assert result == "Invalid: no_update requires unchanged=true"


def test_done_reflecting_requires_done_true() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={"C_01": 0.4})
    updates: list[dict[str, object]] = []

    result, is_terminal = _dispatch_tool(
        "done_reflecting",
        {},
        agent,
        pool,
        updates,
    )

    assert not is_terminal
    assert result == "Invalid: done_reflecting requires done=true"


def test_coerce_text_tool_call_recovers_done_reflecting_braces() -> None:
    tool_calls = _coerce_text_tool_call(
        "done_reflecting{}",
        [
            {
                "type": "function",
                "function": {
                    "name": "done_reflecting",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert tool_calls == [
        {
            "type": "function",
            "function": {
                "name": "done_reflecting",
                "arguments": {"done": True},
            },
        }
    ]


def test_coerce_text_tool_call_recovers_done_reflecting_parentheses() -> None:
    tool_calls = _coerce_text_tool_call(
        "done_reflecting(done=true)",
        [
            {
                "type": "function",
                "function": {
                    "name": "done_reflecting",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert tool_calls == [
        {
            "type": "function",
            "function": {
                "name": "done_reflecting",
                "arguments": {"done": True},
            },
        }
    ]


def test_coerce_text_tool_call_recovers_no_update_braces() -> None:
    tool_calls = _coerce_text_tool_call(
        "no_update{}",
        [
            {
                "type": "function",
                "function": {
                    "name": "no_update",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert tool_calls == [
        {
            "type": "function",
            "function": {
                "name": "no_update",
                "arguments": {"unchanged": True},
            },
        }
    ]


def test_coerce_text_tool_call_recovers_no_update_parentheses() -> None:
    tool_calls = _coerce_text_tool_call(
        "no_update(unchanged=true)",
        [
            {
                "type": "function",
                "function": {
                    "name": "no_update",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert tool_calls == [
        {
            "type": "function",
            "function": {
                "name": "no_update",
                "arguments": {"unchanged": True},
            },
        }
    ]


def test_coerce_gemma_native_no_update_paren_wrapped_json() -> None:
    # Gemma-4 emits `call:name({...})` — JSON payload wrapped in parens. The
    # regex must accept the parens; otherwise a valid terminal tool call is
    # silently dropped and the run crashes on retry exhaustion (observed live).
    tool_calls = _coerce_text_tool_call(
        '<|tool_call>call:no_update({"unchanged":true})<tool_call|>',
        [
            {
                "type": "function",
                "function": {
                    "name": "no_update",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert tool_calls == [
        {
            "type": "function",
            "function": {"name": "no_update", "arguments": {"unchanged": True}},
        }
    ]


def test_coerce_gemma_native_update_weight_paren_wrapped_json() -> None:
    tool_calls = _coerce_text_tool_call(
        '<|tool_call>call:update_weight({"cid":"C_02","weight":0.8,"stance":"endorse"})<tool_call|>',
        [
            {
                "type": "function",
                "function": {
                    "name": "update_weight",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert tool_calls == [
        {
            "type": "function",
            "function": {
                "name": "update_weight",
                "arguments": {"cid": "C_02", "weight": 0.8, "stance": "endorse"},
            },
        }
    ]


def test_coerce_gemma_native_no_paren_still_parses() -> None:
    # The original brace-only form must keep working after the parens change.
    tool_calls = _coerce_text_tool_call(
        '<|tool_call>call:no_update{"unchanged":true}<tool_call|>',
        [
            {
                "type": "function",
                "function": {
                    "name": "no_update",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert tool_calls == [
        {
            "type": "function",
            "function": {"name": "no_update", "arguments": {"unchanged": True}},
        }
    ]


def test_coerce_text_tool_call_ignores_no_update_parenthesis_payload() -> None:
    tool_calls = _coerce_text_tool_call(
        "no_update(anything here is ignored)",
        [
            {
                "type": "function",
                "function": {
                    "name": "no_update",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert tool_calls == [
        {
            "type": "function",
            "function": {
                "name": "no_update",
                "arguments": {"unchanged": True},
            },
        }
    ]


def test_coerce_text_tool_call_recovers_submit_influence_score_fraction() -> None:
    tool_calls = _coerce_text_tool_call(
        "I would rate this argument 65/100 persuasive overall.",
        [
            {
                "type": "function",
                "function": {
                    "name": "submit_influence",
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    assert tool_calls == [
        {
            "type": "function",
            "function": {
                "name": "submit_influence",
                "arguments": {"score": 65},
            },
        }
    ]


def test_voice_user_msg_lists_positive_weight_options() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={"C_01": 0.4, "C_02": -0.8})

    msg = _voice_user_msg(agent, pool)

    assert "VOICE OPTIONS:" in msg
    assert "- C_01: Pro statement" in msg
    assert "C_02" not in msg
    assert "Choose exactly one cid from VOICE OPTIONS." in msg


def test_voice_user_msg_handles_empty_supported_list() -> None:
    pool = _build_pool()
    pool.add(Consideration(id="C_03", label="Counter statement", direction=1.0, persuasiveness=0.5))
    pool.attack_graph.add_attack("C_03", "C_02", 0.9)
    agent = Agent(id="A1", params=AgentParams(), weights={"C_01": -0.4, "C_02": -0.8})

    msg = _voice_user_msg(agent, pool)

    assert "You currently have no positive-weight statements in your repertoire." in msg
    assert "Your strongest opposed statement in the current pool is C_02: Con statement" in msg
    assert "Fallback rule: pick ONE statement from the current pool that strongly attacks that opposed statement." in msg
    assert "VOICE OPTIONS:" in msg
    assert "- C_03 (attacks C_02, strength=0.90): Counter statement" in msg
    assert "Choose exactly one cid from VOICE OPTIONS." in msg


class _VoiceAttackClient:
    async def chat(self, *args, **kwargs):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "submit_voice",
                        "arguments": {"cid": "C_03"},
                    },
                }
            ],
        }


class _EvaluateTextFallbackClient:
    async def chat(self, *args, **kwargs):
        return {
            "role": "assistant",
            "content": "After considering the argument and its conflicts with my views, I would score it 65/100.",
            "tool_calls": [],
        }


class _EvaluateRetryCaptureClient:
    def __init__(self) -> None:
        self.messages_seen = []
        self.calls = 0

    async def chat(self, messages, *args, **kwargs):
        self.calls += 1
        self.messages_seen.append([dict(message) for message in messages])
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "Please provide the argument you would like me to evaluate.",
                "tool_calls": [],
            }
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "submit_influence",
                        "arguments": {"score": 65},
                    },
                }
            ],
        }


class _EvaluateQueryBudgetClient:
    def __init__(self) -> None:
        self.calls = 0
        self.tools_seen = []
        self.messages_seen = []

    async def chat(self, messages, tools, *args, **kwargs):
        self.calls += 1
        self.tools_seen.append([tool["function"]["name"] for tool in tools])
        self.messages_seen.append([dict(message) for message in messages])
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "query_supports",
                        "arguments": {"target_cid": "C_01"},
                    },
                }
            ],
        }


class _ReflectManyUpdatesClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= 5:
            cid = f"C_0{self.calls}"
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "update_weight",
                            "arguments": {"cid": cid, "weight": 0.2, "stance": "endorse"},
                        },
                    }
                ],
            }
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "done_reflecting",
                        "arguments": {"done": True},
                    },
                }
            ],
        }


class _ReflectBudgetExhaustClient:
    def __init__(self) -> None:
        self.calls = 0
        self.messages_seen = []

    async def chat(self, messages, tools, *args, **kwargs):
        self.calls += 1
        self.messages_seen.append([dict(message) for message in messages])
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "update_weight",
                            "arguments": {"cid": "C_01", "weight": 0.2, "stance": "endorse"},
                        },
                    }
                ],
            }
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "query_supports",
                        "arguments": {"target_cid": "C_01"},
                    },
                }
            ],
        }


class _ReflectUpdateThenNoUpdateClient:
    def __init__(self) -> None:
        self.calls = 0
        self.messages_seen = []

    async def chat(self, messages, tools, *args, **kwargs):
        self.calls += 1
        self.messages_seen.append([dict(message) for message in messages])
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "update_weight",
                            "arguments": {"cid": "C_01", "weight": 0.6, "stance": "endorse"},
                        },
                    }
                ],
            }
        if self.calls == 2:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "no_update",
                            "arguments": {"unchanged": True},
                        },
                    }
                ],
            }
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "done_reflecting",
                        "arguments": {"done": True},
                    },
                }
            ],
        }


def test_voice_uses_attack_fallback_option_when_no_supported_statements() -> None:
    pool = _build_pool()
    pool.add(Consideration(id="C_03", label="Counter statement", direction=1.0, persuasiveness=0.5))
    pool.attack_graph.add_attack("C_03", "C_02", 0.9)
    agent = Agent(id="A1", params=AgentParams(), weights={"C_01": -0.4, "C_02": -0.8})
    harness = ToolCallHarness(client=_VoiceAttackClient())

    result = asyncio.run(harness.voice(agent, pool))

    assert result["cid"] == "C_03"
    assert result["_rounds"] == 1
    assert result["_tool_calls"] == [{"name": "submit_voice", "args": {"cid": "C_03"}}]


def test_evaluate_recovers_score_from_plain_text_fraction() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={"C_01": 0.4})
    harness = ToolCallHarness(client=_EvaluateTextFallbackClient())

    result = asyncio.run(harness.evaluate(agent, "C_01", 0.5, pool))

    assert result["score"] == 65
    assert result["_rounds"] == 1
    assert result["_tool_calls"] == [{"name": "submit_influence", "args": {"score": 65}}]


def test_evaluate_retry_restates_current_argument() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={"C_01": 0.4})
    client = _EvaluateRetryCaptureClient()
    harness = ToolCallHarness(client=client)

    result = asyncio.run(harness.evaluate(agent, "C_01", 0.5, pool))

    assert result["score"] == 65
    assert client.calls == 2
    retry_user_msg = client.messages_seen[1][-1]["content"]
    assert "ARGUMENT: C_01: Pro statement" in retry_user_msg
    assert "SPEAKER_OPINION: +0.500" in retry_user_msg
    assert "CALL submit_influence" in retry_user_msg


def test_evaluate_budget_exhaustion_raises() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={"C_01": 0.4})
    client = _EvaluateQueryBudgetClient()
    harness = ToolCallHarness(client=client)

    # No nondescript fallback score: exhausting the budget without a terminal
    # submit_influence must halt loudly with full attribution.
    with pytest.raises(HookLoopError) as exc:
        asyncio.run(harness.evaluate(agent, "C_01", 0.5, pool))

    assert exc.value.agent_id == "A1"
    assert exc.value.hook == "evaluate"
    assert "no terminal tool within budget" in str(exc.value)
    assert client.calls == 5
    assert client.tools_seen[0] == ["query_attacks", "query_supports", "submit_influence"]
    initial_user_msg = client.messages_seen[0][-1]["content"]
    assert "Tool budget:" in initial_user_msg
    assert "must call submit_influence" in initial_user_msg


def test_reflect_allows_one_update_per_heard_argument_plus_done() -> None:
    pool = _build_pool()
    pool.add(Consideration(id="C_03", label="Third statement", direction=1.0, persuasiveness=0.5))
    pool.add(Consideration(id="C_04", label="Fourth statement", direction=-1.0, persuasiveness=0.5))
    pool.add(Consideration(id="C_05", label="Fifth statement", direction=1.0, persuasiveness=0.5))
    agent = Agent(id="A1", params=AgentParams(), weights={})
    harness = ToolCallHarness(client=_ReflectManyUpdatesClient())
    round_updates = [
        ("C_01", 0.1, 50.0),
        ("C_02", -0.1, 50.0),
        ("C_03", 0.2, 50.0),
        ("C_04", -0.2, 50.0),
        ("C_05", 0.3, 50.0),
    ]

    result = asyncio.run(harness.reflect(agent, round_updates, pool))

    assert result["_rounds"] == 6
    assert len(result["updates"]) == 5
    assert result["updates"][-1]["cid"] == "C_05"


def test_reflect_budget_exhaustion_raises() -> None:
    pool = _build_pool()
    pool.add(Consideration(id="C_03", label="Third statement", direction=1.0, persuasiveness=0.5))
    pool.add(Consideration(id="C_04", label="Fourth statement", direction=-1.0, persuasiveness=0.5))
    pool.add(Consideration(id="C_05", label="Fifth statement", direction=1.0, persuasiveness=0.5))
    agent = Agent(id="A1", params=AgentParams(), weights={})
    client = _ReflectBudgetExhaustClient()
    harness = ToolCallHarness(client=client)
    round_updates = [
        ("C_01", 0.1, 50.0),
        ("C_02", -0.1, 50.0),
        ("C_03", 0.2, 50.0),
        ("C_04", -0.2, 50.0),
        ("C_05", 0.3, 50.0),
    ]

    # Partial updates without a terminal done_reflecting/no_update is not a
    # silent "move on" — it halts loudly so the dropped reflection is surfaced.
    with pytest.raises(HookLoopError) as exc:
        asyncio.run(harness.reflect(agent, round_updates, pool))

    assert exc.value.hook == "reflect"
    assert "no terminal tool within budget" in str(exc.value)
    assert client.calls == 6
    initial_user_msg = client.messages_seen[0][-1]["content"]
    assert "Tool budget:" in initial_user_msg
    assert "the run halts with an error" in initial_user_msg


def test_reflect_rejects_no_update_after_update_weight() -> None:
    pool = _build_pool()
    agent = Agent(id="A1", params=AgentParams(), weights={"C_01": 0.4})
    client = _ReflectUpdateThenNoUpdateClient()
    harness = ToolCallHarness(client=client)

    result = asyncio.run(harness.reflect(agent, [("C_01", 0.5, 80.0)], pool))

    assert client.calls == 3
    assert result["updates"] == [
        {
            "cid": "C_01",
            "weight": 0.6,
            "stance": "endorse",
            "new_weight": 0.6,
        }
    ]
    assert agent.weights["C_01"] == 0.6
    tool_messages = [m for m in result["_conversation"] if m.get("role") == "tool"]
    assert any(
        "no_update cannot be used after update_weight" in str(m.get("content", ""))
        for m in tool_messages
    )
