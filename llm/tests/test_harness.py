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


# ---------------------------------------------------------------------------
# Model-conditional harness profile.
#
# The guarantee under test is not "leniency works" but "E2B is untouched": the
# 300-run Arm-B collection was produced by the strict path, and a later cell is
# only comparable to it if that path is bit-for-bit the same code. So the
# central test asserts the new branch is unreachable under `strict`, not merely
# that it happens not to fire.
# ---------------------------------------------------------------------------

class _ReflectBareArgsClient:
    """Emits `{"unchanged": true}` as content, never as a tool call.

    This is E4B's observed behaviour in the model-scale pilot: semantically a
    correct "I am not revising anything", but the argument object without its
    tool-call envelope.
    """

    model = "gemma-4-E4B-it-Q8_0.gguf"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *args, **kwargs):
        self.calls += 1
        return {"role": "assistant", "content": '{"unchanged":true}'}


# reflect() returns early when nothing was voiced this round, so the loop is
# only exercised with a non-empty update list.
_ROUND_UPDATES = [("C_01", 0.2, 0.5)]


def _reflect_agent_and_pool():
    pool = _build_pool()
    agent = Agent(id="TH_02", params=AgentParams(), weights={"C_01": 0.2})
    return agent, pool


def test_profile_resolves_e2b_to_strict_and_others_to_lenient(monkeypatch) -> None:
    monkeypatch.delenv("LLM_HARNESS_PROFILE", raising=False)
    from llm.harness import resolve_harness_profile

    assert resolve_harness_profile("gemma-4-E2B-it-Q8_0.gguf") == "strict"
    assert resolve_harness_profile("gemma-4-E4B-it-Q8_0.gguf") == "lenient"
    assert resolve_harness_profile("gemma-4-26B-A4B-it-Q8_0.gguf") == "lenient"
    # No model information is not a licence to loosen anything.
    assert resolve_harness_profile("") == "strict"
    assert resolve_harness_profile(None) == "strict"


def test_profile_env_override_wins_both_directions(monkeypatch) -> None:
    from llm.harness import resolve_harness_profile

    monkeypatch.setenv("LLM_HARNESS_PROFILE", "strict")
    assert resolve_harness_profile("gemma-4-E4B-it-Q8_0.gguf") == "strict"
    monkeypatch.setenv("LLM_HARNESS_PROFILE", "lenient")
    assert resolve_harness_profile("gemma-4-E2B-it-Q8_0.gguf") == "lenient"
    # Garbage is ignored rather than silently loosening the harness.
    monkeypatch.setenv("LLM_HARNESS_PROFILE", "banana")
    assert resolve_harness_profile("gemma-4-E2B-it-Q8_0.gguf") == "strict"


def test_argshape_coercion_is_narrow() -> None:
    from llm.harness import _coerce_argshape_tool_call

    tools = [
        {"function": {"name": "no_update"}},
        {"function": {"name": "done_reflecting"}},
        {"function": {"name": "update_weight"}},
    ]
    assert _coerce_argshape_tool_call('{"unchanged":true}', tools)[0]["function"]["name"] == "no_update"
    assert _coerce_argshape_tool_call('{"done":true}', tools)[0]["function"]["name"] == "done_reflecting"
    # Everything else must be refused: extra keys, prose, empty, wrong keys,
    # and a tool that this hook does not offer.
    for bad in ('{"unchanged":true,"why":"x"}', "I leave my weights unchanged",
                "{}", '{"cid":"C_01"}', "not json", '[{"unchanged":true}]'):
        assert _coerce_argshape_tool_call(bad, tools) == []
    assert _coerce_argshape_tool_call('{"unchanged":true}',
                                      [{"function": {"name": "update_weight"}}]) == []


def test_strict_profile_never_reaches_the_lenient_branch(monkeypatch) -> None:
    """The E2B guarantee: under strict the new code is dead, not just quiet."""
    monkeypatch.delenv("LLM_HARNESS_PROFILE", raising=False)
    import llm.harness as H

    def _explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("lenient coercion reached from the strict path")

    monkeypatch.setattr(H, "_coerce_argshape_tool_call", _explode)

    client = _ReflectBareArgsClient()
    client.model = "gemma-4-E2B-it-Q8_0.gguf"
    harness = ToolCallHarness(client=client)
    assert harness.harness_profile == "strict"

    agent, pool = _reflect_agent_and_pool()
    # Same failure the pilot saw, and the same one the 300-run collection would
    # have seen: retries exhausted, run raised.
    with pytest.raises(HookLoopError) as exc:
        asyncio.run(harness.reflect(agent, _ROUND_UPDATES, pool))
    assert "no parseable tool call" in str(exc.value)


def test_lenient_profile_recovers_the_bare_argument_object(monkeypatch) -> None:
    monkeypatch.delenv("LLM_HARNESS_PROFILE", raising=False)
    client = _ReflectBareArgsClient()          # E4B -> lenient
    harness = ToolCallHarness(client=client)
    assert harness.harness_profile == "lenient"

    agent, pool = _reflect_agent_and_pool()
    before = dict(agent.weights)
    result = asyncio.run(harness.reflect(agent, _ROUND_UPDATES, pool))

    # Recovered on the FIRST response: no retry storm, no failure.
    assert client.calls == 1
    assert agent.weights == before, "no_update must not alter weights"
    names = [tc["name"] for tc in result.get("_tool_calls", [])]
    assert names == ["no_update"]


def test_explicit_profile_argument_beats_the_model_table() -> None:
    client = _ReflectBareArgsClient()          # E4B, would be lenient
    assert ToolCallHarness(client=client, harness_profile="strict").harness_profile == "strict"
    with pytest.raises(ValueError):
        ToolCallHarness(client=client, harness_profile="permissive")


class _ReflectNeverTerminatesClient:
    """Emits valid update_weight calls forever, never a terminal tool.

    The 26B pilot's failure mode: not malformed output, just more revisions
    than the E2B-tuned round budget allows.
    """

    model = "gemma-4-26B-A4B-it-Q8_0.gguf"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *args, **kwargs):
        self.calls += 1
        return {"role": "assistant", "tool_calls": [{
            "type": "function",
            "function": {"name": "update_weight",
                         "arguments": {"cid": "C_01", "weight": 0.5, "stance": "accept"}},
        }]}


def test_reflect_round_budget_is_profile_conditional(monkeypatch) -> None:
    """Strict keeps E2B's budget exactly; lenient buys slack for more revisions."""
    monkeypatch.delenv("LLM_HARNESS_PROFILE", raising=False)
    from llm.harness import MAX_TOOL_ROUNDS, _REFLECT_SLACK

    assert _REFLECT_SLACK["strict"] == 1, "the frozen E2B budget must not move"
    agent, pool = _reflect_agent_and_pool()
    updates = [("C_01", 0.2, 0.5), ("C_02", 0.1, 0.4), ("C_01", 0.3, 0.6),
               ("C_02", 0.2, 0.5), ("C_01", 0.4, 0.7), ("C_02", 0.3, 0.6)]

    strict_client = _ReflectNeverTerminatesClient()
    strict_client.model = "gemma-4-E2B-it-Q8_0.gguf"
    strict = ToolCallHarness(client=strict_client)
    with pytest.raises(HookLoopError):
        asyncio.run(strict.reflect(agent, updates, pool))
    assert strict_client.calls == max(MAX_TOOL_ROUNDS, len(updates) + 1)

    lenient_client = _ReflectNeverTerminatesClient()   # 26B -> lenient
    lenient = ToolCallHarness(client=lenient_client)
    with pytest.raises(HookLoopError):
        asyncio.run(lenient.reflect(agent, updates, pool))
    assert lenient_client.calls == max(MAX_TOOL_ROUNDS, len(updates) + 4)
    assert lenient_client.calls > strict_client.calls
