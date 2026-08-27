import json

from llm.client import _build_chat_trace_record


def test_chat_trace_record_includes_full_conversation_history() -> None:
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "first user turn"},
        {"role": "assistant", "content": "tool thought"},
        {"role": "user", "content": "retry hint"},
    ]
    response = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "submit_voice",
                    "arguments": '{"cid":"C_02"}',
                },
            }
        ],
    }

    record = _build_chat_trace_record(
        seq=1,
        hook="voice",
        agent_id="TH_01",
        messages=messages,
        msg=response,
        tool_calls=response["tool_calls"],
        prompt_tokens=123,
        gen_tokens=7,
        elapsed_s=0.4,
        model="test-model",
    )

    data = record.to_dict()

    assert data["conversation"] == messages
    assert json.loads(data["prompt"]) == messages
    assert json.loads(data["response_raw"])["tool_calls"][0]["function"]["name"] == "submit_voice"


def test_chat_trace_record_freezes_messages() -> None:
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "first user turn"},
    ]

    record = _build_chat_trace_record(
        seq=1,
        hook="voice",
        agent_id="TH_01",
        messages=messages,
        msg={"role": "assistant", "content": ""},
        tool_calls=[],
        prompt_tokens=1,
        gen_tokens=1,
        elapsed_s=0.1,
        model="test-model",
    )

    messages.append({"role": "user", "content": "mutated later"})

    assert len(record.conversation or []) == 2
    assert json.loads(record.prompt) == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "first user turn"},
    ]