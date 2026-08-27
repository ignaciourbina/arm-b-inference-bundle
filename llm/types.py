"""Shared JSON and tool-calling types for the llm package."""

from __future__ import annotations

from typing import NotRequired, TypeAlias, TypedDict

JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONObject: TypeAlias = dict[str, JSONValue]
ObjectMap: TypeAlias = dict[str, object]
ToolArguments: TypeAlias = dict[str, object]
ToolSchema: TypeAlias = dict[str, object]
HookResult: TypeAlias = dict[str, object]


class ToolFunctionCall(TypedDict):
    name: str
    arguments: str | ToolArguments


class ToolCall(TypedDict, total=False):
    id: str
    type: str
    function: ToolFunctionCall


class ChatMessage(TypedDict, total=False):
    role: str
    content: JSONValue
    tool_calls: list[ToolCall]
    tool_call_id: str
    name: str


class TraceRecordDict(TypedDict):
    seq: int
    hook: str
    agent_id: str
    prompt: str
    response_raw: str
    response_parsed: ObjectMap
    prompt_tokens: int
    gen_tokens: int
    elapsed_s: float
    model: str
    error: str | None
    conversation: NotRequired[list[ChatMessage]]
    tool_calls_made: NotRequired[list[ToolCall]]
    loop_rounds: NotRequired[int]