from llm.tools import (
    DONE_REFLECTING,
    EVALUATE_TOOLS,
    NO_UPDATE,
    QUERY_ATTACKS,
    QUERY_REPERTOIRE,
    QUERY_SUPPORTS,
    REFLECT_TOOLS,
    SUBMIT_INFLUENCE,
    SUBMIT_VOICE,
    UPDATE_WEIGHT,
    VOICE_TOOLS,
)


def _reasoning_property(tool: dict[str, object]) -> dict[str, object]:
    function = tool["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    reasoning = properties["reasoning"]
    assert isinstance(reasoning, dict)
    return reasoning


def test_all_tool_schemas_accept_optional_reasoning() -> None:
    tools = [
        QUERY_ATTACKS,
        QUERY_SUPPORTS,
        QUERY_REPERTOIRE,
        SUBMIT_VOICE,
        SUBMIT_INFLUENCE,
        UPDATE_WEIGHT,
        NO_UPDATE,
        DONE_REFLECTING,
    ]

    for tool in tools:
        reasoning = _reasoning_property(tool)
        assert reasoning["type"] == "string"
        assert "50 words or fewer" in reasoning["description"]


def test_reasoning_is_not_required_for_any_tool_set() -> None:
    for tool_set in (VOICE_TOOLS, EVALUATE_TOOLS, REFLECT_TOOLS):
        for tool in tool_set:
            function = tool["function"]
            assert isinstance(function, dict)
            parameters = function["parameters"]
            assert isinstance(parameters, dict)
            required = parameters.get("required", [])
            assert isinstance(required, list)
            assert "reasoning" not in required


def test_submit_influence_schema_uses_0_to_100_scale() -> None:
    function = SUBMIT_INFLUENCE["function"]
    assert isinstance(function, dict)
    description = function["description"]
    assert isinstance(description, str)
    assert "0-100 persuasiveness scale (min=0, max=100)" in description

    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    score = properties["score"]
    assert isinstance(score, dict)
    assert score["minimum"] == 0
    assert score["maximum"] == 100