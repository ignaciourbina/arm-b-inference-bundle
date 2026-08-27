"""Tool definitions for the LLM agentic deliberation engine.

Each tool maps to a capability the LLM agent can invoke during
voice, evaluate, or reflect hooks. Tools are defined in the
OpenAI-compatible tool-calling format used by llama-server.
"""

from __future__ import annotations

from .influence_scale import (
    INFLUENCE_LIKERT_GUIDANCE,
    INFLUENCE_LIKERT_MAX,
    INFLUENCE_LIKERT_MIN,
    INFLUENCE_LIKERT_SCALE_TEXT,
)
from .types import ToolSchema


REASONING_PROPERTY = {
    "type": "string",
    "description": (
        "Optional brief justification for this tool call. Prefer 50 words or fewer."
    ),
}


def _with_optional_reasoning(properties: dict[str, object]) -> dict[str, object]:
    return {
        **properties,
        "reasoning": REASONING_PROPERTY,
    }

# --- Tool schemas (sent to the LLM) ---

QUERY_ATTACKS: ToolSchema = {
    "type": "function",
    "function": {
        "name": "query_attacks",
        "description": (
            "Get all considerations that attack a target consideration. "
            "Returns a list of {attacker_id, attacker_label, strength}. "
            "Use this to judge whether an argument is undermined."
        ),
        "parameters": {
            "type": "object",
            "properties": _with_optional_reasoning({
                "target_cid": {
                    "type": "string",
                    "description": "The consideration ID to query attacks for",
                },
            }),
            "required": ["target_cid"],
        },
    },
}

SUBMIT_VOICE: ToolSchema = {
    "type": "function",
    "function": {
        "name": "submit_voice",
        "description": (
            "Commit a statement YOU CURRENTLY SUPPORT as your contribution "
            "to the debate. You are advocating for that statement. Submit "
            "only statements you support, not ones you currently disagree with. "
            "This ends your turn."
        ),
        "parameters": {
            "type": "object",
            "properties": _with_optional_reasoning({
                "cid": {
                    "type": "string",
                    "description": "The consideration ID to voice",
                },
            }),
            "required": ["cid"],
        },
    },
}

SUBMIT_INFLUENCE: ToolSchema = {
    "type": "function",
    "function": {
        "name": "submit_influence",
        "description": (
            "Submit your persuasiveness score for the argument you just heard "
            f"on a {INFLUENCE_LIKERT_SCALE_TEXT}. {INFLUENCE_LIKERT_GUIDANCE} "
            "Consider the argument's strength, whether it is attacked, "
            "and how it relates to your stance."
        ),
        "parameters": {
            "type": "object",
            "properties": _with_optional_reasoning({
                "score": {
                    "type": "integer",
                    "minimum": INFLUENCE_LIKERT_MIN,
                    "maximum": INFLUENCE_LIKERT_MAX,
                    "description": (
                        f"Persuasiveness score on the {INFLUENCE_LIKERT_SCALE_TEXT}"
                    ),
                },
            }),
            "required": ["score"],
        },
    },
}

UPDATE_WEIGHT: ToolSchema = {
    "type": "function",
    "function": {
        "name": "update_weight",
        "description": (
            "Update your stance toward a single consideration. "
            "Call this once per consideration you want to change. "
            "Use weight for stance strength in [0, 1] and stance to say whether "
            "you endorse or reject the statement itself."
        ),
        "parameters": {
            "type": "object",
            "properties": _with_optional_reasoning({
                "cid": {
                    "type": "string",
                    "description": "The consideration ID to update",
                },
                "weight": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": (
                        "Stance strength. Values are clipped into [0, 1]; "
                        "0 removes this consideration from your repertoire."
                    ),
                },
                "stance": {
                    "type": "string",
                    "enum": ["endorse", "reject"],
                    "description": (
                        "Statement-level stance. endorse = you accept/support "
                        "this statement; reject = you oppose/disagree with this statement."
                    ),
                },
            }),
            "required": ["cid", "weight", "stance"],
        },
    },
}

NO_UPDATE: ToolSchema = {
    "type": "function",
    "function": {
        "name": "no_update",
        "description": (
            "Explicitly state that you do not want to change any consideration "
            "weights after hearing this round. Use this only after deciding the "
            "arguments did not change your endorsement or rejection strengths."
        ),
        "parameters": {
            "type": "object",
            "properties": _with_optional_reasoning({
                "unchanged": {
                    "type": "boolean",
                    "description": "Set to true to confirm that no weights changed",
                },
            }),
            "required": ["unchanged"],
        },
    },
}

DONE_REFLECTING: ToolSchema = {
    "type": "function",
    "function": {
        "name": "done_reflecting",
        "description": (
            "Signal that you are done after making one or more update_weight "
            "calls. If you made no changes, prefer no_update instead."
        ),
        "parameters": {
            "type": "object",
            "properties": _with_optional_reasoning({
                "done": {
                    "type": "boolean",
                    "description": "Set to true to confirm you are finished updating weights",
                },
            }),
            "required": ["done"],
        },
    },
}

QUERY_SUPPORTS: ToolSchema = {
    "type": "function",
    "function": {
        "name": "query_supports",
        "description": (
            "Get all considerations that support a target consideration. "
            "Returns a list of {supporter_id, supporter_label, strength}. "
            "Use this to find arguments that reinforce a position."
        ),
        "parameters": {
            "type": "object",
            "properties": _with_optional_reasoning({
                "target_cid": {
                    "type": "string",
                    "description": "The consideration ID to query supports for",
                },
            }),
            "required": ["target_cid"],
        },
    },
}

QUERY_REPERTOIRE: ToolSchema = {
    "type": "function",
    "function": {
        "name": "query_repertoire",
        "description": (
            "Re-check your current consideration weights and opinion. "
            "Useful mid-reflection after updating some weights to see "
            "your current state before deciding further changes."
        ),
        "parameters": {
            "type": "object",
            "properties": _with_optional_reasoning({}),
        },
    },
}

# --- Tool sets per hook ---

VOICE_TOOLS: list[ToolSchema] = [QUERY_ATTACKS, QUERY_SUPPORTS, SUBMIT_VOICE]
EVALUATE_TOOLS: list[ToolSchema] = [QUERY_ATTACKS, QUERY_SUPPORTS, SUBMIT_INFLUENCE]
REFLECT_TOOLS: list[ToolSchema] = [
    QUERY_ATTACKS,
    QUERY_SUPPORTS,
    QUERY_REPERTOIRE,
    UPDATE_WEIGHT,
    NO_UPDATE,
    DONE_REFLECTING,
]
