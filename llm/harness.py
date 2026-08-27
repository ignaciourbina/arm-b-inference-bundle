"""Agentic tool-calling harness for LLM deliberation.

Runs an agentic loop: the LLM receives context via system prompt and
makes tool calls to query the argumentation graph and commit actions.
The loop runs until a terminal tool is called (submit_voice,
submit_influence, or done_reflecting).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from agora.agents import Agent  # type: ignore[import-untyped]
from agora.considerations import ArgumentPool  # type: ignore[import-untyped]

from .client import LLMClient
from .influence_scale import (
    INFLUENCE_LIKERT_MAX,
    INFLUENCE_LIKERT_MIN,
)
from .prompts import (
    BASELINE_PROMPT_BUILDER,
    BasePromptBuilder,
    PromptContext,
    STANDARD_PROMPT_BUILDER,
)
from .tools import EVALUATE_TOOLS, REFLECT_TOOLS, VOICE_TOOLS
from .types import ChatMessage, HookResult, ObjectMap, ToolArguments, ToolCall, ToolSchema

logger = logging.getLogger(__name__)

_INFLUENCE_SCORE_TEXT_RE = re.compile(r"\b(100|[1-9]?\d)\s*/\s*100\b")

TERMINAL_TOOLS = {"submit_voice", "submit_influence", "done_reflecting", "no_update"}
MAX_TOOL_ROUNDS = 5
MAX_EVALUATE_QUERY_ROUNDS = 2
# Give the model several chances to make an explicit tool decision before
# falling back to JSON. Silent no-update fallbacks are data-corrupting here:
# no update must be a model action, not a catch-all parser failure.
MAX_TEXT_RETRIES = 4


class HookLoopError(RuntimeError):
    """An agentic hook loop ended without a valid terminal tool call.

    Raised instead of returning a silent default. A nondescript fallback at the
    bottom of the loop (neutral score, empty updates, forced skip) is data-
    corrupting: it is indistinguishable from a real model decision. The run must
    halt loudly with full attribution so the failure is surfaced, fixed, and the
    deliberation resumed from the last checkpoint — never papered over.
    """

    def __init__(
        self, agent_id: str, hook: str, *, kind: str, attempts: int, detail: str = ""
    ) -> None:
        self.agent_id = agent_id
        self.hook = hook
        self.kind = kind
        self.attempts = attempts
        msg = f"{agent_id}/{hook}: {kind} after {attempts} attempts"
        if detail:
            msg = f"{msg}; {detail}"
        super().__init__(msg)


# Hook → required terminal tool + one-line schema hint, injected as a loud
# retry nudge when the model thinks but doesn't emit a tool call.
_RETRY_HINT = {
    "voice": (
        "submit_voice",
        'CALL submit_voice({"cid":"<CID>"}) NOW. '
        "Pick one cid from the VOICE OPTIONS in this message. "
        "Reply with ONLY the tool call, no prose.",
    ),
    "evaluate": (
        "submit_influence",
        f'CALL submit_influence({{"score":<integer {INFLUENCE_LIKERT_MIN}-{INFLUENCE_LIKERT_MAX}>}}) NOW. '
        "Reply with ONLY the tool call, no prose.",
    ),
    "reflect": (
        "done_reflecting/no_update",
        "CALL update_weight({\"cid\":\"<CID>\",\"weight\":<0.0-1.0>,\"stance\":\"endorse|reject\"}) for any changes, "
        "THEN CALL done_reflecting({\"done\":true}). "
        "If nothing should change, CALL no_update({\"unchanged\":true}). "
        "Reply with ONLY the tool calls, no prose.",
    ),
}


def _budget_note(hook: str, max_tool_rounds: int) -> str:
    if hook == "evaluate":
        return (
            f"Tool budget: you may use at most {MAX_EVALUATE_QUERY_ROUNDS} graph queries "
            "(query_attacks/query_supports) before you must call submit_influence. "
            "If you spend the budget without submitting a score, the run halts with an error — always submit a score."
        )
    if hook == "reflect":
        return (
            f"Tool budget: you have at most {max_tool_rounds} tool rounds total. "
            "Use any query/update calls you need, then finish with done_reflecting or no_update. "
            "If you spend the budget without finishing with a terminal tool, the run halts with an error — always finish."
        )
    if hook == "voice":
        return (
            f"Tool budget: you have at most {max_tool_rounds} tool rounds total. "
            "Choose and submit one voice option promptly. If you spend the budget without submitting, the run halts with an error."
        )
    return ""


def _voice_options(agent: Agent, pool: ArgumentPool) -> list[str]:
    """Return currently voiceable cids, strongest supported first."""
    options = [(cid, weight) for cid, weight in agent.weights.items() if weight > 0.0]
    options.sort(key=lambda item: (-abs(item[1]), item[0]))
    return [cid for cid, _ in options if cid in pool.considerations]


def _voice_attack_fallback_options(
    agent: Agent,
    pool: ArgumentPool,
    *,
    max_suggestions: int = 3,
) -> tuple[str | None, list[tuple[str, str, float]]]:
    """Return pool-local attackers of the statement the agent most strongly opposes."""
    opposed = [
        (cid, weight)
        for cid, weight in agent.weights.items()
        if weight < 0.0 and cid in pool.considerations
    ]
    if not opposed:
        return None, []

    target_cid, _ = min(opposed, key=lambda item: (item[1], item[0]))
    attackers = [
        (attacker_id, pool.get(attacker_id).label.strip().replace("\n", " "), strength)
        for attacker_id, strength in pool.attack_graph.get_attackers(target_cid).items()
        if attacker_id in pool.considerations and attacker_id not in agent.weights
    ]
    attackers.sort(key=lambda item: (-item[2], item[0]))
    return target_cid, attackers[:max_suggestions]


def _voice_user_msg(agent: Agent, pool: ArgumentPool) -> str:
    """Construct a compact user-turn reminder with explicit voiceable cids."""
    options = _voice_options(agent, pool)
    if not options:
        target_cid, suggestions = _voice_attack_fallback_options(agent, pool)
        lines = [
            "You currently have no positive-weight statements in your repertoire.",
        ]
        if target_cid is not None:
            target_label = pool.get(target_cid).label.strip().replace("\n", " ")
            lines.append(
                f"Your strongest opposed statement in the current pool is {target_cid}: {target_label}"
            )
            if suggestions:
                lines.append(
                    "Fallback rule: pick ONE statement from the current pool that strongly attacks that opposed statement."
                )
                lines.append("VOICE OPTIONS:")
                lines.append(
                    "These attack options come from the shared pool, so the underlying information set stays fixed:"
                )
                for cid, label, strength in suggestions:
                    lines.append(f"- {cid} (attacks {target_cid}, strength={strength:.2f}): {label}")
                lines.append("Choose exactly one cid from VOICE OPTIONS.")
            else:
                lines.append(
                    "No attack-based voice options are available from the current pool for that statement."
                )
        return "\n".join(lines)

    lines = [
        "Choose a consideration to voice and call submit_voice.",
        "VOICE OPTIONS:",
    ]
    for cid in options:
        label = pool.get(cid).label.strip().replace("\n", " ")
        lines.append(f"- {cid}: {label}")
    lines.append("Choose exactly one cid from VOICE OPTIONS.")
    return "\n".join(lines)


# Gemma-4 emits `<|tool_call>call:name{...}<tool_call|>`, but depending on
# sampling it wraps the brace payload in parentheses — `call:name({...})`. Accept
# either form; group(2) is always the inner text between the braces.
_GEMMA_TOOL_CALL_RE = re.compile(
    r'<\|tool_call\>call:([a-z_][a-z0-9_]*)\(?\{(.*?)\}\)?<tool_call\|>',
    re.DOTALL,
)


def _parse_gemma_tool_args(raw_args: str) -> dict | None:
    """Parse a Gemma-4 tool-call payload (the text between the braces).

    Gemma-4 emits two payload dialects for this model:
      - standard JSON:  "cid":"C_02","weight":0.8,"stance":"endorse"
      - native kv:      key:<|"|>value<|"|>,key2:number
    Try JSON first (the common case), then fall back to the native delimiters.
    """
    raw = raw_args.strip()
    if not raw:
        return {}
    try:
        parsed = json.loads("{" + raw + "}")
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    args: dict = {}
    rest = raw
    while rest:
        key_match = re.match(r'([a-z_][a-z0-9_]*)\s*:\s*', rest)
        if not key_match:
            break
        key = key_match.group(1)
        rest = rest[key_match.end():]
        if rest.startswith('<|"|>'):
            rest = rest[5:]
            end = rest.find('<|"|>')
            if end == -1:
                args[key] = rest
                break
            args[key] = rest[:end]
            rest = rest[end + 5:]
        else:
            num_match = re.match(r'(-?[\d.]+)', rest)
            if num_match:
                val = num_match.group(1)
                args[key] = int(val) if val.isdigit() or (val.startswith('-') and val[1:].isdigit()) else float(val)
                rest = rest[num_match.end():]
            elif rest.startswith('true'):
                args[key] = True
                rest = rest[4:]
            elif rest.startswith('false'):
                args[key] = False
                rest = rest[5:]
            else:
                break
        rest = rest.lstrip(',').strip()
    return args


def _coerce_text_tool_call(content: str, tools: list[ToolSchema]) -> list[ToolCall]:
    """Parse bare tool-like text such as done_reflecting{} into tool calls.

    This is intentionally narrow: it only accepts content that is just a single
    tool-shaped expression with no surrounding prose.
    Supports Gemma-4 native format: <|tool_call>call:name{args}<tool_call|>
    """
    text = content.strip()
    if not text:
        return []

    allowed_names = {
        cast(str, tool["function"]["name"])
        for tool in tools
    }

    gemma_match = _GEMMA_TOOL_CALL_RE.search(text)
    if gemma_match:
        name = gemma_match.group(1)
        if name in allowed_names:
            args = _parse_gemma_tool_args(gemma_match.group(2))
            if args is not None:
                if name == "done_reflecting":
                    args = {"done": True}
                elif name == "no_update":
                    args = {"unchanged": True}
                return [{"type": "function", "function": {"name": name, "arguments": args}}]

    if "submit_influence" in allowed_names:
        score_match = _INFLUENCE_SCORE_TEXT_RE.search(text)
        if score_match is not None:
            score = _parse_influence_likert(int(score_match.group(1)))
            if score is not None:
                return [{
                    "type": "function",
                    "function": {
                        "name": "submit_influence",
                        "arguments": {"score": score},
                    },
                }]

    match = re.fullmatch(r"([a-z_][a-z0-9_]*)\s*(?:\((.*)\)|(\{.*\}))", text, re.DOTALL)
    if match is None:
        return []

    name = match.group(1)
    if name not in allowed_names:
        return []

    paren_payload = match.group(2)
    payload = paren_payload if paren_payload is not None else match.group(3)
    payload = payload.strip() if payload is not None else ""
    if not payload:
        payload = "{}"

    if name == "done_reflecting" and (payload == "{}" or paren_payload is not None):
        args: ToolArguments = {"done": True}
    elif name == "no_update" and (payload == "{}" or paren_payload is not None):
        args = {"unchanged": True}
    else:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, dict):
            return []
        args = cast(ToolArguments, parsed)

    return [{"type": "function", "function": {"name": name, "arguments": args}}]


def _parse_weight_update_args(
    args: ToolArguments,
) -> tuple[float, str, float, float] | None:
    """Validate the strict update_weight contract and return normalized values."""
    weight = args.get("weight")
    if not isinstance(weight, (int, float)):
        return None
    stance = args.get("stance")
    if stance == "endorse":
        sign = 1
    elif stance == "reject":
        sign = -1
    else:
        return None
    submitted_weight = float(weight)
    if not np.isfinite(submitted_weight):
        return None
    weight_value = float(np.clip(submitted_weight, 0.0, 1.0))
    signed_weight = float(weight_value * sign)
    return weight_value, stance, signed_weight, submitted_weight


def _parse_influence_likert(score: object) -> int | None:
    """Return a strict integer persuasiveness score, or None if invalid."""
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    score_float = float(score)
    if not np.isfinite(score_float):
        return None
    score_int = int(score_float)
    if score_float != float(score_int):
        return None
    if not INFLUENCE_LIKERT_MIN <= score_int <= INFLUENCE_LIKERT_MAX:
        return None
    return score_int


def _dispatch_tool(
    name: str,
    args: ToolArguments,
    agent: Agent,
    pool: ArgumentPool,
    weight_updates: list[ObjectMap],
    allowed_voice_cids: set[str] | None = None,
) -> tuple[str, bool]:
    """Dispatch a tool call. Returns (result_text, is_terminal)."""

    if name == "query_attacks":
        target = args.get("target_cid", "")
        attackers = pool.attack_graph.get_attackers(target)
        if not attackers:
            return "No attacks found on this consideration.", False
        atk_list = [
            {
                "attacker_id": aid,
                "label": pool.get(aid).label if aid in pool.considerations else "?",
                "strength": round(s, 2),
            }
            for aid, s in attackers.items()
        ]
        return json.dumps(atk_list), False

    elif name == "query_supports":
        target = args.get("target_cid", "")
        supporters = pool.attack_graph.get_supporters(target)
        if not supporters:
            return "No supports found for this consideration.", False
        sup_list = [
            {
                "supporter_id": sid,
                "label": pool.get(sid).label if sid in pool.considerations else "?",
                "strength": round(s, 2),
            }
            for sid, s in supporters.items()
        ]
        return json.dumps(sup_list), False

    elif name == "query_repertoire":
        rep = {
            "opinion": round(agent.opinion(pool), 4),
            "weights": {cid: round(w, 4) for cid, w in agent.weights.items()},
        }
        return json.dumps(rep), False

    elif name == "submit_voice":
        cid = args.get("cid", "")
        if not isinstance(cid, str) or cid not in pool.considerations:
            return "Invalid: cid must reference a known consideration", False
        if agent.weights.get(cid, 0.0) <= 0.0 and (not allowed_voice_cids or cid not in allowed_voice_cids):
            return (
                "Invalid: submit_voice must reference a currently supported statement "
                "or a listed attack fallback option"
            ), False
        return "", True

    elif name == "submit_influence":
        score = _parse_influence_likert(args.get("score"))
        if score is None:
            return (
                f"Invalid: score must be an integer persuasiveness rating from "
                f"{INFLUENCE_LIKERT_MIN} to {INFLUENCE_LIKERT_MAX}"
            ), False
        args["score"] = score
        return "", True

    elif name == "no_update":
        if args.get("unchanged") is not True:
            return "Invalid: no_update requires unchanged=true", False
        if weight_updates:
            return (
                "Invalid: no_update cannot be used after update_weight in the same reflection. "
                "Call done_reflecting({\"done\":true}) instead.",
                False,
            )
        return "", True

    elif name == "update_weight":
        cid = args.get("cid", "")
        pool_ids = set(pool.all_ids())
        if cid not in pool_ids:
            return f"Invalid: {cid} not in argument pool", False
        parsed = _parse_weight_update_args(args)
        if parsed is None:
            return (
                "Invalid: update_weight requires numeric weight in [0, 1] "
                "and stance in {'endorse', 'reject'}",
                False,
            )
        weight_value, stance, new_w, submitted_weight = parsed
        if weight_value == 0.0:
            removed = cid in agent.weights
            agent.weights.pop(cid, None)
            entry = {
                "cid": cid,
                "weight": weight_value,
                "stance": stance,
                "new_weight": 0.0,
                "_removed": removed,
            }
            if submitted_weight != weight_value:
                entry["submitted_weight"] = submitted_weight
            weight_updates.append(entry)
            return (
                (
                    f"Removed {cid} from repertoire. "
                    "If you are finished, call done_reflecting({\"done\":true}) now. "
                    "Otherwise make one more update_weight call."
                )
                if removed else
                (
                    f"No-op: {cid} is not in repertoire. "
                    "If you are finished, call done_reflecting({\"done\":true}) now. "
                    "Otherwise make one more update_weight call."
                ),
                False,
            )

        adopted = cid not in agent.weights
        agent.weights[cid] = new_w
        entry = {
            "cid": cid,
            "weight": weight_value,
            "stance": stance,
            "new_weight": new_w,
        }
        if submitted_weight != weight_value:
            entry["submitted_weight"] = submitted_weight
        if adopted:
            entry["_adopted"] = True
        weight_updates.append(entry)
        verb = "Adopted" if adopted else "Updated"
        return (
            f"{verb} {cid} to {agent.weights[cid]:+.3f} "
            f"(weight={weight_value:.3f}, stance={stance}). "
            "If you are finished, call done_reflecting({\"done\":true}) now. "
            "Otherwise make one more update_weight call.",
            False,
        )

    elif name == "done_reflecting":
        if args.get("done") is not True:
            return "Invalid: done_reflecting requires done=true", False
        return "", True

    else:
        return f"Unknown tool: {name}", False


@dataclass
class ToolCallHarness:
    """Runs an agentic tool-calling loop against a local LLM."""

    client: LLMClient = field(default_factory=LLMClient)
    confirmation_bias: float = 0.0
    facilitated: bool = False
    group_building_level: int = 3
    prompt_builder: BasePromptBuilder = field(default_factory=lambda: STANDARD_PROMPT_BUILDER)

    def _build_prompt_context(self, agent: Agent) -> PromptContext:
        return PromptContext(
            confirmation_bias=self.confirmation_bias,
            facilitated=self.facilitated,
            group_building_level=self.group_building_level,
        )

    async def _run_loop(
        self,
        agent: Agent,
        pool: ArgumentPool,
        hook: str,
        system_prompt: str,
        tools: list[ToolSchema],
        user_msg: str = "Proceed.",
        allowed_voice_cids: set[str] | None = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> HookResult:
        """Run the agentic loop until a terminal tool is called."""
        budget_note = _budget_note(hook, max_tool_rounds)
        initial_user_msg = user_msg if not budget_note else f"{user_msg}\n\n{budget_note}"
        messages: list[ChatMessage] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_msg},
        ]

        result: HookResult = {}
        weight_updates: list[ObjectMap] = []
        tool_call_log: list[ObjectMap] = []
        text_retries = 0

        for round_idx in range(max_tool_rounds):
            msg = await self.client.chat(
                messages, tools, hook=hook, agent_id=agent.id,
            )

            tool_calls = cast(list[ToolCall], msg.get("tool_calls", []))
            if not tool_calls:
                content_value = msg.get("content", "")
                content = content_value if isinstance(content_value, str) else ""
                tool_calls = _coerce_text_tool_call(content, tools)
                if tool_calls:
                    logger.info(
                        "Recovered pseudo-tool call from plain text for %s/%s: %s",
                        agent.id,
                        hook,
                        content.strip(),
                    )

            if not tool_calls:
                text_retries += 1
                content = msg.get("content", "")
                content_preview = content[:100] if isinstance(content, str) else str(content)[:100]
                logger.warning(
                    "No tool call in round %d for %s/%s (retry %d/%d): %s",
                    round_idx, agent.id, hook, text_retries, MAX_TEXT_RETRIES, content_preview,
                )
                if text_retries > MAX_TEXT_RETRIES:
                    raise HookLoopError(
                        agent.id, hook,
                        kind="no parseable tool call",
                        attempts=text_retries,
                        detail=f"last content: {content_preview!r}",
                    )
                # Loud retry: spell out the exact tool call required.
                messages.append(msg)
                _, hint = _RETRY_HINT.get(hook, ("", "Use one of the available tools to proceed."))
                if hook == "voice":
                    hint = f"{hint}\n\n{_voice_user_msg(agent, pool)}"
                elif hook == "evaluate":
                    hint = f"{user_msg}\n\n{hint}"
                messages.append({"role": "user", "content": hint})
                continue

            # Process tool calls — handle all non-terminal, then check terminal
            messages.append(msg)
            terminal_found = False

            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", {})
                # OpenAI spec: arguments is a JSON string. Ollama returns a dict.
                if isinstance(raw_args, str):
                    try:
                        args = cast(ToolArguments, json.loads(raw_args) if raw_args else {})
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = cast(ToolArguments, raw_args or {})
                tc_id = tc.get("id", "")
                tool_call_log.append({"name": name, "args": args})

                tool_result, is_terminal = _dispatch_tool(
                    name, args, agent, pool, weight_updates, allowed_voice_cids,
                )

                if is_terminal:
                    if name == "submit_voice":
                        result = {"cid": args.get("cid", "")}
                    elif name == "submit_influence":
                        result = {"score": args["score"]}
                    elif name == "done_reflecting":
                        result = {"updates": weight_updates}
                    elif name == "no_update":
                        result = {
                            "updates": [],
                            "no_update": True,
                        }
                    terminal_found = True
                    break
                else:
                    tool_msg: ChatMessage = {"role": "tool", "content": tool_result}
                    if tc_id:
                        tool_msg["tool_call_id"] = tc_id
                    messages.append(tool_msg)

            if terminal_found:
                break

        if not result:
            # No nondescript fallback: a terminal tool is mandatory. Halt loudly
            # with full attribution rather than fabricate a neutral score / empty
            # update / forced skip that is indistinguishable from a real decision.
            raise HookLoopError(
                agent.id, hook,
                kind="no terminal tool within budget",
                attempts=max_tool_rounds,
                detail=f"non-terminal tool calls so far: {tool_call_log}",
            )

        result["_tool_calls"] = tool_call_log
        result["_rounds"] = round_idx + 1
        result["_conversation"] = messages
        return result

    async def voice(self, agent: Agent, pool: ArgumentPool) -> HookResult:
        system = self.prompt_builder.build_voice(
            agent,
            pool,
            context=self._build_prompt_context(agent),
        )
        user_msg = _voice_user_msg(agent, pool)
        allowed_voice_cids = set(_voice_options(agent, pool))
        if not allowed_voice_cids:
            _, suggestions = _voice_attack_fallback_options(agent, pool)
            allowed_voice_cids = {cid for cid, _, _ in suggestions}
            if not allowed_voice_cids:
                return {
                    "skip": True,
                    "reason": user_msg,
                    "_tool_calls": [],
                    "_rounds": 0,
                    "_conversation": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                }
        return await self._run_loop(
            agent, pool, "voice", system, VOICE_TOOLS,
            user_msg=user_msg,
            allowed_voice_cids=allowed_voice_cids,
        )

    async def evaluate(
        self, agent: Agent, cid: str, speaker_opinion: float, pool: ArgumentPool,
    ) -> HookResult:
        consideration = pool.get(cid)
        system = self.prompt_builder.build_evaluate(
            agent,
            cid,
            speaker_opinion,
            pool,
            context=self._build_prompt_context(agent),
        )
        return await self._run_loop(
            agent, pool, "evaluate", system, EVALUATE_TOOLS,
            user_msg=(
                "Evaluate the incoming argument and call submit_influence with your score.\n"
                f"ARGUMENT: {cid}: {consideration.label}\n"
                f"SPEAKER_OPINION: {speaker_opinion:+.3f}"
            ),
        )

    async def reflect(
        self, agent: Agent, round_updates: list[tuple[str, float, float]], pool: ArgumentPool,
    ) -> HookResult:
        if not round_updates:
            return {"updates": []}
        system = self.prompt_builder.build_reflect(
            agent,
            round_updates,
            pool,
            context=self._build_prompt_context(agent),
        )
        return await self._run_loop(
            agent, pool, "reflect", system, REFLECT_TOOLS,
            user_msg=(
                "Update your weights using update_weight(cid, weight, stance), "
                "then call done_reflecting."
            ),
            max_tool_rounds=max(MAX_TOOL_ROUNDS, len(round_updates) + 1),
        )


@dataclass
class BaselineHarness(ToolCallHarness):
    """Harness with baseline prompt framing."""

    prompt_builder: BasePromptBuilder = field(default_factory=lambda: BASELINE_PROMPT_BUILDER)
    topic_description: str = ""

    def _build_prompt_context(self, agent: Agent) -> PromptContext:
        return PromptContext(
            topic_description=self.topic_description,
            confirmation_bias=self.confirmation_bias,
            facilitated=self.facilitated,
            group_building_level=self.group_building_level,
        )


