"""System prompt builders for the agentic LLM deliberation engine.

Instead of mathematical corrections, cognitive characteristics (confirmation
bias, open-mindedness, facilitation effects) are injected as personality
framing. The LLM exhibits these naturally through reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .influence_scale import (
    INFLUENCE_LIKERT_GUIDANCE,
    INFLUENCE_LIKERT_MAX,
    INFLUENCE_LIKERT_MIN,
    INFLUENCE_LIKERT_SCALE_TEXT,
    format_influence_likert,
)

if TYPE_CHECKING:
    from agora.agents import Agent  # type: ignore[import-untyped]
    from agora.considerations import ArgumentPool  # type: ignore[import-untyped]


WEIGHT_THRESHOLD = 0.05
OPINION_LEAN_THRESHOLD = 0.3


def _support_label(weight: float) -> str:
    if weight > WEIGHT_THRESHOLD:
        return "SUPPORT"
    if weight < -WEIGHT_THRESHOLD:
        return "DISAGREE"
    return "UNSURE"


def _policy_side_label(direction: float) -> str:
    if direction > 0:
        return "PRO"
    if direction < 0:
        return "CON"
    return "MIXED"


def _format_likert_score(score: float) -> str:
    return format_influence_likert(score)


def _format_opinion_score(score: float) -> str:
    return f"{score:+.3f} (min=-1, max=+1)"


def _format_statement_line(cid: str, weight: float, consideration) -> str:
    return (
        f"  {cid} [strength={abs(weight):.3f} dir={int(consideration.direction):+d} "
        f"{_policy_side_label(consideration.direction)}]: "
        f"\"{consideration.label}\""
    )


def _bucketed_repertoire(agent: Agent, pool: ArgumentPool) -> tuple[list[str], list[str], list[str]]:
    support_lines: list[str] = []
    disagree_lines: list[str] = []
    unsure_lines: list[str] = []

    items = [(cid, weight, pool.get(cid)) for cid, weight in agent.weights.items()]
    items.sort(key=lambda item: (-abs(item[1]), item[0]))

    for cid, weight, consideration in items:
        line = _format_statement_line(cid, weight, consideration)
        if weight > WEIGHT_THRESHOLD:
            support_lines.append(line)
        elif weight < -WEIGHT_THRESHOLD:
            disagree_lines.append(line)
        else:
            unsure_lines.append(line)

    return support_lines, disagree_lines, unsure_lines


def _format_repertoire(agent: Agent, pool: ArgumentPool) -> str:
    support_lines, disagree_lines, unsure_lines = _bucketed_repertoire(agent, pool)
    blocks = [
        "STATEMENTS YOU CURRENTLY SUPPORT:",
        "\n".join(support_lines) if support_lines else "  (none)",
        "",
        "STATEMENTS YOU CURRENTLY DISAGREE WITH:",
        "\n".join(disagree_lines) if disagree_lines else "  (none)",
    ]
    if unsure_lines:
        blocks += ["", "STATEMENTS YOU FEEL UNSURE ABOUT:", "\n".join(unsure_lines)]
    return "\n".join(blocks)


def _format_supported_repertoire(agent: Agent, pool: ArgumentPool) -> str:
    """Only statements the agent currently supports strongly enough to voice."""
    support_lines, _, _ = _bucketed_repertoire(agent, pool)
    return "\n".join(support_lines) if support_lines else "  (no currently supported statements strong enough to voice)"


def _cognitive_style_block(
    agent: Agent,
    confirmation_bias: float = 0.0,
    facilitated: bool = False,
    group_building_level: int = 3,
) -> str:
    """Build cognitive style description based on engine parameters."""
    lines: list[str] = []

    if confirmation_bias > 0:
        if confirmation_bias < 0.3:
            lines.append(
                "You do not assume your side is automatically right. When someone makes "
                "a serious point from the other side, you are able to consider that they "
                "may be seeing something you missed, and you are open to revising your "
                "view without needing overwhelming proof first."
            )
        elif confirmation_bias < 0.6:
            lines.append(
                "You usually feel that you are on the correct side of the debate. "
                "Opposing views tend to strike you as missing something important, and "
                "it usually takes very strong evidence before you seriously consider that "
                "you may be wrong."
            )
        else:
            lines.append(
                "You usually feel that you are on the correct side of the debate. You "
                "are skeptical of opposing views and tend to see them as mistaken unless "
                "they are backed by irrefutable evidence. It usually takes something very "
                "close to decisive proof before you genuinely think you might be wrong."
            )

    om = agent.params.open_mindedness
    if facilitated:
        om_eff = min(om + 0.076 * (group_building_level - 1), 1.0)
        lines.append(
            f"Because this discussion is moderated, the pace is a little calmer and "
            f"there is more room to sit with a point before reacting to it. In this "
            f"setting, you are somewhat more able to seriously weigh perspectives you "
            f"might otherwise move past (base={om:.2f}, effective={om_eff:.2f})."
        )
    elif om > 0.6:
        lines.append(
            f"You are fairly comfortable engaging with arguments that challenge your "
            f"first instinct ({om:.2f}), and you can sometimes find yourself revising "
            f"your view after thinking one through."
        )
    elif om < 0.3:
        lines.append(
            f"You do not revise your views quickly ({om:.2f}). Before a counterargument "
            f"really lands, it usually needs to feel concrete, credible, and difficult "
            f"to shrug off."
        )

    gbl_descs = {
        1: "The room feels low-trust and somewhat guarded, so people choose their words carefully.",
        2: "Trust is beginning to form, though people are still somewhat careful with one another.",
        3: "There is enough trust in the group for a reasonably open exchange.",
        4: "The group feels fairly cohesive, and people tend to engage openly and constructively.",
        5: "The atmosphere is highly trusting, and people feel relatively safe reconsidering their views aloud.",
    }
    if group_building_level in gbl_descs:
        lines.append(gbl_descs[group_building_level])

    return "\n".join(lines)


def _attack_context(cid: str, agent: Agent, pool: ArgumentPool) -> str:
    """Show which of the agent's considerations attack/are attacked by cid."""
    lines = []
    attackers = pool.attack_graph.get_attackers(cid)
    for aid, strength in attackers.items():
        if aid in agent.weights:
            c = pool.get(aid)
            lines.append(
                f"  * Your {aid} (\"{c.label}\", {_support_label(agent.weights[aid])}, "
                f"strength={abs(agent.weights[aid]):.3f}) "
                f"ATTACKS this argument (strength={strength:.2f})"
            )
    supporters = pool.attack_graph.get_supporters(cid)
    for sid, strength in supporters.items():
        if sid in agent.weights:
            c = pool.get(sid)
            lines.append(
                f"  * Your {sid} (\"{c.label}\", {_support_label(agent.weights[sid])}, "
                f"strength={abs(agent.weights[sid]):.3f}) "
                f"SUPPORTS this argument (strength={strength:.2f})"
            )
    return "\n".join(lines) if lines else ""


@dataclass(frozen=True)
class PromptContext:
    topic_description: str = ""
    confirmation_bias: float = 0.0
    facilitated: bool = False
    group_building_level: int = 3


@dataclass(frozen=True)
class PromptFeatures:
    voice_supported_only: bool = False
    include_attack_context: bool = True
    split_reflect_updates: bool = False


@dataclass(frozen=True)
class PromptSections:
    intro: tuple[str, ...] = ()
    framing: tuple[str, ...] = ()
    repertoire: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    overlay: tuple[str, ...] = ()
    instructions: tuple[str, ...] = ()

    def render(self) -> str:
        blocks: list[str] = []
        for lines in (
            self.intro,
            self.framing,
            self.repertoire,
            self.evidence,
            self.overlay,
            self.instructions,
        ):
            if lines:
                blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


@dataclass(frozen=True)
class PromptProfile:
    voice_intro_style: str = "standard"
    evaluate_intro_style: str = "standard"
    reflect_intro_style: str = "standard"
    evaluate_argument_style: str = "standard"
    own_argument_stance_style: str = "none"
    owned_update_style: str = "standard"
    new_updates_heading: str = (
        "NEW ARGUMENTS (not in your repertoire — you had not considered these before):"
    )
    voice_supported_heading: str = "STATEMENTS YOU CURRENTLY SUPPORT (pick ONE of these to voice):"
    attack_context_heading: str = "RELEVANT RELATIONSHIPS in your repertoire:"
    voice_addition_lines: tuple[str, ...] = ()
    voice_task_lines: tuple[str, ...] = (
        "TASK: Pick ONE statement you currently support and call submit_voice with the cid.",
    )
    evaluate_task_lines: tuple[str, ...] = (
        "TASK: Rate how persuasive this argument is to you. "
        f"Call submit_influence with an integer score on the {INFLUENCE_LIKERT_SCALE_TEXT}.",
        f"  {INFLUENCE_LIKERT_GUIDANCE}",
    )
    reflect_task_lines: tuple[str, ...] = (
        "TASK: Update weights based on what you heard.",
        "Call update_weight(cid, weight, stance) for each change, then call done_reflecting(done=true).",
        "Use weight in [0, 1]; weight=0 removes the statement from your repertoire.",
        'Use stance="endorse" if you support the statement itself.',
        'Use stance="reject" if you oppose the statement itself.',
        'For CON statements, stance="endorse" means you agree with that CON argument; it does not mean PRO policy support.',
        "If nothing changes, call no_update(unchanged=true).",
        "Changes should be proportionate — small shifts, not dramatic.",
    )
    evaluate_addition_lines: tuple[str, ...] = ()
    reflect_addition_lines: tuple[str, ...] = ()


POLICY_SIDE_LINE = "The policy side of each statement is shown separately via dir/PRO/CON."
CORE_VOICE_TASK_LINES = (
    "TASK: Pick ONE statement from your SUPPORTED list and call submit_voice with its cid.",
    "Do not voice a statement you currently disagree with.",
)
CORE_EVALUATE_TASK_LINES = (
    f"TASK: Score how persuasive this argument is on a {INFLUENCE_LIKERT_SCALE_TEXT}. "
    "Call submit_influence with an integer score.",
    f"  {INFLUENCE_LIKERT_GUIDANCE}",
)
CORE_REFLECT_TASK_LINES = (
    "TOOL CALL FORMAT FOR update_weight:",
    "- update_weight(cid, weight, stance)",
    "- weight = stance strength in [0, 1]; weight=0 removes the statement from your repertoire",
    '- stance="endorse" means you SUPPORT/ACCEPT the statement itself',
    '- stance="reject" means you OPPOSE/DISAGREE WITH the statement itself',
    '- For CON statements, endorse means you agree with the CON argument; reject means you oppose that CON argument.',
    "- No update is a valid decision only if the round did not change those strengths.",
    "",
    "TASK: Call update_weight(cid, weight, stance) for ANY consideration you",
    "want to change — whether it is already in your repertoire OR new to you:",
    "- Existing supported statement reinforced → strengthen it slightly.",
    "- Existing disagreed-with statement softened → nudge it toward zero.",
    "- New statement that genuinely resonated → add it with a support or",
    "  disagreement strength.",
    "- New consideration that left you unmoved → do not add it.",
    "You may call update_weight any number of times. Most rounds: 0-3 total calls.",
    "",
    "When done: call done_reflecting(done=true). If nothing changed at all, call no_update(unchanged=true).",
)
COMMON_BASELINE_OWN_STANCE = (
    "  You currently {label} this statement with strength {strength:.3f}."
)
BELIEF_COHERENCE_LINES = (
    "You value coherence in your beliefs — your views reflect real experience "
    "and reflection. When you encounter a new argument, you naturally notice "
    "whether it fits with what you already believe, and you know the difference "
    "between a genuinely compelling argument and a merely clever one.",
)
BELIEF_STABILITY_LINES = (
    "You take your current views seriously and do not revise them lightly.",
)
STANDARD_PROMPT_PROFILE = PromptProfile()
BASELINE_PROMPT_PROFILE = PromptProfile(
    voice_intro_style="baseline",
    evaluate_intro_style="baseline",
    reflect_intro_style="baseline",
    evaluate_argument_style="baseline",
    own_argument_stance_style="baseline",
    voice_addition_lines=BELIEF_COHERENCE_LINES + BELIEF_STABILITY_LINES,
    voice_task_lines=CORE_VOICE_TASK_LINES,
    evaluate_task_lines=CORE_EVALUATE_TASK_LINES,
    reflect_task_lines=CORE_REFLECT_TASK_LINES,
    evaluate_addition_lines=BELIEF_COHERENCE_LINES + BELIEF_STABILITY_LINES,
    reflect_addition_lines=BELIEF_COHERENCE_LINES + BELIEF_STABILITY_LINES,
)


class BasePromptBuilder:
    """Shared scaffold for voice/evaluate/reflect prompt construction."""

    def __init__(
        self,
        features: PromptFeatures | None = None,
        profile: PromptProfile | None = None,
    ) -> None:
        self.features = features or PromptFeatures()
        self.profile = profile or STANDARD_PROMPT_PROFILE

    def build_voice(
        self,
        agent: Agent,
        pool: ArgumentPool,
        *,
        context: PromptContext | None = None,
    ) -> str:
        prompt_context = context or PromptContext()
        return self._voice_sections(agent, pool, prompt_context).render()

    def build_evaluate(
        self,
        agent: Agent,
        cid: str,
        speaker_opinion: float,
        pool: ArgumentPool,
        *,
        context: PromptContext | None = None,
    ) -> str:
        prompt_context = context or PromptContext()
        return self._evaluate_sections(
            agent,
            cid,
            speaker_opinion,
            pool,
            prompt_context,
        ).render()

    def build_reflect(
        self,
        agent: Agent,
        round_updates: list[tuple[str, float, float]],
        pool: ArgumentPool,
        *,
        context: PromptContext | None = None,
    ) -> str:
        prompt_context = context or PromptContext()
        return self._reflect_sections(agent, round_updates, pool, prompt_context).render()

    def _voice_sections(
        self,
        agent: Agent,
        pool: ArgumentPool,
        context: PromptContext,
    ) -> PromptSections:
        return PromptSections(
            intro=tuple(self._voice_intro_lines(agent, pool, context)),
            framing=(POLICY_SIDE_LINE,),
            repertoire=tuple(self._voice_repertoire_lines(agent, pool, context)),
            overlay=tuple(self.profile.voice_addition_lines),
            instructions=tuple(self.profile.voice_task_lines),
        )

    def _evaluate_sections(
        self,
        agent: Agent,
        cid: str,
        speaker_opinion: float,
        pool: ArgumentPool,
        context: PromptContext,
    ) -> PromptSections:
        c = pool.get(cid)
        evidence_lines = list(self._evaluate_argument_lines(agent, cid, speaker_opinion, c, context))

        own_stance_line = self._own_argument_stance_line(agent, cid)
        if own_stance_line:
            evidence_lines.append(own_stance_line)

        if self.features.include_attack_context:
            atk_ctx = _attack_context(cid, agent, pool)
            if atk_ctx:
                evidence_lines += [self.profile.attack_context_heading, atk_ctx]

        return PromptSections(
            intro=tuple(self._evaluate_intro_lines(agent, pool, context)),
            framing=(POLICY_SIDE_LINE,),
            repertoire=("Your current statement repertoire:", _format_repertoire(agent, pool)),
            evidence=tuple(evidence_lines),
            overlay=tuple(self.profile.evaluate_addition_lines),
            instructions=tuple(self.profile.evaluate_task_lines),
        )

    def _reflect_sections(
        self,
        agent: Agent,
        round_updates: list[tuple[str, float, float]],
        pool: ArgumentPool,
        context: PromptContext,
    ) -> PromptSections:
        return PromptSections(
            intro=tuple(self._reflect_intro_lines(agent, pool, context)),
            framing=(POLICY_SIDE_LINE,),
            repertoire=("Your current statement repertoire:", _format_repertoire(agent, pool)),
            evidence=tuple(self._reflect_updates_lines(agent, round_updates, pool)),
            overlay=tuple(self.profile.reflect_addition_lines),
            instructions=tuple(self.profile.reflect_task_lines),
        )

    def _voice_intro_lines(
        self,
        agent: Agent,
        pool: ArgumentPool,
        context: PromptContext,
    ) -> list[str]:
        if self.profile.voice_intro_style == "baseline":
            op = agent.opinion(pool)
            stance = self._stance_label(op)
            return [
                "You are a real Seattle resident polled on:",
                f'  "{context.topic_description}"',
                "",
                f"Your stance: {stance} this policy (opinion {op:+.3f} on -1 to +1). "
                "Your voting pattern below captures your actual convictions — you are "
                "speaking AS this person, not as a neutral analyst.",
            ]
        return [
            f"You are deliberation agent {agent.id}.",
            f"Your current opinion: {agent.opinion(pool):+.3f} (scale: -1 con to +1 pro)",
        ]

    def _voice_repertoire_lines(
        self,
        agent: Agent,
        pool: ArgumentPool,
        context: PromptContext,
    ) -> list[str]:
        if self.features.voice_supported_only:
            return [self.profile.voice_supported_heading, _format_supported_repertoire(agent, pool)]
        return ["Your current statement repertoire:", _format_repertoire(agent, pool)]

    def _evaluate_intro_lines(
        self,
        agent: Agent,
        pool: ArgumentPool,
        context: PromptContext,
    ) -> list[str]:
        if self.profile.evaluate_intro_style == "baseline":
            op = agent.opinion(pool)
            stance = self._stance_label(op)
            return [
                f'Seattle resident polled on: "{context.topic_description}"',
                f"Your stance: {stance} (opinion {_format_opinion_score(op)}). Speak AS this person.",
            ]
        return [
            f"You are deliberation agent {agent.id}.",
            f"Your current opinion: {agent.opinion(pool):+.3f} (scale: -1 con to +1 pro)",
        ]

    def _evaluate_argument_lines(
        self,
        agent: Agent,
        cid: str,
        speaker_opinion: float,
        consideration,
        context: PromptContext,
    ) -> list[str]:
        if self.profile.evaluate_argument_style == "baseline":
            return [
                f"ARGUMENT JUST RAISED (from a participant with opinion {_format_opinion_score(speaker_opinion)}):",
                f"  {cid}: \"{consideration.label}\"",
                f"  direction={int(consideration.direction):+d}",
            ]
        return [
            f"INCOMING ARGUMENT from another agent (their opinion={_format_opinion_score(speaker_opinion)}):",
            f"  {cid}: \"{consideration.label}\"",
            f"  direction={int(consideration.direction):+d}",
        ]

    def _own_argument_stance_line(self, agent: Agent, cid: str) -> str:
        own_w = agent.weights.get(cid)
        if own_w is None:
            return ""
        if self.profile.own_argument_stance_style == "baseline":
            return COMMON_BASELINE_OWN_STANCE.format(
                label=_support_label(own_w),
                strength=abs(own_w),
            )
        return ""

    def _reflect_intro_lines(
        self,
        agent: Agent,
        pool: ArgumentPool,
        context: PromptContext,
    ) -> list[str]:
        if self.profile.reflect_intro_style == "baseline":
            op = agent.opinion(pool)
            stance = self._stance_label(op)
            return [
                "You are a real Seattle resident who participated in a public poll on:",
                f'  "{context.topic_description}"',
                "",
                "Your voting pattern below captures your actual convictions. You are "
                "updating those convictions based on what you just heard — but you "
                "are a real person, not an unbiased analyst.",
                "",
                f"YOUR POSITION: You {stance} this policy "
                f"(opinion score = {op:+.3f} on -1 oppose to +1 support).",
            ]
        return [
            f"You are deliberation agent {agent.id}.",
            f"Your current opinion: {agent.opinion(pool):+.3f} (scale: -1 con to +1 pro)",
        ]

    def _reflect_updates_lines(
        self,
        agent: Agent,
        round_updates: list[tuple[str, float, float]],
        pool: ArgumentPool,
    ) -> list[str]:
        if not self.features.split_reflect_updates:
            lines = ["ARGUMENTS HEARD THIS ROUND:"]
            for cid, sp_op, influence in round_updates:
                c = pool.get(cid)
                lines.append(
                    f"  {cid}: \"{c.label}\" (d={int(c.direction):+d}) "
                    f"| speaker_opinion={_format_opinion_score(sp_op)} "
                    f"| your_persuasiveness_score={_format_likert_score(influence)}"
                )
            return lines

        owned_updates = [(c, s, i) for c, s, i in round_updates if c in agent.weights]
        new_updates = [(c, s, i) for c, s, i in round_updates if c not in agent.weights]
        lines: list[str] = []

        if owned_updates:
            lines.append(self._owned_updates_heading())
            for cid, sp_op, influence in owned_updates:
                c = pool.get(cid)
                lines.append(self._format_owned_update_line(agent, cid, c, sp_op, influence))

        if new_updates:
            if lines:
                lines.append("")
            lines.append(self.profile.new_updates_heading)
            for cid, sp_op, influence in new_updates:
                c = pool.get(cid)
                lines.append(self._format_new_update_line(cid, c, sp_op, influence))

        return lines

    def _owned_updates_heading(self) -> str:
        return "ARGUMENTS YOU HEARD (already in your repertoire):"

    def _format_owned_update_line(
        self,
        agent: Agent,
        cid: str,
        consideration,
        speaker_opinion: float,
        influence: float,
    ) -> str:
        own_w = agent.weights[cid]
        if self.profile.owned_update_style == "baseline":
            return (
                f"  {cid}: \"{consideration.label}\" "
                f"(dir={int(consideration.direction):+d}; speaker_op={_format_opinion_score(speaker_opinion)}; "
                f"your_view={_support_label(own_w)}; "
                f"your_strength={abs(own_w):.3f}; "
                f"your_persuasiveness_score={_format_likert_score(influence)})"
            )
        return (
            f"  {cid}: \"{consideration.label}\" "
            f"(dir={int(consideration.direction):+d}, your_view={_support_label(own_w)}, "
            f"your_strength={abs(own_w):.3f}, "
            f"persuasiveness_you_rated={_format_likert_score(influence)})"
        )

    def _format_new_update_line(
        self,
        cid: str,
        consideration,
        speaker_opinion: float,
        influence: float,
    ) -> str:
        if self.profile.owned_update_style == "baseline":
            return (
                f"  {cid}: \"{consideration.label}\" "
                f"(dir={int(consideration.direction):+d}; speaker_op={_format_opinion_score(speaker_opinion)}; "
                f"your_persuasiveness_score={_format_likert_score(influence)})"
            )
        return (
            f"  {cid}: \"{consideration.label}\" "
            f"(dir={int(consideration.direction):+d}, speaker_op={_format_opinion_score(speaker_opinion)}, "
            f"persuasiveness_you_rated={_format_likert_score(influence)})"
        )


    def _stance_label(self, opinion: float) -> str:
        if opinion >= OPINION_LEAN_THRESHOLD:
            return "STRONGLY SUPPORT"
        if opinion >= 0:
            return "WEAKLY SUPPORT"
        if opinion <= -OPINION_LEAN_THRESHOLD:
            return "STRONGLY OPPOSE"
        return "WEAKLY OPPOSE"


class BaselinePromptBuilder(BasePromptBuilder):
    def __init__(self) -> None:
        super().__init__(
            features=PromptFeatures(
                voice_supported_only=True,
                include_attack_context=True,
                split_reflect_updates=True,
            ),
            profile=BASELINE_PROMPT_PROFILE,
        )


STANDARD_PROMPT_BUILDER = BasePromptBuilder(profile=STANDARD_PROMPT_PROFILE)
BASELINE_PROMPT_BUILDER = BaselinePromptBuilder()


def build_voice_prompt(
    agent: Agent,
    pool: ArgumentPool,
    confirmation_bias: float = 0.0,
    facilitated: bool = False,
    group_building_level: int = 3,
) -> str:
    return STANDARD_PROMPT_BUILDER.build_voice(
        agent,
        pool,
        context=PromptContext(
            confirmation_bias=confirmation_bias,
            facilitated=facilitated,
            group_building_level=group_building_level,
        ),
    )


def build_evaluate_prompt(
    agent: Agent,
    cid: str,
    speaker_opinion: float,
    pool: ArgumentPool,
    confirmation_bias: float = 0.0,
    facilitated: bool = False,
    group_building_level: int = 3,
) -> str:
    return STANDARD_PROMPT_BUILDER.build_evaluate(
        agent,
        cid,
        speaker_opinion,
        pool,
        context=PromptContext(
            confirmation_bias=confirmation_bias,
            facilitated=facilitated,
            group_building_level=group_building_level,
        ),
    )


def build_reflect_prompt(
    agent: Agent,
    round_updates: list[tuple[str, float, float]],
    pool: ArgumentPool,
    confirmation_bias: float = 0.0,
    facilitated: bool = False,
    group_building_level: int = 3,
) -> str:
    return STANDARD_PROMPT_BUILDER.build_reflect(
        agent,
        round_updates,
        pool,
        context=PromptContext(
            confirmation_bias=confirmation_bias,
            facilitated=facilitated,
            group_building_level=group_building_level,
        ),
    )


def build_baseline_voice_prompt(
    agent: Agent,
    pool: ArgumentPool,
    topic_description: str,
    confirmation_bias: float = 0.0,
    facilitated: bool = False,
    group_building_level: int = 3,
) -> str:
    return BASELINE_PROMPT_BUILDER.build_voice(
        agent,
        pool,
        context=PromptContext(
            topic_description=topic_description,
            confirmation_bias=confirmation_bias,
            facilitated=facilitated,
            group_building_level=group_building_level,
        ),
    )


def build_baseline_evaluate_prompt(
    agent: Agent,
    cid: str,
    speaker_opinion: float,
    pool: ArgumentPool,
    topic_description: str,
    confirmation_bias: float = 0.0,
    facilitated: bool = False,
    group_building_level: int = 3,
) -> str:
    return BASELINE_PROMPT_BUILDER.build_evaluate(
        agent,
        cid,
        speaker_opinion,
        pool,
        context=PromptContext(
            topic_description=topic_description,
            confirmation_bias=confirmation_bias,
            facilitated=facilitated,
            group_building_level=group_building_level,
        ),
    )


def build_baseline_reflect_prompt(
    agent: Agent,
    round_updates: list[tuple[str, float, float]],
    pool: ArgumentPool,
    topic_description: str,
    confirmation_bias: float = 0.0,
    facilitated: bool = False,
    group_building_level: int = 3,
) -> str:
    return BASELINE_PROMPT_BUILDER.build_reflect(
        agent,
        round_updates,
        pool,
        context=PromptContext(
            topic_description=topic_description,
            confirmation_bias=confirmation_bias,
            facilitated=facilitated,
            group_building_level=group_building_level,
        ),
    )
