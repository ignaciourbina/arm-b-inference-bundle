"""Named prompt variants for the Sprint-15 ablation programme.

The production Arm-B configuration is `control` — byte-identical to
`BASELINE_PROMPT_BUILDER`, so a control cell run through this registry is
directly comparable with the existing 390-run collection.

Every other variant changes ONE thing relative to control, so an observed
difference is attributable. Each carries the hypothesis it tests:

  anti-repetition    Sprint 14 found an LLM agent voices only ~1.5 distinct
                     considerations over eight rounds (vs ~3.2 for the rule
                     engine), and that this coincides with weight saturation
                     after round 1. Adding one line that discourages restating
                     the previous argument separates "the prompt never asked for
                     variety" from "the agent's state stopped moving".
  explicit-tradeoff  Cross-camp uptake DECLINES in Arm B where it rises in
                     Arm A. Requiring the agent to name the strongest opposing
                     consideration before updating tests whether that decline is
                     recoverable by instruction.
  terse              Strips the cognitive-style overlay (belief coherence /
                     stability) and the attack-context block. Tests how much of
                     the observed behaviour is carried by prompt scaffolding
                     rather than by the engine.

Usage:
    from llm.prompt_variants import resolve_variant, VARIANT_NAMES
    builder = resolve_variant("anti-repetition")
"""

from __future__ import annotations

from llm.prompts import (
    BASELINE_PROMPT_PROFILE,
    BELIEF_COHERENCE_LINES,
    BELIEF_STABILITY_LINES,
    CORE_REFLECT_TASK_LINES,
    CORE_VOICE_TASK_LINES,
    BasePromptBuilder,
    BaselinePromptBuilder,
    PromptContext,
    PromptFeatures,
    _format_opinion_score,
)
from dataclasses import replace

# --- variant-specific wording -------------------------------------------------

ANTI_REPETITION_LINE = (
    "If you voiced a statement in a previous round, prefer a different one that "
    "you also support, unless the earlier statement genuinely remains the "
    "strongest expression of your view this round."
)

EXPLICIT_TRADEOFF_LINES = (
    "",
    "Before updating, briefly consider the strongest statement you heard that "
    "cuts AGAINST your current position. If it has genuine force, reflect that "
    "in your weights; if it does not, leave your weights unchanged.",
)


def _control() -> BasePromptBuilder:
    """Production configuration — identical to BASELINE_PROMPT_BUILDER."""
    return BaselinePromptBuilder()


def _anti_repetition() -> BasePromptBuilder:
    profile = replace(
        BASELINE_PROMPT_PROFILE,
        voice_task_lines=CORE_VOICE_TASK_LINES + (ANTI_REPETITION_LINE,),
    )
    return BasePromptBuilder(
        features=PromptFeatures(
            voice_supported_only=True,
            include_attack_context=True,
            split_reflect_updates=True,
        ),
        profile=profile,
    )


def _explicit_tradeoff() -> BasePromptBuilder:
    profile = replace(
        BASELINE_PROMPT_PROFILE,
        reflect_task_lines=CORE_REFLECT_TASK_LINES + EXPLICIT_TRADEOFF_LINES,
    )
    return BasePromptBuilder(
        features=PromptFeatures(
            voice_supported_only=True,
            include_attack_context=True,
            split_reflect_updates=True,
        ),
        profile=profile,
    )


def _terse() -> BasePromptBuilder:
    """Drop the cognitive-style overlay and the attack-context block."""
    profile = replace(
        BASELINE_PROMPT_PROFILE,
        voice_addition_lines=(),
        evaluate_addition_lines=(),
        reflect_addition_lines=(),
    )
    return BasePromptBuilder(
        features=PromptFeatures(
            voice_supported_only=True,
            include_attack_context=False,
            split_reflect_updates=True,
        ),
        profile=profile,
    )


# --- neutrality wave (register-only levers; information held fixed) -----------
#
# Motivation (see sprint-15 neutrality-ablation-design.md): does the RHETORICAL
# REGISTER of the production prompt prime the headline pattern? Every lever
# below rewrites wording only — same numbers, same tool contract, same
# structure, attack context stays ON, speaker opinions stay visible.


def _baseline_features() -> PromptFeatures:
    return PromptFeatures(
        voice_supported_only=True,
        include_attack_context=True,
        split_reflect_updates=True,
    )


def _degradualized(lines: tuple[str, ...]) -> tuple[str, ...]:
    """Strip magnitude-priming adverbs/anchors from the reflect task text.

    Information-equivalent: the tool contract and every decision option are
    unchanged; only 'slightly'/'nudge'/'0-3 calls' step-size priming goes.
    """
    swaps = {
        "- Existing supported statement reinforced → strengthen it slightly.":
            "- Existing supported statement reinforced → strengthen it.",
        "- Existing disagreed-with statement softened → nudge it toward zero.":
            "- Existing disagreed-with statement softened → move it toward zero.",
        "You may call update_weight any number of times. Most rounds: 0-3 total calls.":
            "You may call update_weight any number of times.",
    }
    out = tuple(swaps.get(line, line) for line in lines)
    assert out != lines, "degradualize matched nothing — CORE_REFLECT_TASK_LINES drifted"
    return out


NO_GRADUALISM_REFLECT_TASK_LINES = _degradualized(CORE_REFLECT_TASK_LINES)


class NeutralRegisterBuilder(BasePromptBuilder):
    """Baseline builder with the persona and/or stance-label register neutralized.

    `neutral_persona` replaces the real-resident/actual-convictions/not-a-
    neutral-analyst roleplay framing with a descriptive participant framing.
    `neutral_stance` replaces the capitalized STRONGLY SUPPORT/OPPOSE verbal
    label with the bare position score. Both keep every number and the topic.
    """

    def __init__(
        self,
        *,
        neutral_persona: bool = False,
        neutral_stance: bool = False,
        profile=None,
        features: PromptFeatures | None = None,
    ) -> None:
        super().__init__(
            features=features or _baseline_features(),
            profile=profile or BASELINE_PROMPT_PROFILE,
        )
        self.neutral_persona = neutral_persona
        self.neutral_stance = neutral_stance

    # -- shared phrase helpers -------------------------------------------------

    def _stance_phrase(self, op: float) -> str:
        if self.neutral_stance:
            return f"Your position score: {op:+.3f} on -1 (oppose) to +1 (support)."
        return f"Your stance: {self._stance_label(op)} this policy (opinion {op:+.3f} on -1 to +1)."

    # -- intros ----------------------------------------------------------------

    def _voice_intro_lines(self, agent, pool, context: PromptContext) -> list[str]:
        if not (self.neutral_persona or self.neutral_stance):
            return super()._voice_intro_lines(agent, pool, context)
        op = agent.opinion(pool)
        if self.neutral_persona:
            return [
                "You are a participant in a discussion of:",
                f'  "{context.topic_description}"',
                "",
                f"{self._stance_phrase(op)} Your ratings of the statements are listed below.",
            ]
        return [
            "You are a real Seattle resident polled on:",
            f'  "{context.topic_description}"',
            "",
            f"{self._stance_phrase(op)} Your voting pattern below captures your "
            "actual convictions — you are speaking AS this person, not as a "
            "neutral analyst.",
        ]

    def _evaluate_intro_lines(self, agent, pool, context: PromptContext) -> list[str]:
        if not (self.neutral_persona or self.neutral_stance):
            return super()._evaluate_intro_lines(agent, pool, context)
        op = agent.opinion(pool)
        if self.neutral_stance:
            stance_part = f"Your position: opinion {_format_opinion_score(op)}."
        else:
            stance_part = f"Your stance: {self._stance_label(op)} (opinion {_format_opinion_score(op)})."
        if self.neutral_persona:
            return [f'Discussion topic: "{context.topic_description}"', stance_part]
        return [
            f'Seattle resident polled on: "{context.topic_description}"',
            f"{stance_part} Speak AS this person.",
        ]

    def _reflect_intro_lines(self, agent, pool, context: PromptContext) -> list[str]:
        if not (self.neutral_persona or self.neutral_stance):
            return super()._reflect_intro_lines(agent, pool, context)
        op = agent.opinion(pool)
        if self.neutral_stance:
            position = f"YOUR POSITION: opinion score = {op:+.3f} (-1 oppose to +1 support)."
        else:
            position = (
                f"YOUR POSITION: You {self._stance_label(op)} this policy "
                f"(opinion score = {op:+.3f} on -1 oppose to +1 support)."
            )
        if self.neutral_persona:
            return [
                "You are a participant in a discussion of:",
                f'  "{context.topic_description}"',
                "",
                "Your ratings of the statements are listed below. You are updating "
                "them based on what you heard this round.",
                "",
                position,
            ]
        return [
            "You are a real Seattle resident who participated in a public poll on:",
            f'  "{context.topic_description}"',
            "",
            "Your voting pattern below captures your actual convictions. You are "
            "updating those convictions based on what you just heard — but you "
            "are a real person, not an unbiased analyst.",
            "",
            position,
        ]


def _neutral_persona() -> BasePromptBuilder:
    return NeutralRegisterBuilder(neutral_persona=True)


def _neutral_stance() -> BasePromptBuilder:
    return NeutralRegisterBuilder(neutral_stance=True)


def _no_overlay() -> BasePromptBuilder:
    """Conviction overlay removed; attack context KEPT (unlike `terse`)."""
    profile = replace(
        BASELINE_PROMPT_PROFILE,
        voice_addition_lines=(),
        evaluate_addition_lines=(),
        reflect_addition_lines=(),
    )
    return BasePromptBuilder(features=_baseline_features(), profile=profile)


def _no_gradualism() -> BasePromptBuilder:
    profile = replace(
        BASELINE_PROMPT_PROFILE,
        reflect_task_lines=NO_GRADUALISM_REFLECT_TASK_LINES,
    )
    return BasePromptBuilder(features=_baseline_features(), profile=profile)


def _neutral_full() -> BasePromptBuilder:
    """All four register levers neutralized — the maximally neutral prompt."""
    profile = replace(
        BASELINE_PROMPT_PROFILE,
        voice_addition_lines=(),
        evaluate_addition_lines=(),
        reflect_addition_lines=(),
        reflect_task_lines=NO_GRADUALISM_REFLECT_TASK_LINES,
    )
    return NeutralRegisterBuilder(
        neutral_persona=True, neutral_stance=True, profile=profile
    )


VARIANTS = {
    "control": _control,
    "anti-repetition": _anti_repetition,
    "explicit-tradeoff": _explicit_tradeoff,
    "terse": _terse,
    "neutral-persona": _neutral_persona,
    "neutral-stance": _neutral_stance,
    "no-overlay": _no_overlay,
    "no-gradualism": _no_gradualism,
    "neutral-full": _neutral_full,
}
VARIANT_NAMES = tuple(VARIANTS)

# One-line descriptions, recorded into each trace's config block so a cell is
# self-describing after the fact.
VARIANT_NOTES = {
    "control": "production BASELINE prompt profile (unchanged)",
    "anti-repetition": "voice task adds a prefer-a-different-statement line",
    "explicit-tradeoff": "reflect task adds a consider-the-strongest-opposing line",
    "terse": "no belief-coherence/stability overlay, no attack context",
    "neutral-persona": "roleplay register -> descriptive participant framing (info unchanged)",
    "neutral-stance": "STRONGLY SUPPORT/OPPOSE labels -> bare position score (info unchanged)",
    "no-overlay": "conviction overlay removed, attack context kept",
    "no-gradualism": "slightly/nudge/0-3-calls step-size priming removed from reflect",
    "neutral-full": "all register levers neutral: persona+stance+overlay+gradualism",
}


def resolve_variant(name: str) -> BasePromptBuilder:
    if name not in VARIANTS:
        raise ValueError(
            f"Unknown prompt variant {name!r}. Options: {list(VARIANT_NAMES)}"
        )
    return VARIANTS[name]()
