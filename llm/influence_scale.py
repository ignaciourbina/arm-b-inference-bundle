"""Shared definitions for persuasiveness / influence scoring."""

from __future__ import annotations

INFLUENCE_LIKERT_MIN = 0
INFLUENCE_LIKERT_MAX = 100
INFLUENCE_LIKERT_FALLBACK = 50

INFLUENCE_LIKERT_SCALE_TEXT = (
    f"{INFLUENCE_LIKERT_MIN}-{INFLUENCE_LIKERT_MAX} persuasiveness scale "
    f"(min={INFLUENCE_LIKERT_MIN}, max={INFLUENCE_LIKERT_MAX})"
)

INFLUENCE_LIKERT_GUIDANCE = (
    f"{INFLUENCE_LIKERT_MIN} = not at all persuasive, "
    f"{INFLUENCE_LIKERT_MAX} = extremely persuasive. "
    "Use values in between to express gradations in how convincing the argument feels."
)


def format_influence_likert(score: float) -> str:
    score_int = int(score)
    return (
        f"{score_int}/{INFLUENCE_LIKERT_MAX} "
        f"(min={INFLUENCE_LIKERT_MIN}, max={INFLUENCE_LIKERT_MAX})"
    )