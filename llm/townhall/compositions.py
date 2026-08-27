"""Shared town-hall composition presets.

These presets are the canonical composition overrides for empirical-init runs.
Both the LLM and rule-based TownHall paths should import from here so a named
composition means the same joint policy/coherence target in both arms.
"""

from __future__ import annotations

CompositionKey = tuple[str, str]
CompositionPreset = dict[CompositionKey, int]


COMPOSITION_PRESETS: dict[str, CompositionPreset] = {
    "symmetric": {
        ("Pro", "Coherent"): 1,
        ("Pro", "Mixed"): 3,
        ("Pro", "Ambivalent"): 1,
        ("Con", "Coherent"): 1,
        ("Con", "Mixed"): 3,
        ("Con", "Ambivalent"): 1,
    },
    "polarized": {
        ("Pro", "Coherent"): 3,
        ("Pro", "Mixed"): 1,
        ("Pro", "Ambivalent"): 1,
        ("Con", "Coherent"): 3,
        ("Con", "Mixed"): 1,
        ("Con", "Ambivalent"): 1,
    },
    "empirical": {
        ("Pro", "Coherent"): 2,
        ("Pro", "Mixed"): 4,
        ("Pro", "Ambivalent"): 1,
        ("Ambivalent", "Ambivalent"): 1,
        ("Con", "Coherent"): 0,
        ("Con", "Mixed"): 1,
        ("Con", "Ambivalent"): 1,
    },
    "con_flipped": {
        ("Pro", "Coherent"): 0,
        ("Pro", "Mixed"): 1,
        ("Pro", "Ambivalent"): 1,
        ("Ambivalent", "Ambivalent"): 1,
        ("Con", "Coherent"): 2,
        ("Con", "Mixed"): 4,
        ("Con", "Ambivalent"): 1,
    },
    "symmetric_n6": {
        ("Pro", "Mixed"): 2,
        ("Pro", "Ambivalent"): 1,
        ("Con", "Mixed"): 2,
        ("Con", "Ambivalent"): 1,
    },
    "polarized_n6": {
        ("Pro", "Coherent"): 2,
        ("Pro", "Mixed"): 1,
        ("Con", "Coherent"): 2,
        ("Con", "Mixed"): 1,
    },
    "three_clusters_n6": {
        ("Pro", "Coherent"): 1,
        ("Pro", "Mixed"): 1,
        ("Ambivalent", "Ambivalent"): 2,
        ("Con", "Mixed"): 1,
        ("Con", "Coherent"): 1,
    },
}


COMPOSITION_NAMES = tuple(COMPOSITION_PRESETS.keys())
LOCAL_SMOKE_COMPOSITION_NAMES = (
    "symmetric_n6",
    "polarized_n6",
    "three_clusters_n6",
)


def resolve_composition(
    composition: str | None,
    *,
    n_agents: int,
) -> CompositionPreset | None:
    if composition is None:
        return None
    if composition not in COMPOSITION_PRESETS:
        raise ValueError(
            f"Unknown composition '{composition}'. Options: {list(COMPOSITION_PRESETS)}"
        )
    preset = COMPOSITION_PRESETS[composition]
    total = sum(preset.values())
    if total != n_agents:
        raise ValueError(
            f"Composition '{composition}' sums to {total}, expected n_agents={n_agents}"
        )
    return preset


# Backward-compatible alias retained while downstream imports are migrated.
COMPOSITIONS_N10 = COMPOSITION_PRESETS