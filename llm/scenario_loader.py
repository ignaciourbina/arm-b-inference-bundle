"""Load consideration pools from JSON scenario files."""

from __future__ import annotations

import json
from pathlib import Path

from agora.considerations import ArgumentPool, AttackGraph, Consideration  # type: ignore[import-untyped]

SCENARIOS_DIR = Path(__file__).parent / "scenarios"


def load_scenario(path: str | Path) -> ArgumentPool:
    """Load an ArgumentPool from a JSON scenario file."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    pool = ArgumentPool()
    for c in data["considerations"]:
        irt_a = c.get("irt_a")
        irt_b = c.get("irt_b")
        pool.add(Consideration(
            id=c["id"],
            label=c["label"],
            direction=float(c["direction"]),
            persuasiveness=float(c["persuasiveness"]),
            irt_a=None if irt_a is None else float(irt_a),
            irt_b=None if irt_b is None else float(irt_b),
        ))

    graph = AttackGraph()
    for a in data.get("attacks", []):
        graph.add_attack(a["attacker"], a["target"], float(a["strength"]))
    for s in data.get("supports", []):
        graph.add_support(s["supporter"], s["target"], float(s["strength"]))
    pool.attack_graph = graph

    return pool


def load_builtin(name: str) -> ArgumentPool:
    """Load a built-in scenario by name (e.g. 'carbon_tax')."""
    return load_scenario(SCENARIOS_DIR / f"{name}.json")


def from_agora_scenario(name: str) -> ArgumentPool:
    """Load an ArgumentPool from the agora hand-crafted scenarios.

    Supported names: 'barabas_consensual', 'barabas_non_consensual',
    'jackman_sniderman_symmetric'.
    """
    from agora.scenarios import (  # type: ignore[import-untyped]
        barabas_consensual,
        barabas_non_consensual,
        jackman_sniderman_symmetric,
    )

    builders = {
        "barabas_consensual": barabas_consensual,
        "barabas_non_consensual": barabas_non_consensual,
        "jackman_sniderman_symmetric": jackman_sniderman_symmetric,
    }
    if name not in builders:
        raise ValueError(f"Unknown agora scenario: {name}. Available: {list(builders)}")
    scenario = builders[name]()
    return scenario.pool
