"""Scenario generation: procedural and hand-crafted deliberation scenarios.

Version : 0.4.2
Module  : agora.scenarios
Spec    : manuscript/theoretical-appendix.tex §8.3

Overview
--------
    Scenario           — frozen dataclass (name, pool: ArgumentPool)
    ScenarioGenerator  — procedural generation with configurable:
                         n_considerations (15), pro_con_balance (0.5),
                         attack_density (0.1), strength_distribution
                         ("uniform" | "beta").

Procedural generation details
-----------------------------
- Strength floors: uniform → s_c ∈ [0.1, 1.0]; beta → clip to [0.01, 1.0].
- Attacks only between opposing-direction pairs, with probability =
  attack_density and strength ~ U(0.3, 1.0).
- Bidirectional with P = 0.5.

Hand-crafted scenarios (all use persuasiveness = 0.5)
---------------------------------------------------
    barabas_consensual       — 13 pro / 2 con, 2 unidirectional attacks (σ=0.6)
    barabas_non_consensual   — 7 pro / 8 con, 2 bidirectional attacks (σ=0.6)
    jackman_sniderman_symmetric — 7 pro / 7 con, 7 symmetric attack pairs (σ=0.5)

Changelog
---------
0.4.2  c5de059  [US-002] J&S scenario fix, direction corrections
0.4.0  e3b2d53  Restore from calibration branch (baseline)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from agora.considerations import ArgumentPool, Consideration

if TYPE_CHECKING:
    from numpy.random import Generator


@dataclass(frozen=True)
class Scenario:
    """A deliberation scenario: named ArgumentPool with considerations and attacks."""

    name: str
    pool: ArgumentPool


@dataclass
class ScenarioGenerator:
    """Parameterized scenario generator.

    Attributes:
        n_considerations: Total number of considerations to generate.
        pro_con_balance: Fraction that are pro (direction=+1). Rest are con.
        attack_density: Probability of an attack between opposing considerations.
        strength_distribution: How persuasiveness is sampled ("uniform" or "beta").
    """

    n_considerations: int = 15
    pro_con_balance: float = 0.5
    attack_density: float = 0.1
    strength_distribution: Literal["uniform", "beta"] = "uniform"

    def generate(self, seed: int = 0) -> Scenario:
        """Generate a scenario with the configured parameters."""
        rng: Generator = np.random.default_rng(seed)
        pool = ArgumentPool()

        n_pro = round(self.n_considerations * self.pro_con_balance)
        n_con = self.n_considerations - n_pro

        for i in range(1, self.n_considerations + 1):
            direction = 1.0 if i <= n_pro else -1.0
            if self.strength_distribution == "uniform":
                strength = float(rng.uniform(0.1, 1.0))
            else:
                strength = float(rng.beta(2.0, 2.0))
            pool.add(Consideration(
                id=f"C_{i:02d}",
                label=f"{'Pro' if direction > 0 else 'Con'} {i}",
                direction=direction,
                persuasiveness=float(np.clip(strength, 0.01, 1.0)),
            ))

        # Generate attacks between opposing considerations
        ids = pool.all_ids()
        for i, a_id in enumerate(ids):
            for j in range(i + 1, len(ids)):
                b_id = ids[j]
                a_dir = pool.get(a_id).direction
                b_dir = pool.get(b_id).direction
                if a_dir * b_dir < 0 and float(rng.random()) < self.attack_density:
                    atk_strength = float(rng.uniform(0.3, 1.0))
                    pool.attack_graph.add_attack(a_id, b_id, atk_strength)
                    if float(rng.random()) < 0.5:
                        pool.attack_graph.add_attack(b_id, a_id, atk_strength)

        _ = n_con  # used implicitly via n_pro
        name = (
            f"gen_{self.n_considerations}c"
            f"_{self.pro_con_balance:.2f}b"
            f"_{self.attack_density:.2f}d"
        )
        return Scenario(name=name, pool=pool)


# ---------------------------------------------------------------------------
# Hand-crafted fixtures
# ---------------------------------------------------------------------------


def barabas_consensual() -> Scenario:
    """Barabas consensual scenario: 13 pro / 2 con (>5:1 ratio)."""
    pool = ArgumentPool()
    for i in range(1, 14):
        pool.add(Consideration(
            id=f"C_{i:02d}", label=f"Pro {i}",
            direction=1.0, persuasiveness=0.5,
        ))
    for i in range(14, 16):
        pool.add(Consideration(
            id=f"C_{i:02d}", label=f"Con {i}",
            direction=-1.0, persuasiveness=0.5,
        ))
    pool.attack_graph.add_attack("C_14", "C_01", 0.6)
    pool.attack_graph.add_attack("C_15", "C_02", 0.6)
    return Scenario(name="barabas_consensual", pool=pool)


def barabas_non_consensual() -> Scenario:
    """Barabas non-consensual scenario: 7 pro / 8 con (~1:1 ratio)."""
    pool = ArgumentPool()
    for i in range(1, 8):
        pool.add(Consideration(
            id=f"C_{i:02d}", label=f"Pro {i}",
            direction=1.0, persuasiveness=0.5,
        ))
    for i in range(8, 16):
        pool.add(Consideration(
            id=f"C_{i:02d}", label=f"Con {i}",
            direction=-1.0, persuasiveness=0.5,
        ))
    pool.attack_graph.add_attack("C_08", "C_01", 0.6)
    pool.attack_graph.add_attack("C_09", "C_02", 0.6)
    pool.attack_graph.add_attack("C_01", "C_08", 0.6)
    pool.attack_graph.add_attack("C_02", "C_09", 0.6)
    return Scenario(name="barabas_non_consensual", pool=pool)


def jackman_sniderman_symmetric() -> Scenario:
    """Jackman-Sniderman symmetric scenario: 7 pro / 7 con, equal strengths."""
    pool = ArgumentPool()
    for i in range(1, 8):
        pool.add(Consideration(
            id=f"C_{i:02d}", label=f"Pro {i}",
            direction=1.0, persuasiveness=0.5,
        ))
    for i in range(8, 15):
        pool.add(Consideration(
            id=f"C_{i:02d}", label=f"Con {i}",
            direction=-1.0, persuasiveness=0.5,
        ))
    # Symmetric attacks: each pro attacks its paired con and vice versa
    for i in range(1, 8):
        pro_id = f"C_{i:02d}"
        con_id = f"C_{i + 7:02d}"
        pool.attack_graph.add_attack(pro_id, con_id, 0.5)
        pool.attack_graph.add_attack(con_id, pro_id, 0.5)
    return Scenario(name="jackman_sniderman_symmetric", pool=pool)
