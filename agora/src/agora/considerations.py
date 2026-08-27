"""Argumentation layer: considerations, attack/support graphs, and argument pool.

Version : 0.4.0
Module  : agora.considerations
Spec    : manuscript/theoretical-appendix.tex §2

Overview
--------
The argumentation layer defines the atomic units of deliberation:

    Consideration  — frozen dataclass (id, label, direction ∈ [-1,1],
                     persuasiveness ∈ [0,1]).  Positive direction = pro,
                     negative = con.
    AttackGraph    — weighted directed edges: attacks and supports between
                     considerations.  Note: the supports relation is defined
                     but currently unused by any engine or scenario.
    ArgumentPool   — the universe of considerations plus its attack graph.
                     Provides sample_repertoire(size, rng) for agent init.

Key details
-----------
- Attack/support strengths are not validated to (0,1]; any float is accepted.
- sample_repertoire returns min(k, M) items; when k ≥ M, the full pool.

Changelog
---------
0.4.0  e3b2d53  Restore from calibration branch (baseline)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator


@dataclass(frozen=True)
class Consideration:
    """A single consideration (argument) in the deliberation pool.

    Attributes:
        id: Unique identifier (e.g., "C_01").
        label: Human-readable label.
        direction: Position supported, in [-1, 1]. Positive = pro, negative = con.
        persuasiveness: Intrinsic persuasive quality, in [0, 1].
        irt_a: Optional 2PL IRT discrimination (signed; + for pro-aligned
               items, - for con-aligned). Used by engines to compute
               dynamic theta. None = fall back to naive opinion formula.
        irt_b: Optional 2PL IRT difficulty. None = fall back to naive
               opinion formula.
    """

    id: str
    label: str
    direction: float  # [-1, 1]
    persuasiveness: float  # [0, 1]
    irt_a: float | None = None
    irt_b: float | None = None

    def __post_init__(self) -> None:
        if not -1.0 <= self.direction <= 1.0:
            raise ValueError(f"direction must be in [-1, 1], got {self.direction}")
        if not 0.0 <= self.persuasiveness <= 1.0:
            raise ValueError(f"persuasiveness must be in [0, 1], got {self.persuasiveness}")


@dataclass
class AttackGraph:
    """Stores attack and support relations between considerations.

    attacks: mapping from (attacker_id, target_id) -> strength.
    supports: mapping from (supporter_id, target_id) -> strength.
    """

    attacks: dict[tuple[str, str], float] = field(default_factory=dict)
    supports: dict[tuple[str, str], float] = field(default_factory=dict)

    def add_attack(self, attacker_id: str, target_id: str, strength: float = 1.0) -> None:
        self.attacks[(attacker_id, target_id)] = strength

    def add_support(self, supporter_id: str, target_id: str, strength: float = 1.0) -> None:
        self.supports[(supporter_id, target_id)] = strength

    def get_attackers(self, target_id: str) -> dict[str, float]:
        """Return {attacker_id: strength} for all attacks on target_id."""
        return {
            attacker: strength
            for (attacker, target), strength in self.attacks.items()
            if target == target_id
        }

    def get_supporters(self, target_id: str) -> dict[str, float]:
        """Return {supporter_id: strength} for all supports of target_id."""
        return {
            supporter: strength
            for (supporter, target), strength in self.supports.items()
            if target == target_id
        }


@dataclass
class ArgumentPool:
    """Pool of all considerations and their relations.

    Attributes:
        considerations: mapping from consideration id to Consideration.
        attack_graph: AttackGraph storing inter-consideration relations.
    """

    considerations: dict[str, Consideration] = field(default_factory=dict)
    attack_graph: AttackGraph = field(default_factory=AttackGraph)

    def add(self, consideration: Consideration) -> None:
        self.considerations[consideration.id] = consideration

    def get(self, cid: str) -> Consideration:
        return self.considerations[cid]

    def all_ids(self) -> list[str]:
        return list(self.considerations.keys())

    def sample_repertoire(
        self, size: int, rng: Generator | None = None
    ) -> list[str]:
        """Return a random subset of consideration ids of the given size."""
        ids = self.all_ids()
        if size >= len(ids):
            return list(ids)
        if rng is None:
            rng = np.random.default_rng()
        return list(rng.choice(ids, size=size, replace=False))
