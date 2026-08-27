"""Simulation history: per-round state snapshots for analysis and replay.

Version : 0.4.0
Module  : agora.history
Spec    : manuscript/theoretical-appendix.tex §9.4

Overview
--------
    AgentSnapshot     — frozen record of (agent_id, opinion, weights, repertoire)
                        at a single point in time.
    StateSnapshot     — frozen tuple of AgentSnapshots for one round, plus metadata.
    SimulationHistory — ordered list of StateSnapshots; provides record(),
                        get_round(n), and n_rounds.

The history sequence H = (S^0, S^1, ..., S^T) is the primary
data structure consumed by metrics and the I/O layer.

Changelog
---------
0.4.0  e3b2d53  Restore from calibration branch (baseline)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentSnapshot:
    """Immutable snapshot of a single agent's state at a point in time."""

    agent_id: str
    opinion: float
    weights: dict[str, float]
    repertoire: frozenset[str]


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable snapshot of the full simulation state at a given round."""

    round_num: int
    agent_snapshots: tuple[AgentSnapshot, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationHistory:
    """Accumulates StateSnapshots across rounds."""

    snapshots: list[StateSnapshot] = field(default_factory=list)

    def record(self, snapshot: StateSnapshot) -> None:
        self.snapshots.append(snapshot)

    def get_round(self, round_num: int) -> StateSnapshot | None:
        for s in self.snapshots:
            if s.round_num == round_num:
                return s
        return None

    @property
    def n_rounds(self) -> int:
        return len(self.snapshots)
