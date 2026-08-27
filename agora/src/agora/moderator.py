"""Facilitation mechanisms: group building, group assignment, and moderation.

Version : 0.4.2
Module  : agora.moderator
Spec    : manuscript/theoretical-appendix.tex §6

Overview
--------
Three institutional-design components:

    GroupBuildingLevel  — ordinal 1–5 encoding trust-building intensity;
                         each level adds 0.076 to agents' open-mindedness.
    GroupAssigner       — static methods for heterogeneous (round-robin by
                         opinion), homogeneous (contiguous blocks), and
                         random partitioning.  Group count uses floor
                         division: n_groups = max(1, N // group_size).
    Moderator          — when active, enforces speaking equity (ascending
                         speak-count order) and injects ≤1 missing
                         consideration per round.  Also provides
                         adjust_open_mindedness() for the GB bonus.

Key details
-----------
- Group sizes are approximate: remainder agents are distributed
  across groups, so some groups get +1 member.
- Substantive intervention returns [] when every pool consideration
  is already held by at least one group member.

Changelog
---------
0.4.2  c5de059  [US-002] Moderator wiring, GroupAssigner strategies
0.4.0  e3b2d53  Restore from calibration branch (baseline)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

    from agora.agents import Agent
    from agora.considerations import ArgumentPool


class GroupBuildingLevel(IntEnum):
    """Niemeyer ordinal scale for group-building intensity (1-5).

    open_mindedness_bonus = 0.076 * (level - 1).
    """

    MINIMAL = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    FULL = 5

    def open_mindedness_bonus(self) -> float:
        """Return the open-mindedness bonus for this level."""
        return 0.076 * (self.value - 1)


@dataclass
class GroupAssigner:
    """Assigns agents to deliberation groups."""

    @staticmethod
    def heterogeneous(
        agents: list[Agent],
        group_size: int,
        pool: ArgumentPool,
    ) -> list[list[Agent]]:
        """Round-robin by sorted opinion for maximum within-group diversity."""
        sorted_agents = sorted(agents, key=lambda a: a.opinion(pool))
        n_groups = max(1, len(agents) // group_size)
        groups: list[list[Agent]] = [[] for _ in range(n_groups)]
        for i, agent in enumerate(sorted_agents):
            groups[i % n_groups].append(agent)
        return groups

    @staticmethod
    def homogeneous(
        agents: list[Agent],
        group_size: int,
        pool: ArgumentPool,
    ) -> list[list[Agent]]:
        """Contiguous blocks by sorted opinion for within-group similarity."""
        sorted_agents = sorted(agents, key=lambda a: a.opinion(pool))
        n_groups = max(1, len(agents) // group_size)
        groups: list[list[Agent]] = []
        base_size = len(sorted_agents) // n_groups
        remainder = len(sorted_agents) % n_groups
        idx = 0
        for g in range(n_groups):
            size = base_size + (1 if g < remainder else 0)
            groups.append(sorted_agents[idx : idx + size])
            idx += size
        return groups

    @staticmethod
    def random(
        agents: list[Agent],
        group_size: int,
        rng: Generator,
    ) -> list[list[Agent]]:
        """Random assignment to groups."""
        shuffled = list(agents)
        rng.shuffle(shuffled)
        n_groups = max(1, len(agents) // group_size)
        groups: list[list[Agent]] = [[] for _ in range(n_groups)]
        for i, agent in enumerate(shuffled):
            groups[i % n_groups].append(agent)
        return groups


@dataclass
class Moderator:
    """Epstein-Leshed triage moderator.

    When active, provides speaking equity enforcement, substantive
    intervention (inject underrepresented considerations), and
    open-mindedness adjustment.
    """

    active: bool = True
    group_building_level: GroupBuildingLevel = GroupBuildingLevel.MINIMAL

    def enforce_speaking_equity(
        self,
        agents: list[Agent],
        rng: Generator,
    ) -> list[Agent]:
        """Sort agents by speaking count ascending; break ties randomly.

        Returns a reordered list placing under-represented speakers first.
        When inactive, returns the input list unchanged.
        """
        if not self.active:
            return list(agents)
        # Assign random tiebreakers
        tiebreakers = {a.id: float(rng.random()) for a in agents}
        return sorted(agents, key=lambda a: (a.speaking_count, tiebreakers[a.id]))

    def intervene_substantive(
        self,
        group: list[Agent],
        pool: ArgumentPool,
        n: int = 1,
        rng: Generator | None = None,
    ) -> list[str]:
        """Return consideration ids not in any group member's repertoire.

        Samples up to n considerations from the pool that no group member
        currently holds. When inactive, returns an empty list.
        """
        if not self.active:
            return []
        group_repertoire: set[str] = set()
        for agent in group:
            group_repertoire.update(agent.repertoire)
        available = [cid for cid in pool.all_ids() if cid not in group_repertoire]
        if not available:
            return []
        if rng is None:
            rng = np.random.default_rng()
        k = min(n, len(available))
        return list(rng.choice(available, size=k, replace=False))

    def adjust_open_mindedness(self, base_open_mindedness: float) -> float:
        """Return adjusted open-mindedness with group-building bonus.

        Does not mutate frozen AgentParams. When inactive, returns the
        base value unchanged.
        """
        if not self.active:
            return base_open_mindedness
        bonus = self.group_building_level.open_mindedness_bonus()
        return min(base_open_mindedness + bonus, 1.0)
