"""Deliberation protocols defining institutional rules for mini-publics.

Version : 0.5.0
Module  : agora.protocols
Spec    : manuscript/theoretical-appendix.tex §5

Overview
--------
Each protocol orchestrates a sequence of deliberation rounds over
a partitioned or full-group agent population.  Five formats are
provided:

    Plenary            — all agents, single group, T rounds
    Jury               — fixed random subset of size n_jury
    TownHall           — plenary with moderator forced inactive
    CitizensAssembly   — info phase → breakout phase → plenary phase
    CommitteePlenary   — committee phase → plenary phase

All protocols share a common round execution kernel
(_run_round_on_group) that implements the speaker loop,
probabilistic update gate (p_update^|o_i - o_s|), repertoire
dynamics, and engine hook dispatch.

Key invariants
--------------
- Round-0 baseline state snapshot is always recorded before
  deliberation begins.
- Repertoire expansion (learning mode) is idempotent: existing
  considerations keep their current weight.
- When moderator is None or inactive, speakers proceed in list order.

Changelog
---------
0.5.0  vectorization  [US-VEC-006/007] Vectorized evaluate + reflect
0.4.3  14fd3f6  [US-003] Probabilistic update gate in protocol layer
0.4.0  e3b2d53  Restore from calibration branch (baseline)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from agora.agents import Agent
from agora.considerations import ArgumentPool
from agora.history import AgentSnapshot, SimulationHistory, StateSnapshot
from agora.moderator import GroupAssigner, GroupBuildingLevel, Moderator

if TYPE_CHECKING:
    from numpy.random import Generator

    from agora.engines import CognitiveEngine
    from agora.state_tensor import StateTensor


@dataclass(frozen=True)
class RoundResult:
    """Immutable record of a single deliberation round."""

    round_num: int
    groups: list[list[str]]
    voiced_arguments: list[tuple[str, str]]  # (speaker_id, cid)
    opinion_changes: dict[str, float]  # agent_id -> delta
    moderator_interventions: list[str]


def _snapshot_agents(agents: list[Agent], pool: ArgumentPool, round_num: int) -> StateSnapshot:
    """Create a StateSnapshot from the current agent states."""
    snaps = tuple(
        AgentSnapshot(
            agent_id=a.id,
            opinion=a.opinion(pool),
            weights=dict(a.weights),
            repertoire=a.repertoire,
        )
        for a in agents
    )
    return StateSnapshot(round_num=round_num, agent_snapshots=snaps)


def _run_round_on_group(
    group: list[Agent],
    engine: CognitiveEngine,
    pool: ArgumentPool,
    rng: Generator,
    moderator: Moderator | None,
    repertoire_dynamics: Literal["static", "learning"],
    p_update: float = 1.0,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Execute one round of deliberation within a single group.

    Returns (voiced_arguments, moderator_interventions).

    Uses vectorized batch operations (evaluate_batch, reflect_batch)
    when the engine supports them. Falls back to per-agent operations
    for engines without batch methods (e.g., MixedEngine).
    """
    from agora.engines import SimpleUpdateEngine

    # Determine speaking order
    if moderator is not None and moderator.active:
        speaking_order = moderator.enforce_speaking_equity(group, rng)
    else:
        speaking_order = list(group)

    # Collect pending updates per agent
    pending: dict[str, list[tuple[str, float, float]]] = {a.id: [] for a in group}
    voiced: list[tuple[str, str]] = []
    interventions: list[str] = []

    # Moderator substantive intervention: inject underrepresented considerations
    if moderator is not None and moderator.active:
        injected = moderator.intervene_substantive(group, pool, n=1, rng=rng)
        for cid in injected:
            interventions.append(cid)
            if repertoire_dynamics == "learning":
                for agent in group:
                    agent.add_to_repertoire(cid, initial_weight=0.0)

    n = len(group)

    # Vectorized evaluate: only for static dynamics + SimpleUpdateEngine
    use_vec_evaluate = (
        repertoire_dynamics == "static"
        and isinstance(engine, SimpleUpdateEngine)
    )
    use_vec_reflect = hasattr(engine, "reflect_batch")

    # Build StateTensor for evaluate_batch (opinions stable in static mode)
    eval_state: StateTensor | None = None
    if use_vec_evaluate:
        from agora.state_tensor import StateTensor as ST

        eval_state = ST.from_agents(group, pool)

    # Build cid_to_idx for vec_pending
    cid_to_idx: dict[str, int] | None = None
    if use_vec_reflect:
        if eval_state is not None:
            cid_to_idx = eval_state.cid_to_idx
        else:
            all_cids = pool.all_ids()
            cid_to_idx = {cid: j for j, cid in enumerate(all_cids)}

    vec_pending: list[tuple[int, float, np.ndarray]] = []

    # Pre-compute opinions for p_update gate (avoids O(N^2) per-agent calls)
    gate_opinions: list[float] | None = None
    if p_update < 1.0 and repertoire_dynamics == "static":
        gate_opinions = [a.opinion(pool) for a in group]

    for speaker in speaking_order:
        cid = engine.voice(speaker, pool, rng)
        voiced.append((speaker.id, cid))
        speaker.speaking_count += 1
        speaker_opinion = speaker.opinion(pool)

        # Vectorized evaluate: all agents at once
        phi_all: np.ndarray | None = None
        if use_vec_evaluate and eval_state is not None:
            cid_idx = eval_state.cid_to_idx[cid]
            phi_all = engine.evaluate_batch(  # type: ignore[attr-defined]
                eval_state, cid_idx, speaker_opinion, rng,
            )

        phi_vec: np.ndarray | None = None
        if use_vec_reflect:
            phi_vec = np.zeros(n, dtype=np.float64)

        for i, listener in enumerate(group):
            if listener.id == speaker.id:
                continue

            if phi_all is not None:
                influence = float(phi_all[i])
            else:
                influence = engine.evaluate(
                    listener, cid, speaker_opinion, pool, rng,
                )

            # Butler Eq.6 probabilistic gate: p = p_update^|delta|
            if p_update >= 1.0 or rng.random() < p_update ** abs(
                (gate_opinions[i] if gate_opinions is not None
                 else listener.opinion(pool)) - speaker_opinion
            ):
                pending[listener.id].append((cid, speaker_opinion, influence))
                if phi_vec is not None:
                    phi_vec[i] = influence

            # repertoire_dynamics='learning': add voiced consideration if not held
            if repertoire_dynamics == "learning" and cid not in listener.weights:
                listener.add_to_repertoire(cid, initial_weight=0.0)

        if use_vec_reflect and cid_to_idx is not None and phi_vec is not None:
            vec_pending.append((cid_to_idx[cid], speaker_opinion, phi_vec))

    # Reflect phase: vectorized when engine supports it
    if use_vec_reflect:
        from agora.state_tensor import StateTensor as ST

        reflect_state = ST.from_agents(group, pool)
        engine.reflect_batch(reflect_state, vec_pending, pool, rng)  # type: ignore[attr-defined]
        reflect_state.sync_back(group)
    else:
        for agent in group:
            engine.reflect(agent, pending[agent.id], pool, rng)

    return voiced, interventions


class Protocol(ABC):
    """Abstract base for deliberation protocols."""

    engine: CognitiveEngine
    moderator: Moderator | None
    group_building: GroupBuildingLevel
    n_rounds: int
    repertoire_dynamics: Literal["static", "learning"]
    p_update: float

    @abstractmethod
    def assign_groups(
        self, agents: list[Agent], pool: ArgumentPool, rng: Generator
    ) -> list[list[Agent]]:
        """Partition agents into deliberation groups for a round."""
        ...

    @abstractmethod
    def run(
        self, agents: list[Agent], pool: ArgumentPool, seed: int = 0
    ) -> SimulationHistory:
        """Run the full protocol and return simulation history."""
        ...

    def is_complete(self, round_num: int) -> bool:
        """Whether the protocol is done after round_num rounds."""
        return round_num >= self.n_rounds


@dataclass
class Plenary(Protocol):
    """All agents in one group; every agent speaks once per round."""

    engine: CognitiveEngine = field(default=None)  # type: ignore[assignment]
    moderator: Moderator | None = None
    group_building: GroupBuildingLevel = GroupBuildingLevel.MINIMAL
    n_rounds: int = 10
    repertoire_dynamics: Literal["static", "learning"] = "static"
    p_update: float = 1.0

    def assign_groups(
        self, agents: list[Agent], pool: ArgumentPool, rng: Generator
    ) -> list[list[Agent]]:
        return [list(agents)]

    def run(
        self, agents: list[Agent], pool: ArgumentPool, seed: int = 0
    ) -> SimulationHistory:
        rng = np.random.default_rng(seed)
        history = SimulationHistory()
        history.record(_snapshot_agents(agents, pool, 0))

        for r in range(1, self.n_rounds + 1):
            groups = self.assign_groups(agents, pool, rng)
            for group in groups:
                _run_round_on_group(
                    group, self.engine, pool, rng,
                    self.moderator, self.repertoire_dynamics,
                    p_update=self.p_update,
                )
            history.record(_snapshot_agents(agents, pool, r))

        return history


@dataclass
class Jury(Protocol):
    """Fixed jury of jury_size agents, moderated."""

    engine: CognitiveEngine = field(default=None)  # type: ignore[assignment]
    moderator: Moderator | None = None
    group_building: GroupBuildingLevel = GroupBuildingLevel.MINIMAL
    n_rounds: int = 10
    repertoire_dynamics: Literal["static", "learning"] = "static"
    p_update: float = 1.0
    jury_size: int = 12

    def assign_groups(
        self, agents: list[Agent], pool: ArgumentPool, rng: Generator
    ) -> list[list[Agent]]:
        if len(agents) <= self.jury_size:
            return [list(agents)]
        indices = rng.choice(len(agents), size=self.jury_size, replace=False)
        return [[agents[i] for i in sorted(indices)]]

    def run(
        self, agents: list[Agent], pool: ArgumentPool, seed: int = 0
    ) -> SimulationHistory:
        rng = np.random.default_rng(seed)
        history = SimulationHistory()

        # Select jury once (fixed)
        groups = self.assign_groups(agents, pool, rng)
        jury = groups[0]

        history.record(_snapshot_agents(agents, pool, 0))

        for r in range(1, self.n_rounds + 1):
            _run_round_on_group(
                jury, self.engine, pool, rng,
                self.moderator, self.repertoire_dynamics,
                p_update=self.p_update,
            )
            history.record(_snapshot_agents(agents, pool, r))

        return history


@dataclass
class CitizensAssembly(Protocol):
    """3-phase protocol: information, breakout, plenary.

    Phase 1 (information): expand all repertoires to the full pool.
    Phase 2 (breakout): small groups (8-12) via GroupAssigner.
    Phase 3 (plenary): all agents in one group.
    """

    engine: CognitiveEngine = field(default=None)  # type: ignore[assignment]
    moderator: Moderator | None = None
    group_building: GroupBuildingLevel = GroupBuildingLevel.MINIMAL
    n_rounds: int = 6
    repertoire_dynamics: Literal["static", "learning"] = "static"
    p_update: float = 1.0
    breakout_size: int = 10
    group_composition: Literal["heterogeneous", "homogeneous", "random"] = "heterogeneous"
    info_rounds: int = 1
    breakout_rounds: int = 3
    plenary_rounds: int = 2

    def __post_init__(self) -> None:
        self.n_rounds = self.info_rounds + self.breakout_rounds + self.plenary_rounds

    def assign_groups(
        self, agents: list[Agent], pool: ArgumentPool, rng: Generator
    ) -> list[list[Agent]]:
        """Default: heterogeneous breakout groups."""
        if self.group_composition == "heterogeneous":
            return GroupAssigner.heterogeneous(agents, self.breakout_size, pool)
        elif self.group_composition == "homogeneous":
            return GroupAssigner.homogeneous(agents, self.breakout_size, pool)
        else:
            return GroupAssigner.random(agents, self.breakout_size, rng)

    def run(
        self, agents: list[Agent], pool: ArgumentPool, seed: int = 0
    ) -> SimulationHistory:
        rng = np.random.default_rng(seed)
        history = SimulationHistory()
        history.record(_snapshot_agents(agents, pool, 0))
        round_num = 0

        # Phase 1: Information — expand repertoires to full pool
        for _ in range(self.info_rounds):
            round_num += 1
            all_cids = pool.all_ids()
            for agent in agents:
                for cid in all_cids:
                    agent.add_to_repertoire(cid, initial_weight=0.0)
            # Run a plenary round so agents process the new considerations
            _run_round_on_group(
                list(agents), self.engine, pool, rng,
                self.moderator, self.repertoire_dynamics,
                p_update=self.p_update,
            )
            history.record(_snapshot_agents(agents, pool, round_num))

        # Phase 2: Breakout groups
        for _ in range(self.breakout_rounds):
            round_num += 1
            groups = self.assign_groups(agents, pool, rng)
            for group in groups:
                _run_round_on_group(
                    group, self.engine, pool, rng,
                    self.moderator, self.repertoire_dynamics,
                    p_update=self.p_update,
                )
            history.record(_snapshot_agents(agents, pool, round_num))

        # Phase 3: Plenary
        for _ in range(self.plenary_rounds):
            round_num += 1
            _run_round_on_group(
                list(agents), self.engine, pool, rng,
                self.moderator, self.repertoire_dynamics,
                p_update=self.p_update,
            )
            history.record(_snapshot_agents(agents, pool, round_num))

        return history


@dataclass
class TownHall(Protocol):
    """Unstructured deliberation, no facilitation.

    Even if a Moderator object is passed, all interventions are disabled.
    """

    engine: CognitiveEngine = field(default=None)  # type: ignore[assignment]
    moderator: Moderator | None = None
    group_building: GroupBuildingLevel = GroupBuildingLevel.MINIMAL
    n_rounds: int = 10
    repertoire_dynamics: Literal["static", "learning"] = "static"
    p_update: float = 1.0

    def assign_groups(
        self, agents: list[Agent], pool: ArgumentPool, rng: Generator
    ) -> list[list[Agent]]:
        return [list(agents)]

    def run(
        self, agents: list[Agent], pool: ArgumentPool, seed: int = 0
    ) -> SimulationHistory:
        rng = np.random.default_rng(seed)
        history = SimulationHistory()
        history.record(_snapshot_agents(agents, pool, 0))

        # Force inactive moderator — TownHall has no facilitation
        inactive_mod = Moderator(active=False)

        for r in range(1, self.n_rounds + 1):
            groups = self.assign_groups(agents, pool, rng)
            for group in groups:
                _run_round_on_group(
                    group, self.engine, pool, rng,
                    inactive_mod, self.repertoire_dynamics,
                    p_update=self.p_update,
                )
            history.record(_snapshot_agents(agents, pool, r))

        return history


@dataclass
class CommitteePlenary(Protocol):
    """K&B format: committee subgroup deliberates, then reports to plenary.

    Committee rounds happen first. Committee members' voiced considerations
    are then added to all plenary agents' repertoires before plenary rounds.
    """

    engine: CognitiveEngine = field(default=None)  # type: ignore[assignment]
    moderator: Moderator | None = None
    group_building: GroupBuildingLevel = GroupBuildingLevel.MINIMAL
    n_rounds: int = 10
    repertoire_dynamics: Literal["static", "learning"] = "static"
    p_update: float = 1.0
    committee_size: int = 5
    committee_rounds: int = 5
    plenary_rounds: int = 5
    group_composition: Literal["heterogeneous", "homogeneous", "random"] = "heterogeneous"

    def __post_init__(self) -> None:
        self.n_rounds = self.committee_rounds + self.plenary_rounds

    def assign_groups(
        self, agents: list[Agent], pool: ArgumentPool, rng: Generator
    ) -> list[list[Agent]]:
        """Assign committee groups using the configured composition."""
        if self.group_composition == "heterogeneous":
            return GroupAssigner.heterogeneous(agents, self.committee_size, pool)
        elif self.group_composition == "homogeneous":
            return GroupAssigner.homogeneous(agents, self.committee_size, pool)
        else:
            return GroupAssigner.random(agents, self.committee_size, rng)

    def run(
        self, agents: list[Agent], pool: ArgumentPool, seed: int = 0
    ) -> SimulationHistory:
        rng = np.random.default_rng(seed)
        history = SimulationHistory()
        history.record(_snapshot_agents(agents, pool, 0))
        round_num = 0

        # Select committee
        groups = self.assign_groups(agents, pool, rng)
        committee = groups[0]
        committee_ids = {a.id for a in committee}

        # Track considerations voiced during committee phase
        committee_voiced_cids: set[str] = set()

        # Phase 1: Committee rounds
        for _ in range(self.committee_rounds):
            round_num += 1
            voiced, _ = _run_round_on_group(
                committee, self.engine, pool, rng,
                self.moderator, self.repertoire_dynamics,
                p_update=self.p_update,
            )
            for _, cid in voiced:
                committee_voiced_cids.add(cid)
            history.record(_snapshot_agents(agents, pool, round_num))

        # Committee report: expand plenary agents' repertoires with committee findings
        non_committee = [a for a in agents if a.id not in committee_ids]
        for agent in non_committee:
            for cid in committee_voiced_cids:
                agent.add_to_repertoire(cid, initial_weight=0.0)

        # Phase 2: Plenary rounds
        for _ in range(self.plenary_rounds):
            round_num += 1
            _run_round_on_group(
                list(agents), self.engine, pool, rng,
                self.moderator, self.repertoire_dynamics,
                p_update=self.p_update,
            )
            history.record(_snapshot_agents(agents, pool, round_num))

        return history
