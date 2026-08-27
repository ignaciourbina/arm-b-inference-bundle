"""Experiment orchestration: configuration, factorial design, and runner.

Version : 0.4.5
Module  : agora.experiment
Spec    : manuscript/theoretical-appendix.tex §9

Overview
--------
A single simulation run is fully specified by an ExperimentConfig
frozen dataclass.  The Runner builds a scenario, generates a
population, executes the selected protocol, and computes all
outcome metrics.

    ExperimentConfig  — 20+ parameter tuple (see §9.1 table)
    FactorialDesign   — full-factorial sweep with deterministic seeding:
                        seed(j,l) = base_seed + j * replications + l
    Runner            — run_single() / run_batch() execution pipeline

Engine string mapping
---------------------
    "degroot"              → DeGrootEngine
    "bayesian"             → BayesianEngine
    "bounded_confidence"   → BoundedConfidenceEngine
    "structural"           → StructuralAlignmentEngine
    "argument_based"       → ArgumentBasedEngine

Note: MixedEngine is not reachable via string config; it must be
constructed programmatically.

Changelog
---------
0.4.5  2f8312e  Factorial sweep script integration
0.4.5  5daff0a  [US-005] ArgumentBasedEngine config wiring
0.4.4  f8be5e0  [US-004] Group-building level config
0.4.3  14fd3f6  [US-003] p_update config parameter
0.4.0  e3b2d53  Restore from calibration branch (baseline)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from agora.agents import AgentPopulation
from agora.engines import (
    ArgumentBasedEngine,
    BayesianEngine,
    BoundedConfidenceEngine,
    CognitiveEngine,
    DeGrootEngine,
    StructuralAlignmentEngine,
)
from agora.history import AgentSnapshot, StateSnapshot
from agora.metrics import (
    AlphaFairness,
    DRI,
    DistributionalMetrics,
    MetricsSuite,
)
from agora.moderator import GroupBuildingLevel, Moderator
from agora.protocols import (
    CitizensAssembly,
    CommitteePlenary,
    Jury,
    Plenary,
    Protocol,
    TownHall,
)
from agora.scenarios import (
    Scenario,
    ScenarioGenerator,
    barabas_consensual,
    barabas_non_consensual,
    jackman_sniderman_symmetric,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen configuration for a single experiment run."""

    name: str
    seed: int
    n_agents: int = 20
    protocol: str = "plenary"
    engine: str = "degroot"
    group_size: int = 10
    group_composition: str = "heterogeneous"
    facilitation: bool = True
    group_building_level: int = 1
    n_rounds: int = 10
    repertoire_dynamics: str = "static"
    scenario: str = "generated"
    pro_con_balance: float = 0.5
    n_considerations: int = 15
    attack_density: float = 0.1
    alpha_values: tuple[float, ...] = (0.0, 1.0, 2.0)
    engine_mu: float = 0.006
    engine_signal_precision: float = 0.02
    engine_strategic_voicing: bool = True
    engine_p_accept_base: float = 0.28
    engine_p_reject_base: float = 0.12
    engine_confirmation_bias: float = 0.0
    engine_lambda_salience: float = 0.05
    engine_p_update: float = 1.0
    graph_aware_eval: bool = False
    graph_aware_propagation: bool = True
    propagation_rate: float = 0.1
    repertoire_correlation: float = 0.0


@dataclass
class FactorialDesign:
    """Full factorial cross of factor levels.

    Attributes:
        factors: Mapping from ExperimentConfig field names to lists of values.
        replications: Number of replications per factor combination.
        base_seed: Starting seed; each config gets a deterministic non-colliding seed.
    """

    factors: dict[str, list[Any]]
    replications: int = 1
    base_seed: int = 42

    def generate_configs(self, name_prefix: str = "exp") -> list[ExperimentConfig]:
        """Generate the full cross-product of configs."""
        keys = list(self.factors.keys())
        values = list(self.factors.values())
        configs: list[ExperimentConfig] = []

        for i, combo in enumerate(itertools.product(*values)):
            kwargs = dict(zip(keys, combo))
            for rep in range(self.replications):
                seed = self.base_seed + i * self.replications + rep
                config = ExperimentConfig(
                    name=f"{name_prefix}_{i:04d}_r{rep:02d}",
                    seed=seed,
                    **kwargs,
                )
                configs.append(config)

        return configs


class _SnapshotProxy:
    """Lightweight adapter so DRI.compute() can consume AgentSnapshots."""

    __slots__ = ("weights", "repertoire", "_opinion")

    def __init__(self, snap: AgentSnapshot) -> None:
        self.weights = snap.weights
        self.repertoire = snap.repertoire
        self._opinion = snap.opinion

    def opinion(self, pool: Any) -> float:  # noqa: ARG002
        return self._opinion


def compute_tick_metrics(
    snapshot: StateSnapshot,
    pool: Any,
    initial_opinions: list[float],
    alpha_values: tuple[float, ...] = (0.0, 1.0, 2.0),
) -> dict[str, float]:
    """Compute aggregate metrics from a single StateSnapshot.

    Returns the same metric keys as MetricsSuite.compute_all(), minus
    procedural metrics that require cumulative data (speaking_equity,
    argument_diversity).
    """
    proxies = [_SnapshotProxy(s) for s in snapshot.agent_snapshots]
    opinions = [s.opinion for s in snapshot.agent_snapshots]

    result: dict[str, float] = {"tick": float(snapshot.round_num)}

    # DRI
    result["dri"] = DRI.compute(proxies, pool)  # type: ignore[arg-type]

    # Alpha fairness (backward-compat opinion-based utilities)
    utilities = [(o + 1.0) / 2.0 for o in opinions]
    for alpha in alpha_values:
        result[f"alpha_fairness_{alpha:.0f}"] = AlphaFairness.compute(utilities, alpha)

    # Distributional
    result["opinion_variance"] = DistributionalMetrics.opinion_variance(opinions)
    result["extremist_proportion"] = DistributionalMetrics.extremist_proportion(opinions)
    result["herfindahl"] = DistributionalMetrics.herfindahl(opinions)
    result["opinion_shifts"] = DistributionalMetrics.opinion_shifts(initial_opinions, opinions)
    result["butler_shift"] = DistributionalMetrics.butler_shift(initial_opinions, opinions)

    return result


@dataclass
class Runner:
    """Executes experiment configurations and collects results."""

    def _build_scenario(self, config: ExperimentConfig) -> Scenario:
        if config.scenario == "barabas_consensual":
            return barabas_consensual()
        elif config.scenario == "barabas_non_consensual":
            return barabas_non_consensual()
        elif config.scenario == "jackman_sniderman":
            return jackman_sniderman_symmetric()
        else:
            gen = ScenarioGenerator(
                n_considerations=config.n_considerations,
                pro_con_balance=config.pro_con_balance,
                attack_density=config.attack_density,
            )
            return gen.generate(seed=config.seed)

    def _build_engine(self, config: ExperimentConfig) -> CognitiveEngine:
        bias = config.engine_confirmation_bias
        gae = config.graph_aware_eval
        gap = config.graph_aware_propagation
        pr = config.propagation_rate
        gbl = config.group_building_level
        fac = config.facilitation
        if config.engine == "degroot":
            return DeGrootEngine(
                confirmation_bias=bias, mu=config.engine_mu,
                graph_aware_eval=gae, graph_aware_propagation=gap,
                propagation_rate=pr,
                group_building_level=gbl, facilitated=fac,
            )
        elif config.engine == "bayesian":
            return BayesianEngine(
                confirmation_bias=bias,
                signal_precision=config.engine_signal_precision,
                graph_aware_eval=gae, graph_aware_propagation=gap,
                propagation_rate=pr,
                group_building_level=gbl, facilitated=fac,
            )
        elif config.engine == "bounded_confidence":
            return BoundedConfidenceEngine(
                confirmation_bias=bias, mu=config.engine_mu,
                graph_aware_eval=gae, graph_aware_propagation=gap,
                propagation_rate=pr,
                group_building_level=gbl, facilitated=fac,
            )
        elif config.engine == "argument_based":
            return ArgumentBasedEngine(
                strategic_voicing=config.engine_strategic_voicing,
                p_accept_base=config.engine_p_accept_base,
                p_reject_base=config.engine_p_reject_base,
                use_grounded_semantics=gae,
                group_building_level=gbl, facilitated=fac,
            )
        elif config.engine == "structural":
            return StructuralAlignmentEngine(
                confirmation_bias=bias,
                lambda_salience=config.engine_lambda_salience,
                group_building_level=gbl,
                graph_aware_eval=gae, graph_aware_propagation=gap,
                propagation_rate=pr,
                facilitated=fac,
            )
        else:
            raise ValueError(f"Unknown engine: {config.engine}")

    def _build_protocol(
        self, config: ExperimentConfig, engine: CognitiveEngine
    ) -> Protocol:
        moderator = Moderator(
            active=config.facilitation,
            group_building_level=GroupBuildingLevel(config.group_building_level),
        )
        gbl = GroupBuildingLevel(config.group_building_level)
        rd: Literal["static", "learning"] = (
            "learning" if config.repertoire_dynamics == "learning" else "static"
        )
        gc: Literal["heterogeneous", "homogeneous", "random"]
        if config.group_composition in ("heterogeneous", "homogeneous", "random"):
            gc = config.group_composition  # type: ignore[assignment]
        else:
            gc = "heterogeneous"

        if config.protocol == "plenary":
            return Plenary(
                engine=engine, moderator=moderator, group_building=gbl,
                n_rounds=config.n_rounds, repertoire_dynamics=rd,
                p_update=config.engine_p_update,
            )
        elif config.protocol == "jury":
            return Jury(
                engine=engine, moderator=moderator, group_building=gbl,
                n_rounds=config.n_rounds, repertoire_dynamics=rd,
                p_update=config.engine_p_update,
                jury_size=config.group_size,
            )
        elif config.protocol == "citizens_assembly":
            return CitizensAssembly(
                engine=engine, moderator=moderator, group_building=gbl,
                repertoire_dynamics=rd, breakout_size=config.group_size,
                group_composition=gc,
                p_update=config.engine_p_update,
            )
        elif config.protocol == "town_hall":
            return TownHall(
                engine=engine, moderator=moderator, group_building=gbl,
                n_rounds=config.n_rounds, repertoire_dynamics=rd,
                p_update=config.engine_p_update,
            )
        elif config.protocol == "committee_plenary":
            return CommitteePlenary(
                engine=engine, moderator=moderator, group_building=gbl,
                repertoire_dynamics=rd, committee_size=config.group_size,
                group_composition=gc,
                p_update=config.engine_p_update,
            )
        else:
            raise ValueError(f"Unknown protocol: {config.protocol}")

    def run_single(self, config: ExperimentConfig) -> dict[str, Any]:
        """Run a single experiment and return merged config params + metrics.

        The returned dict includes a 'history' key with the SimulationHistory
        object (non-scalar, excluded by results_to_csv).
        """
        scenario = self._build_scenario(config)
        pool = scenario.pool

        pop_rng = np.random.default_rng(config.seed)
        pop = AgentPopulation(pool=pool)
        agents = pop.generate(
            config.n_agents, pop_rng,
            repertoire_correlation=config.repertoire_correlation,
        )

        initial_opinions = [a.opinion(pool) for a in agents]

        engine = self._build_engine(config)
        protocol = self._build_protocol(config, engine)
        history = protocol.run(agents, pool, seed=config.seed)

        speaking_counts = [a.speaking_count for a in agents]

        metrics = MetricsSuite.compute_all(
            agents, pool,
            initial_opinions=initial_opinions,
            speaking_counts=speaking_counts,
            alpha_values=config.alpha_values,
        )

        result: dict[str, Any] = {
            "name": config.name,
            "seed": config.seed,
            "n_agents": config.n_agents,
            "protocol": config.protocol,
            "engine": config.engine,
            "group_size": config.group_size,
            "group_composition": config.group_composition,
            "facilitation": config.facilitation,
            "group_building_level": config.group_building_level,
            "n_rounds": config.n_rounds,
            "repertoire_dynamics": config.repertoire_dynamics,
            "scenario": config.scenario,
            "pro_con_balance": config.pro_con_balance,
            "n_considerations": config.n_considerations,
            "attack_density": config.attack_density,
            "engine_mu": config.engine_mu,
            "engine_signal_precision": config.engine_signal_precision,
            "engine_confirmation_bias": config.engine_confirmation_bias,
            "engine_p_update": config.engine_p_update,
            "engine_lambda_salience": config.engine_lambda_salience,
            "graph_aware_eval": config.graph_aware_eval,
            "graph_aware_propagation": config.graph_aware_propagation,
            "propagation_rate": config.propagation_rate,
            "repertoire_correlation": config.repertoire_correlation,
        }
        result["initial_opinion_variance"] = float(np.var(initial_opinions))
        result.update(metrics)
        result["history"] = history

        # Per-tick metric trajectory
        trajectory: list[dict[str, float]] = []
        for snap in history.snapshots:
            tick_metrics = compute_tick_metrics(
                snap, pool, initial_opinions, config.alpha_values,
            )
            trajectory.append(tick_metrics)
        result["trajectory"] = trajectory

        return result

    def run_batch(self, configs: list[ExperimentConfig]) -> list[dict[str, Any]]:
        """Run a batch of configs. Deterministic — identical to running each individually."""
        return [self.run_single(c) for c in configs]
