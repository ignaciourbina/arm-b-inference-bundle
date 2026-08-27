"""Agent model: state, parameters, opinion derivation, and population generation.

Version : 0.4.2
Module  : agora.agents
Spec    : manuscript/theoretical-appendix.tex §3

Overview
--------
An Agent holds a weighted consideration repertoire from which its
expressed opinion is derived endogenously:

    o_i = clip( Σ w_ic · d_c / |R_i|,  -1,  1 )

Agents never hold opinions directly; all attitude change operates
through weight updates applied by cognitive engines.

Exports
-------
    OpinionFn          — callable protocol for pluggable opinion functions
    default_opinion_fn — the standard repertoire-weighted mean
    AgentParams        — frozen dataclass of individual-difference parameters
    PrincipalProfile   — placeholder for principal-agent extensions
    Agent              — mutable agent state (weights, precision, speaking count)
    AgentPopulation    — factory for i.i.d. agent populations

Key conventions
---------------
- Empty repertoire → o_i = 0 by convention.
- Weights are always clamped to [-1, 1] after engine updates.
- add_to_repertoire is idempotent (preserves existing weight).

Changelog
---------
0.4.2  c5de059  [US-002] AgentParams validation, latitude constraint
0.4.1  eba9aad  [US-001] Weight clamping and mutable precision
0.4.0  e3b2d53  Restore from calibration branch (baseline)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Protocol

import numpy as np

from agora.considerations import ArgumentPool

if TYPE_CHECKING:
    from numpy.random import Generator


class OpinionFn(Protocol):
    """Callable that computes an agent's opinion from weights and pool."""

    def __call__(self, weights: dict[str, float], pool: ArgumentPool) -> float: ...


def default_opinion_fn(weights: dict[str, float], pool: ArgumentPool) -> float:
    """sum(weight * direction) / len(repertoire), clamped to [-1, 1]."""
    if not weights:
        return 0.0
    total = sum(w * pool.get(cid).direction for cid, w in weights.items())
    raw = total / len(weights)
    return float(np.clip(raw, -1.0, 1.0))


def _draw_mvn_weights(
    rep_ids: list[str],
    pool: ArgumentPool,
    rng: Generator,
    rho: float = 0.0,
    sigma: float = 0.5,
) -> dict[str, float]:
    """Draw initial weights from a multivariate normal with pro-con correlation.

    Covariance: Sigma_pq = sigma^2 * (rho * d_p * d_q + (1-rho) * delta_pq)
    When rho=0, reduces to i.i.d. N(0, sigma^2) clipped to [-1, 1].

    Note: used only in the rho=0 (backward-compatible) initialization path.
    For rho>0, repertoire selection uses _draw_mvn_repertoire() instead,
    and weights are drawn i.i.d. U(-1, 1).
    """
    k = len(rep_ids)
    if k == 0:
        return {}

    if rho == 0.0:
        raw = rng.normal(0.0, sigma, size=k)
        return {cid: float(np.clip(v, -1.0, 1.0)) for cid, v in zip(rep_ids, raw)}

    directions = np.array([pool.get(cid).direction for cid in rep_ids])
    d_outer = np.outer(directions, directions)
    cov = sigma**2 * (rho * d_outer + (1.0 - rho) * np.eye(k))

    # PD safety net (always PD for rho < 1, but guard numerics)
    min_eig = float(np.linalg.eigvalsh(cov).min())
    if min_eig < 1e-10:
        cov += (1e-10 - min_eig) * np.eye(k)

    raw = rng.multivariate_normal(np.zeros(k), cov)
    clipped = np.clip(raw, -1.0, 1.0)
    return {cid: float(clipped[i]) for i, cid in enumerate(rep_ids)}


def _draw_mvn_repertoire(
    pool: ArgumentPool,
    rng: Generator,
    rho: float,
    k_target: int,
    sigma: float = 1.0,
    sigma_tau: float = 0.3,
    min_size: int = 3,
    max_frac: float = 0.5,
) -> dict[str, float]:
    """Draw a direction-biased repertoire via MVN latent binarization.

    1. Build M x M covariance: Sigma_pq = sigma^2 (rho d_p d_q + (1-rho) I_pq)
    2. Draw z ~ MVN(0, Sigma)  -- latent awareness per consideration
    3. Draw per-consideration thresholds tau_c ~ N(mu_tau, sigma_tau)
       where mu_tau = sigma * Phi^{-1}(1 - k_target / M)
    4. Binarize: R = {c : z_c > tau_c}
    5. Edge guards enforce min_size <= |R| <= floor(max_frac * M)
    6. Weights drawn i.i.d. U(0, 1) for each c in R
       (endorsement-only: agents hold considerations they find
       persuasive, so opinion tracks repertoire composition)

    Returns dict mapping consideration id -> weight.
    """
    from scipy.stats import norm as _norm  # local; only needed for rho > 0

    all_ids = pool.all_ids()
    M = len(all_ids)
    if M == 0:
        return {}

    directions = np.array([pool.get(cid).direction for cid in all_ids])

    # --- covariance ---
    d_outer = np.outer(directions, directions)
    cov = sigma**2 * (rho * d_outer + (1.0 - rho) * np.eye(M))
    min_eig = float(np.linalg.eigvalsh(cov).min())
    if min_eig < 1e-10:
        cov += (1e-10 - min_eig) * np.eye(M)

    # --- latent draw ---
    z = rng.multivariate_normal(np.zeros(M), cov)

    # --- thresholds ---
    frac = max(min(k_target / M, 0.99), 0.01)  # guard quantile domain
    mu_tau = sigma * float(_norm.ppf(1.0 - frac))
    tau = rng.normal(mu_tau, sigma_tau, size=M)

    # --- binarize ---
    included = z > tau
    max_k = max(min_size, int(np.floor(max_frac * M)))

    # edge guard: too few
    if int(included.sum()) < min_size:
        top_idx = np.argsort(z - tau)[-min_size:]
        included[:] = False
        included[top_idx] = True

    # edge guard: too many
    if int(included.sum()) > max_k:
        inc_idx = np.where(included)[0]
        margin = z[inc_idx] - tau[inc_idx]
        drop = inc_idx[np.argsort(margin)[: int(included.sum()) - max_k]]
        included[drop] = False

    # --- build repertoire with i.i.d. U(0, 1) weights ---
    # Endorsement-only: held considerations are persuasive (w > 0),
    # so opinion direction tracks repertoire composition (pro/con).
    rep_ids = [all_ids[j] for j in range(M) if included[j]]
    weights = {cid: float(rng.uniform(0.0, 1.0)) for cid in rep_ids}
    return weights


@dataclass(frozen=True)
class AgentParams:
    """Frozen parameter set for an agent.

    Attributes:
        prior_precision: Confidence in prior beliefs [0.1, 10].
        open_mindedness: Willingness to consider new arguments [0, 1].
        elaboration_quality: Quality of argument processing [0, 1].
        latitude_acceptance: Bounded-confidence acceptance range [0, 2].
        latitude_rejection: Bounded-confidence rejection range [0, 2].
        repertoire_size: Target number of considerations [3, 30].
    """

    prior_precision: float = 1.0
    open_mindedness: float = 0.5
    elaboration_quality: float = 0.5
    latitude_acceptance: float = 0.5
    latitude_rejection: float = 1.5
    repertoire_size: int = 10

    def __post_init__(self) -> None:
        if self.prior_precision < 0:
            raise ValueError(
                f"prior_precision must be non-negative, got {self.prior_precision}"
            )
        if self.latitude_rejection < self.latitude_acceptance:
            raise ValueError(
                f"latitude_rejection ({self.latitude_rejection}) must be >= "
                f"latitude_acceptance ({self.latitude_acceptance})"
            )


@dataclass(frozen=True)
class PrincipalProfile:
    """Frozen demographic profile for ANES-style principal agents.

    Placeholder for future survey-data integration.
    """

    profile_id: str
    attributes: dict[str, float] = field(default_factory=dict)


@dataclass
class Agent:
    """A deliberating agent with a consideration-weight repertoire.

    Opinion is a derived property, not stored independently.
    """

    id: str
    params: AgentParams
    weights: dict[str, float] = field(default_factory=dict)
    _principal: PrincipalProfile | None = field(default=None, repr=False)
    _opinion_fn: Callable[[dict[str, float], ArgumentPool], float] = field(
        default=default_opinion_fn, repr=False
    )
    speaking_count: int = 0
    precision: float = 0.0
    latent_theta: float | None = None
    salience_prior: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.precision == 0.0:
            self.precision = self.params.prior_precision

    def clamp_weights(self) -> None:
        """Clip all weights to [-1, 1]."""
        for cid in self.weights:
            self.weights[cid] = float(np.clip(self.weights[cid], -1.0, 1.0))

    @property
    def principal(self) -> PrincipalProfile | None:
        return self._principal

    @property
    def repertoire(self) -> frozenset[str]:
        return frozenset(self.weights.keys())

    def opinion(self, pool: ArgumentPool) -> float:
        """Derived opinion via opinion_fn(weights, pool)."""
        return self._opinion_fn(self.weights, pool)

    def add_to_repertoire(self, cid: str, initial_weight: float = 0.0) -> None:
        """Add a consideration to this agent's repertoire without disrupting existing weights."""
        if cid not in self.weights:
            self.weights[cid] = initial_weight


@dataclass
class AgentPopulation:
    """Factory for generating heterogeneous agent populations."""

    pool: ArgumentPool
    default_params: AgentParams = field(default_factory=AgentParams)

    def generate(
        self,
        n: int,
        rng: Generator | None = None,
        opinion_fn: Callable[[dict[str, float], ArgumentPool], float] | None = None,
        repertoire_correlation: float = 0.0,
    ) -> list[Agent]:
        """Produce n agents with varying params and repertoires.

        When repertoire_correlation > 0, repertoires are drawn via MVN
        direction-biased binarization (pro-leaning agents hold more pro
        considerations). Weights are always i.i.d. U(-1, 1) in this path.

        When repertoire_correlation == 0, the old uniform path is used
        (exact backward compatibility).
        """
        if rng is None:
            rng = np.random.default_rng()

        agents: list[Agent] = []
        for i in range(n):
            lat_acceptance = float(rng.uniform(0.0, 2.0))
            lat_rejection = float(rng.uniform(lat_acceptance, 2.0))
            params = AgentParams(
                prior_precision=float(rng.uniform(0.1, 10.0)),
                open_mindedness=float(rng.uniform(0.0, 1.0)),
                elaboration_quality=float(rng.uniform(0.0, 1.0)),
                latitude_acceptance=lat_acceptance,
                latitude_rejection=lat_rejection,
                repertoire_size=int(rng.integers(
                    3, max(4, len(self.pool.all_ids()) // 2 + 1),
                )),
            )
            if repertoire_correlation > 0.0:
                weights = _draw_mvn_repertoire(
                    self.pool, rng,
                    rho=repertoire_correlation,
                    k_target=params.repertoire_size,
                )
            else:
                rep_ids = self.pool.sample_repertoire(params.repertoire_size, rng)
                weights = _draw_mvn_weights(
                    rep_ids, self.pool, rng,
                    rho=0.0, sigma=0.5,
                )
            fn = opinion_fn if opinion_fn is not None else default_opinion_fn
            agents.append(
                Agent(
                    id=f"agent_{i:03d}",
                    params=params,
                    weights=weights,
                    _opinion_fn=fn,
                )
            )
        return agents

    def from_ising_profiles(
        self,
        profiles_path: str | Path,
        n: int,
        rng: Generator | None = None,
        theta_path: str | Path | None = None,
        precision_exponent: float = 0.5,
        composition: dict[tuple[str, str], int] | None = None,
        opinion_fn: Callable[[dict[str, float], ArgumentPool], float] | None = None,
    ) -> list[Agent]:
        """Build agents from empirical Ising participant profiles.

        Profiles come from the Polis-derived ``ising_profiles.json`` (see
        ``polis-analysis/``): each carries a sparse ``weights`` repertoire, a
        ``coherence`` z-score, a ``latent_theta``, and the derived
        ``policy_cell`` / ``coherence_cell`` (3x3 taxonomy).

        Agent parameters are derived from coherence:
          - ``prior_precision = clip(10 ** (precision_exponent * z), 0.1, 10)``
            (exact inverse of ``z = 2 * log10(precision)``).
          - ``open_mindedness = clip(0.5 - 0.25 * z, 0, 1)`` (decreasing in
            coherence; affects only the LLM arm, reconstructed monotone mapping).

        When ``composition`` is given (``{(policy_cell, coherence_cell): count}``,
        e.g. the presets in ``llm/townhall/compositions.py``), agents are
        stratified-sampled to match those cell counts (with replacement within a
        cell if it is thinner than requested). Otherwise ``n`` profiles are drawn
        uniformly at random.
        """
        import json

        if rng is None:
            rng = np.random.default_rng()

        with open(profiles_path) as f:
            data = json.load(f)
        profiles = data["profiles"] if isinstance(data, dict) else data

        # Optional external theta map overrides per-profile latent_theta.
        theta_map: dict[str, float] = {}
        if theta_path is not None:
            with open(theta_path) as f:
                theta_map = {str(k): float(v) for k, v in json.load(f).items()}

        def _theta(profile: dict) -> float | None:
            pid = str(profile.get("participant_id", profile.get("participant")))
            if pid in theta_map:
                return theta_map[pid]
            lt = profile.get("latent_theta")
            return None if lt is None else float(lt)

        # Choose the profiles that will seed agents.
        if composition is not None:
            by_cell: dict[tuple[str, str], list[dict]] = {}
            for p in profiles:
                key = (p["policy_cell"], p["coherence_cell"])
                by_cell.setdefault(key, []).append(p)
            chosen: list[dict] = []
            for key, count in composition.items():
                if count <= 0:
                    continue
                pool_cell = by_cell.get(key, [])
                if not pool_cell:
                    raise ValueError(
                        f"No empirical profiles in cell {key}; cannot satisfy composition."
                    )
                replace = len(pool_cell) < count
                idx = rng.choice(len(pool_cell), size=count, replace=replace)
                chosen.extend(pool_cell[i] for i in idx)
        else:
            idx = rng.choice(len(profiles), size=n, replace=len(profiles) < n)
            chosen = [profiles[i] for i in idx]

        fn = opinion_fn if opinion_fn is not None else default_opinion_fn
        agents: list[Agent] = []
        for i, profile in enumerate(chosen):
            z = float(profile.get("coherence", 0.0))
            precision = float(np.clip(10.0 ** (precision_exponent * z), 0.1, 10.0))
            openness = float(np.clip(0.5 - 0.25 * z, 0.0, 1.0))
            # NOTE (Sprint 12): the manuscript §"Ising Profile Construction"
            # defines the agent weight w_{pi} as the observed vote where
            # available and the mean-field imputed value m_{pi} otherwise — i.e.
            # the dense ``full_weights`` field. This recovered code instead uses
            # the sparse observed ``weights``. Sprint 12 found that neither
            # choice reproduces the published descriptives cleanly (full_weights
            # fixes the meta-consensus level but inverts its slope and leaves
            # opinion_variance ~3x too high; observed keeps opinion_variance
            # nearer but MC too low), so the seeding is left as recovered
            # pending a design decision. See
            # agora/analysis/sprint-12-rule-based-reproduction/.
            params = AgentParams(
                prior_precision=precision,
                open_mindedness=openness,
                repertoire_size=len(profile["weights"]),
            )
            full_weights = profile.get("full_weights")
            salience_prior = (
                {k: float(v) for k, v in full_weights.items()}
                if full_weights is not None
                else None
            )
            agents.append(
                Agent(
                    id=f"agent_{i:03d}",
                    params=params,
                    weights={k: float(v) for k, v in profile["weights"].items()},
                    _opinion_fn=fn,
                    latent_theta=_theta(profile),
                    salience_prior=salience_prior,
                )
            )
        return agents
