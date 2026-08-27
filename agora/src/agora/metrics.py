"""Outcome metrics: deliberative quality, fairness, distributional, and procedural.

Version : 0.4.2
Module  : agora.metrics
Spec    : manuscript/theoretical-appendix.tex §7

Overview
--------
Four metric families, collected by MetricsSuite.compute_all():

    DRI                  — Deliberative Reasoning Index (Niemeyer et al. 2024):
                           Pearson correlation between cosine(weight vectors)
                           and opinion similarity across all agent pairs.
                           Uses union-of-repertoires projection, not full pool.
    AlphaFairness        — Welfare W_α for α ∈ {0,1,2} and price-of-fairness.
                           Note: POF is available but NOT computed by default.
    DistributionalMetrics — variance, extremist proportion, mean shift, HHI,
                           Butler shift.
    ProceduralMetrics    — speaking equity (1 - Gini), argument diversity,
                           responsiveness.
                           Note: argument_diversity and responsiveness require
                           voiced_cids / influence data not currently passed
                           by Runner.run_single().

Edge cases
----------
- N < 2 or all-identical vectors → DRI = 0.0 (not 1.0 or NaN)
- Zero-norm weight vector → cosine = 0.0
- α ≈ 1 (within 1e-10) → switches to log formula

Changelog
---------
0.4.2  3e0d09d  DRI docstrings, Farrar bounds review
0.4.2  c5de059  [US-002] DRI full-pool projection fix
0.4.0  e3b2d53  Restore from calibration branch (baseline)
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import pearsonr

if TYPE_CHECKING:
    from agora.agents import Agent
    from agora.considerations import ArgumentPool


# ---------------------------------------------------------------------------
# Congruence (manuscript appendix: chi_i(c) in {-1, 0, +1})
# ---------------------------------------------------------------------------


def compute_congruence(
    weights: dict[str, float], pool: ArgumentPool, direction: float
) -> int:
    """Congruence chi of a consideration's direction with a repertoire's net stance.

    Returns +1 if the consideration is *pro-attitudinal* (its direction agrees
    with the repertoire's net opinion direction), -1 if *counter-attitudinal*,
    and 0 for neutral or tied cases. This is the chi_i(c) of the manuscript
    methods appendix (eq:v3-ccu context), used by EmpiricalArgumentEngine for
    confirmation-bias weighting and the legacy congruence gate.

    The repertoire's net stance is sum_c w_c * d_c over the held considerations;
    only its sign matters, so normalization is irrelevant.
    """
    net = 0.0
    for cid, w in weights.items():
        c = pool.considerations.get(cid)
        if c is not None:
            net += w * c.direction
    if abs(net) < 1e-12 or abs(direction) < 1e-12:
        return 0
    return int(np.sign(net) * np.sign(direction))


# ---------------------------------------------------------------------------
# DRI (Niemeyer 2024)
# ---------------------------------------------------------------------------


@dataclass
class DRI:
    """Deliberative Reasoning Index.

    Pearson correlation between pairwise consideration-cosine-similarity
    and pairwise opinion-similarity (1 - |o_i - o_j|).
    """

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity between two vectors. Returns 0.0 for zero vectors."""
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    @staticmethod
    def compute(agents: list[Agent], pool: ArgumentPool) -> float:
        """Compute DRI across all agent pairs.

        Returns 1.0 when all agents have identical weight vectors.
        Returns ~0.0 when weights and opinions are uncorrelated.
        Projects weight vectors to the union of all agent repertoires
        (zero-filling missing considerations).
        """
        if len(agents) < 2:
            return 0.0

        cosine_sims: list[float] = []
        opinion_sims: list[float] = []

        all_cids = sorted(set().union(*(a.repertoire for a in agents)))
        if not all_cids:
            return 0.0

        for a, b in combinations(agents, 2):
            v_a = np.array([a.weights.get(cid, 0.0) for cid in all_cids])
            v_b = np.array([b.weights.get(cid, 0.0) for cid in all_cids])

            cos = DRI._cosine_similarity(v_a, v_b)
            op_sim = 1.0 - abs(a.opinion(pool) - b.opinion(pool))

            cosine_sims.append(cos)
            opinion_sims.append(op_sim)

        if len(cosine_sims) < 2:
            return 0.0

        # If all values are identical, correlation is undefined; return 1.0
        eps = 1e-12
        std_cos = float(np.std(cosine_sims))
        std_op = float(np.std(opinion_sims))
        if std_cos < eps and std_op < eps:
            return 1.0
        if std_cos < eps or std_op < eps:
            return 0.0

        r, _ = pearsonr(cosine_sims, opinion_sims)
        return float(r)


# ---------------------------------------------------------------------------
# Alpha-Fairness (Bertsimas 2012)
# ---------------------------------------------------------------------------


@dataclass
class AlphaFairness:
    """Alpha-fairness welfare function.

    W_alpha = sum(u^(1-a)/(1-a)) for a != 1
    W_alpha = sum(log(u))         for a == 1
    """

    @staticmethod
    def compute(
        utilities: list[float],
        alpha: float,
    ) -> float:
        """Compute W_alpha welfare.

        For alpha=0: sum of utilities.
        For alpha=1: sum of log(utilities) (proportional fairness).
        For alpha=2: -sum(1/u) (max-min fairness).
        """
        arr = np.array(utilities, dtype=float)
        # Clamp to small positive value to avoid log(0) or division by 0
        arr = np.maximum(arr, 1e-12)

        if abs(alpha - 1.0) < 1e-10:
            return float(np.sum(np.log(arr)))
        else:
            return float(np.sum(arr ** (1.0 - alpha) / (1.0 - alpha)))

    @staticmethod
    def price_of_fairness(
        utilities: list[float],
        alpha: float,
    ) -> float:
        """POF = 1 - W_alpha / W_0."""
        w_alpha = AlphaFairness.compute(utilities, alpha)
        w_0 = AlphaFairness.compute(utilities, 0.0)
        if abs(w_0) < 1e-12:
            return 0.0
        return 1.0 - w_alpha / w_0


# ---------------------------------------------------------------------------
# Utility functions (§7.2)
# ---------------------------------------------------------------------------


def compute_representational_utility(
    agents: list[Agent],
    voiced_considerations: set[str],
) -> list[float]:
    """eq:u-rep §7.2.1 — u_rep_i = |voiced ∩ R_i| / |R_i|.

    Returns a value in [0, 1] per agent. Agents with empty repertoire get 0.
    """
    result: list[float] = []
    for agent in agents:
        repertoire = set(agent.weights.keys())
        if not repertoire:
            result.append(0.0)
        else:
            result.append(len(voiced_considerations & repertoire) / len(repertoire))
    return result


def compute_policy_outcome_utility(
    agents: list[Agent],
    pool: ArgumentPool,
    aggregation_rule: str | None,
) -> list[float] | None:
    """eq:u-pol §7.2.2 — u_pol_i = 1 - |o_i(T) - o_agg|.

    aggregation_rule: 'mean', 'median', 'supermajority', or None.
    Returns None when aggregation_rule is None (no closure overlay).
    """
    if aggregation_rule is None:
        return None
    opinions = [a.opinion(pool) for a in agents]
    arr = np.array(opinions)
    if aggregation_rule == "mean":
        o_agg = float(np.mean(arr))
    elif aggregation_rule == "median":
        o_agg = float(np.median(arr))
    elif aggregation_rule == "supermajority":
        o_agg = float(np.mean(arr))  # fallback: same as mean for now
    else:
        raise ValueError(f"Unknown aggregation rule: {aggregation_rule}")
    return [1.0 - abs(o - o_agg) for o in opinions]


# ---------------------------------------------------------------------------
# Distributional metrics
# ---------------------------------------------------------------------------


@dataclass
class DistributionalMetrics:
    """Opinion-distribution measures: variance, extremism, shifts, Herfindahl."""

    @staticmethod
    def opinion_variance(opinions: list[float]) -> float:
        """Population variance of opinions."""
        return float(np.var(opinions))

    @staticmethod
    def extremist_proportion(
        opinions: list[float],
        threshold: float = 0.75,
    ) -> float:
        """Proportion of agents with |opinion| >= threshold."""
        arr = np.array(opinions)
        return float(np.mean(np.abs(arr) >= threshold))

    @staticmethod
    def opinion_shifts(
        before: list[float],
        after: list[float],
    ) -> float:
        """Mean absolute opinion shift."""
        return float(np.mean(np.abs(np.array(after) - np.array(before))))

    @staticmethod
    def herfindahl(opinions: list[float], n_bins: int = 10) -> float:
        """Herfindahl index on binned opinion distribution.

        HHI = sum(s_i^2) where s_i is share in bin i.
        Returns 1/n_bins for uniform, 1.0 for perfect concentration.
        """
        counts, _ = np.histogram(opinions, bins=n_bins, range=(-1.0, 1.0))
        total = counts.sum()
        if total == 0:
            return 0.0
        shares = counts / total
        return float(np.sum(shares ** 2))

    @staticmethod
    def butler_shift(
        initial_opinions: list[float],
        final_opinions: list[float],
    ) -> float:
        """Butler aggregate shift: 2 * sum|o_initial - o_final| / (max - min).

        Uses opinion range [-1, 1] so denominator = 2.
        """
        diffs = np.abs(np.array(initial_opinions) - np.array(final_opinions))
        return float(2.0 * np.sum(diffs) / 2.0)


# ---------------------------------------------------------------------------
# Procedural metrics
# ---------------------------------------------------------------------------


@dataclass
class ProceduralMetrics:
    """Procedural fairness measures."""

    @staticmethod
    def _gini(values: list[float]) -> float:
        """Gini coefficient of a list of non-negative values."""
        arr = np.array(values, dtype=float)
        if len(arr) == 0 or np.sum(arr) == 0.0:
            return 0.0
        arr = np.sort(arr)
        n = len(arr)
        index = np.arange(1, n + 1)
        return float((2.0 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))

    @staticmethod
    def speaking_equity(speaking_counts: list[int]) -> float:
        """Speaking equity = 1 - Gini(speaking_counts).

        Returns 1.0 when all agents speak equally.
        Returns < 1.0 when speaking is unequal.
        """
        return 1.0 - ProceduralMetrics._gini([float(c) for c in speaking_counts])

    @staticmethod
    def argument_diversity(
        voiced_cids: list[str],
        pool: ArgumentPool,
    ) -> float:
        """Fraction of pool considerations that were voiced at least once."""
        total = len(pool.all_ids())
        if total == 0:
            return 0.0
        unique = len(set(voiced_cids))
        return unique / total

    @staticmethod
    def responsiveness(
        opinion_changes: list[float],
        influences: list[float],
    ) -> float:
        """Correlation between received influence and opinion change magnitude.

        Returns 0.0 if correlation is undefined.
        """
        if len(opinion_changes) < 2:
            return 0.0
        if np.std(opinion_changes) == 0.0 or np.std(influences) == 0.0:
            return 0.0
        r, _ = pearsonr(opinion_changes, influences)
        return float(r)


# ---------------------------------------------------------------------------
# Meta-consensus and plurality (manuscript final-manuscript.tex §F.2)
# ---------------------------------------------------------------------------
#
# Reconstructed 2026-06-30 after the tmp2/ data loss. The original definitions
# were authored in the lost tmp2 working tree and survive in NO source channel
# (git, VS Code local history, or any recovered session transcript) — the
# recovered analysis pipeline (pipeline/) only ever *imports* them. They are,
# however, formally specified in the manuscript, and are re-implemented here
# verbatim from those canonical equations (not fitted to the reference tables):
#   MC        eq:v3-meta-consensus  (final-manuscript.tex)
#   Plurality eq:v3-plurality       (final-manuscript.tex)
# Fidelity is verified against the published t0/t7 per-composition anchors.


def meta_consensus_agreement(agents: list[Agent], pool: ArgumentPool) -> float:
    """Meta-consensus on salience (manuscript eq:v3-meta-consensus).

        MC = 1 - (1/M) * sum_{c in pool} sd(|w_1c|, ..., |w_Nc|)

    where M = |pool|, the standard deviation is taken across all N agents of the
    absolute salience |w_ic| that agent i assigns consideration c, and absent
    considerations contribute zero weight by convention (so the vector for each
    c is length N, zero-filled for agents that do not hold c). Population sd
    (ddof=0), matching the codebase convention (cf. DistributionalMetrics).

    High MC = agents agree about *which* considerations matter (shared salience
    map), even if they disagree on direction. Returns 0.0 for an empty pool or
    fewer than 2 agents (sd undefined).

    Provenance (Sprint 14 forensic reconstruction, 2026-08-13): this is the
    pre-loss implementation, recovered from unreachable git blob fa226e3f69e2
    and matching the manuscript's App. F.2 wording ("absent considerations
    contribute zero weight by convention"). The post-loss reconstruction had
    substituted each agent's dense co-vote-imputed ``salience_prior`` for
    absent considerations; that variant inverted the rule arm's meta-consensus
    trajectory (slopes -5.00/-2.22/-3.42 against the manuscript's
    +0.21/+0.89/+1.32). Restoring the zero-fill reproduces the published
    signs and ordering (+0.52/+1.17/+1.72). ``Agent.salience_prior`` remains
    on the model for other consumers; this metric does not read it.
    """
    if len(agents) < 2:
        return 0.0
    cids = pool.all_ids()
    M = len(cids)
    if M == 0:
        return 0.0
    total_sd = 0.0
    for cid in cids:
        saliences = [abs(float(a.weights.get(cid, 0.0))) for a in agents]
        total_sd += float(np.std(saliences))  # population sd (ddof=0)
    return 1.0 - total_sd / M


def plurality_index(agents: list[Agent], pool: ArgumentPool) -> float:
    """Plurality: viability of both opinion camps (manuscript eq:v3-plurality).

        Plurality = min(n_pro, n_con) / N
        n_pro = #{i : o_i >  tau_camp},  n_con = #{i : o_i < -tau_camp},
        tau_camp = 0.1

    High plurality = both non-ambivalent camps remain substantively present; low
    plurality = one side has been compressed or eliminated. Opinion o_i is the
    emergent mean(w * d) exposed by ``agent.opinion(pool)``. Returns 0.0 for an
    empty population.
    """
    tau_camp = 0.1
    n = len(agents)
    if n == 0:
        return 0.0
    opinions = [a.opinion(pool) for a in agents]
    n_pro = sum(1 for o in opinions if o > tau_camp)
    n_con = sum(1 for o in opinions if o < -tau_camp)
    return min(n_pro, n_con) / n


# ---------------------------------------------------------------------------
# MetricsSuite convenience wrapper
# ---------------------------------------------------------------------------


@dataclass
class MetricsSuite:
    """Compute all metrics in one call."""

    @staticmethod
    def compute_all(
        agents: list[Agent],
        pool: ArgumentPool,
        initial_opinions: list[float] | None = None,
        speaking_counts: list[int] | None = None,
        voiced_cids: list[str] | None = None,
        alpha_values: tuple[float, ...] = (0.0, 1.0, 2.0),
        voiced_considerations: set[str] | None = None,
        aggregation_rule: str | None = None,
    ) -> dict[str, float]:
        """Return a dict with all metric names as keys."""
        opinions = [a.opinion(pool) for a in agents]

        results: dict[str, float] = {}

        # DRI
        results["dri"] = DRI.compute(agents, pool)

        if voiced_considerations is not None:
            # eq:u-rep §7.2.1 — representational utility welfare
            u_rep = compute_representational_utility(agents, voiced_considerations)
            for alpha in alpha_values:
                results[f"welfare_rep_{alpha:.0f}"] = AlphaFairness.compute(
                    u_rep, alpha,
                )
                results[f"pof_rep_{alpha:.0f}"] = AlphaFairness.price_of_fairness(
                    u_rep, alpha,
                )
            # eq:u-pol §7.2.2 — policy-outcome utility welfare
            u_pol = compute_policy_outcome_utility(agents, pool, aggregation_rule)
            if u_pol is not None:
                for alpha in alpha_values:
                    results[f"welfare_pol_{alpha:.0f}"] = AlphaFairness.compute(
                        u_pol, alpha,
                    )
                    results[f"pof_pol_{alpha:.0f}"] = AlphaFairness.price_of_fairness(
                        u_pol, alpha,
                    )
        else:
            # Backward compat: opinion-based utilities when voiced set unavailable
            utilities = [(o + 1.0) / 2.0 for o in opinions]
            for alpha in alpha_values:
                key = f"alpha_fairness_{alpha:.0f}"
                results[key] = AlphaFairness.compute(utilities, alpha)

        # Distributional
        results["opinion_variance"] = DistributionalMetrics.opinion_variance(opinions)
        results["extremist_proportion"] = DistributionalMetrics.extremist_proportion(
            opinions
        )
        results["herfindahl"] = DistributionalMetrics.herfindahl(opinions)

        if initial_opinions is not None:
            results["opinion_shifts"] = DistributionalMetrics.opinion_shifts(
                initial_opinions, opinions
            )
            results["butler_shift"] = DistributionalMetrics.butler_shift(
                initial_opinions, opinions
            )

        # Procedural
        if speaking_counts is not None:
            results["speaking_equity"] = ProceduralMetrics.speaking_equity(
                speaking_counts
            )

        if voiced_cids is not None:
            results["argument_diversity"] = ProceduralMetrics.argument_diversity(
                voiced_cids, pool
            )

        return results
