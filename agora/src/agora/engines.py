"""Cognitive engines governing agent cognition during deliberation.

Version : 0.4.5
Module  : agora.engines
Spec    : manuscript/theoretical-appendix.tex §4

Overview
--------
Each cognitive engine implements the three-hook interface
(voice → evaluate → reflect) that the protocol layer invokes once
per agent per round.  Six concrete engines are provided:

    DeGrootEngine            — weighted-average opinion convergence
    BayesianEngine           — normal-normal posterior with consensus gate
    BoundedConfidenceEngine  — SJT assimilation / contrast zones
    StructuralAlignmentEngine — sign-preserving salience convergence
    ArgumentBasedEngine       — grounded semantics + CI/IC transitions
    MixedEngine              — stochastic delegation across engines

Engines 1–4 share a SimpleUpdateEngine base that factors out
confirmation-biased voicing (Eq. voice), evaluation (Eq. evaluate),
and post-round amplification (Eq. amplify).

Key constants
-------------
- Dead-zone threshold for sgn(o_i): |o_i| ≤ 1e-12 → 0
- Amplification skip threshold:     |o_i| < 1e-6  → no-op
- Near-zero delta guard:            |δ|  < 1e-12  → no-op

Changelog
---------
0.4.5  5daff0a  [US-005] ArgumentBasedEngine latent-orientation transitions
0.4.4  f8be5e0  [US-004] Group-building modulation for StructuralAlignmentEngine
0.4.1  eba9aad  [US-001] Weight clamping and mutable Bayesian precision
0.4.0  e3b2d53  Restore from calibration branch (baseline)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from agora.agents import Agent
from agora.considerations import ArgumentPool
from agora.metrics import compute_congruence

if TYPE_CHECKING:
    from numpy.random import Generator

    from agora.state_tensor import StateTensor


class CognitiveEngine(ABC):
    """Abstract base for all deliberation engines.

    Three hooks decompose the deliberation cycle:
      - voice: select which consideration to speak
      - evaluate: assess influence of a voiced argument
      - reflect: update agent state after processing round's arguments
    """

    @abstractmethod
    def voice(
        self, agent: Agent, pool: ArgumentPool, rng: Generator
    ) -> str:
        """Select a consideration to voice. Returns cid."""
        ...

    @abstractmethod
    def evaluate(
        self,
        agent: Agent,
        cid: str,
        speaker_opinion: float,
        pool: ArgumentPool,
        rng: Generator,
    ) -> float:
        """Evaluate influence of a voiced argument. Returns influence in [0, 1]."""
        ...

    @abstractmethod
    def reflect(
        self,
        agent: Agent,
        round_updates: list[tuple[str, float, float]],
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        """Update agent after a round.

        round_updates: list of (cid, speaker_opinion, influence) tuples.
        """
        ...


@dataclass
class SimpleUpdateEngine(CognitiveEngine, ABC):
    """Base for engines that collapse to a single update() per argument.

    Provides default voice (prob proportional to |weight|), default evaluate
    (returns 1.0), and default reflect (calls update per argument).
    Subclasses only implement update().

    confirmation_bias (0–1): when > 0, activates two mechanisms:
      1. Directional voicing — consonant considerations voiced more often.
      2. Biased evaluation — confirming arguments get influence > 1.
    """

    confirmation_bias: float = 0.0
    graph_aware_eval: bool = False
    graph_aware_propagation: bool = False
    propagation_rate: float = 0.1
    group_building_level: int = 3
    facilitated: bool = False

    def voice(
        self, agent: Agent, pool: ArgumentPool, rng: Generator
    ) -> str:
        """Voice a consideration with probability proportional to |weight|.

        eq:voice §4.1.1 — Pr(c* = c) ∝ |w_ic| · max(1 + β·sgn(d_c)·sgn(o_i), 0).
        When confirmation_bias > 0, consonant considerations (weight sign
        matches opinion sign) are boosted up to 2×.
        """
        if not agent.weights:
            ids = pool.all_ids()
            return str(rng.choice(ids))

        cids = list(agent.weights.keys())
        abs_weights = np.array([abs(agent.weights[c]) * pool.get(c).persuasiveness for c in cids])

        if self.confirmation_bias != 0.0:
            opinion = agent.opinion(pool)
            op_sign = float(np.sign(opinion)) if abs(opinion) > 1e-12 else 0.0
            boosts = np.array([
                max(1.0 + self.confirmation_bias
                    * float(np.sign(pool.get(c).direction)) * op_sign, 0.0)
                for c in cids
            ])
            abs_weights = abs_weights * boosts

        total = abs_weights.sum()
        if total == 0:
            probs = np.ones(len(cids)) / len(cids)
        else:
            probs = abs_weights / total
        idx = int(rng.choice(len(cids), p=probs))
        return cids[idx]

    def evaluate(
        self,
        agent: Agent,
        cid: str,
        speaker_opinion: float,
        pool: ArgumentPool,
        rng: Generator,
    ) -> float:
        """Evaluate influence of a voiced argument.

        When confirmation_bias > 0, confirming arguments (direction aligns
        with opinion sign) get influence > 1, disconfirming get < 1.
        """
        if self.confirmation_bias == 0.0:
            return 1.0
        # Facilitation reduces confirmation bias via open-mindedness bonus
        cb = self.confirmation_bias
        if self.facilitated:
            soph_eff = min(
                agent.params.open_mindedness
                + 0.076 * (self.group_building_level - 1),
                1.0,
            )
            cb = cb * (1.0 - 0.3 * soph_eff)
        direction = pool.get(cid).direction
        opinion = agent.opinion(pool)
        op_sign = float(np.sign(opinion)) if abs(opinion) > 1e-12 else 0.0
        influence = (
            1.0
            + cb
            * float(np.sign(direction))
            * op_sign
            * 0.5
        )
        return float(np.clip(influence, 0.1, 1.5))

    def evaluate_batch(
        self,
        state: StateTensor,
        cid_idx: int,
        speaker_opinion: float,
        rng: Generator,
    ) -> np.ndarray:
        """Vectorized evaluate: influence scores for all agents at once.

        Returns (N,) array matching per-agent evaluate() exactly.
        """
        n = state.W.shape[0]
        if self.confirmation_bias == 0.0:
            return np.ones(n, dtype=np.float64)
        cb = self.confirmation_bias
        if self.facilitated:
            soph_eff = np.minimum(
                state.open_mindedness
                + 0.076 * (self.group_building_level - 1),
                1.0,
            )
            cb = cb * (1.0 - 0.3 * soph_eff)
        d_c = state.directions[cid_idx]
        opinions = state.opinions()
        op_sign = np.sign(opinions)
        op_sign = np.where(np.abs(opinions) <= 1e-12, 0.0, op_sign)
        phi = 1.0 + cb * float(np.sign(d_c)) * op_sign * 0.5
        return np.clip(phi, 0.1, 1.5)

    def reflect(
        self,
        agent: Agent,
        round_updates: list[tuple[str, float, float]],
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        """Call update() for each argument, then apply directional amplification.

        When graph_aware_eval is True, influence is modulated by the attack
        graph before each update (eq:graph-modulate §4.1.5).

        When graph_aware_propagation is True, after each update the weight
        change propagates through attack/support edges (eq:graph-propagate §4.1.5).
        """
        for cid, speaker_opinion, influence in round_updates:
            if self.graph_aware_eval:
                attackers = pool.attack_graph.get_attackers(cid)
                attenuation = sum(
                    sigma * abs(agent.weights[c_prime])
                    for c_prime, sigma in attackers.items()
                    if c_prime in agent.weights
                )
                influence = influence * max(0.0, min(1.0, 1.0 - attenuation))
            self.update(agent, cid, speaker_opinion, influence, pool, rng)
            if self.graph_aware_propagation and cid in agent.weights:
                self._propagate_graph(agent, cid, pool)

    def _propagate_graph(
        self,
        agent: Agent,
        cid: str,
        pool: ArgumentPool,
    ) -> None:
        """eq:graph-propagate §4.1.5 — Propagate weight update through graph edges.

        Attack: w_ic' *= (1 - mu_g * sigma * |w_ic|).
        Support: w_ic' += mu_g * sigma * |w_ic| * sgn(w_ic').
        """
        mu_g = self.propagation_rate
        w_ic_abs = abs(agent.weights[cid])
        for (src, tgt), sigma in pool.attack_graph.attacks.items():
            if src == cid and tgt in agent.weights:
                agent.weights[tgt] *= 1.0 - mu_g * sigma * w_ic_abs
        for (src, tgt), sigma in pool.attack_graph.supports.items():
            if src == cid and tgt in agent.weights:
                sign = 1.0 if agent.weights[tgt] >= 0 else -1.0
                agent.weights[tgt] += mu_g * sigma * w_ic_abs * sign
        agent.clamp_weights()

    # ------------------------------------------------------------------
    # Vectorized batch methods
    # ------------------------------------------------------------------

    def reflect_batch(
        self,
        state: StateTensor,
        pending: list[tuple[int, float, np.ndarray]],
        pool: ArgumentPool,
        rng: Generator,
        *,
        received_masks: list[np.ndarray] | None = None,
    ) -> None:
        """Vectorized reflect: loop over arguments, vectorize across agents.

        pending: list of (cid_idx, speaker_opinion, phi) where phi is (N,).
        received_masks: optional per-argument boolean masks indicating which
            agents actually received the argument. If None, inferred from phi > 0.
        """
        for idx, (cid_idx, speaker_opinion, phi) in enumerate(pending):
            effective_phi = phi.copy()
            if self.graph_aware_eval:
                effective_phi = self._attenuate_phi_batch(
                    state, cid_idx, effective_phi, pool,
                )
            self._update_batch_step(
                state, cid_idx, speaker_opinion, effective_phi, pool, rng,
            )
            state.clamp()
            if self.graph_aware_propagation:
                if received_masks is not None:
                    received = received_masks[idx]
                else:
                    received = phi > 0
                self._propagate_graph_batch(state, cid_idx, pool, received)

    def _update_batch_step(
        self,
        state: StateTensor,
        cid_idx: int,
        speaker_opinion: float,
        phi: np.ndarray,
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        """Engine-specific vectorized update. Override in subclasses."""
        raise NotImplementedError

    def _attenuate_phi_batch(
        self,
        state: StateTensor,
        cid_idx: int,
        phi: np.ndarray,
        pool: ArgumentPool,
    ) -> np.ndarray:
        """eq:graph-modulate §4.1.5 — Attenuate phi via attack graph for all agents."""
        cid = state.idx_to_cid[cid_idx]
        attackers = pool.attack_graph.get_attackers(cid)
        if not attackers:
            return phi
        attenuation = np.zeros(state.W.shape[0], dtype=np.float64)
        for c_prime, sigma in attackers.items():
            if c_prime not in state.cid_to_idx:
                continue
            c_prime_idx = state.cid_to_idx[c_prime]
            mask = state.R[:, c_prime_idx]
            attenuation += mask * sigma * np.abs(state.W[:, c_prime_idx])
        return phi * np.clip(1.0 - attenuation, 0.0, 1.0)

    def _propagate_graph_batch(
        self,
        state: StateTensor,
        cid_idx: int,
        pool: ArgumentPool,
        received: np.ndarray | None = None,
    ) -> None:
        """eq:graph-propagate §4.1.5 — Vectorized graph propagation."""
        mu_g = self.propagation_rate
        cid = state.idx_to_cid[cid_idx]
        w_ic_abs = np.abs(state.W[:, cid_idx])
        src_mask = state.R[:, cid_idx]
        if received is not None:
            src_mask = src_mask & received

        for (src, tgt), sigma in pool.attack_graph.attacks.items():
            if src != cid or tgt not in state.cid_to_idx:
                continue
            tgt_idx = state.cid_to_idx[tgt]
            tgt_mask = state.R[:, tgt_idx]
            mask = src_mask & tgt_mask
            factor = np.where(mask, 1.0 - mu_g * sigma * w_ic_abs, 1.0)
            state.W[:, tgt_idx] *= factor

        for (src, tgt), sigma in pool.attack_graph.supports.items():
            if src != cid or tgt not in state.cid_to_idx:
                continue
            tgt_idx = state.cid_to_idx[tgt]
            tgt_mask = state.R[:, tgt_idx]
            mask = src_mask & tgt_mask
            sign = np.where(state.W[:, tgt_idx] >= 0, 1.0, -1.0)
            state.W[:, tgt_idx] += mask * mu_g * sigma * w_ic_abs * sign

        state.clamp()

    @abstractmethod
    def update(
        self,
        agent: Agent,
        cid: str,
        speaker_opinion: float,
        influence: float,
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        """Apply a single argument's effect on the agent."""
        ...


@dataclass
class DeGrootEngine(SimpleUpdateEngine):
    """DeGroot averaging — eq:degroot §4.2.

    Consideration-level update: w_ic <- w_ic + mu * phi * (d_c * s_c - w_ic).
    Opinion shift is emergent via eq:opinion.

    Scaling formula: mu = 0.576 / K, where K is the expected number of
    update events per agent per session. At K ~ 96 (20 agents, 5 rounds):
    mu = 0.006. Empirical target: 7-28% change rate (Butler/Hansen).
    """

    mu: float = 0.006

    def update(
        self,
        agent: Agent,
        cid: str,
        speaker_opinion: float,
        influence: float,
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        """eq:degroot §4.2 — w_ic <- w_ic + mu * phi * (d_c * s_c - w_ic)."""
        if cid not in agent.weights:
            return
        c = pool.get(cid)
        target = c.direction * c.persuasiveness
        gb_mult = max(1.0 + 0.5 * (self.group_building_level - 3), 0.1)
        agent.weights[cid] += self.mu * gb_mult * influence * (target - agent.weights[cid])
        agent.clamp_weights()

    def _update_batch_step(
        self,
        state: StateTensor,
        cid_idx: int,
        speaker_opinion: float,
        phi: np.ndarray,
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        """Vectorized DeGroot: W[:, c] += mask * mu * phi * (target - W[:, c])."""
        c = pool.get(state.idx_to_cid[cid_idx])
        target = c.direction * c.persuasiveness
        mask = state.R[:, cid_idx].astype(np.float64)
        gb_mult = max(1.0 + 0.5 * (self.group_building_level - 3), 0.1)
        state.W[:, cid_idx] += mask * self.mu * gb_mult * phi * (target - state.W[:, cid_idx])


@dataclass
class BayesianEngine(SimpleUpdateEngine):
    """Bayesian updating: posterior precision-weighted combination.

    posterior = (prior_prec * prior + signal_prec * signal) / (prior_prec + signal_prec)
    Low prior_precision agents update more; high prior_precision agents update less.

    Scaling formula: signal_precision = 2 / K, where K is the expected
    number of update events per agent per session. At K ~ 100:
    sp = 0.02. Self-dampening: as precision accumulates, each update
    shrinks automatically, so magnitude is stable across session lengths.
    """

    signal_precision: float = 0.02

    def reflect(
        self,
        agent: Agent,
        round_updates: list[tuple[str, float, float]],
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        """Consensus-gated reflect (Barabas 2004).

        Computes directional consensus from the round's voiced arguments
        and gates each argument's influence proportionally.  Under balanced
        argument pools (equal pro/con), consensus ≈ 0 and opinions barely
        move; under imbalanced pools, consensus is high and updates proceed
        normally.  The signal is changed from speaker_opinion to
        ``c.direction * c.persuasiveness`` (argument-directional signal).
        """
        if not round_updates:
            return
        # Directional consensus: how aligned the round's arguments are
        net_direction = sum(
            pool.get(cid).direction * influence
            for cid, _, influence in round_updates
        )
        total_influence = sum(
            abs(influence) for _, _, influence in round_updates
        )
        consensus = (
            abs(net_direction) / total_influence
            if total_influence > 0 else 0.0
        )
        # Build gated updates: argument-directional signal, consensus-gated
        gated_updates: list[tuple[str, float, float]] = [
            (
                cid,
                pool.get(cid).direction * pool.get(cid).persuasiveness,
                influence * consensus,
            )
            for cid, _speaker_opinion, influence in round_updates
        ]
        super().reflect(agent, gated_updates, pool, rng)

    def update(
        self,
        agent: Agent,
        cid: str,
        speaker_opinion: float,
        influence: float,
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        """eq:bayesian-update §4.3 — Precision-weighted per-consideration posterior.

        speaker_opinion is sigma_j = d_c * s_c (after consensus rewrite).
        influence is phi' = phi * gamma (consensus-gated).
        """
        if cid not in agent.weights:
            return
        gb_mult = max(1.0 + 0.5 * (self.group_building_level - 3), 0.1)
        sigma_signal = self.signal_precision * gb_mult * influence
        tau = agent.precision
        tau_plus = tau + sigma_signal
        if tau_plus < 1e-12:
            return
        agent.weights[cid] = (
            tau * agent.weights[cid] + sigma_signal * speaker_opinion
        ) / tau_plus
        agent.precision = tau_plus
        agent.clamp_weights()

    def reflect_batch(
        self,
        state: StateTensor,
        pending: list[tuple[int, float, np.ndarray]],
        pool: ArgumentPool,
        rng: Generator,
        *,
        received_masks: list[np.ndarray] | None = None,
    ) -> None:
        """Consensus-gated vectorized reflect for Bayesian engine.

        Per-agent consensus is computed from the original phi values,
        then phi is gated and speaker_opinion rewritten to d_c * s_c.
        """
        if not pending:
            return

        n = state.W.shape[0]

        # Per-agent consensus from original influences
        net_direction = np.zeros(n, dtype=np.float64)
        total_influence = np.zeros(n, dtype=np.float64)
        for cid_idx, _sp, phi in pending:
            d_c = pool.get(state.idx_to_cid[cid_idx]).direction
            net_direction += d_c * phi
            total_influence += np.abs(phi)

        consensus = np.where(
            total_influence > 0,
            np.abs(net_direction) / total_influence,
            0.0,
        )

        # Track which agents received each argument (before gating)
        rm = received_masks if received_masks is not None else [
            phi > 0 for _, _, phi in pending
        ]

        # Gate phi and rewrite speaker_opinion
        gated_pending: list[tuple[int, float, np.ndarray]] = []
        for cid_idx, _sp, phi in pending:
            c = pool.get(state.idx_to_cid[cid_idx])
            new_sp = c.direction * c.persuasiveness
            gated_phi = phi * consensus
            gated_pending.append((cid_idx, new_sp, gated_phi))

        super().reflect_batch(
            state, gated_pending, pool, rng, received_masks=rm,
        )

    def _update_batch_step(
        self,
        state: StateTensor,
        cid_idx: int,
        speaker_opinion: float,
        phi: np.ndarray,
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        """Vectorized Bayesian: precision-weighted posterior."""
        mask = state.R[:, cid_idx]
        gb_mult = max(1.0 + 0.5 * (self.group_building_level - 3), 0.1)
        sigma_signal = self.signal_precision * gb_mult * phi
        tau = state.precisions.copy()
        tau_plus = tau + sigma_signal

        safe = tau_plus >= 1e-12
        active = mask & safe
        safe_tau_plus = np.where(active, tau_plus, 1.0)
        new_w = (
            tau * state.W[:, cid_idx] + sigma_signal * speaker_opinion
        ) / safe_tau_plus
        state.W[:, cid_idx] = np.where(active, new_w, state.W[:, cid_idx])
        state.precisions = np.where(active, tau_plus, state.precisions)


@dataclass
class BoundedConfidenceEngine(SimpleUpdateEngine):
    """Bounded confidence (Butler Eq.5): assimilate, contrast, or ignore.

    - distance < latitude_acceptance: assimilate (move toward)
    - distance > latitude_rejection: contrast (move away)
    - intermediate zone: no change

    Scaling formula: mu = 0.576 / K, where K is the expected number of
    update events per agent per session. At K ~ 96: mu = 0.006.
    Empirical target: 7-28% change rate (Butler/Hansen).
    """

    mu: float = 0.006

    def update(
        self,
        agent: Agent,
        cid: str,
        speaker_opinion: float,
        influence: float,
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        """eq:bc-update §4.4 — Three-zone per-consideration update.

        Delta_c = |w_ic - d_c * s_c|.
        Assimilation (Delta_c <= lat_acc): w_ic += mu * phi * (d_c*s_c - w_ic).
        Indifference (lat_acc < Delta_c < lat_rej): no change.
        Contrast (Delta_c >= lat_rej): w_ic -= mu * phi * (d_c*s_c - w_ic).
        """
        if cid not in agent.weights:
            return
        c = pool.get(cid)
        target = c.direction * c.persuasiveness
        delta_c = abs(agent.weights[cid] - target)

        lat_accept = agent.params.latitude_acceptance
        lat_reject = agent.params.latitude_rejection

        gb_mult = max(1.0 + 0.5 * (self.group_building_level - 3), 0.1)
        effective_mu = self.mu * gb_mult
        if delta_c <= lat_accept:
            agent.weights[cid] += effective_mu * influence * (target - agent.weights[cid])
        elif delta_c >= lat_reject:
            agent.weights[cid] -= effective_mu * influence * (target - agent.weights[cid])
        else:
            return
        agent.clamp_weights()

    def _update_batch_step(
        self,
        state: StateTensor,
        cid_idx: int,
        speaker_opinion: float,
        phi: np.ndarray,
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        """Vectorized BC: three-zone classification via np.where."""
        c = pool.get(state.idx_to_cid[cid_idx])
        target = c.direction * c.persuasiveness
        w_col = state.W[:, cid_idx]
        delta_c = np.abs(w_col - target)
        diff = target - w_col

        assimilate = delta_c <= state.lat_accept
        contrast = delta_c >= state.lat_reject
        mask = state.R[:, cid_idx]

        gb_mult = max(1.0 + 0.5 * (self.group_building_level - 3), 0.1)
        effective_mu = self.mu * gb_mult
        update = np.where(assimilate, effective_mu * phi * diff, 0.0)
        update = np.where(contrast & ~assimilate, -effective_mu * phi * diff, update)
        state.W[:, cid_idx] += mask * update


@dataclass
class StructuralAlignmentEngine(SimpleUpdateEngine):
    """Structural alignment: converge weight magnitudes, preserve signs.

    Agents align the *salience* they assign to each consideration (how
    important it is) while keeping their *stances* (pro/con sign).  This
    produces intersubjective consistency — agents agree on which issues
    matter even when they disagree on direction — which increases DRI
    (Niemeyer 2024).

    Scaling formula: lambda_salience = 5 / K, where K is the expected
    number of update events per agent per session.  At K ~ 100:
    lambda_salience = 0.05.  noise_sd adds small per-weight jitter to
    avoid degenerate convergence.
    """

    lambda_salience: float = 0.05
    noise_sd: float = 0.01

    def update(
        self,
        agent: Agent,
        cid: str,
        speaker_opinion: float,
        influence: float,
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        """Converge magnitude toward shared salience with coherence bias.

        Two combined mechanisms:
        1. **Shared salience** — magnitude moves toward ``influence *
           persuasiveness``, making agents agree on which considerations
           are important.
        2. **Coherence bias** — the target is shifted up for weights that
           contribute in the same direction as the agent's opinion, and
           down for opposing weights.  This tightens the correlation
           between weight structure and opinion (DRI ↑).

        The sign of each weight is never changed, preserving stances.
        """
        if cid not in agent.weights:
            return
        c = pool.get(cid)
        opinion = agent.opinion(pool)

        old_w = agent.weights[cid]
        sign = 1.0 if old_w >= 0 else -1.0
        old_mag = abs(old_w)

        # Base target: shared salience (all agents converge toward this)
        base_target = influence * c.persuasiveness

        # Coherence: positive when weight contribution aligns with opinion
        contribution_sign = float(np.sign(old_w * c.direction))
        opinion_sign = (
            float(np.sign(opinion)) if abs(opinion) > 1e-12 else 0.0
        )
        coherence = contribution_sign * opinion_sign  # +1, 0, or -1

        # GB modulation: scale both step size and coherence bias
        gb_multiplier = max(1.0 + 0.5 * (self.group_building_level - 3), 0.1)
        effective_lambda = self.lambda_salience * gb_multiplier

        # Coherence-biased target: consistent weights aim higher
        target = base_target * (1.0 + 0.5 * gb_multiplier * coherence)

        new_mag = old_mag + effective_lambda * (target - old_mag)
        new_mag += float(rng.normal(0.0, self.noise_sd))
        new_mag = max(new_mag, 0.0)
        agent.weights[cid] = float(np.clip(sign * new_mag, -1.0, 1.0))

    def reflect_batch(
        self,
        state: StateTensor,
        pending: list[tuple[int, float, np.ndarray]],
        pool: ArgumentPool,
        rng: Generator,
        *,
        received_masks: list[np.ndarray] | None = None,
    ) -> None:
        """Pre-generate noise in agent-first order, then delegate to base."""
        n = state.W.shape[0]
        k = len(pending)
        self._batch_noise = np.zeros((n, k), dtype=np.float64)
        self._batch_received = np.zeros((n, k), dtype=np.bool_)
        for i in range(n):
            for j, (cid_idx, _sp, phi) in enumerate(pending):
                if state.R[i, cid_idx] and phi[i] > 0:
                    self._batch_received[i, j] = True
                    self._batch_noise[i, j] = float(
                        rng.normal(0.0, self.noise_sd),
                    )
        self._batch_arg_counter = 0
        super().reflect_batch(
            state, pending, pool, rng, received_masks=received_masks,
        )
        del self._batch_noise
        del self._batch_received
        del self._batch_arg_counter

    def _update_batch_step(
        self,
        state: StateTensor,
        cid_idx: int,
        speaker_opinion: float,
        phi: np.ndarray,
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        """Vectorized structural alignment: sign-preserving salience convergence."""
        j = self._batch_arg_counter
        self._batch_arg_counter += 1
        received = self._batch_received[:, j]
        noise = self._batch_noise[:, j]

        c = pool.get(state.idx_to_cid[cid_idx])
        opinions = state.opinions()
        w_col = state.W[:, cid_idx]

        sign = np.where(w_col >= 0, 1.0, -1.0)
        old_mag = np.abs(w_col)

        base_target = phi * c.persuasiveness

        contribution_sign = np.sign(w_col * c.direction)
        opinion_sign = np.sign(opinions)
        opinion_sign = np.where(np.abs(opinions) <= 1e-12, 0.0, opinion_sign)
        coherence = contribution_sign * opinion_sign

        gb_multiplier = max(1.0 + 0.5 * (self.group_building_level - 3), 0.1)
        effective_lambda = self.lambda_salience * gb_multiplier
        target = base_target * (1.0 + 0.5 * gb_multiplier * coherence)

        new_mag = old_mag + effective_lambda * (target - old_mag) + noise
        new_mag = np.maximum(new_mag, 0.0)
        new_w = np.clip(sign * new_mag, -1.0, 1.0)
        state.W[:, cid_idx] = np.where(received, new_w, w_col)


@dataclass
class ArgumentBasedEngine(CognitiveEngine):
    """Argumentation-based engine using grounded semantics.

    voice: strategic voicing selects highest-impact consideration.
    evaluate: reduced influence for attacked arguments (grounded semantics).
    reflect: latent-orientation-dependent transition probabilities.

    Transition model (J&S impossibility):
      - Consistent agents (latent == expressed): p_CI = p_reject_base *
        (1 - 0.6*soph) * influence * persuasiveness.  Low probability of
        losing consistency; sophisticated agents resist even more.
      - Inconsistent agents (latent != expressed): p_IC = p_accept_base *
        (0.5 + 0.5*soph + soph^2) * influence * persuasiveness.

    group_building_level scales p_accept/p_reject; facilitated=True
    applies the open-mindedness bonus (omega_eff = min(omega + 0.076*(g-1), 1)).

    Scaling formulas: p_accept = 26.6 / K, p_reject = 11.4 / K, where K
    is the expected number of update events per agent per session. At
    K ~ 95: p_accept = 0.28, p_reject = 0.12.
    """

    strategic_voicing: bool = True
    p_accept_base: float = 0.28
    p_reject_base: float = 0.12
    use_grounded_semantics: bool = True
    group_building_level: int = 3
    facilitated: bool = False

    def voice(
        self, agent: Agent, pool: ArgumentPool, rng: Generator
    ) -> str:
        if not agent.weights:
            ids = pool.all_ids()
            return str(rng.choice(ids))

        if self.strategic_voicing:
            # Select the consideration with highest |weight * persuasiveness|
            return max(
                agent.weights,
                key=lambda c: abs(agent.weights[c]) * pool.get(c).persuasiveness,
            )
        else:
            # Random selection proportional to |weight|
            cids = list(agent.weights.keys())
            abs_w = np.array([abs(agent.weights[c]) for c in cids])
            total = abs_w.sum()
            if total == 0:
                probs = np.ones(len(cids)) / len(cids)
            else:
                probs = abs_w / total
            idx = int(rng.choice(len(cids), p=probs))
            return cids[idx]

    def evaluate(
        self,
        agent: Agent,
        cid: str,
        speaker_opinion: float,
        pool: ArgumentPool,
        rng: Generator,
    ) -> float:
        """Graded influence attenuation by held attackers' weight magnitudes.

        Attenuation is proportional to sigma * |w_ic'| for each attacker
        in the agent's repertoire, matching the graph_aware_eval formula
        used by Engines 1-4.
        When use_grounded_semantics is False, always returns 1.0.
        """
        if not self.use_grounded_semantics:
            return 1.0
        attackers = pool.attack_graph.get_attackers(cid)
        if not attackers:
            return 1.0

        attenuation = sum(
            attack_strength * abs(agent.weights[attacker_id])
            for attacker_id, attack_strength in attackers.items()
            if attacker_id in agent.weights
        )
        return float(np.clip(1.0 - attenuation, 0.0, 1.0))

    def reflect(
        self,
        agent: Agent,
        round_updates: list[tuple[str, float, float]],
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        """Orientation-dependent weight transitions, recomputed per argument.

        D17: orientation = sgn(o_i) (unified latent/expressed).
        D14: opinion recomputed after each weight update (online).
        D18: consistency recomputed per argument.
        """
        soph = agent.params.open_mindedness
        if self.facilitated:
            soph = min(soph + 0.076 * (self.group_building_level - 1), 1.0)
        gb_mult = max(1.0 + 0.5 * (self.group_building_level - 3), 0.1)

        for cid, speaker_opinion, influence in round_updates:
            if cid not in agent.weights:
                continue
            c = pool.get(cid)

            # D14+D17+D18: recompute opinion and consistency per argument
            opinion = agent.opinion(pool)
            orientation = float(np.sign(opinion)) if abs(opinion) > 1e-10 else 0.0
            is_consistent = orientation != 0.0

            if is_consistent:
                p_transition = (
                    self.p_reject_base * gb_mult
                    * (1.0 - 0.6 * soph)
                    * influence
                    * c.persuasiveness
                )
            else:
                p_transition = (
                    self.p_accept_base * gb_mult
                    * (0.5 + 0.5 * soph + soph**2)
                    * influence
                    * c.persuasiveness
                )
            p_transition = float(np.clip(p_transition, 0.0, 1.0))

            if float(rng.random()) < p_transition:
                if is_consistent:
                    shift_sign = -orientation * c.direction
                else:
                    shift_sign = c.direction
                agent.weights[cid] += shift_sign * influence
        agent.clamp_weights()

    def reflect_batch(
        self,
        state: StateTensor,
        pending: list[tuple[int, float, np.ndarray]],
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        """Vectorized ArgumentBased: per-argument consistency recomputation."""
        n = state.W.shape[0]
        k = len(pending)

        # Pre-generate random values in agent-first order (matching sequential)
        rand_vals = np.full((n, k), 2.0, dtype=np.float64)
        for i in range(n):
            for j, (cid_idx, _sp, phi) in enumerate(pending):
                if state.R[i, cid_idx] and phi[i] > 0:
                    rand_vals[i, j] = float(rng.random())

        soph = state.open_mindedness.copy()
        if self.facilitated:
            soph = np.minimum(soph + 0.076 * (self.group_building_level - 1), 1.0)
        gb_mult = max(1.0 + 0.5 * (self.group_building_level - 3), 0.1)

        for j, (cid_idx, _sp, phi) in enumerate(pending):
            c = pool.get(state.idx_to_cid[cid_idx])
            mask = state.R[:, cid_idx] & (phi > 0)

            # D14+D17+D18: recompute opinion and consistency per argument
            opinions = state.opinions()
            orientation = np.sign(opinions)
            orientation = np.where(np.abs(opinions) <= 1e-10, 0.0, orientation)
            is_consistent = orientation != 0.0

            p_ci = (
                self.p_reject_base * gb_mult
                * (1.0 - 0.6 * soph)
                * phi
                * c.persuasiveness
            )
            p_ic = (
                self.p_accept_base * gb_mult
                * (0.5 + 0.5 * soph + soph ** 2)
                * phi
                * c.persuasiveness
            )
            p_transition = np.where(is_consistent, p_ci, p_ic)
            p_transition = np.clip(p_transition, 0.0, 1.0)

            do_transition = mask & (rand_vals[:, j] < p_transition)

            ci_shift = -orientation * c.direction
            ic_shift = np.full(n, c.direction)
            shift_sign = np.where(is_consistent, ci_shift, ic_shift)

            state.W[:, cid_idx] += do_transition * shift_sign * phi

        state.clamp()


@dataclass
class EmpiricalArgumentEngine(CognitiveEngine):
    """Empirical comparison engine with aligned statement-level defaults.

    Canonical defaults now mirror the live LLM path on the structural
    decision grammar rather than on hidden psychology:
      - Voice from currently supported statements only, with a narrow
        attack-based fallback when no supported statement is available.
      - Evaluate via the local attack/support neighborhood already held in
        the listener's repertoire.
      - Reflect through explicit statement-level stance updates with a
        no-update region and adoption threshold, instead of a hard
        congruence gate plus directional-delta kernel.

    The legacy congruence-gated directional-delta configuration is retained
    for backward compatibility and ablations, but it is no longer the
    canonical comparison default.
    """

    # Voicing mode:
    #   "impact_prop" -- probabilistic, probs proportional to
    #       |w_c| * persuasiveness. Strong arguments are more likely but
    #       weaker ones can still surface, so the aired-argument pool
    #       reflects the full repertoire probabilistically.
    #   "argmax"                -- deterministic argmax over |w_c| * beta.
    #       Every voice call surfaces the single highest-impact consideration;
    #       diversity of voiced arguments is constrained to the top-k the
    #       agent holds most strongly.
    #   "weight_prop"           -- probs proportional to |w_c| only
    #       (ignores persuasiveness). Tests what happens when voicing is
    #       personal-salience-driven rather than impact-aware.
    voicing_mode: str = "weight_prop"
    supported_only_voice: bool = True
    fallback_attack_max: int = 3
    base_lr: float = 0.20
    precision_power: float = 1.0
    support_weight: float = 0.5
    apply_congruence_gate: bool = False
    reflect_mode: str = "explicit_stance"
    neutral_influence: float = 1.0
    no_update_band: float = 0.15
    adoption_threshold: float = 0.25
    p_pro_base: float = 0.28
    p_counter_base: float = 0.12
    confirmation_bias: float = 0.0

    def _voice_candidates(
        self,
        agent: Agent,
        pool: ArgumentPool,
    ) -> tuple[list[str], np.ndarray]:
        if self.supported_only_voice:
            supported = [
                (cid, weight)
                for cid, weight in agent.weights.items()
                if weight > 0.0 and cid in pool.considerations
            ]
            if supported:
                return [cid for cid, _ in supported], np.array([weight for _, weight in supported])

            opposed = [
                (cid, weight)
                for cid, weight in agent.weights.items()
                if weight < 0.0 and cid in pool.considerations
            ]
            if opposed:
                target_cid, _ = min(opposed, key=lambda item: (item[1], item[0]))
                attackers = [
                    (attacker_id, strength)
                    for attacker_id, strength in pool.attack_graph.get_attackers(target_cid).items()
                    if attacker_id in pool.considerations and attacker_id not in agent.weights
                ]
                attackers.sort(key=lambda item: (-item[1], item[0]))
                attackers = attackers[: self.fallback_attack_max]
                if attackers:
                    return [cid for cid, _ in attackers], np.array([strength for _, strength in attackers])

        cids = [cid for cid in agent.weights if cid in pool.considerations]
        return cids, np.array([abs(agent.weights[cid]) for cid in cids])

    def _influence_signal(self, influence: float) -> float:
        if influence >= self.neutral_influence:
            upper_span = max(1.5 - self.neutral_influence, 1e-6)
            return float(np.clip((influence - self.neutral_influence) / upper_span, 0.0, 1.0))
        lower_span = max(self.neutral_influence, 1e-6)
        return float(np.clip((influence - self.neutral_influence) / lower_span, -1.0, 0.0))

    def _reflect_explicit_stance(
        self,
        agent: Agent,
        cid: str,
        influence: float,
        pool: ArgumentPool,
        eta: float,
    ) -> None:
        signal = self._influence_signal(influence)
        strength = abs(signal)
        if strength <= self.no_update_band:
            return
        if cid not in agent.weights and strength < self.adoption_threshold:
            return

        target = 1.0 if signal > 0.0 else -1.0
        current = agent.weights.get(cid, 0.0)
        step = float(np.clip(eta * pool.get(cid).persuasiveness * strength, 0.0, 1.0))
        new_weight = current + step * (target - current)
        if abs(new_weight) < 1e-6:
            agent.weights.pop(cid, None)
            return
        agent.weights[cid] = new_weight

    def voice(
        self, agent: Agent, pool: ArgumentPool, rng: Generator
    ) -> str:
        if not agent.weights:
            ids = pool.all_ids()
            return str(rng.choice(ids))

        cids, base_scores = self._voice_candidates(agent, pool)
        if not cids:
            ids = pool.all_ids()
            return str(rng.choice(ids))

        if self.voicing_mode == "impact_prop":
            arg_persuasiveness = np.array([pool.get(c).persuasiveness for c in cids])
            scores = base_scores * arg_persuasiveness
        elif self.voicing_mode == "weight_prop":
            scores = base_scores
        elif self.voicing_mode == "argmax":
            scores = base_scores * np.array([pool.get(c).persuasiveness for c in cids])
        else:
            raise ValueError(
                f"Unknown voicing_mode: {self.voicing_mode!r}. "
                "Expected 'impact_prop', 'argmax', or 'weight_prop'."
            )

        if self.voicing_mode == "argmax":
            idx = int(np.argmax(scores))
            return cids[idx]

        total = scores.sum()
        probs = scores / total if total > 0 else np.ones(len(cids)) / len(cids)
        idx = int(rng.choice(len(cids), p=probs))
        return cids[idx]

    def evaluate(
        self,
        agent: Agent,
        cid: str,
        speaker_opinion: float,
        pool: ArgumentPool,
        rng: Generator,
    ) -> float:
        """Influence = (1 - held-attackers + support_weight * held-supporters)
        * (1 + cb * congruence), clipped [0, 1.5].
        """
        influence = 1.0
        for atk_id, atk_str in pool.attack_graph.get_attackers(cid).items():
            if atk_id in agent.weights:
                influence -= abs(agent.weights[atk_id]) * atk_str
        for sup_id, sup_str in pool.attack_graph.get_supporters(cid).items():
            if sup_id in agent.weights:
                influence += (
                    self.support_weight * abs(agent.weights[sup_id]) * sup_str
                )

        if self.confirmation_bias > 0.0:
            c = pool.get(cid)
            congruence = compute_congruence(agent.weights, pool, c.direction)
            influence *= (1.0 + self.confirmation_bias * congruence)

        return float(np.clip(influence, 0.0, 1.5))

    def reflect(
        self,
        agent: Agent,
        round_updates: list[tuple[str, float, float]],
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        """Congruence-gated proportional weight update.

        For each heard (cid, speaker_opinion, influence):
          1. Compute congruence of the agent's repertoire with c.direction
             (compute_congruence -> {-1, 0, +1}).
          2. Gate: pro-attitudinal fires with p_pro_base*influence*beta;
             counter-attitudinal with p_counter_base*influence*beta;
             neutral with the mean. Clipped [0, 1].
          3. If fired, dw = (base_lr / precision) * influence * beta * c.direction.
        """
        precision = max(agent.params.prior_precision, 1e-3)
        eta = self.base_lr / (precision ** self.precision_power)

        for cid, _speaker_opinion, influence in round_updates:
            if influence <= 0.0:
                continue
            if self.reflect_mode == "explicit_stance":
                self._reflect_explicit_stance(agent, cid, influence, pool, eta)
                continue
            if cid not in agent.weights:
                continue
            c = pool.get(cid)
            congruence = 0
            fired = True

            if self.apply_congruence_gate:
                congruence = compute_congruence(agent.weights, pool, c.direction)
                if congruence > 0:
                    p_gate = self.p_pro_base * influence * c.persuasiveness
                elif congruence < 0:
                    p_gate = self.p_counter_base * influence * c.persuasiveness
                else:
                    p_gate = (
                        0.5 * (self.p_pro_base + self.p_counter_base)
                        * influence * c.persuasiveness
                    )
                p_gate = float(np.clip(p_gate, 0.0, 1.0))
                if float(rng.random()) >= p_gate:
                    self._on_gate_result(
                        agent, cid, congruence, fired=False,
                        influence=influence, persuasiveness=c.persuasiveness,
                    )
                    continue

            self._on_gate_result(
                agent, cid, congruence, fired=fired,
                influence=influence, persuasiveness=c.persuasiveness,
            )
            delta = eta * influence * c.persuasiveness * c.direction
            agent.weights[cid] += delta
        agent.clamp_weights()

    def _on_gate_result(
        self,
        agent: Agent,
        cid: str,
        congruence: int,
        fired: bool,
        influence: float,
        persuasiveness: float,
    ) -> None:
        """Hook for subclasses to record per-event gate results. No-op here."""
        pass


@dataclass
class MixedEngine(CognitiveEngine):
    """Delegates to constituent engines selected by probability.

    Each hook call independently selects an engine.
    """

    engines: list[tuple[CognitiveEngine, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.engines:
            total = sum(p for _, p in self.engines)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"Engine probabilities must sum to 1.0, got {total}"
                )

    def _select(self, rng: Generator) -> CognitiveEngine:
        """Pick an engine by probability."""
        probs = np.array([p for _, p in self.engines])
        idx = int(rng.choice(len(self.engines), p=probs))
        return self.engines[idx][0]

    def voice(
        self, agent: Agent, pool: ArgumentPool, rng: Generator
    ) -> str:
        return self._select(rng).voice(agent, pool, rng)

    def evaluate(
        self,
        agent: Agent,
        cid: str,
        speaker_opinion: float,
        pool: ArgumentPool,
        rng: Generator,
    ) -> float:
        return self._select(rng).evaluate(
            agent, cid, speaker_opinion, pool, rng
        )

    def reflect(
        self,
        agent: Agent,
        round_updates: list[tuple[str, float, float]],
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        self._select(rng).reflect(agent, round_updates, pool, rng)
