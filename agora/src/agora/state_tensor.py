"""Packed array view of agent weights for vectorized deliberation.

Version : 0.1.0
Module  : agora.state_tensor
Spec    : agents/vectorization/RESEARCH.md

Provides StateTensor: a (N, C) numpy array W that mirrors the weight
dicts of N agents over C considerations. All hot-path operations
(opinion, clamp, update) become array operations on W instead of
per-agent Python loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from agora.agents import Agent
    from agora.considerations import ArgumentPool


@dataclass
class StateTensor:
    """Packed array view of all agent weights for a group.

    Attributes:
        W: (N, C) weight matrix.
        R: (N, C) boolean mask — True where agent i holds consideration c.
        cid_to_idx: consideration id -> column index.
        idx_to_cid: column index -> consideration id.
        agent_to_idx: agent id -> row index.
        directions: (C,) direction vector for opinion computation.
        R_sizes: (N,) number of considerations per agent.
        precisions: (N,) Bayesian precision per agent (mutable).
        lat_accept: (N,) latitude_acceptance per agent.
        lat_reject: (N,) latitude_rejection per agent.
        open_mindedness: (N,) open_mindedness per agent.
    """

    W: np.ndarray
    R: np.ndarray
    cid_to_idx: dict[str, int]
    idx_to_cid: dict[int, str] = field(repr=False)
    agent_to_idx: dict[str, int]
    directions: np.ndarray
    R_sizes: np.ndarray
    precisions: np.ndarray
    lat_accept: np.ndarray
    lat_reject: np.ndarray
    open_mindedness: np.ndarray

    @classmethod
    def from_agents(
        cls,
        agents: list[Agent],
        pool: ArgumentPool,
    ) -> StateTensor:
        """Pack agent weight dicts into a (N, C) array.

        Column order follows pool.all_ids(). Row order follows the
        agents list. Non-repertoire entries are zero in W and False in R.
        """
        all_cids = pool.all_ids()
        C = len(all_cids)
        N = len(agents)

        cid_to_idx = {cid: j for j, cid in enumerate(all_cids)}
        idx_to_cid = {j: cid for cid, j in cid_to_idx.items()}
        agent_to_idx = {a.id: i for i, a in enumerate(agents)}

        W = np.zeros((N, C), dtype=np.float64)
        R = np.zeros((N, C), dtype=np.bool_)

        for i, agent in enumerate(agents):
            for cid, w in agent.weights.items():
                j = cid_to_idx[cid]
                W[i, j] = w
                R[i, j] = True

        directions = np.array(
            [pool.get(cid).direction for cid in all_cids],
            dtype=np.float64,
        )
        R_sizes = R.sum(axis=1).astype(np.float64)

        precisions = np.array(
            [a.precision for a in agents], dtype=np.float64,
        )
        lat_accept = np.array(
            [a.params.latitude_acceptance for a in agents], dtype=np.float64,
        )
        lat_reject = np.array(
            [a.params.latitude_rejection for a in agents], dtype=np.float64,
        )
        open_mindedness = np.array(
            [a.params.open_mindedness for a in agents], dtype=np.float64,
        )

        return cls(
            W=W,
            R=R,
            cid_to_idx=cid_to_idx,
            idx_to_cid=idx_to_cid,
            agent_to_idx=agent_to_idx,
            directions=directions,
            R_sizes=R_sizes,
            precisions=precisions,
            lat_accept=lat_accept,
            lat_reject=lat_reject,
            open_mindedness=open_mindedness,
        )

    def sync_back(self, agents: list[Agent]) -> None:
        """Write W and mutable state back to agents.

        Only updates weights for considerations already in each agent's
        repertoire (R mask). Does not add or remove considerations.
        Also syncs precision (modified by BayesianEngine).
        """
        for i, agent in enumerate(agents):
            for cid in agent.weights:
                j = self.cid_to_idx[cid]
                agent.weights[cid] = float(self.W[i, j])
            agent.precision = float(self.precisions[i])

    def opinions(self) -> np.ndarray:
        """Compute all opinions: O = clip(W @ d / |R_i|, -1, 1).

        Empty repertoire (R_sizes == 0) yields opinion 0.0.
        Returns (N,) array.
        """
        raw = self.W @ self.directions  # (N,)
        safe_sizes = np.where(self.R_sizes > 0, self.R_sizes, 1.0)
        result = raw / safe_sizes
        result = np.where(self.R_sizes > 0, result, 0.0)
        return np.clip(result, -1.0, 1.0)

    def clamp(self) -> None:
        """Clip all weights to [-1, 1] in place."""
        np.clip(self.W, -1.0, 1.0, out=self.W)
