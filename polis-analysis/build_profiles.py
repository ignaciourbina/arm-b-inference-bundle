#!/usr/bin/env python3
"""Stage 2-3, 6-7: from the co-vote matrix, build the scenario edges and the
Ising participant profiles. Validates against the manuscript methods appendix.

Run:
    source agora/.venv/bin/activate
    python polis-analysis/build_profiles.py
"""
from __future__ import annotations

import numpy as np

from build_covote import (
    PAIR_MIN_JOINT,
    build_q_matrix,
    load_vote_matrix,
)

# Stage 2: the 10 selected considerations (manuscript Table 5).
# (cid, polis_comment_id, direction)  direction: +1 Pro, -1 Con
SELECTED = [
    ("C_01", "48", +1),
    ("C_02", "11", +1),
    ("C_03", "32", -1),
    ("C_04", "8",  +1),
    ("C_05", "45", +1),
    ("C_06", "46", -1),
    ("C_07", "36", -1),
    ("C_08", "39", +1),
    ("C_09", "0",  -1),
    ("C_10", "20", -1),
]
CIDS = [s[0] for s in SELECTED]
POLIS_IDS = [s[1] for s in SELECTED]
DIRECTIONS = np.array([s[2] for s in SELECTED], dtype=float)

EDGE_MIN_Q = 0.2


def edge_strength(q: float) -> float:
    return 0.2 + 0.6 * min(abs(q), 1.0)


def build_edges(Q: np.ndarray) -> tuple[list, list]:
    """Stage 3: attacks (Q<0) and supports (Q>0) where |Q|>=0.2."""
    attacks, supports = [], []
    n = len(CIDS)
    for i in range(n):
        for j in range(i + 1, n):
            q = Q[i, j]
            if np.isnan(q) or abs(q) < EDGE_MIN_Q:
                continue
            s = round(edge_strength(q), 3)
            if q < 0:
                attacks.append((CIDS[i], CIDS[j], s))
            else:
                supports.append((CIDS[i], CIDS[j], s))
    return attacks, supports


def build_ising_profiles(V: np.ndarray, col_index: dict[str, int], Q: np.ndarray):
    """Stages 6-7: mean-field imputation + policy/coherence scores.

    Returns list of profile dicts for the engaged pool (>=3 directional votes).
    """
    cols = [col_index[p] for p in POLIS_IDS]
    Vsel = V[:, cols]                       # (n_participants, 10), {+1,-1,0,nan}
    directional = np.isin(Vsel, (-1.0, 1.0))

    # External field h_i = atanh(mean directional vote), clipped off +-1.
    h = np.zeros(len(CIDS))
    for i in range(len(CIDS)):
        col = Vsel[:, i]
        dvotes = col[np.isin(col, (-1.0, 1.0))]
        vbar = np.clip(dvotes.mean(), -0.999, 0.999)
        h[i] = np.arctanh(vbar)

    Qz = np.nan_to_num(Q, nan=0.0)          # couplings; nan -> 0

    profiles = []
    for p in range(V.shape[0]):
        obs = directional[p]                # boolean (10,)
        n_obs = int(obs.sum())
        if n_obs < 3:
            continue
        v = Vsel[p].copy()
        w = np.empty(len(CIDS))
        for i in range(len(CIDS)):
            if obs[i]:
                w[i] = v[i]
            else:
                # mean-field over observed neighbours
                s = h[i] + float(np.sum(Qz[i, obs] * v[obs]))
                w[i] = np.tanh(s)
        r_p = float(np.mean(DIRECTIONS * w))                  # policy score (full vector)
        E_p = -float(np.sum(np.triu(Qz, 1) * np.outer(w, w))) # Ising energy (full vector)
        profiles.append({
            "participant": p,
            "n_observed": n_obs,
            # Sparse starting repertoire: ONLY observed considerations (their
            # clamped votes). Unobserved enter later via adoption. The full
            # imputed vector (below) is used only to derive policy/coherence.
            "weights": {CIDS[i]: float(v[i]) for i in range(len(CIDS)) if obs[i]},
            "full_weights": {CIDS[i]: float(w[i]) for i in range(len(CIDS))},
            "policy_score": r_p,
            "neg_energy": -E_p,   # standardized later
        })

    # Coherence = z-standardized negative energy over the engaged pool.
    negE = np.array([pr["neg_energy"] for pr in profiles])
    z = (negE - negE.mean()) / negE.std(ddof=0)
    for pr, zz in zip(profiles, z):
        pr["coherence"] = float(zz)
        r = pr["policy_score"]
        pr["policy_cell"] = "Pro" if r > 0.1 else "Con" if r < -0.1 else "Ambivalent"
        pr["coherence_cell"] = "Coherent" if zz > 1 else "Ambivalent" if zz < -1 else "Mixed"
    return profiles


def main() -> None:
    V, comment_cols = load_vote_matrix()
    col_index = {cid: k for k, cid in enumerate(comment_cols)}

    # Q among the 10 selected
    Q10 = build_q_matrix(V, col_index, POLIS_IDS)

    # --- Stage 3: edges ---
    attacks, supports = build_edges(Q10)
    print("=== Stage 3 VALIDATION (edges) ===")
    print(f"  attacks: {len(attacks)}   (manuscript 15)")
    print(f"  supports: {len(supports)}  (manuscript 19)")
    print("  sample attacks:", attacks[:3])
    print("  sample supports:", supports[:3])
    # spot-checks vs Table 6
    amap = {(a, b): s for a, b, s in attacks}
    smap = {(a, b): s for a, b, s in supports}
    print(f"  Attack C_01-C_03 = {amap.get(('C_01','C_03'))}  (manuscript .457)")
    print(f"  Attack C_01-C_10 = {amap.get(('C_01','C_10'))}  (manuscript .615)")
    print(f"  Support C_01-C_02 = {smap.get(('C_01','C_02'))}  (manuscript .645)")
    print(f"  Support C_05-C_08 = {smap.get(('C_05','C_08'))}  (manuscript .643)")

    # --- Stages 6-7: profiles ---
    profiles = build_ising_profiles(V, col_index, Q10)
    n = len(profiles)
    n_obs = np.array([p["n_observed"] for p in profiles])
    print("\n=== Stages 6-7 VALIDATION (Ising profiles) ===")
    print(f"  engaged profiles: {n}   (manuscript 108)")
    print(f"  mean observed: {n_obs.mean():.2f}  median {int(np.median(n_obs))}  range {n_obs.min()}-{n_obs.max()}   (manuscript 5.40, 5, 3-10)")
    pct_obs = 100 * n_obs.sum() / (n * 10)
    print(f"  %% observed cells: {pct_obs:.1f}%  (manuscript 54.0%)")

    pol = [p["policy_cell"] for p in profiles]
    coh = [p["coherence_cell"] for p in profiles]
    def pct(lst, k): return f"{lst.count(k)} ({100*lst.count(k)/len(lst):.1f}%)"
    print(f"  policy:    Pro {pct(pol,'Pro')}  Con {pct(pol,'Con')}  Amb {pct(pol,'Ambivalent')}")
    print(f"             (manuscript 69/63.9%, 31/28.7%, 8/7.4%)")
    print(f"  coherence: Coh {pct(coh,'Coherent')}  Mix {pct(coh,'Mixed')}  Amb {pct(coh,'Ambivalent')}")
    print(f"             (manuscript 22/20.4%, 59/54.6%, 27/25.0%)")


if __name__ == "__main__":
    main()
