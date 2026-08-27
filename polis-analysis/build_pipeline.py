#!/usr/bin/env python3
"""Consolidated generator: rebuild the lost MPSA artifacts from the recovered
raw Polis data, faithful to the manuscript methods appendix.

Outputs (to polis-analysis/output/):
  - minimum_wage_seattle_crossover.json   scenario (considerations + edges)
  - ising_profiles.json                    108 engaged participant profiles
  - irt_ising_theta.json                   {participant_id: latent theta}

Fidelity policy:
  - Co-vote matrix, edges, profiles: re-derived from raw votes; validated
    EXACTLY against the manuscript (Q stats, 15/19 edges, 108 profiles, splits).
  - Persuasiveness: uses the PUBLISHED Table 5 values (exact). Our IRT cross-over
    reconstruction tracks them at corr ~0.85 but is not bit-exact, so we defer to
    the published numbers for the scenario.
  - latent_theta: reconstructed via bounded 2PL IRT (corr ~0.97 with policy
    score). Never published per-participant, so this is the best faithful proxy.

Run:
    source agora/.venv/bin/activate
    python polis-analysis/build_pipeline.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from build_covote import build_q_matrix, load_vote_matrix
from build_profiles import (
    CIDS,
    DIRECTIONS,
    POLIS_IDS,
    build_edges,
    build_ising_profiles,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "output"
COMMENTS = ROOT / "datasets" / "polis-openData" / "15-per-hour-seattle" / "comments.csv"

# Published Table 5 persuasiveness (exact, from the manuscript).
PERS_PUBLISHED = {
    "C_01": 0.817, "C_02": 0.900, "C_03": 0.300, "C_04": 0.633, "C_05": 0.734,
    "C_06": 0.416, "C_07": 0.768, "C_08": 0.852, "C_09": 0.750, "C_10": 0.333,
}
# Short human labels (cluster themes from the manuscript).
LABELS = {
    "C_01": "living wage / subsidy", "C_02": "inflation protection for workers",
    "C_03": "inflation and fairness", "C_04": "older-worker income",
    "C_05": "income inequality", "C_06": "small-business survival",
    "C_07": "automation", "C_08": "basic-income transition",
    "C_09": "service redesign", "C_10": "systemic critique",
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def fit_irt_theta(V, col_index, a_max: float = 2.0):
    """Bounded 2PL with N(0,1) theta prior on the 10 selected, pro-polarity coded.

    Returns (theta[P], a[10], b[10]) oriented so high theta = pro-policy.
    """
    cols = [col_index[p] for p in POLIS_IDS]
    Vsel = V[:, cols]
    mask = np.isin(Vsel, (-1.0, 1.0))
    Y = ((Vsel * DIRECTIONS[None, :]) == 1.0).astype(float)  # pro-policy response
    P, I = Y.shape

    def unpack(x):
        return x[:P], x[P:P + I], x[P + I:]

    def nll(x):
        th, a, b = unpack(x)
        z = a[None, :] * (th[:, None] - b[None, :])
        p = _sigmoid(z)
        eps = 1e-9
        val = -np.sum((Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps))[mask])
        val += 0.5 * np.sum(th ** 2)   # N(0,1) prior -> proper MML-style regularization
        return val

    th0 = np.array([(Y[p, mask[p]].mean() - 0.5) * 2 if mask[p].any() else 0.0
                    for p in range(P)])
    x0 = np.concatenate([th0, np.ones(I), np.zeros(I)])
    bounds = [(-5, 5)] * P + [(0.2, a_max)] * I + [(-4, 4)] * I
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 3000, "maxfun": 80000})
    th, a, b = unpack(res.x)
    sd = th.std(ddof=0) or 1.0
    th = (th - th.mean()) / sd
    a = a * sd
    # pro-polarity coding => a already oriented positive; ensure high theta = pro
    if np.sum(DIRECTIONS * np.sign(a)) < 0:  # safety
        th = -th
    return th, a, b


def statement_text() -> dict[str, str]:
    want = {pid: cid for cid, pid in zip(CIDS, POLIS_IDS)}
    out = {}
    with open(COMMENTS) as f:
        for row in csv.DictReader(f):
            if row["comment-id"] in want:
                out[want[row["comment-id"]]] = row["comment-body"].strip()
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    V, comment_cols = load_vote_matrix()
    col_index = {c: k for k, c in enumerate(comment_cols)}

    Q10 = build_q_matrix(V, col_index, POLIS_IDS)
    attacks, supports = build_edges(Q10)
    theta, a, b = fit_irt_theta(V, col_index)
    profiles = build_ising_profiles(V, col_index, Q10)
    texts = statement_text()

    # ---- scenario JSON ----
    considerations = []
    for k, cid in enumerate(CIDS):
        considerations.append({
            "id": cid,
            "label": LABELS[cid],
            "direction": float(DIRECTIONS[k]),
            "persuasiveness": PERS_PUBLISHED[cid],
            "irt_a": round(float(a[k]), 4),
            "irt_b": round(float(b[k]), 4),
            "polis_comment_id": POLIS_IDS[k],
            "statement": texts.get(cid, ""),
        })
    scenario = {
        "name": "minimum_wage_seattle_crossover",
        "source": "Polis open data 15-per-hour-seattle (pol.is/2demo)",
        "provenance": "Reconstructed from raw votes per manuscript appendix G; "
                      "persuasiveness from published Table 5; edges/profiles re-derived.",
        "considerations": considerations,
        "attacks": [{"attacker": x, "target": y, "strength": s} for x, y, s in attacks],
        "supports": [{"supporter": x, "target": y, "strength": s} for x, y, s in supports],
    }
    (OUT / "minimum_wage_seattle_crossover.json").write_text(json.dumps(scenario, indent=2))

    # ---- profiles JSON ----
    profile_records = []
    for pr in profiles:
        pid = pr["participant"]
        profile_records.append({
            "participant_id": int(pid),
            "n_observed": pr["n_observed"],
            "weights": pr["weights"],            # sparse: observed considerations only
            "full_weights": pr["full_weights"],  # full imputed vector (scoring only)
            "policy_score": round(pr["policy_score"], 6),
            "coherence": round(pr["coherence"], 6),
            "policy_cell": pr["policy_cell"],
            "coherence_cell": pr["coherence_cell"],
            "latent_theta": round(float(theta[pid]), 6),
        })
    (OUT / "ising_profiles.json").write_text(json.dumps({
        "n_profiles": len(profile_records),
        "source": "Polis 15-per-hour-seattle; mean-field Ising imputation (appendix G.6).",
        "profiles": profile_records,
    }, indent=2))

    # ---- theta JSON ----
    theta_map = {str(pr["participant"]): round(float(theta[pr["participant"]]), 6)
                 for pr in profiles}
    (OUT / "irt_ising_theta.json").write_text(json.dumps(theta_map, indent=2))

    # ---- summary ----
    print("Wrote 3 artifacts to", OUT)
    print(f"  scenario: 10 considerations, {len(attacks)} attacks, {len(supports)} supports")
    print(f"  profiles: {len(profile_records)} engaged")
    pol = [p["policy_cell"] for p in profile_records]
    coh = [p["coherence_cell"] for p in profile_records]
    print(f"  policy:    Pro {pol.count('Pro')}  Con {pol.count('Con')}  Amb {pol.count('Ambivalent')}")
    print(f"  coherence: Coh {coh.count('Coherent')}  Mix {coh.count('Mixed')}  Amb {coh.count('Ambivalent')}")
    th = np.array([p["latent_theta"] for p in profile_records])
    rp = np.array([p["policy_score"] for p in profile_records])
    print(f"  corr(theta, policy_score): {np.corrcoef(th, rp)[0,1]:.3f}")


if __name__ == "__main__":
    main()
