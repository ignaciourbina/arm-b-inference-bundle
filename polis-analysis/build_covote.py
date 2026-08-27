#!/usr/bin/env python3
"""Stage 0-1: load the Seattle Polis vote matrix and build the Yule's Q co-vote
coupling matrix (the "Ising coupling matrix").

Validates against the manuscript methods appendix:
  - 30 moderated -> 29 eligible (>=34 directional votes)
  - off-diagonal Q (29-item): mean +0.099, sd 0.339, range [-0.714, +0.803],
    59.4% positive, 37.2% negative

Run:
    source agora/.venv/bin/activate
    python polis-analysis/build_covote.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POLIS_DIR = ROOT / "datasets" / "polis-openData" / "15-per-hour-seattle"

ELIGIBILITY_MIN_DIRECTIONAL = 34   # ~10% of 339
PAIR_MIN_JOINT = 10                 # min joint directional voters for a Q estimate


def load_vote_matrix() -> tuple[np.ndarray, list[str]]:
    """Return (V, comment_ids) where V[p, c] in {+1,-1,0} or NaN (missing).

    Columns are the raw Polis comment-ids (strings) in file order.
    """
    path = POLIS_DIR / "participants-votes.csv"
    with open(path) as f:
        r = csv.reader(f)
        hdr = next(r)
        comment_cols = hdr[6:]  # first 6 are metadata
        rows = []
        for row in r:
            cells = row[6:]
            vals = [np.nan if c == "" else float(c) for c in cells]
            rows.append(vals)
    V = np.array(rows, dtype=float)
    return V, comment_cols


def moderated_comment_ids() -> list[str]:
    """Comment-ids of the 30 moderated statements (moderated == 1)."""
    path = POLIS_DIR / "comments.csv"
    with open(path) as f:
        r = csv.DictReader(f)
        return [row["comment-id"] for row in r if row.get("moderated") == "1"]


def yules_q(vi: np.ndarray, vj: np.ndarray) -> tuple[float, int]:
    """Yule's Q for two directional vote vectors. Returns (Q, n_joint).

    Only cells where BOTH are directional (+1/-1) count. Q undefined -> nan.
    """
    mask = np.isin(vi, (-1.0, 1.0)) & np.isin(vj, (-1.0, 1.0))
    n = int(mask.sum())
    if n < PAIR_MIN_JOINT:
        return np.nan, n
    a = int(((vi == vj) & mask).sum())          # same direction
    d = int(((vi != vj) & mask).sum())          # opposite direction
    if a + d == 0:
        return np.nan, n
    return (a - d) / (a + d), n


def directional_counts(V: np.ndarray, col_index: dict[str, int], ids: list[str]) -> dict[str, int]:
    return {cid: int(np.isin(V[:, col_index[cid]], (-1.0, 1.0)).sum()) for cid in ids}


def build_q_matrix(V: np.ndarray, col_index: dict[str, int], ids: list[str]) -> np.ndarray:
    """Full symmetric Q matrix over `ids` (diagonal = nan)."""
    n = len(ids)
    Q = np.full((n, n), np.nan)
    for ii in range(n):
        for jj in range(ii + 1, n):
            q, _ = yules_q(V[:, col_index[ids[ii]]], V[:, col_index[ids[jj]]])
            Q[ii, jj] = Q[jj, ii] = q
    return Q


def main() -> None:
    V, comment_cols = load_vote_matrix()
    col_index = {cid: k for k, cid in enumerate(comment_cols)}
    n_participants = V.shape[0]
    print(f"participants: {n_participants}, comment columns: {len(comment_cols)}")

    moderated = moderated_comment_ids()
    print(f"moderated statements: {len(moderated)}")

    # Stage 0: eligibility
    dcounts = directional_counts(V, col_index, moderated)
    eligible = [c for c in moderated if dcounts[c] >= ELIGIBILITY_MIN_DIRECTIONAL]
    dropped = [c for c in moderated if dcounts[c] < ELIGIBILITY_MIN_DIRECTIONAL]
    print(f"eligible (>= {ELIGIBILITY_MIN_DIRECTIONAL} directional): {len(eligible)}")
    print(f"  dropped: {[(c, dcounts[c]) for c in dropped]}")

    # Stage 1: Q over eligible set
    Q = build_q_matrix(V, col_index, eligible)
    off = Q[~np.isnan(Q)]
    off = off[np.triu_indices_from(Q, k=1)[0].size and slice(None)]  # keep all non-nan
    # recompute strictly upper-triangle non-nan values
    iu = np.triu_indices(len(eligible), k=1)
    vals = Q[iu]
    vals = vals[~np.isnan(vals)]
    n_pairs_total = len(iu[0])
    n_pos = int((vals > 0).sum())
    n_neg = int((vals < 0).sum())

    print("\n=== Stage 1 VALIDATION (off-diagonal Q, eligible set) ===")
    print(f"  n pairs (defined / total):  {len(vals)} / {n_pairs_total}")
    print(f"  mean   : {vals.mean():+.3f}   (manuscript +0.099)")
    print(f"  sd     : {vals.std(ddof=0):.3f}    (manuscript 0.339)")
    print(f"  range  : [{vals.min():+.3f}, {vals.max():+.3f}]   (manuscript [-0.714, +0.803])")
    print(f"  %% positive (of all pairs): {100*n_pos/n_pairs_total:.1f}%   (manuscript 59.4%)")
    print(f"  %% negative (of all pairs): {100*n_neg/n_pairs_total:.1f}%   (manuscript 37.2%)")
    # also report relative to defined pairs
    print(f"  [alt] %% pos/neg of DEFINED pairs: {100*n_pos/len(vals):.1f}% / {100*n_neg/len(vals):.1f}%")


if __name__ == "__main__":
    main()
