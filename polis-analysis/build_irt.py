#!/usr/bin/env python3
"""Stage 4-5: fit a 1-D 2PL IRT on the directional vote matrix, then compute
cross-over endorsement -> persuasiveness. Validates against Table 5 (Pers column),
which also pins the IRT sign orientation.

Run:
    source agora/.venv/bin/activate
    python polis-analysis/build_irt.py
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from build_covote import load_vote_matrix, moderated_comment_ids
from build_profiles import CIDS, DIRECTIONS, POLIS_IDS

# Manuscript Table 5 persuasiveness targets (for validation).
PERS_TARGET = {
    "C_01": .817, "C_02": .900, "C_03": .300, "C_04": .633, "C_05": .734,
    "C_06": .416, "C_07": .768, "C_08": .852, "C_09": .750, "C_10": .333,
}
CROSS_TARGET = {
    "C_01": .471, "C_02": .541, "C_03": .032, "C_04": .314, "C_05": .400,
    "C_06": .130, "C_07": .429, "C_08": .500, "C_09": .414, "C_10": .061,
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def fit_2pl(Y: np.ndarray, mask: np.ndarray, n_iter_restart: int = 1):
    """Joint-ML 2PL. Y in {0,1} (agree=1), mask=observed directional cell.

    Returns (theta[P], a[I], b[I]) with theta standardized to mean0/sd1.
    """
    P, I = Y.shape

    def unpack(x):
        theta = x[:P]
        a = x[P:P + I]
        b = x[P + I:]
        return theta, a, b

    def nll(x):
        theta, a, b = unpack(x)
        z = a[None, :] * (theta[:, None] - b[None, :])
        p = _sigmoid(z)
        eps = 1e-9
        ll = Y * np.log(p + eps) + (1 - Y) * np.log(1 - p + eps)
        val = -np.sum(ll[mask])
        # mild ridge for identifiability
        val += 1e-3 * (np.sum(theta**2) + np.sum(a**2) + np.sum(b**2))
        return val

    # init: theta from row mean agree-rate, a=1, b=0
    theta0 = np.zeros(P)
    for p in range(P):
        m = mask[p]
        theta0[p] = (Y[p, m].mean() - 0.5) * 2 if m.any() else 0.0
    x0 = np.concatenate([theta0, np.ones(I), np.zeros(I)])
    res = minimize(nll, x0, method="L-BFGS-B",
                   options={"maxiter": 2000, "maxfun": 50000})
    theta, a, b = unpack(res.x)

    # standardize theta (positive scaling preserves agree-prob ordering)
    sd = theta.std(ddof=0) or 1.0
    theta = (theta - theta.mean()) / sd
    a = a * sd
    return theta, a, b


def main() -> None:
    V, comment_cols = load_vote_matrix()
    col_index = {cid: k for k, cid in enumerate(comment_cols)}

    for item_set_name, polis_ids, dirs in [
        ("10-selected", POLIS_IDS, DIRECTIONS),
        ("29-eligible", None, None),
    ]:
        if polis_ids is None:
            # eligible = moderated with >=34 directional
            moderated = moderated_comment_ids()
            polis_ids = [c for c in moderated
                         if np.isin(V[:, col_index[c]], (-1.0, 1.0)).sum() >= 34]
            dirs = None  # unknown content directions for the non-selected items

        cols = [col_index[c] for c in polis_ids]
        Vsel = V[:, cols]
        mask = np.isin(Vsel, (-1.0, 1.0))
        Y = (Vsel == 1.0).astype(float)   # agree=1, disagree=0

        theta, a, b = fit_2pl(Y, mask)

        # Orient theta so pro-policy = high theta.
        # Use the 10 selected items' known directions to set the sign.
        sel_idx = {c: k for k, c in enumerate(polis_ids)}
        orient = 0.0
        for cid, pid, d in zip(CIDS, POLIS_IDS, DIRECTIONS):
            if pid in sel_idx:
                orient += d * a[sel_idx[pid]]
        if orient < 0:
            theta = -theta
            a = -a

        # Cross-over endorsement for the 10 selected, using this theta.
        print(f"\n=== IRT fit on {item_set_name}: cross-over -> persuasiveness ===")
        cross = {}
        for cid, pid, d in zip(CIDS, POLIS_IDS, DIRECTIONS):
            c = col_index[pid]
            col = V[:, c]
            directional = np.isin(col, (-1.0, 1.0))
            opposing = (theta < 0) if d > 0 else (theta > 0)
            sel = directional & opposing
            if sel.sum() == 0:
                cross[cid] = np.nan
            else:
                cross[cid] = float((col[sel] == 1.0).mean())

        # Min-max rescale to [0.3, 0.9]
        cv = np.array([cross[c] for c in CIDS])
        lo, hi = np.nanmin(cv), np.nanmax(cv)
        pers = 0.3 + 0.6 * (cv - lo) / (hi - lo)

        print(f"  {'cid':5s} {'cross':>7s} {'(tgt)':>7s}  {'pers':>6s} {'(tgt)':>7s}")
        err = 0.0
        for k, cid in enumerate(CIDS):
            ct, pt = CROSS_TARGET[cid], PERS_TARGET[cid]
            print(f"  {cid:5s} {cross[cid]:7.3f} {ct:7.3f}  {pers[k]:6.3f} {pt:7.3f}")
            err += abs(pers[k] - pt)
        print(f"  mean |pers error|: {err/len(CIDS):.3f}")


if __name__ == "__main__":
    main()
