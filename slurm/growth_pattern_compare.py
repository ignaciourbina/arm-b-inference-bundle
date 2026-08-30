#!/usr/bin/env python3
"""Compare deliberation GROWTH PATTERNS across model scales.

The substantive question: does a more capable model deliberate with a different
SHAPE, not just a better score? Arm-B at E2B concentrates ~40% of opinion
movement in round 1 and the median agent stops revising at round 6. If capacity
changes that profile -- later peak, longer horizon, more sustained revision --
that is a finding about deliberation dynamics, not a leaderboard delta.

Reasoning ON is a LOCKED project decision (2026-08-30), so the default cell
list is reasoning-on only: the model-scale curve is the live question. The
reasoning-off cells still load if named explicitly -- they are the historical
evidence that closed that axis, not a comparison arm.

Two families of indicator, reported separately because they answer different
questions:

  MOVEMENT   mean |d opinion| by round, round-1 share, median last-change
             round, frozen-after-r2 fraction, cumulative profile.
             Cheap, low-variance, and it is what "settling" means.

  SUBSTANCE  DRI and meta-consensus BY ROUND, plus argumentative diversity by
             round. This is the debate indicator proper: DRI asks whether
             agents who agree on the bottom line also share the *reasons*,
             after the shared directional component is projected out.

DRI definitions are imported from `llm/score_ablation_cells.py` rather than
reimplemented, so the curve and the paired scorer cannot drift apart.

CAVEAT, and it is not a small one: DRI is very noisy (CV ~110% per run, range
-0.27 to +0.81). A 15-run pilot cell cannot support a DRI *conclusion*. Every
DRI/mc row therefore carries a standard error, and a curve whose swings are
inside +/- 2 SE should be read as flat.

Usage: growth_pattern_compare.py [cell ...]
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.score_ablation_cells import trajectory_at  # noqa: E402

ROOT = "llm/traces/collection"
SC = json.load(open("llm/scenarios/minimum_wage_seattle_crossover.json"))
DIR = {c["id"]: float(c["direction"]) for c in SC["considerations"]}
IDS = [c["id"] for c in SC["considerations"]]
CIDS = sorted(DIR)
M = len(IDS)


def runs(cell):
    for f in glob.glob(f"{ROOT}/{cell}/townhall_*.json"):
        if "_checkpoint" in f or "_trace_" in f:
            continue
        try:
            yield json.load(open(f))
        except Exception:
            pass


def profile(cell):
    dO, last, clip, tot, n = [], [], 0, 0, 0
    dri_runs, mc_runs, var_runs = [], [], []
    voiced_runs, topshare_runs = [], []
    for rec in runs(cell):
        snaps = rec.get("snapshots") or []
        if len(snaps) < 2:
            continue
        n += 1
        O = np.array([[a["opinion"] for a in s] for s in snaps])
        W = np.array([[[a["weights"].get(i, 0.0) for i in IDS] for a in s] for s in snaps])
        raw = (W * np.array([DIR[i] for i in IDS])).sum(axis=2) / M
        clip += int((np.abs(raw) > 1 + 1e-9).sum())
        tot += raw.size
        dO.append(np.abs(np.diff(O, axis=0)).mean(axis=1))
        for col in O.T:
            ch = np.nonzero(np.abs(np.diff(col)) > 1e-9)[0]
            last.append(int(ch[-1]) + 1 if len(ch) else 0)

        # Substantive indicators at every snapshot, same definitions the paired
        # scorer uses. snapshot 0 is pre-deliberation, so the curve is r0..rN.
        tr = [trajectory_at(s, DIR, CIDS) for s in snaps]
        dri_runs.append([x[0] for x in tr])
        mc_runs.append([x[1] for x in tr])
        var_runs.append([x[2] for x in tr])

        # Argumentative diversity per round (voicing side, not weight side).
        dv, ts = [], []
        for rnd in rec.get("rounds", []):
            cids = [v["cid"] for v in (rnd.get("voices") or [])]
            dv.append(len(set(cids)))
            ts.append(max(cids.count(c) for c in set(cids)) / len(cids) if cids else np.nan)
        voiced_runs.append(dv)
        topshare_runs.append(ts)

    if not n:
        return None

    def stack(rows):
        w = min(len(r) for r in rows)
        return np.array([r[:w] for r in rows], dtype=float)

    dO = np.array(dO).mean(axis=0)
    last = np.array(last)
    D, C, V = stack(dri_runs), stack(mc_runs), stack(var_runs)
    return dict(
        n=n, traj=len(last), dO=dO,
        r1_share=100 * dO[0] / dO.sum(),
        median_last=float(np.median(last)),
        frozen_r2=100 * (last <= 2).sum() / len(last),
        clip=100 * clip / max(tot, 1),
        cum=np.cumsum(dO) / dO.sum(),
        dri=np.nanmean(D, axis=0), dri_se=np.nanstd(D, axis=0) / np.sqrt(n),
        mc=np.nanmean(C, axis=0), mc_se=np.nanstd(C, axis=0) / np.sqrt(n),
        var=np.nanmean(V, axis=0),
        voiced=stack(voiced_runs).mean(axis=0),
        topshare=np.nanmean(stack(topshare_runs), axis=0),
    )


def horizon(curve):
    """Round at which the indicator has covered 50% / 90% of its total change.

    The substantive analogue of median-last-change: a model that keeps
    integrating reasons late has a later 90% round even if its endpoint is the
    same. NaN when the net change is too small to locate meaningfully.
    """
    net = curve[-1] - curve[0]
    if not np.isfinite(net) or abs(net) < 1e-6:
        return None, None
    frac = (curve - curve[0]) / net
    r50 = next((i for i, f in enumerate(frac) if f >= 0.5), None)
    r90 = next((i for i, f in enumerate(frac) if f >= 0.9), None)
    return r50, r90


def row(label, vals, width=8, fmt="{:>8.4f}"):
    return f"{label:<24}" + "".join(fmt.format(v) for v in vals)


def main():
    cells = sys.argv[1:] or ["armb_sota_r256_t10", "ms_E4B", "ms_26B_a4b"]
    res = {}
    for c in cells:
        p = profile(c)
        if p:
            res[c] = p
    if not res:
        print("no cells with data yet")
        return 0

    R = max(len(p["dO"]) for p in res.values())
    RS = max(len(p["dri"]) for p in res.values())
    print("=" * 100)
    print("GROWTH-PATTERN COMPARISON ACROSS MODEL SCALE")
    print("=" * 100)

    print("\n-- MOVEMENT ------------------------------------------------------")
    print(f"\nmean |d opinion| by round\n{'cell':<24}" + "".join(f"{'r' + str(i + 1):>8}" for i in range(R)))
    for c, p in res.items():
        print(row(c, p["dO"]))

    print(f"\n{'cell':<24}{'runs':>5}{'traj':>6}{'r1 share':>10}{'med last':>10}{'frozen r2':>11}{'clipped':>9}")
    for c, p in res.items():
        print(f"{c:<24}{p['n']:>5}{p['traj']:>6}{p['r1_share']:>9.1f}%{p['median_last']:>10.0f}"
              f"{p['frozen_r2']:>10.0f}%{p['clip']:>8.2f}%")

    print(f"\ncumulative share of total movement by round (curve shape)\n{'cell':<24}"
          + "".join(f"{'r' + str(i + 1):>8}" for i in range(R)))
    for c, p in res.items():
        print(row(c, p["cum"], fmt="{:>8.2f}"))

    print("\n-- SUBSTANCE -----------------------------------------------------")
    hdr = f"{'cell':<24}" + "".join(f"{'r' + str(i):>8}" for i in range(RS))
    for key, se, name in (("dri", "dri_se", "DRI (direction-orthogonal reason sharing)"),
                          ("mc", "mc_se", "meta-consensus")):
        print(f"\n{name} by round   (r0 = pre-deliberation)")
        print(hdr)
        for c, p in res.items():
            print(row(c, p[key]))
            print(row("  +/- SE", p[se], fmt="{:>8.3f}"))

    print("\nargumentative diversity by round")
    print(f"{'cell':<24}" + "".join(f"{'r' + str(i + 1):>8}" for i in range(R)))
    for c, p in res.items():
        print(row(c + " distinct", p["voiced"], fmt="{:>8.2f}"))
        print(row("  top-cid share", p["topshare"], fmt="{:>8.2f}"))

    print(f"\nsubstantive horizon: round reaching 50% / 90% of net change")
    print(f"{'cell':<24}{'DRI r50':>10}{'DRI r90':>10}{'mc r50':>10}{'mc r90':>10}"
          f"{'DRI r0':>9}{'DRI end':>9}")
    for c, p in res.items():
        d50, d90 = horizon(p["dri"])
        m50, m90 = horizon(p["mc"])
        f = lambda x: "-" if x is None else str(x)  # noqa: E731
        print(f"{c:<24}{f(d50):>10}{f(d90):>10}{f(m50):>10}{f(m90):>10}"
              f"{p['dri'][0]:>9.3f}{p['dri'][-1]:>9.3f}")

    print("\nDRI is noisy (CV ~110% per run). Read a curve as flat unless its")
    print("swings exceed ~2 SE. n=15 cells cannot support a DRI conclusion.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
