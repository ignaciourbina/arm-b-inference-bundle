#!/usr/bin/env python3
"""Elective tool use per run — a candidate mediator, measured properly.

`query_supports` / `query_attacks` / `query_repertoire` are ELECTIVE: the engine
offers them and the model's weights decide whether to spend a call. So a query
count is behavior, not a resource fact (IU) — and because what an agent queries
shapes what it learns, and what it learns shapes its weight updates, elective
exploration is a plausible MEDIATOR of deliberative outcomes rather than an
epiphenomenon.

That makes it testable on data already collected: the 300-run E2B collection can
establish, at full power and zero GPU cost, whether exploration relates to
deliberative outcomes at all. If it does, a between-model difference in
exploration is interpretable; if it does not, it is a curiosity.

Per run this reports elective-query share alongside the substantive outcomes
(DRI, meta-consensus, opinion variance), joined by (composition, seed), and the
rank correlation between them. Rank correlation because DRI is heavy-tailed and
its outliers would dominate a Pearson estimate.

Usage: tool_usage.py CELL [CELL...]  [--limit N]
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.score_ablation_cells import trajectory_at  # noqa: E402

ROOT = "llm/traces/collection"
FINAL_RE = re.compile(r"_(?P<comp>polarized|symmetric|three_clusters)_n6_s(?P<seed>\d+)_(?P<ts>\d+)\.json$")
ELECTIVE = ("query_supports", "query_attacks", "query_repertoire")


def tool_counts(trace_path):
    try:
        d = json.load(open(trace_path))
    except Exception:
        return None
    agg = {}
    for c in d.get("calls", []):
        for x in ((c.get("response_parsed") or {}).get("tool_calls") or []):
            n = (x.get("function") or {}).get("name")
            agg[n] = agg.get(n, 0) + 1
    return agg


def outcomes(final_path):
    try:
        rec = json.load(open(final_path))
    except Exception:
        return None
    cons = rec.get("scenario", {}).get("considerations", [])
    dirs = {c["id"]: float(c["direction"]) for c in cons}
    snaps = rec.get("snapshots") or []
    if not dirs or len(snaps) < 2:
        return None
    dri, mc, var = trajectory_at(snaps[-1], dirs, sorted(dirs))
    return dict(dri=dri, mc=mc, var=var,
                wchg=rec["summary"].get("total_weight_changes"),
                calls=rec["summary"].get("total_llm_calls"))


def cell_rows(cell, limit=None):
    rows = []
    finals = sorted(f for f in glob.glob(f"{ROOT}/{cell}/townhall_*.json")
                    if "_checkpoint" not in f and "_trace_" not in f and "_live" not in f)
    if limit:
        finals = finals[:limit]
    for fin in finals:
        m = FINAL_RE.search(os.path.basename(fin))
        if not m:
            continue
        tr = fin.replace(f"_s{m.group('seed')}_{m.group('ts')}.json",
                         f"_s{m.group('seed')}_trace_{m.group('ts')}.json")
        agg = tool_counts(tr) if os.path.exists(tr) else None
        out = outcomes(fin)
        if agg is None or out is None:
            continue
        tot = sum(agg.values())
        if not tot:
            continue
        q = sum(agg.get(k, 0) for k in ELECTIVE)
        rows.append(dict(comp=m.group("comp"), seed=int(m.group("seed")),
                         tool_calls=tot, elective=q, q_share=100.0 * q / tot, **out))
    return rows


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = None
    for a in sys.argv[1:]:
        if a.startswith("--limit"):
            limit = int(a.split("=", 1)[1]) if "=" in a else None
    if not args:
        print(__doc__)
        return 2
    for cell in args:
        rows = cell_rows(cell, limit)
        if not rows:
            print(f"=== {cell}: no runs with both a final and a trace ===\n")
            continue
        qs = np.array([r["q_share"] for r in rows])
        print(f"=== {cell}  (n={len(rows)} runs with paired trace) ===")
        print(f"  elective-query share %   median {np.median(qs):.1f}   "
              f"range {qs.min():.1f}–{qs.max():.1f}   IQR "
              f"{np.percentile(qs, 25):.1f}–{np.percentile(qs, 75):.1f}")
        print(f"  elective calls / run     median {np.median([r['elective'] for r in rows]):.0f}"
              f"   of {np.median([r['tool_calls'] for r in rows]):.0f} tool calls")
        if len(rows) >= 10:
            print(f"\n  Does exploration relate to outcomes? (Spearman, n={len(rows)})")
            print(f"  {'outcome':<22}{'rho':>8}{'95% CI':>18}{'p':>10}")
            for key, name in (("dri", "DRI (final)"), ("mc", "meta-consensus"),
                              ("var", "opinion variance"), ("wchg", "weight changes"),
                              ("calls", "LLM calls")):
                v = np.array([r[key] for r in rows], dtype=float)
                ok = np.isfinite(v) & np.isfinite(qs)
                if ok.sum() < 10:
                    continue
                rho, p = stats.spearmanr(qs[ok], v[ok])
                # Fisher z interval. A null without an interval says nothing:
                # it cannot distinguish "no effect" from "no power".
                n_ok = int(ok.sum())
                z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
                se = 1.0 / np.sqrt(n_ok - 3)
                lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
                flag = "  *" if p < 0.05 else ""
                print(f"  {name:<22}{rho:>8.3f}   [{lo:>6.3f},{hi:>6.3f}]{p:>10.4f}{flag}")
            print("\n  A non-significant rho with a tight CI rules out effects larger than")
            print("  the interval; it is not evidence of exactly zero.")
        else:
            print(f"\n  n={len(rows)} — too few for a correlation; description only.")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
