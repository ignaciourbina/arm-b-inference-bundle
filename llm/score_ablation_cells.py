#!/usr/bin/env python3
"""Score Sprint-15 ablation cells against the control, paired by seed.

Every cell runs the same (composition, seed) grid, so each cell is compared to
control as a set of PAIRED differences rather than two independent samples.
With five seeds per composition that matters: a paired test removes the
seed-to-seed variation that would otherwise swamp a small prompt effect.

Outcomes are chosen to answer the questions the ablations were designed for:

  voicing        distinct considerations per agent, and the share of a room's
                 utterances taken by its most-voiced consideration. Sprint 14
                 identified these as the sharpest arm difference (1.5 vs 3.2
                 distinct per agent), so they are the primary prompt outcome.
  saturation     fraction of weights at |w| >= 0.99 after round 1. This is the
                 proposed *mechanism* for the repetition; if a prompt variant
                 changes voicing without changing saturation, the repetition was
                 prompt-induced, and vice versa.
  trajectory     dri_orthogonal, mc_agreement and opinion_variance at the final
                 round, to confirm a variant does not silently move the headline
                 results while fixing a secondary one.
  reliability    recorded LLM call errors and tool-loop retries, which is where
                 heavier quantization is most likely to show damage.

Control defaults to the validated production collection, whose seeds 1-5 were
produced by the configuration this programme calls `control`.

Usage:
    python llm/score_ablation_cells.py --cells prompt_anti-repetition prompt_terse
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import pearsonr

BASE_DIR = Path(__file__).resolve().parent.parent
CONTROL_DIR = BASE_DIR / "llm/traces/beta_local/arm_b_local_p6cache"
ABLATION_ROOT = BASE_DIR / "llm/traces/ablation"

FINAL_RE = re.compile(r"_(?P<comp>polarized|symmetric|three_clusters)_n6_s(?P<seed>\d+)_\d+\.json$")


def load_runs(directory: Path, seeds: set[int]) -> dict[tuple[str, int], dict]:
    """Map (composition, seed) -> trace record, ignoring checkpoints/call traces."""
    out: dict[tuple[str, int], dict] = {}
    if not directory.exists():
        return out
    for path in sorted(directory.iterdir()):
        if "_trace_" in path.name or "checkpoint" in path.name:
            continue
        m = FINAL_RE.search(path.name)
        if not m:
            continue
        seed = int(m.group("seed"))
        if seed not in seeds:
            continue
        try:
            out[(m.group("comp"), seed)] = json.loads(path.read_bytes())
        except Exception:
            continue
    return out


def metrics_for(record: dict) -> dict:
    """Per-run outcomes: voicing, saturation, trajectory endpoints."""
    cons = record.get("scenario", {}).get("considerations", [])
    dirs = {c["id"]: float(c["direction"]) for c in cons}
    cids = sorted(dirs)
    M = len(cids) or 1

    voiced: list[str] = []
    by_agent: dict[str, list[str]] = {}
    for rnd in record.get("rounds", []):
        for v in (rnd.get("voices") or []):
            voiced.append(v["cid"])
            by_agent.setdefault(v.get("agent_id", "?"), []).append(v["cid"])

    snaps = record.get("snapshots", [])
    def sat(idx: int) -> float:
        if idx >= len(snaps):
            return float("nan")
        ws = [w for a in snaps[idx] for w in a["weights"].values()]
        return sum(1 for w in ws if abs(w) >= 0.99) / len(ws) if ws else float("nan")

    def traj(idx: int) -> tuple[float, float, float]:
        if idx >= len(snaps):
            return (float("nan"),) * 3
        snap = snaps[idx]
        W = [dict(a["weights"]) for a in snap]
        ops = [float(np.clip(sum(v * dirs[c] for c, v in w.items() if c in dirs) / M, -1, 1))
               for w in W]
        mc = 1.0 - sum(float(np.std([abs(w.get(c, 0.0)) for w in W])) for c in cids) / M
        union = sorted({c for w in W for c in w})
        d = np.array([dirs[c] for c in union], dtype=float)
        dn = float(d @ d) or 1.0
        cs, os_ = [], []
        for (wa, oa), (wb, ob) in combinations(zip(W, ops), 2):
            va = np.array([wa.get(c, 0.0) for c in union], dtype=float)
            vb = np.array([wb.get(c, 0.0) for c in union], dtype=float)
            va = va - (float(va @ d) / dn) * d
            vb = vb - (float(vb @ d) / dn) * d
            na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
            cs.append(0.0 if na < 1e-12 or nb < 1e-12 else float(va @ vb / (na * nb)))
            os_.append(1.0 - abs(oa - ob))
        dri = (float(pearsonr(cs, os_)[0])
               if len(cs) >= 2 and np.std(cs) > 1e-12 and np.std(os_) > 1e-12 else float("nan"))
        return dri, mc, float(np.var(ops))

    dri8, mc8, var8 = traj(len(snaps) - 1)
    return {
        "voice_events": float(len(voiced)),
        "distinct_room": float(len(set(voiced))),
        "distinct_per_agent": st.mean(len(set(v)) for v in by_agent.values()) if by_agent else float("nan"),
        "top_cid_share": (Counter(voiced).most_common(1)[0][1] / len(voiced)) if voiced else float("nan"),
        "saturation_r1": sat(1),
        "dri_final": dri8,
        "mc_final": mc8,
        "opinion_var_final": var8,
        "llm_calls": float(sum(r.get("llm_calls", 0) for r in record.get("rounds", []))),
    }


OUTCOMES = [
    ("distinct_per_agent", "distinct args / agent"),
    ("top_cid_share", "top-cid share"),
    ("distinct_room", "distinct args / room"),
    ("saturation_r1", "saturation @r1"),
    ("dri_final", "DRI (final)"),
    ("mc_final", "meta-consensus (final)"),
    ("opinion_var_final", "opinion var (final)"),
    ("llm_calls", "LLM calls / run"),
]


def paired_report(control: dict, cell: dict, name: str) -> list[dict]:
    keys = sorted(set(control) & set(cell))
    rows = []
    if not keys:
        print(f"  [warn] {name}: no paired runs against control")
        return rows
    cm = {k: metrics_for(control[k]) for k in keys}
    xm = {k: metrics_for(cell[k]) for k in keys}
    for field, label in OUTCOMES:
        a = np.array([cm[k][field] for k in keys], dtype=float)
        b = np.array([xm[k][field] for k in keys], dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 2:
            continue
        diff = b[ok] - a[ok]
        t = p = float("nan")
        if diff.std(ddof=1) > 1e-12:
            t, p = stats.ttest_rel(b[ok], a[ok])
        rows.append({
            "cell": name, "outcome": label, "n_pairs": int(ok.sum()),
            "control": float(a[ok].mean()), "variant": float(b[ok].mean()),
            "delta": float(diff.mean()),
            "t": float(t) if not math.isnan(t) else None,
            "p": float(p) if not math.isnan(p) else None,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--control-dir", type=Path, default=CONTROL_DIR)
    ap.add_argument("--ablation-root", type=Path, default=ABLATION_ROOT)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", type=Path,
                    default=BASE_DIR / "agora/analysis/sprint-15-ablation-prep/ablation-scores.json")
    args = ap.parse_args()

    seeds = set(args.seeds)
    control = load_runs(args.control_dir, seeds)
    print(f"control: {len(control)} paired runs from {args.control_dir.name}\n")

    all_rows: list[dict] = []
    for cell in args.cells:
        runs = load_runs(args.ablation_root / cell, seeds)
        print(f"=== {cell}  ({len(runs)} runs) ===")
        rows = paired_report(control, runs, cell)
        if rows:
            print(f"  {'outcome':<24}{'control':>10}{'variant':>10}{'delta':>10}{'t':>8}{'p':>9}")
            for r in rows:
                p = "n/a" if r["p"] is None else ("<.001" if r["p"] < 0.001 else f"{r['p']:.3f}")
                t = "n/a" if r["t"] is None else f"{r['t']:.2f}"
                print(f"  {r['outcome']:<24}{r['control']:>10.3f}{r['variant']:>10.3f}"
                      f"{r['delta']:>+10.3f}{t:>8}{p:>9}")
        all_rows.extend(rows)
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_rows, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
