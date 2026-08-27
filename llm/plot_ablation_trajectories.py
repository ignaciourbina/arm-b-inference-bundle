#!/usr/bin/env python3
"""Round-by-round trajectories for each ablation cell against the control.

The paired-delta table compares only the final round. This shows the *path*:
for each cell, four per-round outcomes overlaid on the control, so you can see
whether a variant shifts the whole dynamic or only the endpoint.

Outcomes (all cheap from the per-round snapshots / voice logs):
  * saturation            fraction of weights at |w| >= 0.99.
  * opinion variance      var over agents of o_i = clip(sum(w·d)/M, -1, 1).
  * meta-consensus        1 - mean_c std_agents(|w_c|) / M.
  * distinct args/agent   cumulative distinct considerations voiced per agent up
                          to that round (the sharpest Sprint-14 arm difference).

Bands are 95% CIs on the mean across the pooled (composition × seed) runs —
Student's t on n-1 df. Pooling across compositions is deliberate: the question
here is whether a variant moves the aggregate dynamic, matching how the paired
deltas pool. One page per cell.

Usage:
    python llm/plot_ablation_trajectories.py [--cells NAME ...] [--out PDF]
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
CONTROL_DIR = BASE_DIR / "llm/traces/beta_local/arm_b_local_p6cache"
ABLATION_ROOT = BASE_DIR / "llm/traces/ablation"
DEFAULT_OUT = BASE_DIR / "pipeline/output/reports/ablation/ablation_trajectories.pdf"

FINAL_RE = re.compile(r"_(?P<comp>polarized|symmetric|three_clusters)_n6_s(?P<seed>\d+)_\d+\.json$")
CONTROL_COLOR = "#888888"
VARIANT_COLOR = "#c0392b"


def load_runs(directory: Path, seeds: set[int]) -> dict[tuple[str, int], dict]:
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
            out[(m.group("comp"), seed)] = __import__("json").loads(path.read_bytes())
        except Exception:
            continue
    return out


def per_round_series(record: dict) -> dict[str, list[float]]:
    """Return {outcome: [value per round]} aligned to snapshot index."""
    cons = record.get("scenario", {}).get("considerations", [])
    dirs = {c["id"]: float(c["direction"]) for c in cons}
    cids = sorted(dirs)
    M = len(cids) or 1
    snaps = record.get("snapshots", [])

    sat, ovar, mc = [], [], []
    for snap in snaps:
        ws = [w for a in snap for w in a["weights"].values()]
        sat.append(sum(1 for w in ws if abs(w) >= 0.99) / len(ws) if ws else np.nan)
        W = [dict(a["weights"]) for a in snap]
        ops = [float(np.clip(sum(v * dirs[c] for c, v in w.items() if c in dirs) / M, -1, 1))
               for w in W]
        ovar.append(float(np.var(ops)) if ops else np.nan)
        mc.append(1.0 - sum(float(np.std([abs(w.get(c, 0.0)) for w in W])) for c in cids) / M
                  if W else np.nan)

    # cumulative distinct considerations voiced per agent, indexed to snapshot 1..n
    seen: dict[str, set] = {}
    dpa = [np.nan]  # snapshot 0 = pre-deliberation, no voices yet
    for rnd in record.get("rounds", []):
        for v in (rnd.get("voices") or []):
            seen.setdefault(v.get("agent_id", "?"), set()).add(v["cid"])
        dpa.append(float(np.mean([len(s) for s in seen.values()])) if seen else np.nan)
    # align length to snapshots
    while len(dpa) < len(snaps):
        dpa.append(dpa[-1] if dpa else np.nan)
    return {"saturation": sat, "opinion variance": ovar,
            "meta-consensus": mc, "distinct args / agent": dpa[:len(snaps)]}


def mean_ci(matrix: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Per-round mean and 95% t-CI half-width across runs (ragged-safe)."""
    maxlen = max((len(r) for r in matrix), default=0)
    arr = np.full((len(matrix), maxlen), np.nan)
    for i, r in enumerate(matrix):
        arr[i, :len(r)] = r
    if not maxlen:
        return np.array([]), np.array([])
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rounds -> NaN, plotted as gaps
        means = np.nanmean(arr, axis=0)
        n = np.sum(np.isfinite(arr), axis=0)
        sd = np.nanstd(arr, axis=0, ddof=1)
        half = np.where(n > 1, stats.t.ppf(0.975, np.maximum(n - 1, 1)) * sd / np.sqrt(n), np.nan)
    return means, half


def try_fonts() -> None:
    try:
        sys.path.insert(0, str(BASE_DIR / "pipeline"))
        from plot_style import configure_matplotlib_defaults
        configure_matplotlib_defaults(BASE_DIR)
    except Exception:
        pass


def series_for(runs: dict, outcome: str) -> list[list[float]]:
    return [per_round_series(rec)[outcome] for rec in runs.values()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="*", default=None,
                    help="Cell names; default = every dir under the ablation root.")
    ap.add_argument("--control-dir", type=Path, default=CONTROL_DIR)
    ap.add_argument("--ablation-root", type=Path, default=ABLATION_ROOT)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    seeds = set(args.seeds)
    control = load_runs(args.control_dir, seeds)
    if not control:
        print(f"[abort] no control runs in {args.control_dir}")
        return 2

    cell_names = args.cells or [d.name for d in sorted(args.ablation_root.iterdir())
                                if (d / "CELL.json").exists()]
    outcomes = ["saturation", "opinion variance", "meta-consensus", "distinct args / agent"]
    try_fonts()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ctrl_series = {o: mean_ci(series_for(control, o)) for o in outcomes}

    with PdfPages(args.out) as pdf:
        for cell in cell_names:
            runs = load_runs(args.ablation_root / cell, seeds)
            if not runs:
                print(f"  [skip] {cell}: no runs")
                continue
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            for ax, outcome in zip(axes.flatten(), outcomes):
                x = np.arange(len(ctrl_series[outcome][0]))
                cm, ch = ctrl_series[outcome]
                ax.plot(x, cm, color=CONTROL_COLOR, lw=1.5, label="control")
                ax.fill_between(x, cm - ch, cm + ch, color=CONTROL_COLOR, alpha=0.18)
                vm, vh = mean_ci(series_for(runs, outcome))
                xv = np.arange(len(vm))
                ax.plot(xv, vm, color=VARIANT_COLOR, lw=1.5, label=cell)
                ax.fill_between(xv, vm - vh, vm + vh, color=VARIANT_COLOR, alpha=0.18)
                ax.set_title(outcome, fontsize=10)
                ax.set_xlabel("round", fontsize=8)
                ax.tick_params(labelsize=8)
                ax.grid(color="#f0f0f0")
            axes[0, 0].legend(fontsize=8, frameon=False, loc="best")
            fig.suptitle(f"{cell}  vs control  (n_variant={len(runs)}, "
                         f"n_control={len(control)}; bands = 95% CI)", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  page: {cell}  ({len(runs)} runs)")

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
