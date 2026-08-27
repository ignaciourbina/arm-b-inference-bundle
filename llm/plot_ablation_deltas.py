#!/usr/bin/env python3
"""Forest plots of the Sprint-15 ablation paired deltas, one panel per outcome.

Reads `ablation-scores.json` (written by `score_ablation_cells.py`) and draws,
for each outcome, a forest plot: one row per cell, the mean paired difference
against control, with a 95% confidence interval and a significance mark. Cells
are colored by axis (prompt / reasoning / quant) so the three ablation axes read
at a glance.

Why per-outcome panels rather than one chart: the outcomes live on different
scales (distinct-args ~1-5, saturation 0-1, DRI -1..1, LLM-calls in the
hundreds), so a shared x-axis would be dishonest. Each panel carries its own
x-scale in the outcome's own units, centered on Δ=0.

The 95% CI is reconstructed from the paired t-test the scorer already ran:
SE = |Δ / t|, CI = Δ ± t_(.975, n-1)·SE — the exact interval behind the p-value,
not a re-derived one. Rows where the scorer found no within-pair variance
(t = null) are drawn as a point with no whisker and flagged.

Usage:
    python llm/plot_ablation_deltas.py [--scores JSON] [--out PDF]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCORES = BASE_DIR / "agora/analysis/sprint-15-ablation-prep/ablation-scores.json"
DEFAULT_OUT = BASE_DIR / "pipeline/output/reports/ablation/ablation_deltas.pdf"

# Cell-name prefix -> ablation axis, and a color per axis.
AXIS_COLORS = {"prompt": "#c0392b", "reasoning": "#1a7a1a", "quant": "#8e44ad"}


def axis_of(cell: str) -> str:
    if cell.startswith("prompt_"):
        return "prompt"
    if cell.startswith("reasoning_"):
        return "reasoning"
    if cell.startswith("quant_"):
        return "quant"
    return "other"


def ci95(delta: float, t: float | None, n: int) -> float | None:
    """Half-width of the 95% CI behind the paired t-test, or None if undefined."""
    if t is None or abs(t) < 1e-12 or n < 2:
        return None
    se = abs(delta / t)
    return float(stats.t.ppf(0.975, df=n - 1) * se)


def try_fonts() -> None:
    """Best-effort Minion Pro; fall back silently to matplotlib defaults."""
    try:
        sys.path.insert(0, str(BASE_DIR / "pipeline"))
        from plot_style import configure_matplotlib_defaults
        configure_matplotlib_defaults(BASE_DIR)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.scores.exists():
        print(f"[abort] {args.scores} not found — run score_ablation_cells.py first")
        return 2
    rows = json.loads(args.scores.read_text())
    if not rows:
        print("[abort] no scored rows")
        return 2

    try_fonts()

    # Group rows by outcome; preserve first-seen outcome order.
    by_outcome: dict[str, list[dict]] = defaultdict(list)
    outcome_order: list[str] = []
    for r in rows:
        if r["outcome"] not in by_outcome:
            outcome_order.append(r["outcome"])
        by_outcome[r["outcome"]].append(r)

    cells = sorted({r["cell"] for r in rows})
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(args.out) as pdf:
        n = len(outcome_order)
        ncols = 2
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows + 1))
        axes = axes.flatten() if n > 1 else [axes]

        for ax, outcome in zip(axes, outcome_order):
            entries = {e["cell"]: e for e in by_outcome[outcome]}
            ys = list(range(len(cells)))
            any_sig = False
            for y, cell in zip(ys, cells):
                e = entries.get(cell)
                color = AXIS_COLORS.get(axis_of(cell), "#555555")
                if e is None:
                    ax.plot(0, y, "x", color="#bbbbbb", markersize=6)
                    continue
                half = ci95(e["delta"], e.get("t"), e.get("n_pairs", 0))
                sig = e.get("p") is not None and e["p"] < 0.05
                any_sig = any_sig or sig
                if half is not None:
                    ax.errorbar(e["delta"], y, xerr=half, fmt="o", color=color,
                                markersize=6 if sig else 5, capsize=3,
                                markerfacecolor=color if sig else "white",
                                markeredgecolor=color, elinewidth=1.2, zorder=3)
                else:
                    ax.plot(e["delta"], y, "o", color=color, markersize=5,
                            markerfacecolor="white", markeredgecolor=color, zorder=3)
                if sig:
                    ax.annotate("*", (e["delta"], y), textcoords="offset points",
                                xytext=(6, 3), color=color, fontsize=12, fontweight="bold")
            ax.axvline(0, color="#333333", lw=0.8, ls="--", zorder=1)
            ax.set_yticks(ys)
            ax.set_yticklabels([c.replace("_", " ") for c in cells], fontsize=8)
            ax.set_ylim(-0.6, len(cells) - 0.4)
            ax.invert_yaxis()
            ax.set_title(outcome, fontsize=10)
            ax.set_xlabel("Δ vs control (variant − control)", fontsize=8)
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(axis="x", color="#eeeeee", zorder=0)

        for ax in axes[n:]:
            ax.axis("off")

        # Legend for axes + significance.
        handles = [plt.Line2D([], [], marker="o", color=c, linestyle="",
                              markerfacecolor=c, label=a)
                   for a, c in AXIS_COLORS.items()]
        handles.append(plt.Line2D([], [], marker="o", color="#555", linestyle="",
                                  markerfacecolor="white", label="n.s. (hollow)"))
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle("Sprint-15 ablation: paired Δ vs control, 95% CI "
                     "(filled = p<.05)", fontsize=12)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        pdf.savefig(fig)
        plt.close(fig)

    print(f"wrote {args.out}")
    print(f"  outcomes: {len(outcome_order)}  cells: {len(cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
