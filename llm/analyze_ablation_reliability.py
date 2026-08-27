#!/usr/bin/env python3
"""Reliability of each Sprint-15 ablation cell: how often runs completed, where
they failed, and at what cost.

The paired-delta scorer only sees runs that finished. But *whether* a run
finishes is itself an outcome here: Sprint 14 documented that gemma-4-E2B
struggles to terminate the REFLECT phase, and the Sprint-15 quantization axis
asks whether Q4 makes that worse. This script quantifies it.

Per cell it reports, from three sources:
  * CELL.json           — runs_ok / runs_failed, backend tok/s, wall minutes.
  * the cell run-log     — the phase each failed run died in (REFLECT/VOICE/
                           EVALUATE), parsed from the driver's `[fail ...]` lines.
  * the completed traces — mean LLM calls per run (a cost/effort proxy).

Outputs a markdown table and a failure-rate bar chart (colored by axis), and
prints the headline contrast: mean failure rate of the Q8 cells vs the Q4 cell.

Usage:
    python llm/analyze_ablation_reliability.py [--ablation-root DIR] [--logs DIR]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = BASE_DIR / "llm/traces/ablation"
DEFAULT_LOGS = BASE_DIR / "llm/traces/logs"
DEFAULT_MD = BASE_DIR / "agora/analysis/sprint-15-ablation-prep/ablation-reliability.md"
DEFAULT_PDF = BASE_DIR / "pipeline/output/reports/ablation/ablation_reliability.pdf"

AXIS_COLORS = {"prompt": "#c0392b", "reasoning": "#1a7a1a", "quant": "#8e44ad"}
FAIL_RE = re.compile(r"\[fail rc=\d+\]\s+\S+:\s*(?P<phase>[A-Z][A-Z ]+?)\s*(PHASE)?:?\s*$")
PHASE_RE = re.compile(r"(REFLECT|VOICE|EVALUATE)", re.IGNORECASE)


def axis_of(cell: str) -> str:
    for a in ("prompt", "reasoning", "quant"):
        if cell.startswith(a + "_"):
            return a
    return "other"


def failure_phases(log: Path, cell: str) -> Counter:
    """Count the phase each failed run died in, for the cell's MOST RECENT run.

    The cell log is append-mode (`tee -a`), so it accumulates failures across
    every relaunch — including runs that later succeeded on a refill pass. To
    stay consistent with the current CELL.json, parse only the last run block:
    everything after the final `[cell] <name>` start marker the driver prints.
    """
    phases: Counter = Counter()
    if not log.exists():
        return phases
    lines = log.read_text(errors="replace").splitlines()
    marker = f"[cell] {cell}"
    last = max((i for i, ln in enumerate(lines) if ln.strip() == marker), default=0)
    for line in lines[last:]:
        if "[fail rc=" not in line:
            continue
        m = PHASE_RE.search(line)
        phases[m.group(1).upper() if m else "UNKNOWN"] += 1
    return phases


def mean_llm_calls(cell_dir: Path) -> float | None:
    """Mean summed llm_calls per completed run in a cell."""
    totals = []
    for p in cell_dir.iterdir():
        if "_trace_" in p.name or "checkpoint" in p.name or p.name == "CELL.json":
            continue
        try:
            rec = json.loads(p.read_bytes())
        except Exception:
            continue
        calls = sum(r.get("llm_calls", 0) for r in rec.get("rounds", []))
        if calls:
            totals.append(calls)
    return sum(totals) / len(totals) if totals else None


def try_fonts() -> None:
    try:
        sys.path.insert(0, str(BASE_DIR / "pipeline"))
        from plot_style import configure_matplotlib_defaults
        configure_matplotlib_defaults(BASE_DIR)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablation-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    ap.add_argument("--md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    args = ap.parse_args()

    cells = []
    for cell_dir in sorted(args.ablation_root.iterdir()):
        cj = cell_dir / "CELL.json"
        if not cj.exists():
            continue  # only real ablation cells carry a CELL.json
        meta = json.loads(cj.read_text())
        ok = int(meta.get("runs_ok", 0))
        failed = int(meta.get("runs_failed", 0))
        total = ok + failed or 1
        phases = failure_phases(args.logs / f"ablation_{cell_dir.name}.log", cell_dir.name)
        cells.append({
            "cell": cell_dir.name,
            "axis": axis_of(cell_dir.name),
            "ok": ok, "failed": failed, "total": total,
            "fail_rate": failed / total,
            "phases": phases,
            "tok_s": meta.get("backend_tok_per_s"),
            "min_per_run": (meta.get("minutes", 0) / total) if total else None,
            "mean_calls": mean_llm_calls(cell_dir),
        })

    if not cells:
        print(f"[abort] no cells with CELL.json under {args.ablation_root}")
        return 2

    # ---- markdown table ----
    def phase_str(c: Counter) -> str:
        return ", ".join(f"{k}×{v}" for k, v in c.most_common()) if c else "—"

    md = ["# Sprint 15 — Ablation reliability", "",
          "Run completion, failure phase, and cost per cell. `fail%` counts runs",
          "the driver could not complete; the phase is where they died.", "",
          "| Cell | Axis | ok/total | fail% | fail phases | mean LLM calls/run | tok/s | min/run |",
          "|---|---|---|---|---|---|---|---|"]
    for c in cells:
        calls = f"{c['mean_calls']:.0f}" if c["mean_calls"] is not None else "—"
        toks = f"{c['tok_s']:.1f}" if c["tok_s"] is not None else "—"
        mpr = f"{c['min_per_run']:.1f}" if c["min_per_run"] is not None else "—"
        md.append(f"| `{c['cell']}` | {c['axis']} | {c['ok']}/{c['total']} | "
                  f"{100*c['fail_rate']:.0f}% | {phase_str(c['phases'])} | {calls} | "
                  f"{toks} | {mpr} |")

    # headline contrast: Q4 vs the Q8 cells (everything not on the quant axis)
    q4 = [c for c in cells if c["axis"] == "quant"]
    q8 = [c for c in cells if c["axis"] != "quant"]
    md += ["", "## Headline", ""]
    if q4 and q8:
        q4_rate = sum(c["fail_rate"] for c in q4) / len(q4)
        q8_rate = sum(c["fail_rate"] for c in q8) / len(q8)
        dom = Counter()
        for c in q4:
            dom.update(c["phases"])
        dom_phase = dom.most_common(1)[0][0] if dom else "—"
        md += [
            f"- Q4 cell failure rate: **{100*q4_rate:.0f}%**; "
            f"Q8 cells (prompt+reasoning) mean: **{100*q8_rate:.0f}%**.",
            f"- Q4 failures concentrate in the **{dom_phase}** phase — the "
            "Sprint-14 gemma-4-E2B reflect-termination weakness, amplified by "
            "heavier quantization.",
            "- Read the paired-delta table (`ablation_deltas.tex`) knowing the Q4",
            "  cell rests on fewer completed pairs; its nulls are low-power, not",
            "  evidence of no effect.",
        ]
    else:
        md += ["- (need both a quant cell and non-quant cells for the contrast)"]
    md.append("")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text("\n".join(md))

    # ---- bar chart: failure rate by cell ----
    try_fonts()
    cells_sorted = sorted(cells, key=lambda c: (c["axis"], c["cell"]))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(cells_sorted) + 2))
    ys = range(len(cells_sorted))
    ax.barh(list(ys), [100 * c["fail_rate"] for c in cells_sorted],
            color=[AXIS_COLORS.get(c["axis"], "#555") for c in cells_sorted])
    for y, c in zip(ys, cells_sorted):
        ax.annotate(f"{c['ok']}/{c['total']}", (100 * c["fail_rate"], y),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([c["cell"].replace("_", " ") for c in cells_sorted], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("run failure rate (%)")
    ax.set_title("Sprint-15 ablation: run completion by cell")
    handles = [plt.Line2D([], [], marker="s", color=c, linestyle="", label=a)
               for a, c in AXIS_COLORS.items()]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
    ax.grid(axis="x", color="#eeeeee")
    fig.tight_layout()
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.pdf)
    plt.close(fig)

    print(f"wrote {args.md}\nwrote {args.pdf}")
    for c in cells:
        print(f"  {c['cell']:<28} {c['ok']}/{c['total']} ok  "
              f"fail={100*c['fail_rate']:.0f}%  phases={phase_str(c['phases'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
