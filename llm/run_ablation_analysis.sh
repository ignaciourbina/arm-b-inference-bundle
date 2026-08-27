#!/usr/bin/env bash
# Sprint-15 ablation analysis suite — run after the programme's P5 scoring.
#
# Consumes the scored cells + traces and emits the analysis layer:
#   1. ablation_deltas.pdf        paired Δ vs control, 95% CI, per outcome
#   2. ablation_reliability.{md,pdf}  run completion + failure phase per cell
#   3. ablation_trajectories.pdf  variant-vs-control round-by-round dynamics
#
# Prereq: ablation-scores.json exists (run llm/score_ablation_cells.py +
# llm/report_ablation.py first, or the whole programme's p5 phase).
#
#   bash llm/run_ablation_analysis.sh
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
PY=agora/.venv/bin/python

echo "### 1/3 paired-delta forest plots"
$PY llm/plot_ablation_deltas.py || echo "  [warn] deltas failed"
echo "### 2/3 reliability / failure analysis"
$PY llm/analyze_ablation_reliability.py || echo "  [warn] reliability failed"
echo "### 3/3 variant-vs-control trajectories"
$PY llm/plot_ablation_trajectories.py || echo "  [warn] trajectories failed"

echo
echo "outputs in pipeline/output/reports/ablation/ and the sprint folder:"
ls -1 pipeline/output/reports/ablation/*.pdf \
      agora/analysis/sprint-15-ablation-prep/ablation-reliability.md 2>/dev/null | sed 's/^/  /'
