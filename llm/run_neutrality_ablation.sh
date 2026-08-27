#!/usr/bin/env bash
# Prompt-neutrality ablation — register-only levers, information held fixed.
# Design: agora/analysis/sprint-15-ablation-prep/neutrality-ablation-design.md
#
# Server config is the PRODUCTION collection signature (reasoning off, Q8_0,
# p6-cache) so every cell pairs against the existing 390-run control. Do not
# run these under reasoning-on: neutrality-vs-control must be established on
# the configuration the control was collected under.
#
#   bash llm/run_neutrality_ablation.sh          # 5 cells + scoring
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
PY=agora/.venv/bin/python
LOGS=llm/traces/logs; mkdir -p "$LOGS"

Q8=~/.cache/llama-models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf

# Production p6-cache signature, reasoning off (matches the control collection).
pkill -x llama-server 2>/dev/null; sleep 3
nohup llama-server --model "$Q8" --host 127.0.0.1 --port 20434 \
  --ctx-size 24576 --parallel 6 --batch-size 2048 --ubatch-size 512 \
  --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --threads 8 --cache-reuse 256 --cont-batching --reasoning off \
  > "$LOGS/neutrality_server.log" 2>&1 &
for _ in $(seq 1 40); do sleep 3; curl -sf localhost:20434/v1/models >/dev/null && break; done
curl -sf localhost:20434/v1/models >/dev/null || { echo "[abort] server failed to start"; exit 1; }

cell() {  # cell <name> <variant>
  echo; echo "######## CELL $1 ########"
  $PY llm/run_ablation_cell.py --cell "$1" --prompt-variant "$2" \
    --infra "reasoning=off,quant=Q8_0,wave=neutrality" \
    2>&1 | tee -a "$LOGS/ablation_$1.log"
}

# Single-lever cells first, the combined cell last (most informative if time is cut).
cell prompt_neutral-persona neutral-persona
cell prompt_neutral-stance  neutral-stance
cell prompt_no-overlay      no-overlay
cell prompt_no-gradualism   no-gradualism
cell prompt_neutral-full    neutral-full

echo; echo "######## SCORING ########"
CELLS=$(ls llm/traces/ablation 2>/dev/null | grep -vE '_p0check|smoke|pipeline_validation' | tr '\n' ' ')
$PY llm/score_ablation_cells.py --cells $CELLS 2>&1 | tee "$LOGS/neutrality_scores.log"
$PY llm/report_ablation.py 2>&1 | tee -a "$LOGS/neutrality_scores.log"
bash llm/run_ablation_analysis.sh 2>&1 | tail -8
