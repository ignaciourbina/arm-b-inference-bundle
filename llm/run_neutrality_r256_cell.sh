#!/usr/bin/env bash
# Neutrality × reasoning interaction cell: neutral-full under reasoning-on-256.
#
# Completes the 2x2 the neutrality design needs for the FORWARD config:
#   control prompt  × reasoning-off   = production 390-run collection (control)
#   neutral-full    × reasoning-off   = prompt_neutral-full (main wave)
#   control prompt  × reasoning-on256 = reasoning_on_b256 (wave 1, 15/15)
#   neutral-full    × reasoning-on256 = THIS CELL
#
# Scored against reasoning_on_b256 as the paired control (same seeds/pools),
# so the register effect is measured on the configuration the 1200-run
# extension will actually use.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
PY=agora/.venv/bin/python
LOGS=llm/traces/logs; mkdir -p "$LOGS"
Q8=~/.cache/llama-models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf

pkill -x llama-server 2>/dev/null; sleep 3
nohup llama-server --model "$Q8" --host 127.0.0.1 --port 20434 \
  --ctx-size 24576 --parallel 6 --batch-size 2048 --ubatch-size 512 \
  --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --threads 8 --cache-reuse 256 --cont-batching \
  --reasoning on --reasoning-budget 256 \
  > "$LOGS/neutrality_r256_server.log" 2>&1 &
for _ in $(seq 1 40); do sleep 3; curl -sf localhost:20434/v1/models >/dev/null && break; done
curl -sf localhost:20434/v1/models >/dev/null || { echo "[abort] server failed"; exit 1; }

$PY llm/run_ablation_cell.py --cell prompt_neutral-full-r256 \
  --prompt-variant neutral-full \
  --infra "reasoning=on:256,quant=Q8_0,wave=neutrality-x-reasoning" \
  2>&1 | tee -a "$LOGS/ablation_prompt_neutral-full-r256.log"

echo; echo "######## SCORING vs reasoning_on_b256 (the reasoning-on control) ########"
$PY llm/score_ablation_cells.py --cells prompt_neutral-full-r256 \
  --control-dir llm/traces/ablation/reasoning_on_b256 \
  --out agora/analysis/sprint-15-ablation-prep/neutrality-r256-scores.json \
  2>&1 | tee "$LOGS/neutrality_r256_scores.log"

# Restore the production serving configuration.
pkill -x llama-server 2>/dev/null; sleep 3
nohup llama-server --model "$Q8" --host 127.0.0.1 --port 20434 \
  --ctx-size 24576 --parallel 6 --batch-size 2048 --ubatch-size 512 \
  --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --threads 8 --cache-reuse 256 --cont-batching --reasoning off \
  > "$LOGS/server_restore.log" 2>&1 &
echo "[done] production server restored"
