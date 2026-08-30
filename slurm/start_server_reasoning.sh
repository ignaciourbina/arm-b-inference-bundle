#!/bin/bash
# llama-server with reasoning ON — mirrors llm/run_collection_server.sh's
# signature but adds PORT + CUDA_VISIBLE_DEVICES so one server can be pinned
# per GPU. Used for the SoTA (reasoning-on) cells.
#   CUDA_VISIBLE_DEVICES=0 PORT=20434 BUDGET=256 bash slurm/start_server_reasoning.sh
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
PORT="${PORT:-20434}"; BUDGET="${BUDGET:-256}"
PARALLEL="${N_PARALLEL:-24}"; CTX="${CTX:-98304}"
MODEL="$HOME/.cache/llama-models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf"
exec llama-server --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --ctx-size "$CTX" --parallel "$PARALLEL" --batch-size 2048 --ubatch-size 512 \
  --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --threads 12 --cont-batching \
  --reasoning on --reasoning-budget "$BUDGET"
