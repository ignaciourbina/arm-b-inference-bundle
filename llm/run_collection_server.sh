#!/usr/bin/env bash
# High-occupancy collection server (Sprint-16 scale-out config).
#
# Evidence base: frontier report 2026-08-27 — aggregate decode scales to
# ~434 tok/s at 24-way with no plateau; real runs average only ~2.7 busy
# slots, so the win comes from running MANY runs concurrently against one
# server. Config: 24 slots x 4096 ctx/slot, q8 KV (f16 measured 0.52x —
# rejected), coopmat2 source binary, refreshed (Jul-15) E2B Q8_0 weights.
#
#   bash llm/run_collection_server.sh [128|256]     # reasoning budget (default 128)
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
BUDGET="${1:-128}"
PARALLEL="${PARALLEL:-24}"
CTX="${CTX:-98304}"   # 4096/slot at defaults; override e.g. PARALLEL=12 CTX=49152
BIN="$HOME/src/llama.cpp/build/bin/llama-server"
MODEL="$HOME/.cache/llama-models/gemma-4-E2B-it-GGUF-r0715/gemma-4-E2B-it-Q8_0.gguf"
LOGS=llm/traces/logs; mkdir -p "$LOGS"

[ -x "$BIN" ] || { echo "[abort] coopmat2 binary missing: $BIN"; exit 1; }
[ -f "$MODEL" ] || { echo "[abort] refreshed weights missing: $MODEL"; exit 1; }
case "$BUDGET" in 128|256) ;; *) echo "[abort] budget must be 128 or 256"; exit 1;; esac

pkill -x llama-server 2>/dev/null; sleep 3
nohup "$BIN" --model "$MODEL" --host 127.0.0.1 --port 20434 \
  --ctx-size "$CTX" --parallel "$PARALLEL" --batch-size 2048 --ubatch-size 512 \
  --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --threads 8 --cont-batching \
  --reasoning on --reasoning-budget "$BUDGET" \
  > "$LOGS/collection_server.log" 2>&1 &

for _ in $(seq 1 60); do
  sleep 3
  curl -sf localhost:20434/v1/models >/dev/null && break
done
curl -sf localhost:20434/v1/models >/dev/null || { echo "[abort] server failed to come up"; tail -5 "$LOGS/collection_server.log"; exit 1; }

VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || echo "n/a")
echo "[up] collection server: p$PARALLEL, ctx $CTX ($((CTX/PARALLEL))/slot), q8 KV, reasoning b$BUDGET"
echo "[up] VRAM after load: $VRAM (gate: < 6.5 GB per sprint-16 plan)"
