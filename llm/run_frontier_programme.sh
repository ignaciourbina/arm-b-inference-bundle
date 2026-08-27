#!/usr/bin/env bash
# Efficiency/performance frontier programme (overnight, autonomous).
#
# Motivation: reasoning-on-256 (the adopted quality config) costs 18.4 min/run,
# pricing the 1200-run extension at ~370 GPU-hours. This programme measures the
# levers that could cut that without giving up the measured quality gains:
#
#   A. bench matrix (fast): KV f16 vs q8, ngram self-speculation, E2B QAT-Q4_0
#      weights, parallel 6 vs 12 — single-stream + concurrent aggregate tok/s
#      on a deliberation-shaped prompt.
#   B. quality/efficiency cells (paired, forward config reasoning-on):
#      B1 reasoning-budget 128 (vs the b256 control cell) — the budget lever.
#      B2 E2B QAT-Q4_0 at b256 (vs the same control) — the weight-quant lever.
#   C. scoring + reliability sweep.
#
# Control for both cells: newstack_e2b_r0715 (refreshed E2B Q8, b256, 15/15).
#
#   nohup bash llm/run_frontier_programme.sh > llm/traces/logs/frontier.out 2>&1 &
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
PY=agora/.venv/bin/python
LOGS=llm/traces/logs; mkdir -p "$LOGS"
SPRINT=agora/analysis/sprint-15-ablation-prep
RESULTS="$SPRINT/frontier-bench.jsonl"; : > "$RESULTS"

BIN=~/src/llama.cpp/build/bin/llama-server
E2B_R=~/.cache/llama-models/gemma-4-E2B-it-GGUF-r0715/gemma-4-E2B-it-Q8_0.gguf
E2B_Q4=~/.cache/llama-models/gemma-4-E2B-it-GGUF-qat/gemma-4-E2B-it-Q4_0.gguf

start() {  # start <model> <parallel> <ctx> <extra...>
  local model=$1 par=$2 ctx=$3; shift 3
  pkill -x llama-server 2>/dev/null; sleep 3
  nohup "$BIN" --model "$model" --host 127.0.0.1 --port 20434 \
    --ctx-size "$ctx" --parallel "$par" --batch-size 2048 --ubatch-size 512 \
    --n-gpu-layers 99 --flash-attn on --jinja --threads 8 --cont-batching \
    --reasoning on --reasoning-budget 256 "$@" \
    > "$LOGS/frontier_server.log" 2>&1 &
  for _ in $(seq 1 40); do sleep 3; curl -sf localhost:20434/v1/models >/dev/null && return 0; done
  echo "[skip] server failed: $(basename "$model") par=$par $*"; return 1
}

bench() {  # bench <label> <concurrent>
  $PY llm/bench_frontier_probe.py --label "$1" --concurrent "$2" | tee -a "$RESULTS"
}

echo "######## PHASE A: bench matrix ########"
# A1 baseline: refreshed E2B Q8, q8 KV, p6
start "$E2B_R" 6 24576 --cache-type-k q8_0 --cache-type-v q8_0 --cache-reuse 256 \
  && bench base_q8kv_p6 6
# A2 KV f16 (VRAM headroom exists; Gemma is KV-quant-sensitive per verified claims)
start "$E2B_R" 6 24576 --cache-reuse 256 \
  && bench kv_f16_p6 6
# A3 ngram self-speculation (lossless; only throughput can change)
start "$E2B_R" 6 24576 --cache-type-k q8_0 --cache-type-v q8_0 --cache-reuse 256 \
  --spec-type ngram-simple \
  && bench spec_ngram_p6 6
# A4 E2B QAT-Q4_0 weights (if downloaded)
if [ -f "$E2B_Q4" ]; then
  start "$E2B_Q4" 6 24576 --cache-type-k q8_0 --cache-type-v q8_0 --cache-reuse 256 \
    && bench qat_q4_p6 6
else
  echo "[skip] E2B QAT-Q4_0 not on disk yet"
fi
# A5 parallel 12 (macro-throughput probe; ctx doubled to keep 4096/slot)
start "$E2B_R" 12 49152 --cache-type-k q8_0 --cache-type-v q8_0 --cache-reuse 256 \
  && bench base_q8kv_p12 12

echo; echo "######## PHASE B1: reasoning-budget-128 cell ########"
pkill -x llama-server 2>/dev/null; sleep 3
nohup "$BIN" --model "$E2B_R" --host 127.0.0.1 --port 20434 \
  --ctx-size 24576 --parallel 6 --batch-size 2048 --ubatch-size 512 \
  --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja --threads 8 --cache-reuse 256 --cont-batching \
  --reasoning on --reasoning-budget 128 \
  > "$LOGS/frontier_server.log" 2>&1 &
for _ in $(seq 1 40); do sleep 3; curl -sf localhost:20434/v1/models >/dev/null && break; done
$PY llm/run_ablation_cell.py --cell frontier_e2b_b128 --prompt-variant control \
  --infra "reasoning=on:128,quant=Q8_0,weights=r0715,binary=coopmat2" \
  2>&1 | tee -a "$LOGS/ablation_frontier_e2b_b128.log"

echo; echo "######## PHASE B2: E2B QAT-Q4_0 cell (b256) ########"
if [ -f "$E2B_Q4" ]; then
  pkill -x llama-server 2>/dev/null; sleep 3
  nohup "$BIN" --model "$E2B_Q4" --host 127.0.0.1 --port 20434 \
    --ctx-size 24576 --parallel 6 --batch-size 2048 --ubatch-size 512 \
    --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --jinja --threads 8 --cache-reuse 256 --cont-batching \
    --reasoning on --reasoning-budget 256 \
    > "$LOGS/frontier_server.log" 2>&1 &
  for _ in $(seq 1 40); do sleep 3; curl -sf localhost:20434/v1/models >/dev/null && break; done
  $PY llm/run_ablation_cell.py --cell frontier_e2bqat_b256 --prompt-variant control \
    --infra "reasoning=on:256,quant=QAT-Q4_0,weights=unsloth-qat,binary=coopmat2" \
    2>&1 | tee -a "$LOGS/ablation_frontier_e2bqat_b256.log"
else
  echo "[skip] B2: QAT weights absent"
fi

echo; echo "######## PHASE C: scoring vs newstack_e2b_r0715 ########"
for cell in frontier_e2b_b128 frontier_e2bqat_b256; do
  [ -d "llm/traces/ablation/$cell" ] || continue
  $PY llm/score_ablation_cells.py --cells "$cell" \
    --control-dir llm/traces/ablation/newstack_e2b_r0715 \
    --out "$SPRINT/${cell}-scores.json" 2>&1 | tee -a "$LOGS/frontier_scores.log"
done
$PY llm/analyze_ablation_reliability.py 2>&1 | tail -8

pkill -x llama-server 2>/dev/null
echo "[done] bench: $RESULTS ; scores: $SPRINT/frontier_*-scores.json"
