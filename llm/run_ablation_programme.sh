#!/usr/bin/env bash
# Sprint-15 ablation programme — the whole 24h budget in one command.
#
# Preconditions (P0): the GPU must be restored. Every cell self-checks the
# backend and aborts if it is on the CPU-fallback path, so running this before
# the driver is fixed costs seconds, not hours.
#
#   bash llm/run_ablation_programme.sh            # run everything
#   bash llm/run_ablation_programme.sh p2         # one phase only
#
# Phases: p2 = prompt variants, p3 = reasoning, p4 = quantization, p5 = scoring.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
PY=agora/.venv/bin/python
LOGS=llm/traces/logs; mkdir -p "$LOGS"
PHASE="${1:-all}"

Q8=~/.cache/llama-models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf
Q4=~/.cache/llama-models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf

start_server() {  # start_server <extra-flags...>
  pkill -x llama-server 2>/dev/null; sleep 3
  MODEL="${MODEL_OVERRIDE:-$Q8}"
  nohup llama-server --model "$MODEL" --host 127.0.0.1 --port 20434 \
    --ctx-size 24576 --parallel 6 --batch-size 2048 --ubatch-size 512 \
    --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --jinja --threads 8 --cache-reuse 256 --cont-batching "$@" \
    > "$LOGS/ablation_server.log" 2>&1 &
  for _ in $(seq 1 40); do sleep 3; curl -sf localhost:20434/v1/models >/dev/null && return 0; done
  echo "[abort] server failed to start"; return 1
}

cell() {  # cell <name> <prompt-variant> <infra-description>
  echo; echo "######## CELL $1 ########"
  $PY llm/run_ablation_cell.py --cell "$1" --prompt-variant "$2" --infra "$3" \
    2>&1 | tee -a "$LOGS/ablation_$1.log"
}

# ---------------------------------------------------------------- P2: prompts
if [[ "$PHASE" == "all" || "$PHASE" == "p2" ]]; then
  start_server --reasoning off || exit 1
  cell prompt_anti-repetition   anti-repetition   "reasoning=off,quant=Q8_0"
  cell prompt_explicit-tradeoff explicit-tradeoff "reasoning=off,quant=Q8_0"
  cell prompt_terse             terse             "reasoning=off,quant=Q8_0"
fi

# -------------------------------------------------------------- P3: reasoning
if [[ "$PHASE" == "all" || "$PHASE" == "p3" ]]; then
  # Calibrate first: per-slot context is 4096 and prompts already run 1.8-2.9k,
  # so an unbounded thinking budget risks context exhaustion and the tool-call
  # parse-500 failure mode. Try the bounded settings before the expensive one.
  for BUDGET in 256 512; do
    start_server --reasoning on --reasoning-budget "$BUDGET" || exit 1
    cell "reasoning_on_b${BUDGET}" control "reasoning=on:${BUDGET},quant=Q8_0"
  done
fi

# ----------------------------------------------------------- P4: quantization
if [[ "$PHASE" == "all" || "$PHASE" == "p4" ]]; then
  if [[ ! -f "$Q4" ]]; then
    echo "[p4] quantizing Q8_0 -> Q4_K_M (requantization; documented caveat)"
    llama-quantize "$Q8" "$Q4" Q4_K_M 2>&1 | tail -5
  fi
  if [[ -f "$Q4" ]]; then
    MODEL_OVERRIDE="$Q4" start_server --reasoning off || exit 1
    cell quant_Q4_K_M control "reasoning=off,quant=Q4_K_M"
  else
    echo "[p4] SKIPPED — quantization produced no output"
  fi
fi

# ------------------------------------------------------------------ P5: score
if [[ "$PHASE" == "all" || "$PHASE" == "p5" ]]; then
  echo; echo "######## P5 SCORING ########"
  CELLS=$(ls llm/traces/ablation 2>/dev/null | tr '\n' ' ')
  if [[ -n "$CELLS" ]]; then
    $PY llm/score_ablation_cells.py --cells $CELLS 2>&1 | tee "$LOGS/ablation_scores.log"
    $PY llm/report_ablation.py 2>&1 | tee -a "$LOGS/ablation_scores.log"
  else
    echo "[p5] no cells collected yet"
  fi
fi

# Restore the production serving configuration.
start_server --reasoning off >/dev/null 2>&1 && echo "[done] production server restored"
