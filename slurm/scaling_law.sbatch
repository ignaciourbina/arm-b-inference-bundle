#!/bin/bash
#SBATCH --job-name=scaling-law
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scratch/%u/scaling_law_%j.log
#
# MEASURED SCALING LAW — objective function is MAKESPAN PER RUN (s/run),
# never aggregate tok/s (SCALING-FINDINGS: "never infer collection wall-clock
# from an aggregate tok/s curve").
#
# Two axes run concurrently, one per GPU, same orchestrator throughout so the
# only thing varying within an axis is the knob under test:
#   GPU 0  LANES  L in {1,2,4,6}   at fixed p24
#   GPU 1  SLOTS  p in {6,12,24,48} at fixed lanes=4
# Fixed elsewhere: reasoning OFF, temp 0.3, t=8, run-parallel 6, Q8_0,
# 12 runs/cell (4 seeds x 3 compositions -> divides 1,2,4,6 evenly).
#
# Distinct --cell per point: --resume would otherwise mark the grid done after
# the first cell and every later point would read as instant.
set -uo pipefail
ROOT="$HOME/projects/arm-b-inference-bundle"; cd "$ROOT"
export PATH="$HOME/src/llama.cpp/build/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
export PYTHONSAFEPATH=1
export LLM_API_FLAVOR=openai LLM_MODEL="gemma-4-E2B-it-Q8_0.gguf"
export COLLECT_MAX_VRAM_MB=32000
export LLM_TEMPERATURE=0.3
mkdir -p llm/traces/logs
echo "[node] $(hostname)"

start_srv() {  # $1=gpu $2=port $3=nparallel
  pkill -f "llama-server.*--port $2" 2>/dev/null; sleep 3
  local ctx=$(( $3 * 4096 ))
  CUDA_VISIBLE_DEVICES=$1 PORT=$2 N_PARALLEL=$3 CTX=$ctx CACHE_REUSE=256 UBATCH=512 THREADS=8 \
    bash pipeline/runpod/scripts/start_llama_server_local.sh \
    > "llm/traces/logs/sl_srv${1}_p$3_${SLURM_JOB_ID}.log" 2>&1 &
  for _ in $(seq 1 150); do curl -sf localhost:$2/v1/models >/dev/null 2>&1 && break; sleep 4; done
  curl -sf localhost:$2/v1/models >/dev/null 2>&1 || { echo "[abort] :$2 p$3 down"; return 1; }
  echo "[srv] gpu$1 :$2 p$3 UP  vram=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader)"
}
cell() {  # $1=cell $2=lanes $3=port $4=label
  local t0=$SECONDS
  LLM_BASE_URL="http://localhost:$3" agora/.venv/bin/python llm/run_collection_parallel.py \
    --cell "$1" --seeds 501-504 --lanes "$2" --rounds 8 --agents 6 \
    --run-parallel 6 --lane-stagger 15 --status-interval 600 \
    --base-url "http://localhost:$3" > "llm/traces/logs/$1.out" 2>&1
  local el=$(( SECONDS - t0 ))
  local n=$(find llm/traces/collection/$1 -name 'townhall_*.json' ! -name '*_checkpoint.json' ! -name '*_trace_*' 2>/dev/null | wc -l)
  echo "RESULT $4 cell=$1 runs=$n makespan_s=$el makespan_per_run=$(( n>0 ? el/n : -1 ))"
}

axis_lanes() {   # GPU 0 : lanes sweep at p24
  start_srv 0 20434 24 || return 1
  for L in 1 2 4 6; do cell "sl_lanes_L${L}" "$L" 20434 "LANES"; done
}
axis_slots() {   # GPU 1 : slot sweep at lanes=4
  for P in 6 12 24 48; do
    start_srv 1 20435 $P || continue
    cell "sl_slots_p${P}" 4 20435 "SLOTS"
  done
}
trap 'pkill -f "llama-server" 2>/dev/null' EXIT
axis_lanes & A=$!
axis_slots & B=$!
wait $A; wait $B
echo "===== SCALING LAW SWEEP COMPLETE ====="
grep -h "^RESULT" /gpfs/scratch/$USER/scaling_law_${SLURM_JOB_ID}.log 2>/dev/null || true
