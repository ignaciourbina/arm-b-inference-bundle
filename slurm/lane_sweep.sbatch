#!/bin/bash
#SBATCH --job-name=lane-sweep
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/scratch/%u/lane_sweep_%j.log
#
# Sprint-16 P4a replication on Tesla V100-32GB.
#
# Hypothesis: the measured ~4-lane ceiling (1.9-2.1x) was found on an RTX 2060
# SUPER (~448 GB/s). Decode is memory-bandwidth-bound, and V100 HBM2 is
# ~900 GB/s, so the ceiling should sit higher here. This sweeps it directly.
#
# Method (mirrors P4a): measure solo per-run time, then makespan of a fixed
# 12-run grid at L lanes with server slots scaled to lanes x run-parallel.
# Reasoning stays OFF throughout -> comparability-safe (SCALING-FINDINGS.md).
set -uo pipefail
ROOT="$HOME/projects/arm-b-inference-bundle"; cd "$ROOT"
export PATH="$HOME/src/llama.cpp/build/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
export PYTHONSAFEPATH=1
export LLM_BASE_URL="http://localhost:20434"
export LLM_API_FLAVOR=openai
export LLM_MODEL="gemma-4-E2B-it-Q8_0.gguf"
export COLLECT_MAX_VRAM_MB=32000     # 6500 gate is sized for an 8GB card
PY=agora/.venv/bin/python
STAGGER=15                            # 90s default would dominate makespan at 8 lanes
mkdir -p llm/traces/logs

start_server() {   # $1=n_parallel  $2=ctx
  pkill -f 'llama-server' 2>/dev/null; sleep 5
  echo "[srv] starting p$1 ctx$2 (reasoning OFF)"
  N_PARALLEL=$1 CTX=$2 CACHE_REUSE=256 UBATCH=512 \
    bash pipeline/runpod/scripts/start_llama_server_local.sh \
    > "llm/traces/logs/lanesweep_srv_p$1.log" 2>&1 &
  for _ in $(seq 1 120); do
    curl -sf localhost:20434/v1/models >/dev/null 2>&1 && break; sleep 4
  done
  curl -sf localhost:20434/v1/models >/dev/null 2>&1 || { echo "[abort] server p$1 failed"; tail -20 "llm/traces/logs/lanesweep_srv_p$1.log"; return 1; }
  echo "[srv] up. VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
}

echo "===== NODE: $(hostname) ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# ---- solo baseline: 3 runs, 1 lane, p6 ----
start_server 6 24576 || exit 1
echo "===== BASELINE (1 lane, 3 runs, t=4) ====="
/usr/bin/time -f "BASELINE_WALL=%e" $PY llm/run_collection_parallel.py \
  --cell lanesweep_solo --seeds 900 --lanes 1 --rounds 4 --agents 6 \
  --run-parallel 6 --lane-stagger 0 --status-interval 600 2>&1 | tail -20

# ---- lane sweep: fixed 12-run grid (3 comps x seeds 901-904) ----
# 12-run grid divides evenly at L=2,4,6 (6/3/2 per lane). L=8 would slice
# 2,2,2,2,1,1,1,1 -- its second wave runs at half occupancy and would understate
# it, so 8 is a conditional follow-up on a 24-run grid if L=6 is still scaling.
for L in 2 4 6; do
  NP=$(( L * 6 )); CTX=$(( NP * 4096 ))
  start_server $NP $CTX || { echo "[skip] L=$L server failed"; continue; }
  echo "===== LANES=$L  (server p$NP ctx$CTX, 12 runs, t=4) ====="
  /usr/bin/time -f "LANES${L}_WALL=%e" $PY llm/run_collection_parallel.py \
    --cell lanesweep_L$L --seeds 901-904 --lanes $L --rounds 4 --agents 6 \
    --run-parallel 6 --lane-stagger $STAGGER --status-interval 600 2>&1 | tail -20
done

pkill -f 'llama-server' 2>/dev/null
echo "===== SWEEP COMPLETE ====="
