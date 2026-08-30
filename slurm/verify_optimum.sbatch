#!/bin/bash
#SBATCH --job-name=verify-opt
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scratch/%u/verify_opt_%j.log
#
# The sweep's two axes point opposite ways and never met:
#   lanes wants MORE (L6 fastest), slots wants FEWER (p6 fastest).
# Their combination L=6 @ p6 is the law's predicted optimum and was never run.
# Test it against the best measured point (p6 @ L4 = 260 s/run), one cell per
# GPU so both are measured under IDENTICAL node conditions -- the sweep's L6
# ran on a quieter node, which may have flattered it.
set -uo pipefail
ROOT="$HOME/projects/arm-b-inference-bundle"; cd "$ROOT"
export PATH="$HOME/src/llama.cpp/build/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
export PYTHONSAFEPATH=1
export LLM_API_FLAVOR=openai LLM_MODEL="gemma-4-E2B-it-Q8_0.gguf"
export COLLECT_MAX_VRAM_MB=32000 LLM_TEMPERATURE=0.3
mkdir -p llm/traces/logs
start_srv() {  # gpu port nparallel
  local ctx=$(( $3 * 4096 ))
  CUDA_VISIBLE_DEVICES=$1 PORT=$2 N_PARALLEL=$3 CTX=$ctx CACHE_REUSE=256 UBATCH=512 THREADS=8 \
    bash pipeline/runpod/scripts/start_llama_server_local.sh > "llm/traces/logs/vo_srv$1_${SLURM_JOB_ID}.log" 2>&1 &
  for _ in $(seq 1 150); do curl -sf localhost:$2/v1/models >/dev/null 2>&1 && break; sleep 4; done
  curl -sf localhost:$2/v1/models >/dev/null 2>&1 || { echo "[abort] :$2 down"; return 1; }
  echo "[srv] gpu$1 :$2 p$3 UP"
}
trap 'pkill -f "llama-server" 2>/dev/null' EXIT
start_srv 0 20434 6 || exit 1
start_srv 1 20435 6 || exit 1
cell() {  # cell lanes port label
  local t0=$SECONDS
  LLM_BASE_URL="http://localhost:$3" agora/.venv/bin/python llm/run_collection_parallel.py \
    --cell "$1" --seeds 601-604 --lanes "$2" --rounds 8 --agents 6 \
    --run-parallel 6 --lane-stagger 15 --status-interval 600 \
    --base-url "http://localhost:$3" > "llm/traces/logs/$1.out" 2>&1
  local el=$(( SECONDS - t0 ))
  local n=$(find llm/traces/collection/$1 -name 'townhall_*.json' ! -name '*_checkpoint.json' ! -name '*_trace_*' 2>/dev/null | wc -l)
  echo "RESULT $4 cell=$1 runs=$n makespan_s=$el makespan_per_run=$(( n>0 ? el/n : -1 ))"
}
cell vo_L6_p6 6 20434 "OPTIMUM_L6p6" &  A=$!
cell vo_L4_p6 4 20435 "CONTROL_L4p6" &  B=$!
wait $A; wait $B
echo "===== VERIFY COMPLETE ====="
