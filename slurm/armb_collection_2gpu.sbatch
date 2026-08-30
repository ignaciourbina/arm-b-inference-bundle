#!/bin/bash
#SBATCH --job-name=armb-2gpu
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scratch/%u/armb_2gpu_%j.log
#
# Arm-B TEST collection, sharded across BOTH GPUs of the node.
#
# Why: the v100 partition allocates WHOLE NODES (28 CPUs), so a --gres=gpu:1 job
# already consumes both GPUs' worth of allocation and leaves one idle. Runs are
# independent over the (composition x seed) grid, so sharding by seed across
# GPUs is embarrassingly parallel -- unlike lanes, which contend for one GPU's
# decode bandwidth and plateau at ~4 (1.9-2.1x). GPUs scale linearly.
#
#   GPU 0 : port 20434 : seeds 1-5
#   GPU 1 : port 20435 : seeds 6-10
# Same --cell, disjoint seeds => traces land together, no collision.
# --resume makes this idempotent: the 4 runs already done are skipped.
set -uo pipefail
ROOT="$HOME/projects/arm-b-inference-bundle"; cd "$ROOT"
export PATH="$HOME/src/llama.cpp/build/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
export PYTHONSAFEPATH=1
export LLM_API_FLAVOR=openai
export LLM_MODEL="gemma-4-E2B-it-Q8_0.gguf"
export COLLECT_MAX_VRAM_MB=32000
mkdir -p llm/traces/logs
echo "[node] $(hostname)"; nvidia-smi --query-gpu=index,name --format=csv,noheader

start_gpu_server() {  # $1=gpu index  $2=port
  echo "[srv$1] starting on GPU $1, port $2 (p24/98304, reasoning OFF)"
  CUDA_VISIBLE_DEVICES=$1 PORT=$2 N_PARALLEL=24 CTX=98304 CACHE_REUSE=256 UBATCH=512 THREADS=12 \
    bash pipeline/runpod/scripts/start_llama_server_local.sh \
    > "llm/traces/logs/2gpu_srv${1}_${SLURM_JOB_ID}.log" 2>&1 &
}
start_gpu_server 0 20434
start_gpu_server 1 20435
trap 'pkill -f "llama-server" 2>/dev/null' EXIT

for p in 20434 20435; do
  for _ in $(seq 1 150); do curl -sf localhost:$p/v1/models >/dev/null 2>&1 && break; sleep 4; done
  curl -sf localhost:$p/v1/models >/dev/null 2>&1 \
    || { echo "[abort] server on :$p failed"; tail -25 llm/traces/logs/2gpu_srv*_${SLURM_JOB_ID}.log; exit 1; }
  echo "[srv] :$p UP"
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

run_shard() {  # $1=seeds  $2=port  $3=label
  LLM_BASE_URL="http://localhost:$2" agora/.venv/bin/python llm/run_collection_parallel.py \
    --cell seawulf_test_l4 --seeds "$1" --lanes 4 --rounds 8 --agents 6 \
    --run-parallel 6 --lane-stagger 15 --status-interval 300 \
    --base-url "http://localhost:$2" 2>&1 | sed "s/^/[$3] /"
}
run_shard 1-5  20434 gpu0 &  P0=$!
run_shard 6-10 20435 gpu1 &  P1=$!
wait $P0; R0=$?
wait $P1; R1=$?
echo "[done] gpu0 rc=$R0  gpu1 rc=$R1"
