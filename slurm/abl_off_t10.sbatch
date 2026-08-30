#!/bin/bash
#SBATCH --job-name=abl-off-t10
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scratch/%u/abl_off_t10_%j.log
#
# Completes the 2x2 (reasoning x temperature) factorial. Three cells exist:
#   (off,0.3) seawulf_test_l4   (on,0.3) sota_r256_t03   (on,1.0) sota_r256_t10
# This adds the missing (off,1.0) so temperature can be tested independently of
# reasoning, and the interaction estimated. Sharded across both GPUs.
# Server p6 (measured fastest); slot count is throughput-only and
# comparability-safe -- reasoning and temperature are the semantic knobs.
set -uo pipefail
ROOT="$HOME/projects/arm-b-inference-bundle"; cd "$ROOT"
export PATH="$HOME/src/llama.cpp/build/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
export PYTHONSAFEPATH=1
export LLM_API_FLAVOR=openai LLM_MODEL="gemma-4-E2B-it-Q8_0.gguf"
export COLLECT_MAX_VRAM_MB=32000
export LLM_TEMPERATURE=1.0
export LLM_ABLATION_INFRA="reasoning=off;quant=Q8_0;temp=1.0;server=p6/24576"
mkdir -p llm/traces/logs
for gpu in 0 1; do
  port=$((20434+gpu))
  CUDA_VISIBLE_DEVICES=$gpu PORT=$port N_PARALLEL=6 CTX=24576 CACHE_REUSE=256 UBATCH=512 THREADS=8 \
    bash pipeline/runpod/scripts/start_llama_server_local.sh > llm/traces/logs/abl_srv${gpu}_${SLURM_JOB_ID}.log 2>&1 &
done
trap 'pkill -f "llama-server" 2>/dev/null' EXIT
for port in 20434 20435; do
  for _ in $(seq 1 150); do curl -sf localhost:$port/v1/models >/dev/null 2>&1 && break; sleep 4; done
  curl -sf localhost:$port/v1/models >/dev/null 2>&1 || { echo "[abort] :$port down"; exit 1; }
  echo "[srv] :$port UP (reasoning off, temp 1.0)"
done
run() { LLM_BASE_URL="http://localhost:$2" agora/.venv/bin/python llm/run_collection_parallel.py \
   --cell abl_off_t10 --seeds "$1" --lanes 4 --rounds 8 --agents 6 --run-parallel 6 \
   --lane-stagger 15 --status-interval 300 --base-url "http://localhost:$2" 2>&1 | sed "s/^/[$3] /"; }
run 1-3 20434 gpu0 & A=$!
run 4-5 20435 gpu1 & B=$!
wait $A; wait $B
echo "[done]"
