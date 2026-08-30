#!/bin/bash
#SBATCH --job-name=armb-finish
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/%u/armb_finish_%j.log
# Completes the pilot: polarized_n6 seed 8, the single persistent failure.
# Runs the POST-PULL module, so a repeat failure writes
# llm/traces/logs/FAIL_seawulf_test_l4_baseline_polarized_n6_s8.log
set -uo pipefail
ROOT="$HOME/projects/arm-b-inference-bundle"; cd "$ROOT"
export PATH="$HOME/src/llama.cpp/build/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
export PYTHONSAFEPATH=1
export LLM_API_FLAVOR=openai LLM_MODEL="gemma-4-E2B-it-Q8_0.gguf"
mkdir -p llm/traces/logs
# single lane on one card: removes contention as a variable for this diagnosis
CUDA_VISIBLE_DEVICES=0 PORT=20434 N_PARALLEL=6 CTX=24576 CACHE_REUSE=256 UBATCH=512 \
  bash pipeline/runpod/scripts/start_llama_server_local.sh > llm/traces/logs/finish_srv_${SLURM_JOB_ID}.log 2>&1 &
trap 'pkill -f "llama-server" 2>/dev/null' EXIT
for _ in $(seq 1 150); do curl -sf localhost:20434/v1/models >/dev/null 2>&1 && break; sleep 4; done
curl -sf localhost:20434/v1/models >/dev/null 2>&1 || { echo "[abort] server down"; exit 1; }
echo "[srv] UP (p6, single lane, no contention)"
LLM_BASE_URL=http://localhost:20434 agora/.venv/bin/python llm/run_collection_parallel.py \
  --cell seawulf_test_l4 --seeds 8 --compositions polarized_n6 \
  --lanes 1 --rounds 8 --agents 6 --run-parallel 6 --lane-stagger 0 \
  --base-url http://localhost:20434
echo "[done] rc=$?"
