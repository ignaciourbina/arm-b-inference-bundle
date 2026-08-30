#!/bin/bash
#SBATCH --job-name=armb-test30
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scratch/%u/armb_l4_%j.log
#
# Arm-B TEST collection: 10 seeds x 3 compositions = 30 runs, n=6, t=8.
# Uses the sprint-16 lanes orchestrator at the MEASURED-safe --lanes 4
# (1.9-2.1x; the default 8 is above the real-run ceiling).
# Server slots = lanes x run-parallel = 24, reasoning OFF (comparability).
set -uo pipefail
ROOT="$HOME/projects/arm-b-inference-bundle"; cd "$ROOT"
export PATH="$HOME/src/llama.cpp/build/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
export PYTHONSAFEPATH=1
export LLM_BASE_URL="http://localhost:20434"
export LLM_API_FLAVOR=openai
export LLM_MODEL="gemma-4-E2B-it-Q8_0.gguf"
export COLLECT_MAX_VRAM_MB=32000
mkdir -p llm/traces/logs

echo "[node] $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
pkill -f 'llama-server' 2>/dev/null; sleep 4
echo "[srv] starting p24/98304, reasoning OFF"
N_PARALLEL=24 CTX=98304 CACHE_REUSE=256 UBATCH=512 \
  bash pipeline/runpod/scripts/start_llama_server_local.sh \
  > llm/traces/logs/l4_server_${SLURM_JOB_ID}.log 2>&1 &
trap 'pkill -f "llama-server" 2>/dev/null' EXIT
for _ in $(seq 1 150); do curl -sf localhost:20434/v1/models >/dev/null 2>&1 && break; sleep 4; done
curl -sf localhost:20434/v1/models >/dev/null 2>&1 || { echo "[abort] server down"; tail -30 llm/traces/logs/l4_server_${SLURM_JOB_ID}.log; exit 1; }
echo "[srv] UP. VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader|head -1)"

agora/.venv/bin/python llm/run_collection_parallel.py \
  --cell seawulf_test_l4 --seeds 1-10 --lanes 4 --rounds 8 --agents 6 \
  --run-parallel 6 --lane-stagger 15 --status-interval 300
echo "[done] rc=$?"
