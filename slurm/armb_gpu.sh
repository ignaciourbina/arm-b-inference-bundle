#!/bin/bash
# Arm-B on SeaWulf: starts the containerised llama-server on the job's GPU,
# waits for :20434, then runs the make/sweep command passed as arguments.
#
#   sbatch/srun ... slurm/armb_gpu.sh <command...>
#
# Server signature = the pinned p6-cache Arm-B config
# (N_PARALLEL=6 CTX=24576 CACHE_REUSE=256, --reasoning off), per
# llm/LOCAL-INFERENCE-READINESS.md addendum 2026-07-20.
set -uo pipefail

ROOT="$HOME/projects/arm-b-inference-bundle"
cd "$ROOT"

export PATH="$HOME/src/llama.cpp/build/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
# llm/types.py shadows the stdlib `types` module when a script inside llm/ is
# run by path (sys.path[0] = llm/). PYTHONSAFEPATH=1 stops that prepending.
export PYTHONSAFEPATH=1
export LLM_BASE_URL="http://localhost:20434"
export LLM_API_FLAVOR=openai
export LLM_MODEL="gemma-4-E2B-it-Q8_0.gguf"
export SINGULARITY_CACHEDIR="/gpfs/scratch/$USER/sing-cache"

echo "[node] $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

mkdir -p llm/traces/logs
SRV_LOG="llm/traces/logs/llama_server_${SLURM_JOB_ID:-local}.log"

echo "[srv] starting llama-server (p6-cache signature) -> $SRV_LOG"
N_PARALLEL=6 CTX=24576 CACHE_REUSE=256 \
  bash pipeline/runpod/scripts/start_llama_server_local.sh > "$SRV_LOG" 2>&1 &
SRV_PID=$!
cleanup() { echo "[srv] stopping"; kill $SRV_PID 2>/dev/null; pkill -u "$USER" -f 'llama-server' 2>/dev/null; }
trap cleanup EXIT

echo "[srv] waiting for :20434 ..."
for i in $(seq 1 150); do
    curl -sf localhost:20434/v1/models >/dev/null 2>&1 && break
    kill -0 $SRV_PID 2>/dev/null || { echo "[abort] server died:"; tail -30 "$SRV_LOG"; exit 1; }
    sleep 4
done
curl -sf localhost:20434/v1/models >/dev/null 2>&1 || { echo "[abort] server never came up:"; tail -40 "$SRV_LOG"; exit 1; }

echo "[srv] UP. models:"; curl -sf localhost:20434/v1/models
echo "[gpu] $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"
echo "[run] $*"
"$@"
RC=$?
echo "[done] rc=$RC"
exit $RC
