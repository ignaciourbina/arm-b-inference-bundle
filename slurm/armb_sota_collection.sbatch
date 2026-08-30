#!/bin/bash
#SBATCH --job-name=armb-sota
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=23:00:00
#SBATCH --output=/gpfs/scratch/%u/armb_sota_%j.log
#
# Arm-B SoTA collection: 100 unique seeds x 3 composition pools = 300 runs.
#   reasoning ON (budget 256) · temperature 1.0 · Q8_0 · control prompt · n=6, t=8
#
# temp 1.0 is a deliberate methodological choice (IU): it samples the model's
# actual next-token distribution. Lower temperatures sharpen toward the mode, so
# the agents would not be the cognition the study claims to observe.
#
# Server: p6 per GPU (measured fastest: p6 ~292 s/run vs p24 ~390 at lanes=4;
# these runs are latency-bound, not VRAM-bound). Lanes 4 (measured ceiling;
# L=6 was only 2% better under matched conditions). Slot count is
# throughput-only and comparability-safe.
#
# SELF-RESUBMITTING: ~21h projected against a 24h cap. The orchestrator is
# idempotent (--resume replays finished runs in ~1s), so the job re-queues
# itself until the grid is complete. CHAIN guards against runaway resubmission.
set -uo pipefail
ROOT="$HOME/projects/arm-b-inference-bundle"; cd "$ROOT"
export PATH="$HOME/src/llama.cpp/build/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
export PYTHONSAFEPATH=1
export LLM_API_FLAVOR=openai LLM_MODEL="gemma-4-E2B-it-Q8_0.gguf"
export COLLECT_MAX_VRAM_MB=32000
export LLM_TEMPERATURE=1.0
export LLM_ABLATION_INFRA="reasoning=on;budget=256;quant=Q8_0;temp=1.0;server=p6/24576;lanes=4"
CELL=armb_sota_r256_t10
TARGET=300
CHAIN="${CHAIN:-1}"; MAX_CHAIN=4
mkdir -p llm/traces/logs
echo "[chain $CHAIN/$MAX_CHAIN] node=$(hostname) job=$SLURM_JOB_ID"

for gpu in 0 1; do
  port=$((20434+gpu))
  CUDA_VISIBLE_DEVICES=$gpu PORT=$port BUDGET=256 N_PARALLEL=6 CTX=24576 \
    bash slurm/start_server_reasoning.sh > llm/traces/logs/sota_c${CHAIN}_srv${gpu}_${SLURM_JOB_ID}.log 2>&1 &
done
trap 'pkill -f "llama-server" 2>/dev/null' EXIT
for port in 20434 20435; do
  for _ in $(seq 1 150); do curl -sf localhost:$port/v1/models >/dev/null 2>&1 && break; sleep 4; done
  curl -sf localhost:$port/v1/models >/dev/null 2>&1 || { echo "[abort] :$port down"; exit 1; }
  echo "[srv] :$port UP (reasoning b256, temp 1.0, p6)"
done

shard() { LLM_BASE_URL="http://localhost:$2" agora/.venv/bin/python llm/run_collection_parallel.py \
    --cell "$CELL" --seeds "$1" --lanes 4 --rounds 8 --agents 6 --run-parallel 6 \
    --lane-stagger 15 --status-interval 900 --base-url "http://localhost:$2" 2>&1 | sed "s/^/[$3] /"; }
# stop ~40 min before walltime so the chain can hand over cleanly
timeout -s TERM 79200 bash -c '
  shard() { LLM_BASE_URL="http://localhost:$2" agora/.venv/bin/python llm/run_collection_parallel.py \
    --cell '"$CELL"' --seeds "$1" --lanes 4 --rounds 8 --agents 6 --run-parallel 6 \
    --lane-stagger 15 --status-interval 900 --base-url "http://localhost:$2" 2>&1 | sed "s/^/[$3] /"; }
  shard 1-50   20434 gpu0 & A=$!
  shard 51-100 20435 gpu1 & B=$!
  wait $A; wait $B'

N=$(find llm/traces/collection/$CELL -name 'townhall_*.json' ! -name '*_checkpoint.json' ! -name '*_trace_*' 2>/dev/null | wc -l)
echo "[chain $CHAIN] finals: $N/$TARGET"
if [ "$N" -lt "$TARGET" ] && [ "$CHAIN" -lt "$MAX_CHAIN" ]; then
  echo "[chain] incomplete -> resubmitting as chain $((CHAIN+1))"
  CHAIN=$((CHAIN+1)) sbatch --export=ALL,CHAIN=$((CHAIN+1)) "$ROOT/slurm/armb_sota_collection.sbatch"
else
  echo "[chain] done or max chain reached"
fi
