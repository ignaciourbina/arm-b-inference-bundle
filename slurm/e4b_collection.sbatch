#!/bin/bash
#SBATCH --job-name=e4b-coll
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=23:00:00
#SBATCH --output=/gpfs/scratch/%u/e4b_coll_%j.log
#
# Model-scale collection: gemma-4-E4B-it-Q8_0, 100 seeds x 3 pools = 300 runs.
#
# The question this answers is the one the 15-run pilots structurally cannot:
# does a more capable model deliberate with a different SHAPE -- later peak,
# longer horizon, sustained argument diversity -- or only a different score?
# At E2B that question is settled at n=300; here it gets the same n.
#
# EVERY axis is held to the 300-run E2B collection except the model:
#   seeds 1-100 x 3 pools      identical grid, so pairing by (comp, seed) is real
#   6 slots x 4096 ctx         identical server signature
#   lanes 4, run-parallel 6    identical orchestration
#   reasoning b256, temp 1.0   identical sampling
#   prompt control             identical prompts
#
# HARNESS: `format`, set explicitly. This is the whole reason the profile was
# split. Its one rule -- accept a bare argument object as its tool call -- was
# measured against all 179,605 calls of the E2B collection and would have fired
# ZERO times, so it cannot have changed that data. It fixes 100% of E4B's
# observed failures (20.0% -> 0.0%). The reflect-slack knob is deliberately NOT
# enabled: two of E2B's own failures were budget exhaustion, so that knob is not
# comparability-safe and `lenient` would have made this cell model-or-harness.
#
# SELF-RESUBMITTING: ~33 h projected against a 23 h cap, so the chain hands over
# with a checkpoint. The orchestrator is idempotent (--resume replays finished
# runs in ~1 s). CHAIN guards against runaway resubmission.
set -uo pipefail
ROOT="$HOME/projects/arm-b-inference-bundle"; cd "$ROOT"
export PATH="$HOME/src/llama.cpp/build-multi/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
export PYTHONSAFEPATH=1
export LLM_API_FLAVOR=openai LLM_MODEL="gemma-4-E4B-it-Q8_0.gguf"
export LLM_TEMPERATURE=1.0
export LLM_HARNESS_PROFILE=format
export LLM_ABLATION_INFRA="reasoning=on;budget=256;quant=Q8_0;temp=1.0;harness=format;server=p6/24576;lanes=4;model=gemma-4-E4B-it-Q8_0.gguf"
CELL=e4b_r256_t10
TARGET=300
CHAIN="${CHAIN:-1}"; MAX_CHAIN=4
MD="$HOME/.cache/llama-models"
mkdir -p llm/traces/logs
echo "[chain $CHAIN/$MAX_CHAIN] node=$(hostname) job=$SLURM_JOB_ID"
echo "[harness] LLM_HARNESS_PROFILE=$LLM_HARNESS_PROFILE (explicit; pinned into every run)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for gpu in 0 1; do
  port=$((20434+gpu))
  CUDA_VISIBLE_DEVICES=$gpu llama-server \
    --model "$MD/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf" \
    --host 127.0.0.1 --port $port \
    --ctx-size 24576 --parallel 6 --batch-size 2048 --ubatch-size 512 \
    --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --jinja --threads 12 --cont-batching --cache-reuse 256 \
    --reasoning on --reasoning-budget 256 \
    > "llm/traces/logs/e4b_c${CHAIN}_srv${gpu}_${SLURM_JOB_ID}.log" 2>&1 &
done
trap 'pkill -f "llama-server" 2>/dev/null' EXIT
for port in 20434 20435; do
  for _ in $(seq 1 200); do curl -sf localhost:$port/v1/models >/dev/null 2>&1 && break; sleep 4; done
  curl -sf localhost:$port/v1/models >/dev/null 2>&1 || { echo "[abort] :$port never came up"; exit 1; }
  echo "[srv] :$port UP  E4B Q8_0  slots=6 ctx/slot=4096  VRAM=$(nvidia-smi -i $((port-20434)) --query-gpu=memory.used --format=csv,noheader)"
done

# Stop ~40 min before the walltime so the chain hands over cleanly rather than
# being killed mid-run.
timeout -s TERM 79200 bash -c '
  shard() { LLM_BASE_URL="http://localhost:$2" COLLECT_GPU_INDEX="$3" \
    agora/.venv/bin/python -u llm/run_collection_parallel.py \
      --cell '"$CELL"' --seeds "$1" --lanes 4 --rounds 8 --agents 6 --run-parallel 6 \
      --lane-stagger 15 --status-interval 900 --base-url "http://localhost:$2" 2>&1 | sed "s/^/[gpu$3] /"; }
  shard 1-50   20434 0 & A=$!
  shard 51-100 20435 1 & B=$!
  wait $A; wait $B'

N=$(find llm/traces/collection/$CELL -name 'townhall_*.json' ! -name '*_checkpoint.json' ! -name '*_trace_*' 2>/dev/null | wc -l)
F=$(grep -c '"run_fail"' llm/traces/logs/${CELL}_progress.jsonl 2>/dev/null || echo 0)
echo "[chain $CHAIN] finals: $N/$TARGET   cumulative run_fail events: $F"
if [ "$N" -lt "$TARGET" ] && [ "$CHAIN" -lt "$MAX_CHAIN" ]; then
  echo "[chain] incomplete -> resubmitting as chain $((CHAIN+1))"
  sbatch --export=ALL,CHAIN=$((CHAIN+1)) "$ROOT/slurm/e4b_collection.sbatch"
else
  echo "[chain] done or max chain reached"
fi
