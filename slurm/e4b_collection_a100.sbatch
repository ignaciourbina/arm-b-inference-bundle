#!/bin/bash
#SBATCH --job-name=e4b-a100
#SBATCH --partition=a100-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/scratch/%u/e4b_a100_%j.log
#
# A100 port of slurm/e4b_collection.sbatch: the same 300-run E4B collection
# (gemma-4-E4B-it-Q8_0, 100 seeds x 3 pools), for the seawulf3 scheduler
# (milan1/milan2), which has no v100 partition. Submit from milan1/2.
#
# HELD IDENTICAL to the V100 script -- grid, server signature (6 slots x 4096
# ctx, cache-reuse 256, q8_0 KV, flash-attn), lanes 4 / run-parallel 6,
# reasoning on b256, temp 1.0, prompt control, harness `format`, same
# llama.cpp commit (1844325), same 2-shard topology (seeds 1-50 / 51-100).
#
# DIFFERS, on purpose:
#   GPU            A100 instead of V100, recorded as gpu=... in
#                  LLM_ABLATION_INFRA so every trace carries it. The paired
#                  comparison to armb_sota_r256_t10 (V100) therefore differs in
#                  model AND hardware; hardware changes numerics, not prompts or
#                  sampling.
#   build          build-a100 (80-real) instead of build-multi (60/70-real).
#   ports          picked free at runtime: a100 nodes are shared (OverSubscribe).
#   walltime       2 days (a100-long cap) instead of 23 h; MAX_CHAIN 3.
#   chain guard    a chain that completes ZERO new runs does not resubmit. The
#                  V100 chain burned all 4 links in ~2 min on a disk-space abort.
set -uo pipefail
ROOT="$HOME/projects/arm-b-inference-bundle"; cd "$ROOT"
export PATH="$HOME/src/llama.cpp/build-a100/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
export PYTHONSAFEPATH=1
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-')
export LLM_API_FLAVOR=openai LLM_MODEL="gemma-4-E4B-it-Q8_0.gguf"
export LLM_TEMPERATURE=1.0
export LLM_HARNESS_PROFILE=format
export LLM_ABLATION_INFRA="reasoning=on;budget=256;quant=Q8_0;temp=1.0;harness=format;server=p6/24576;lanes=4;model=gemma-4-E4B-it-Q8_0.gguf;gpu=$GPU_NAME"
CELL=e4b_r256_t10
TARGET=300
CHAIN="${CHAIN:-1}"; MAX_CHAIN=3
MD="$HOME/.cache/llama-models"
mkdir -p llm/traces/logs
count_finals() { find llm/traces/collection/$CELL -name 'townhall_*.json' ! -name '*_checkpoint.json' ! -name '*_trace_*' 2>/dev/null | wc -l; }
N0=$(count_finals)
echo "[chain $CHAIN/$MAX_CHAIN] node=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES finals_at_start=$N0"
echo "[harness] LLM_HARNESS_PROFILE=$LLM_HARNESS_PROFILE (explicit; pinned into every run)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

free_port() { python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])'; }
PORTS=()
for gpu in 0 1; do
  port=$(free_port); PORTS+=("$port")
  CUDA_VISIBLE_DEVICES=$gpu llama-server \
    --model "$MD/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf" \
    --host 127.0.0.1 --port $port \
    --ctx-size 24576 --parallel 6 --batch-size 2048 --ubatch-size 512 \
    --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --jinja --threads 12 --cont-batching --cache-reuse 256 \
    --reasoning on --reasoning-budget 256 \
    > "llm/traces/logs/e4b_a100_c${CHAIN}_srv${gpu}_${SLURM_JOB_ID}.log" 2>&1 &
done
# Kill only this job's servers -- the node is shared, possibly with our own jobs.
trap 'for p in "${PORTS[@]}"; do pkill -u "$USER" -f "llama-server.*--port $p" 2>/dev/null; done' EXIT
for i in 0 1; do
  port=${PORTS[$i]}
  for _ in $(seq 1 200); do curl -sf localhost:$port/v1/models >/dev/null 2>&1 && break; sleep 4; done
  curl -sf localhost:$port/v1/models >/dev/null 2>&1 || { echo "[abort] :$port never came up"; exit 1; }
  echo "[srv] gpu$i :$port UP  E4B Q8_0  slots=6 ctx/slot=4096  VRAM=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader)"
done

# Stop ~40 min before the 48 h walltime so the chain hands over cleanly.
P0=${PORTS[0]} P1=${PORTS[1]} CELL=$CELL timeout -s TERM 170400 bash -c '
  shard() { LLM_BASE_URL="http://localhost:$2" COLLECT_GPU_INDEX="$3" \
    agora/.venv/bin/python -u llm/run_collection_parallel.py \
      --cell "$CELL" --seeds "$1" --lanes 4 --rounds 8 --agents 6 --run-parallel 6 \
      --lane-stagger 15 --status-interval 900 --base-url "http://localhost:$2" 2>&1 | sed "s/^/[gpu$3] /"; }
  shard 1-50   "$P0" 0 & A=$!
  shard 51-100 "$P1" 1 & B=$!
  wait $A; wait $B'

N=$(count_finals)
F=$(grep -c '"run_fail"' llm/traces/logs/${CELL}_progress.jsonl 2>/dev/null || echo 0)
echo "[chain $CHAIN] finals: $N/$TARGET (+$((N-N0)) this chain)   cumulative run_fail events: $F"
if [ "$N" -ge "$TARGET" ]; then
  echo "[chain] done"
elif [ "$N" -le "$N0" ]; then
  echo "[chain] NO PROGRESS this chain -> NOT resubmitting; read the log above"
elif [ "$CHAIN" -lt "$MAX_CHAIN" ]; then
  echo "[chain] incomplete -> resubmitting as chain $((CHAIN+1))"
  sbatch --export=ALL,CHAIN=$((CHAIN+1)) "$ROOT/slurm/e4b_collection_a100.sbatch"
else
  echo "[chain] max chain reached"
fi
