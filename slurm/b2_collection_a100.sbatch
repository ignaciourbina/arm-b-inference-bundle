#!/bin/bash
#SBATCH --job-name=b2-a100
#SBATCH --partition=a100-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gpfs/scratch/%u/b2_a100_%j.log
#
# B2 extension to 200 seeds/room, as a SEPARATE cell on A100 (IU, 2026-11-24):
# armb_sota_r256_t10_a100, seeds 101-200 x 3 rooms = 300 runs. B2 itself
# (armb_sota_r256_t10, seeds 1-100) ran on V100, which milan cannot submit to.
#
# HELD IDENTICAL to slurm/armb_sota_collection.sbatch + start_server_reasoning.sh:
#   gemma-4-E2B-it-Q8_0 (sha256 996d0877..., the refreshed Jul-15 weights);
#   server p6 / ctx 24576 / batch 2048 / ubatch 512 / flash-attn on / q8_0 KV /
#   --jinja / threads 12 / cont-batching / reasoning on, budget 256, and NO
#   --cache-reuse (B2 never set it); temp 1.0; lanes 4 / run-parallel 6 /
#   stagger 15; control prompt; harness `strict` (the frozen E2B path, which the
#   harness documents as byte-identical to what produced B2).
# DIFFERS: GPU (A100 80GB, recorded as gpu=... in ablation_infra), build
#   (build-a100, 80-real), ports picked at runtime, 2-day walltime. B2 traces do
#   not record harness_profile; these record `strict`.
set -uo pipefail
ROOT="$HOME/projects/arm-b-inference-bundle"; cd "$ROOT"
export PATH="$HOME/src/llama.cpp/build-a100/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
export PYTHONSAFEPATH=1
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-')
export LLM_API_FLAVOR=openai LLM_MODEL="gemma-4-E2B-it-Q8_0.gguf"
export COLLECT_MAX_VRAM_MB=32000
export LLM_TEMPERATURE=1.0
export LLM_HARNESS_PROFILE=strict
export LLM_ABLATION_INFRA="reasoning=on;budget=256;quant=Q8_0;temp=1.0;server=p6/24576;lanes=4;harness=strict;gpu=$GPU_NAME"
CELL=armb_sota_r256_t10_a100
# Seed ranges per GPU shard; defaults reproduce the first 300-run collection.
SEEDS_A="${SEEDS_A:-101-150}"; SEEDS_B="${SEEDS_B:-151-200}"
TARGET="${TARGET:-300}"
CHAIN="${CHAIN:-1}"; MAX_CHAIN=3
MD="$HOME/.cache/llama-models"
mkdir -p llm/traces/logs
count_finals() { find llm/traces/collection/$CELL/ -name 'townhall_*.json' ! -name '*_checkpoint.json' ! -name '*_trace_*' 2>/dev/null | wc -l; }
N0=$(count_finals)
echo "[chain $CHAIN/$MAX_CHAIN] node=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES finals_at_start=$N0 seeds=$SEEDS_A/$SEEDS_B target=$TARGET"
echo "[harness] LLM_HARNESS_PROFILE=$LLM_HARNESS_PROFILE (explicit; pinned into every run)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

free_port() { python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])'; }
PORTS=()
for gpu in 0 1; do
  port=$(free_port); PORTS+=("$port")
  CUDA_VISIBLE_DEVICES=$gpu llama-server \
    --model "$MD/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf" \
    --host 127.0.0.1 --port $port \
    --ctx-size 24576 --parallel 6 --batch-size 2048 --ubatch-size 512 \
    --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --jinja --threads 12 --cont-batching \
    --reasoning on --reasoning-budget 256 \
    > "llm/traces/logs/b2_a100_c${CHAIN}_srv${gpu}_${SLURM_JOB_ID}.log" 2>&1 &
done
# Kill only this job's servers -- the node is shared, possibly with our own jobs.
trap 'for p in "${PORTS[@]}"; do pkill -u "$USER" -f "llama-server.*--port $p" 2>/dev/null; done' EXIT
for i in 0 1; do
  port=${PORTS[$i]}
  for _ in $(seq 1 200); do curl -sf localhost:$port/v1/models >/dev/null 2>&1 && break; sleep 4; done
  curl -sf localhost:$port/v1/models >/dev/null 2>&1 || { echo "[abort] :$port never came up"; exit 1; }
  echo "[srv] gpu$i :$port UP  E2B Q8_0  slots=6 ctx/slot=4096  VRAM=$(nvidia-smi -i $i --query-gpu=memory.used --format=csv,noheader)"
done

# Stop ~40 min before the 48 h walltime so the chain hands over cleanly.
P0=${PORTS[0]} P1=${PORTS[1]} CELL=$CELL SEEDS_A=$SEEDS_A SEEDS_B=$SEEDS_B timeout -s TERM 170400 bash -c '
  shard() { LLM_BASE_URL="http://localhost:$2" COLLECT_GPU_INDEX="$3" \
    agora/.venv/bin/python -u llm/run_collection_parallel.py \
      --cell "$CELL" --seeds "$1" --lanes 4 --rounds 8 --agents 6 --run-parallel 6 \
      --lane-stagger 15 --status-interval 900 --base-url "http://localhost:$2" 2>&1 | sed "s/^/[gpu$3] /"; }
  shard "$SEEDS_A" "$P0" 0 & A=$!
  shard "$SEEDS_B" "$P1" 1 & B=$!
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
  sbatch --export=ALL,CHAIN=$((CHAIN+1)),SEEDS_A=$SEEDS_A,SEEDS_B=$SEEDS_B,TARGET=$TARGET "$ROOT/slurm/b2_collection_a100.sbatch"
else
  echo "[chain] max chain reached"
fi
