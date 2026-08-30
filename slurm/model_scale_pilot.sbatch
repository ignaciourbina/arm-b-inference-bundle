#!/bin/bash
#SBATCH --job-name=model-scale
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=23:00:00
#SBATCH --output=/gpfs/scratch/%u/model_scale_%j.log
#
# Model-scale pilot: does the Arm-B deliberation finding hold as capacity grows?
# This is a NEW ARM, not a faster Arm B (Arm B is defined as E2B-Q8_0).
#
#   GPU 0  cell ms_26B_a4b   gemma-4-26B-A4B-it Q8_0  (26B total / 3.8B active)
#   GPU 1  cell ms_E4B       gemma-4-E4B-it     Q8_0  ( 8B total / 4.0B active)
#
# With the existing E2B collection this gives a 3-point model-scale curve.
# Same SoTA config otherwise: reasoning b256, temp 1.0, control prompt, lanes 4,
# seeds 1-5 x 3 pools = 15 runs/cell, paired by seed against armb_sota_r256_t10.
#
# PREFLIGHT gates the expensive part: both servers must load and answer, and we
# record VRAM. 26B-A4B is 25 GB of weights on a 32 GB card, so KV headroom is
# the real risk -- slots start at 4 (not 6) and ctx at 4096/slot.
set -uo pipefail
ROOT="$HOME/projects/arm-b-inference-bundle"; cd "$ROOT"
export PATH="$HOME/src/llama.cpp/build-multi/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
export PYTHONSAFEPATH=1
export LLM_API_FLAVOR=openai
export LLM_TEMPERATURE=1.0
MD="$HOME/.cache/llama-models"
mkdir -p llm/traces/logs
echo "[node] $(hostname)"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

srv() {  # gpu port model slots
  CUDA_VISIBLE_DEVICES=$1 llama-server --model "$3" --host 127.0.0.1 --port $2 \
    --ctx-size $(( $4 * 4096 )) --parallel $4 --batch-size 2048 --ubatch-size 512 \
    --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --jinja --threads 12 --cont-batching --cache-reuse 256 --reasoning-budget 256 \
    > "llm/traces/logs/ms_srv$1_${SLURM_JOB_ID}.log" 2>&1 &
  for _ in $(seq 1 300); do curl -sf localhost:$2/v1/models >/dev/null 2>&1 && break; sleep 5; done
  if ! curl -sf localhost:$2/v1/models >/dev/null 2>&1; then
    echo "[PREFLIGHT FAIL] gpu$1 could not load $(basename $3)"
    grep -iE "error|unknown|unsupported|arch" "llm/traces/logs/ms_srv$1_${SLURM_JOB_ID}.log" | head -8
    return 1
  fi
  echo "[srv] gpu$1 :$2 UP  $(basename $3)  slots=$4  VRAM=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader)"
}
trap 'pkill -f "llama-server" 2>/dev/null' EXIT

srv 0 20434 "$MD/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q8_0.gguf" 4 || exit 1
srv 1 20435 "$MD/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf"         6 || exit 1

echo "[preflight] tool-call smoke on each server"
for p in 20434 20435; do
  curl -s localhost:$p/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"m","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":400}' \
    | head -c 220; echo
done

# COLLECT_GPU_INDEX: nvidia-smi does not honour CUDA_VISIBLE_DEVICES, so each
# shard must be told which card its server is on or it reads GPU 0's memory.
# COLLECT_MAX_VRAM_MB: the 6500 default is sized for E2B at p6. Set per cell,
# just above the model's measured footprint, so the gate still catches a
# misconfigured server instead of being disabled outright.
cell() {  # cell port model label gpu vram_gate_mb
  local t0=$SECONDS
  LLM_BASE_URL="http://localhost:$2" LLM_MODEL="$3" \
  COLLECT_GPU_INDEX="$5" COLLECT_MAX_VRAM_MB="$6" \
  LLM_ABLATION_INFRA="reasoning=on;budget=256;quant=Q8_0;temp=1.0;model=$3" \
  agora/.venv/bin/python llm/run_collection_parallel.py \
    --cell "$1" --seeds 1-5 --lanes 4 --rounds 8 --agents 6 --run-parallel 6 \
    --lane-stagger 15 --status-interval 900 --base-url "http://localhost:$2" 2>&1 | sed "s/^/[$4] /"
  local n=$(find llm/traces/collection/$1 -name 'townhall_*.json' ! -name '*_checkpoint.json' ! -name '*_trace_*' 2>/dev/null | wc -l)
  echo "RESULT $4 cell=$1 runs=$n makespan_s=$((SECONDS-t0)) s_per_run=$(( n>0 ? (SECONDS-t0)/n : -1 ))"
}
cell ms_26B_a4b 20434 gemma-4-26B-A4B-it-Q8_0.gguf 26B 0 30000 &  A=$!
cell ms_E4B     20435 gemma-4-E4B-it-Q8_0.gguf     E4B 1  9000 &  B=$!
wait $A; wait $B
echo "===== MODEL-SCALE PILOT COMPLETE ====="
