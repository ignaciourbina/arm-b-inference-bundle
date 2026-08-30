#!/bin/bash
#SBATCH --job-name=abl-q4-b128
#SBATCH --partition=p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=56G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scratch/%u/abl_qb_%j.log
#
# Two SoTA axes at once, one per P100 GPU. Runs on the p100 queue -- a SEPARATE
# 1-job allowance from v100 -- so it never contends with the 300-run collection
# and stays inside documented fair use.
#
#   GPU 0  cell abl_q4_b256   Q4_0 @ budget 256   -> quantization axis
#   GPU 1  cell abl_q8_b128   Q8_0 @ budget 128   -> reasoning-budget axis
#
# Both at temp 1.0, control prompt, n=6 t=8, seeds 1-5 x 3 pools = 15 runs,
# paired against the main collection (Q8_0 @ b256 @ temp 1.0) on the same seeds.
# Q4 is from ggml-org -- the SAME publisher as our Q8 -- which removes the
# unsloth-vs-ggml-org quantizer confound in the Sprint-15 rejection.
# Binary: build-multi (60-real;70-real) so sm_60 runs native SASS, no PTX JIT.
set -uo pipefail
ROOT="$HOME/projects/arm-b-inference-bundle"; cd "$ROOT"
export PATH="$HOME/src/llama.cpp/build-multi/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
export PYTHONSAFEPATH=1
export LLM_API_FLAVOR=openai LLM_MODEL="gemma-4-E2B-it-Q8_0.gguf"
export COLLECT_MAX_VRAM_MB=32000
export LLM_TEMPERATURE=1.0
MODELS="$HOME/.cache/llama-models/gemma-4-E2B-it-GGUF"
mkdir -p llm/traces/logs
echo "[node] $(hostname)"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

srv() {  # gpu port model budget
  CUDA_VISIBLE_DEVICES=$1 llama-server --model "$3" --host 127.0.0.1 --port $2 \
    --ctx-size 24576 --parallel 6 --batch-size 2048 --ubatch-size 512 \
    --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --jinja --threads 6 --cont-batching --cache-reuse 256 \
    --reasoning-budget $4 > "llm/traces/logs/abl_srv$1_${SLURM_JOB_ID}.log" 2>&1 &
}
srv 0 20434 "$MODELS/gemma-4-E2B-it-Q4_0.gguf" 256
srv 1 20435 "$MODELS/gemma-4-E2B-it-Q8_0.gguf" 128
trap 'pkill -f "llama-server" 2>/dev/null' EXIT
for p in 20434 20435; do
  for _ in $(seq 1 200); do curl -sf localhost:$p/v1/models >/dev/null 2>&1 && break; sleep 5; done
  curl -sf localhost:$p/v1/models >/dev/null 2>&1 || { echo "[abort] :$p down"; tail -25 llm/traces/logs/abl_srv*_${SLURM_JOB_ID}.log; exit 1; }
  echo "[srv] :$p UP"
done
cell() {  # cell port infra label
  local t0=$SECONDS
  LLM_BASE_URL="http://localhost:$2" LLM_ABLATION_INFRA="$3" \
  agora/.venv/bin/python llm/run_collection_parallel.py \
    --cell "$1" --seeds 1-5 --lanes 4 --rounds 8 --agents 6 --run-parallel 6 \
    --lane-stagger 15 --status-interval 900 --base-url "http://localhost:$2" 2>&1 | sed "s/^/[$4] /"
  local n=$(find llm/traces/collection/$1 -name 'townhall_*.json' ! -name '*_checkpoint.json' ! -name '*_trace_*' 2>/dev/null | wc -l)
  echo "RESULT $4 cell=$1 runs=$n makespan_s=$((SECONDS-t0)) s_per_run=$(( n>0 ? (SECONDS-t0)/n : -1 ))"
}
cell abl_q4_b256 20434 "reasoning=on;budget=256;quant=Q4_0;temp=1.0;gpu=P100" q4 &  A=$!
cell abl_q8_b128 20435 "reasoning=on;budget=128;quant=Q8_0;temp=1.0;gpu=P100" b128 & B=$!
wait $A; wait $B
echo "===== ABLATION CELLS COMPLETE ====="
