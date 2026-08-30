#!/bin/bash
#SBATCH --job-name=axes-cd
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --output=/gpfs/scratch/%u/axes_cd_%j.log
#
# Axes (c) and (d) of the SoTA memo, one per GPU, BOTH AT REASONING-ON so the
# results apply to the actual collection config:
#   GPU 0  (c) prefix cache-reuse: 0 / 256 / 1024 / 4096   (p6, lanes 4)
#   GPU 1  (d) slot curve:         p6 / p12 / p24          (cache-reuse 256, lanes 4)
#
# Both knobs are throughput-only and comparability-safe (same model, sampling,
# reasoning), so these are wall-clock measurements; quality is not expected to
# move and any drift would itself be a finding.
# 9 runs/cell (seeds 701-703 x 3 pools). Distinct --cell per point, else
# --resume marks the grid done after the first and later points read as instant.
set -uo pipefail
ROOT="$HOME/projects/arm-b-inference-bundle"; cd "$ROOT"
export PATH="$HOME/src/llama.cpp/build-multi/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
export PYTHONSAFEPATH=1
export LLM_API_FLAVOR=openai LLM_MODEL="gemma-4-E2B-it-Q8_0.gguf"
export COLLECT_MAX_VRAM_MB=32000 LLM_TEMPERATURE=1.0
MODEL="$HOME/.cache/llama-models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf"
mkdir -p llm/traces/logs
echo "[node] $(hostname)"

srv() {  # gpu port nparallel cachereuse
  pkill -f "llama-server.*--port $2" 2>/dev/null; sleep 4
  local ctx=$(( $3 * 4096 ))
  CUDA_VISIBLE_DEVICES=$1 llama-server --model "$MODEL" --host 127.0.0.1 --port $2 \
    --ctx-size $ctx --parallel $3 --batch-size 2048 --ubatch-size 512 \
    --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --jinja --threads 12 --cont-batching --cache-reuse $4 --reasoning-budget 256 \
    > "llm/traces/logs/cd_srv$1_p$3_c$4_${SLURM_JOB_ID}.log" 2>&1 &
  for _ in $(seq 1 150); do curl -sf localhost:$2/v1/models >/dev/null 2>&1 && break; sleep 4; done
  curl -sf localhost:$2/v1/models >/dev/null 2>&1 || { echo "[abort] gpu$1 p$3 c$4 down"; return 1; }
  echo "[srv] gpu$1 :$2 p$3 cache-reuse=$4 UP"
}
cell() {  # cell port label
  local t0=$SECONDS
  LLM_BASE_URL="http://localhost:$2" agora/.venv/bin/python llm/run_collection_parallel.py \
    --cell "$1" --seeds 701-703 --lanes 4 --rounds 8 --agents 6 --run-parallel 6 \
    --lane-stagger 15 --status-interval 900 --base-url "http://localhost:$2" \
    > "llm/traces/logs/$1.out" 2>&1
  local n=$(find llm/traces/collection/$1 -name 'townhall_*.json' ! -name '*_checkpoint.json' ! -name '*_trace_*' 2>/dev/null | wc -l)
  echo "RESULT $3 cell=$1 runs=$n makespan_s=$((SECONDS-t0)) s_per_run=$(( n>0 ? (SECONDS-t0)/n : -1 ))"
}
axis_c() { for CR in 0 256 1024 4096; do srv 0 20434 6 $CR || continue; cell "cd_cache_${CR}" 20434 CACHE; done; }
axis_d() { for P in 6 12 24;        do srv 1 20435 $P 256 || continue; cell "cd_slots_p${P}" 20435 SLOTS; done; }
trap 'pkill -f "llama-server" 2>/dev/null' EXIT
axis_c & A=$!
axis_d & B=$!
wait $A; wait $B
echo "===== AXES C+D COMPLETE ====="
grep -h "^RESULT" /gpfs/scratch/$USER/axes_cd_${SLURM_JOB_ID}.log 2>/dev/null || true
