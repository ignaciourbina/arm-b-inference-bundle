#!/bin/bash
#SBATCH --job-name=ms-rerun
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=23:00:00
#SBATCH --output=/gpfs/scratch/%u/ms_rerun_%j.log
#
# Model-scale pilot, RERUN. The first pilot (job 80089) produced three numbers
# that could not be quoted, for three different reasons. Each is fixed here:
#
#  1. E4B failed 20% of first attempts, all on `{"unchanged":true}` -- a correct
#     intent in a form the E2B-tuned loop rejects. -> LLM_HARNESS_PROFILE=lenient
#  2. 26B hit "no terminal tool within budget" while emitting VALID update_weight
#     calls. -> lenient also buys reflect-round slack (_REFLECT_SLACK).
#  3. 26B took a 400: the server said "request (4121 tokens) exceeds the
#     available context size (4096 tokens)". -> 6144 ctx/slot for the 26B.
#
# And two comparability fixes:
#  4. 26B ran at 4 slots vs E2B/E4B's 6 while --run-parallel 6 asks six agents to
#     act at once, so it was slot-starved by construction. -> 6 slots for both.
#  5. The old ms_26B_a4b is NOT UNIFORM (harness landed mid-collection). NEW cell
#     names, so nothing resumes into a split cell.
#
# The harness profile is set EXPLICITLY rather than left to the model table:
# the orchestrator pins it into every run subprocess and the runner records it
# in each trace, so these cells state which harness produced them.
set -uo pipefail
ROOT="$HOME/projects/arm-b-inference-bundle"; cd "$ROOT"
export PATH="$HOME/src/llama.cpp/build-multi/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
export PYTHONSAFEPATH=1
export LLM_API_FLAVOR=openai
export LLM_TEMPERATURE=1.0
export LLM_HARNESS_PROFILE=lenient
MD="$HOME/.cache/llama-models"
mkdir -p llm/traces/logs
echo "[node] $(hostname)"; nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "[harness] LLM_HARNESS_PROFILE=$LLM_HARNESS_PROFILE (explicit; pinned per cell)"

srv() {  # gpu port model slots ctx_per_slot
  CUDA_VISIBLE_DEVICES=$1 llama-server --model "$3" --host 127.0.0.1 --port $2 \
    --ctx-size $(( $4 * $5 )) --parallel $4 --batch-size 2048 --ubatch-size 512 \
    --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --jinja --threads 12 --cont-batching --cache-reuse 256 \
    --reasoning on --reasoning-budget 256 \
    > "llm/traces/logs/ms2_srv$1_${SLURM_JOB_ID}.log" 2>&1 &
  for _ in $(seq 1 300); do curl -sf localhost:$2/v1/models >/dev/null 2>&1 && break; sleep 5; done
  if ! curl -sf localhost:$2/v1/models >/dev/null 2>&1; then
    echo "[PREFLIGHT FAIL] gpu$1 could not load $(basename $3) at ${4}x${5}"
    grep -iE "error|unknown|unsupported|arch|out of memory" "llm/traces/logs/ms2_srv$1_${SLURM_JOB_ID}.log" | head -8
    return 1
  fi
  echo "[srv] gpu$1 :$2 UP  $(basename $3)  slots=$4 ctx/slot=$5  VRAM=$(nvidia-smi -i $1 --query-gpu=memory.used --format=csv,noheader)"
}
trap 'pkill -f "llama-server" 2>/dev/null' EXIT

# 26B: 6 slots x 6144 = 36864 ctx. Measured KV cost on the first pilot was
# ~0.079 MiB/token (26900 MiB used at 4x4096 against 25610 MiB of weights), so
# this projects to ~28.5 GB of 32 GB. The gate below still catches a runaway.
srv 0 20434 "$MD/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-Q8_0.gguf" 6 6144 || exit 1
# E4B: 6 x 4096, identical to the 300-run E2B server signature. Only the model
# and the harness profile differ, so the pairing means something.
srv 1 20435 "$MD/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf"          6 4096 || exit 1

echo "[preflight] tool-call smoke on each server"
for p in 20434 20435; do
  curl -s localhost:$p/v1/chat/completions -H "Content-Type: application/json" \
    -d '{"model":"m","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":400}' \
    | head -c 160; echo
done

cell() {  # cell port model label gpu vram_gate_mb
  local t0=$SECONDS
  LLM_BASE_URL="http://localhost:$2" LLM_MODEL="$3" \
  COLLECT_GPU_INDEX="$5" COLLECT_MAX_VRAM_MB="$6" \
  LLM_ABLATION_INFRA="reasoning=on;budget=256;quant=Q8_0;temp=1.0;harness=lenient;model=$3" \
  agora/.venv/bin/python -u llm/run_collection_parallel.py \
    --cell "$1" --seeds 1-5 --lanes 4 --rounds 8 --agents 6 --run-parallel 6 \
    --lane-stagger 15 --status-interval 900 --base-url "http://localhost:$2" 2>&1 | sed "s/^/[$4] /"
  local n=$(find llm/traces/collection/$1 -name 'townhall_*.json' ! -name '*_checkpoint.json' ! -name '*_trace_*' 2>/dev/null | wc -l)
  echo "RESULT $4 cell=$1 runs=$n makespan_s=$((SECONDS-t0)) s_per_run=$(( n>0 ? (SECONDS-t0)/n : -1 ))"
}
cell ms2_26B_a4b 20434 gemma-4-26B-A4B-it-Q8_0.gguf 26B 0 31000 &  A=$!
cell ms2_E4B     20435 gemma-4-E4B-it-Q8_0.gguf     E4B 1  9000 &  B=$!
wait $A; wait $B
echo "===== MODEL-SCALE RERUN COMPLETE ====="
