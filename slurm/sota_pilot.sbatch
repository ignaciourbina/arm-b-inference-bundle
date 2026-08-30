#!/bin/bash
#SBATCH --job-name=sota-pilot
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch/%u/sota_pilot_%j.log
#
# SoTA pilot — 2 cells, one per GPU, both reasoning-on b256 / Q8_0 / control prompt.
# They differ ONLY in sampling temperature, so the temp axis is attributable:
#
#   GPU 0  cell sota_r256_t10   temp 1.0   <- the target SoTA config
#   GPU 1  cell sota_r256_t03   temp 0.3   <- isolates reasoning from temp
#
# Both pair by seed against the completed reasoning-off temp-0.3 baseline
# (seawulf_test_l4, seeds 1-5), giving three points: (off,0.3) (on,0.3) (on,1.0).
# NOT comparable to the pinned Arm-B set -- reasoning-on changes generation
# semantics. This is a new self-consistent set by design.
set -uo pipefail
ROOT="$HOME/projects/arm-b-inference-bundle"; cd "$ROOT"
export PATH="$HOME/src/llama.cpp/build/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
export PYTHONSAFEPATH=1
export LLM_API_FLAVOR=openai LLM_MODEL="gemma-4-E2B-it-Q8_0.gguf"
export COLLECT_MAX_VRAM_MB=32000
mkdir -p llm/traces/logs
echo "[node] $(hostname)"; nvidia-smi --query-gpu=index,name --format=csv,noheader

for gpu in 0 1; do
  port=$((20434 + gpu))
  CUDA_VISIBLE_DEVICES=$gpu PORT=$port BUDGET=256 N_PARALLEL=24 CTX=98304 \
    bash slurm/start_server_reasoning.sh > llm/traces/logs/sota_srv${gpu}_${SLURM_JOB_ID}.log 2>&1 &
done
trap 'pkill -f "llama-server" 2>/dev/null' EXIT
for port in 20434 20435; do
  for _ in $(seq 1 150); do curl -sf localhost:$port/v1/models >/dev/null 2>&1 && break; sleep 4; done
  curl -sf localhost:$port/v1/models >/dev/null 2>&1 || { echo "[abort] :$port down"; tail -25 llm/traces/logs/sota_srv*_${SLURM_JOB_ID}.log; exit 1; }
  echo "[srv] :$port UP (reasoning on, budget 256)"
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

cell() {  # $1=cell  $2=temp  $3=port  $4=label
  LLM_TEMPERATURE="$2" \
  LLM_ABLATION_INFRA="reasoning=on;budget=256;quant=Q8_0;temp=$2;server=p24/98304" \
  LLM_BASE_URL="http://localhost:$3" \
  agora/.venv/bin/python llm/run_collection_parallel.py \
    --cell "$1" --seeds 1-5 --lanes 4 --rounds 8 --agents 6 \
    --run-parallel 6 --lane-stagger 15 --status-interval 300 \
    --base-url "http://localhost:$3" 2>&1 | sed "s/^/[$4] /"
}
cell sota_r256_t10 1.0 20434 t1.0 &  A=$!
cell sota_r256_t03 0.3 20435 t0.3 &  B=$!
wait $A; RA=$?; wait $B; RB=$?
echo "[done] t1.0 rc=$RA  t0.3 rc=$RB"
