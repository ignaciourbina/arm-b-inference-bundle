#!/bin/bash
#SBATCH --job-name=armb-s8
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/scratch/%u/armb_s8_%j.log
# polarized_n6 seed 8 hits a DETERMINISTIC reflect dead-end (TH_03: no parseable
# tool call after 5 attempts). --resume replays the identical checkpoint state,
# so the orchestrator's retry pass can never clear it. Run FRESH (checkpoint
# removed, no --resume) so the run takes a different sampling path.
set -uo pipefail
ROOT="$HOME/projects/arm-b-inference-bundle"; cd "$ROOT"
export PATH="$HOME/src/llama.cpp/build/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/agora/src"
export PYTHONSAFEPATH=1
export LLM_API_FLAVOR=openai LLM_MODEL="gemma-4-E2B-it-Q8_0.gguf"
export LLM_BASE_URL=http://localhost:20434
CUDA_VISIBLE_DEVICES=0 PORT=20434 N_PARALLEL=6 CTX=24576 CACHE_REUSE=256 UBATCH=512 \
  bash pipeline/runpod/scripts/start_llama_server_local.sh > llm/traces/logs/s8_srv_${SLURM_JOB_ID}.log 2>&1 &
trap 'pkill -f "llama-server" 2>/dev/null' EXIT
for _ in $(seq 1 150); do curl -sf localhost:20434/v1/models >/dev/null 2>&1 && break; sleep 4; done
curl -sf localhost:20434/v1/models >/dev/null 2>&1 || { echo "[abort] server down"; exit 1; }
echo "[srv] UP"
rm -f llm/traces/collection/seawulf_test_l4/*polarized_n6_s8_checkpoint.json
for attempt in 1 2 3; do
  echo "=== fresh attempt $attempt (no --resume) ==="
  agora/.venv/bin/python -m llm.townhall.runner \
    --topic minimum_wage_seattle \
    --scenario-path llm/scenarios/minimum_wage_seattle_crossover.json \
    --agents 6 --rounds 8 --seed 8 --condition baseline --empirical-init \
    --profiles-path polis-analysis/output/ising_profiles.json \
    --theta-path polis-analysis/output/irt_ising_theta.json \
    --composition polarized_n6 --parallel 6 --prompt-variant control \
    --run-tag seawulf_test_l4_baseline_polarized_n6_s8 \
    --output-dir llm/traces/collection/seawulf_test_l4 2>&1 | tail -15
  n=$(find llm/traces/collection/seawulf_test_l4 -name 'townhall_*polarized_n6_s8_*.json' ! -name '*_checkpoint*' ! -name '*_trace_*' | wc -l)
  [ "$n" -ge 1 ] && { echo "[ok] seed 8 completed on attempt $attempt"; break; }
  rm -f llm/traces/collection/seawulf_test_l4/*polarized_n6_s8_checkpoint.json
done
echo "[done]"
