#!/usr/bin/env bash
# Arm-B local collection — pinned p6-cache configuration (adopted 2026-07-20).
#
# Single source of truth for the local 390-run Arm-B collection:
#   130 seeds (1..130) x 3 compositions (symmetric_n6 polarized_n6
#   three_clusters_n6), n=6 agents, t=8 rounds, condition=baseline,
#   AgenticLLMEngine via the TownHall runner.
#
# Server signature (recorded in llm/LOCAL-INFERENCE-READINESS.md addendum;
# benchmark evidence: outputs/llm_engine/bench_local_config_cache/scoreboard.md):
#   N_PARALLEL=6  CTX=24576 (= 4096/slot)  CACHE_REUSE=256  UBATCH=512
#   + canonical flags from pipeline/runpod/scripts/start_llama_server_local.sh
#     (--jinja --reasoning off --flash-attn on --cache-type-k/v q8_0).
#
# The run-tag prefix is FIXED (no timestamp): skip-existing and checkpoint
# resume both key off it, so pilot runs count toward the 390 and any relaunch
# of `sweep` is idempotent.
#
# Usage:
#   bash llm/run_arm_b_local_collection.sh server   # start llama-server :20434
#   bash llm/run_arm_b_local_collection.sh sweep    # supervised sweep (resumable)
#   bash llm/run_arm_b_local_collection.sh status   # progress / GPU one-liner
#   SEEDS="1" bash llm/run_arm_b_local_collection.sh sweep   # subset (pilot)
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

ARM_B_PREFIX="${ARM_B_PREFIX:-arm_b_local_p6cache}"
ARM_B_OUTPUT_DIR="llm/traces/beta_local/${ARM_B_PREFIX}"
SEEDS_DEFAULT="$(seq 1 130 | tr '\n' ' ')"

case "${1:-}" in
  server)
    exec env N_PARALLEL=6 CTX=24576 CACHE_REUSE=256 \
      bash pipeline/runpod/scripts/start_llama_server_local.sh
    ;;
  sweep)
    # BETA_TRACE_ROOT must match the output root or skip-existing goes blind
    # (the retry loop does not forward it on the make command line; env does).
    export BETA_TRACE_ROOT=llm/traces/beta_local
    export LOG_DIR=llm/traces/logs
    exec make -f llm/Makefile townhall-beta-local-retry-loop \
      BETA_LOCAL_SEEDS="${SEEDS:-$SEEDS_DEFAULT}" \
      BETA_LOCAL_COMPOSITIONS="symmetric_n6 polarized_n6 three_clusters_n6" \
      BETA_LOCAL_ROUNDS=8 \
      BETA_LOCAL_PARALLEL=6 \
      BETA_LOCAL_RUN_TAG_PREFIX="$ARM_B_PREFIX" \
      BETA_LOCAL_OUTPUT_DIR="$ARM_B_OUTPUT_DIR" \
      BETA_LOCAL_RESUME=true
    ;;
  status)
    n=$(find "$ARM_B_OUTPUT_DIR" -name 'townhall_*.json' \
        ! -name '*_checkpoint.json' ! -name '*_trace_*' 2>/dev/null | wc -l)
    echo "finals: ${n}/390  (${ARM_B_OUTPUT_DIR})"
    tail -n 3 "llm/traces/logs/${ARM_B_PREFIX}_supervisor.log" 2>/dev/null || true
    nvidia-smi --query-gpu=temperature.gpu,memory.used,utilization.gpu \
      --format=csv,noheader 2>/dev/null || true
    ;;
  *)
    echo "usage: $0 {server|sweep|status}" >&2
    exit 2
    ;;
esac
