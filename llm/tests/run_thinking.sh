#!/bin/bash
# Run the overnight deliberation with medium thinking enabled.
#
# Config: 6 agents, 5 rounds, carbon_tax scenario, confirmation_bias=0.3
# Thinking: enabled (1024 token budget per tool call)
# Backend: llama-server on localhost:20434 (OpenAI-compatible API)
#
# Estimated runtime:
#   ~55 min on RTX 2060 (CUDA, ~12s/call with thinking)
#   ~10 min on T4 spot (AWS)
#
# Output:
#   llm/traces/overnight_trace_<ts>.json   — full LLM I/O trace
#   llm/traces/overnight_results_<ts>.json — per-round opinion snapshots
#   stdout — live progress
#
# Usage:
#   bash llm/tests/run_thinking.sh          # foreground
#   nohup bash llm/tests/run_thinking.sh &  # background (overnight)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Verify the OpenAI-compatible inference endpoint is reachable.
if ! curl -sf http://localhost:20434/v1/models > /dev/null 2>&1; then
    echo "ERROR: llama-server is not reachable at http://localhost:20434"
    echo "Start the configured LLM backend before running this script."
    exit 1
fi

echo "Starting thinking run: $(date)"
echo "Config: 6 agents, 5 rounds, carbon_tax, cb=0.3, thinking=on"
echo ""

PYTHONPATH=.:agora/src llm/.venv/bin/python llm/tests/overnight_run.py

echo ""
echo "Finished: $(date)"
