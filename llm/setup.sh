#!/bin/bash
# Setup script for the LLM deliberation engine.
# Run from project root: bash llm/setup.sh
#
# What this does:
# 1. Ensures llm dependencies in the canonical venv (agora/.venv)
# 2. Verifies imports against the active code path
# 3. Checks that llama-server is reachable on localhost:20434
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:20434}"
PYBIN="agora/.venv/bin/python"

echo "=== LLM Engine Setup ==="
echo ""

# --- 1. Python venv (canonical: agora/.venv; see CLAUDE.md) ---
echo ""
echo "[1/3] Ensuring llm dependencies in the canonical venv (agora/.venv)..."
if [ ! -x "$PYBIN" ]; then
    python3 -m venv agora/.venv
fi
agora/.venv/bin/pip install -q --upgrade pip
agora/.venv/bin/pip install -q aiohttp numpy scipy pytest mypy ruff
echo "  ✓ deps ready in agora/.venv"

# --- 2. Verify imports ---
echo ""
echo "[2/3] Verifying imports..."
PYTHONPATH=.:agora/src "$PYBIN" -c "
from llm.engine import AgenticLLMEngine
from llm.client import LLMClient
print('  ✓ All imports OK')
"

# --- 3. Verify backend ---
echo ""
echo "[3/3] Checking llama-server endpoint..."
if curl -sf "$LLM_BASE_URL/v1/models" > /dev/null 2>&1; then
    echo "  ✓ llama-server reachable at $LLM_BASE_URL"
else
    echo "  ! llama-server not reachable at $LLM_BASE_URL"
    echo "    Launch it with the canonical local script:"
    echo "      bash pipeline/runpod/scripts/start_llama_server_local.sh"
    echo "    then rerun health or test targets."
fi

echo ""
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  No nvidia-smi available"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage:"
echo "  make -f llm/Makefile help     # show all targets"
echo "  make -f llm/Makefile test     # run minimal 2-agent test"
echo "  make -f llm/Makefile health   # check server + GPU"
echo ""
echo "Backend notes:"
echo "  The supported runtime is llama-server via the OpenAI-compatible API"
echo "  at $LLM_BASE_URL (model: gemma-4-E2B-it-Q8_0.gguf, full filename)."
echo "  See CLAUDE.md and llm/README.md for the endpoint contract."
