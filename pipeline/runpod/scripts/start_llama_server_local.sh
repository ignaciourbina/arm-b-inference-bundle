#!/bin/bash
# Local llama-server on RTX 2060 SUPER (8GB) via Homebrew's Vulkan build.
# Drops in as an OpenAI-compatible endpoint at :20434.
#
# Homebrew's llama-server is Vulkan-backed — faster than Ollama's CUDA path
# on Turing for small models + small batches, which is what we saw earlier.
set -euo pipefail

# Text-only gemma-4-E2B-it GGUF (Q8_0) from ggml-org.
# Ollama's multimodal blob doesn't load under llama.cpp (expects 2012 tensors,
# the GGUF only exposes 601 text tensors — the vision/audio projectors live in
# a separate mmproj blob Ollama doesn't expose).
GGUF_PATH="${GGUF_PATH:-$HOME/.cache/llama-models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf}"
PORT="${PORT:-20434}"
N_PARALLEL="${N_PARALLEL:-1}"     # 8GB is tight; 1 slot matches our Ollama discipline
CTX="${CTX:-4096}"                # per-slot context (8GB budget)
UBATCH="${UBATCH:-512}"
THREADS="${THREADS:-8}"
# Prompt-prefix KV reuse (tokens). Deliberation prompts are ~95% shared-prefix
# prefill (persona+repertoire), so this is a large lever; 0 disables.
CACHE_REUSE="${CACHE_REUSE:-0}"

# --parallel 1      -> one sequence at a time (8GB VRAM budget)
# --n-gpu-layers 99 -> offload all layers to Vulkan (2060 Super)
# --jinja           -> use model's chat template for tool calling
# --reasoning off   -> disable gemma4 thinking; it blows past max_tokens before
#                     the tool call completes on a small GPU, causing 500s.
# --flash-attn on   -> ggml flash-attention kernels (Vulkan-compat)
# --cache-type-k q8_0 -> quantized KV; frees VRAM
exec llama-server \
    --model "$GGUF_PATH" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --ctx-size "$CTX" \
    --parallel "$N_PARALLEL" \
    --batch-size 2048 \
    --ubatch-size "$UBATCH" \
    --n-gpu-layers 99 \
    --flash-attn on \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --jinja \
    --reasoning off \
    --threads "$THREADS" \
    --cache-reuse "$CACHE_REUSE" \
    --cont-batching
