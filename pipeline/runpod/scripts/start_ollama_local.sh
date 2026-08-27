#!/bin/bash
# Legacy local Ollama startup for RTX 2060 (8GB).
# Retained for comparison only; the active local path uses llama-server.
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_CONTEXT_LENGTH=4096
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MAX_QUEUE=1024
exec ollama serve
