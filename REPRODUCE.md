# Arm-B LLM Inference — Reproduction Bundle

Self-contained, lightweight snapshot of everything needed to reproduce the
**Arm-B LLM inference collection** of *Simulating Open Democracy* on a fresh
machine. This is a curated export (no git history) of the parent research repo;
data, model weights, traces, and the analysis pipeline are **not** included —
only the code, scripts, docs, and infrastructure to *run the inference*.

**Target output:** 390 deliberation traces = **130 seeds × 3 composition pools**
(`polarized_n6`, `symmetric_n6`, `three_clusters_n6`), n=6 agents, t=8 rounds,
condition=baseline, produced by `AgenticLLMEngine` driven through the `TownHall`
protocol against a local `llama-server` backend.

---

## 1. What's in this bundle

| Path | Role |
|------|------|
| `llm/` | The inference engine: `engine.py` (`AgenticLLMEngine`), `client.py`, `harness.py`, `prompts.py`, `tools.py`, `townhall/` (runner + `compositions.py`), `scenarios/minimum_wage_seattle_crossover.json`, all `run_*` collectors, `Makefile`, `setup.sh`, `tests/`, and docs. |
| `agora/src/agora/` | The engine package `llm` imports (`agents`, `considerations`, `engines`, `scenarios`). |
| `agora/pyproject.toml` | Dependency manifest / installable `agora` package. |
| `pipeline/runpod/scripts/start_llama_server_local.sh` | Canonical local backend launcher. |
| `pipeline/validate_llm_collection.py` | Post-collection validator (checks the 390-trace set). |

**Deliberately excluded** (not needed to reproduce inference, or unsafe to
publish): `llm/traces/` (multi-GB outputs), `agora/output/`, `.venv/`,
`llm/.env` (holds a private key — see §2), the cloud pod-management scripts, the
downstream analysis/figure pipeline, datasets, literature, and manuscript.

---

## 2. Prerequisites (must be provided on the target machine)

These are **documented, not bundled** — fetch/build them locally.

### 2a. Model weights
- **File:** `gemma-4-E2B-it-Q8_0.gguf` (the **r0715** refreshed E2B Q8_0 weights).
- **Place at:** `~/.cache/llama-models/gemma-4-E2B-it-GGUF-r0715/gemma-4-E2B-it-Q8_0.gguf`
  (the path `llm/run_collection_server.sh` expects; override `MODEL=` to relocate).
- The model identifier used on the OpenAI-compatible API is the **full GGUF
  filename**: `gemma-4-E2B-it-Q8_0.gguf`.

### 2b. Inference runtime — `llama.cpp`
- Build `llama-server` from `llama.cpp` with the **coopmat2** source binary
  (the config the throughput numbers were measured against). A standard
  CUDA/Vulkan build works for correctness; coopmat2 is the throughput-tuned one.
- Expected binary path: `~/src/llama.cpp/build/bin/llama-server`
  (override `BIN=` in the launch scripts).

### 2c. Python environment
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install aiohttp numpy scipy pytest        # runtime + tests
pip install -e agora                          # installs the `agora` engine package
```
All entry points expect `agora` importable and the repo root on `PYTHONPATH`:
```bash
export PYTHONPATH="$PWD:$PWD/agora/src"
```
(`.env.example` shows the one optional secret, `RUNPOD_AI_KEY`, only needed for
the cloud backend path — irrelevant to local reproduction.)

---

## 3. Reproduce the collection

```bash
# 0. env (see §2c) — venv active, PYTHONPATH set

# 1. sanity: offline unit tests (no server needed)
pytest llm/tests/            # ~41 tests

# 2. start the backend (24-slot scale-out config, q8 KV, reasoning b128)
bash llm/run_collection_server.sh 128        # serves :20434

#    (8GB-card single-slot alternative:)
#    N_PARALLEL=6 CTX=24576 CACHE_REUSE=256 \
#      bash pipeline/runpod/scripts/start_llama_server_local.sh

# 3. verify the server
curl -sf localhost:20434/v1/models

# 4. run the 390-trace Arm-B sweep (resumable; keys off a fixed run-tag prefix)
bash llm/run_arm_b_local_collection.sh sweep
#    subset / pilot:  SEEDS="1" bash llm/run_arm_b_local_collection.sh sweep
#    progress:        bash llm/run_arm_b_local_collection.sh status

# 5. validate the collected set
python pipeline/validate_llm_collection.py
```

Traces land under `llm/traces/beta_local/arm_b_local_p6cache/`. Re-running
`sweep` is idempotent (skip-existing + checkpoint resume).

### Canonical server signature
```
--parallel 24 --ctx-size 98304 (4096/slot) --batch-size 2048 --ubatch-size 512
--n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0
--jinja --cont-batching --reasoning on --reasoning-budget 128
```
(See `llm/run_collection_server.sh` and `llm/LOCAL-INFERENCE-READINESS.md`.)

---

## 4. Notes & gotchas

- **`llm/townhall/data_loader.py`** references a `datasets/debate-gpt-x/` profile
  table. That is an **alternate agent-init path and is NOT used** by the 3-pool
  baseline collection (which is defined entirely by `llm/townhall/compositions.py`).
  The dataset is intentionally absent.
- The scenario (`llm/scenarios/minimum_wage_seattle_crossover.json`) is a
  provenance copy of the Polis Seattle $15/hour minimum-wage crossover scenario;
  it is self-contained.
- GPU discipline: **one `llama-server` at a time.** The scale-out throughput
  comes from running many concurrent *runs* against a single multi-slot server,
  not multiple servers.
- The rule-based comparison arm (Arm A) and all downstream trajectory/growth-model
  analysis live in the parent repo and are **out of scope** for this bundle.

---

*Generated as a reproduction export of `simulating-open-democracy` (Arm-B LLM
inference scope). Dissertation chapter — Ignacio Urbina, Stony Brook.*
