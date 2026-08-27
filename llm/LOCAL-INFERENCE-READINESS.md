# Local GPU Inference Scaffold — Readiness Scorecard

> **Status: READY — validated 2026-07-02 on the RTX 2060 SUPER (8GB), all
> checks PASS.** The local scaffold's role (per GPU discipline): smoke tests
> and prompt inspection. Data collection happens on the pod (H100) via the
> tunnel — see `ops/arm-b-rerun-runbook.md`.

## Scorecard

| # | Check | Result | Evidence |
|---|---|---|---|
| V1 | Environment setup (`bash llm/setup.sh`) | **PASS** | deps in canonical `agora/.venv`; imports OK; backend reachable |
| V2 | Unit tests (no server needed) | **PASS** | 41/41 in 0.8s (`pytest llm/tests/` from project root) |
| V3 | `make -f llm/Makefile health` | **PASS** | model listed; GPU 2.8/8.2 GB |
| V4 | Tool-call round-trip on `/v1/chat/completions` | **PASS** | `finish_reason=tool_calls`, valid args JSON; multi-turn re-submission OK (no null-content 400 — the Ollama-path bug stays dead) |
| V5 | `make test` (2-agent engine round) | **PASS** | full voice→evaluate→reflect, 13 calls / 23.6s |
| V6 | `make townhall-smoke` (6 empirical agents, 2 rounds) | **PASS** | 172 calls / 185s (~3.2 min), real opinion movement, results+trace in `outputs/llm_engine/` |
| V7 | Checkpoint/resume | **PASS** | same-tag `SMOKE_RESUME=true` rerun short-circuits in **0.9s**, resumed content identical to original |
| V8 | Server discipline conformance | **PASS** | running flags = canonical script: `--parallel 1 --ctx-size 4096 --jinja --reasoning off --flash-attn on --cache-type-k/v q8_0` |

**Latency benchmark (canonical config, Vulkan, Q8):** median 0.98s/call,
mean 1.08s, p95 1.54s, max 13.8s (first call = warm-up). 275-call runs ≈ 5 min.

**Out of scope here:** the cloud-tunnel path (`:21435`) — requires the pod;
validated as part of Arm-B execution (runbook pre-flight).

## Defects found and fixed in this polish pass (2026-07-02)

1. **`llm/Makefile` was fully broken** — `PYTHON := llm/.venv/bin/python`
   pointed at a venv that doesn't exist. Now auto-falls back to the canonical
   `agora/.venv` (honors `llm/.venv` if someone creates one; `PYTHON=...`
   overrides).
2. **`llm/setup.sh`** created a stray second venv and health-checked the old
   Ollama port `11434`. Now installs into `agora/.venv` and checks `:20434`.
3. **`llm/tests/run_thinking.sh`** — same stale `11434` port, fixed.
4. **`llm/scenarios/` was missing entirely** (lost in the April incident) —
   the Makefile smoke target, `polis_scenario.py` (writes there), and five
   pipeline scripts all expect it. Restored with a provenance copy of the
   canonical scenario (`sha256 c01e52d1…` — byte-identical to
   `polis-analysis/output/minimum_wage_seattle_crossover.json`; re-copy if the
   canonical ever regenerates).
5. **Stale `pipeline/polis-analysis/` profile paths** (pre-recovery layout)
   in the Makefile smoke knobs, `run_empirical_argbased.py`,
   `run_local_llama_queue.py`, `run_llm_sweep.py` → `polis-analysis/` (root).
6. **`refresh_trajectory_inventory.py` / `refresh_pooled_arm_growth_n6_t8.py`**
   hard-coded the nonexistent `llm/.venv` python → canonical fallback.
7. **The running llama-server violated the scaffold's own discipline**
   (`--parallel 4`, `--ctx-size 32768`, `--skip-chat-parsing`, no `--jinja`,
   no `--reasoning off`) — an ad-hoc launch. Replaced with the canonical
   script; VRAM headroom improved and tool-calling template handling
   (`--jinja`) restored.
8. **`llm/README.md` quick start** updated to the canonical venv + launch
   instructions.

## Canonical local workflow (verified end-to-end)

```bash
# 1. one-time / after pulls
bash llm/setup.sh

# 2. launch the backend (canonical flags for the 8GB card)
bash pipeline/runpod/scripts/start_llama_server_local.sh   # :20434

# 3. verify
make -f llm/Makefile health
make -f llm/Makefile test              # 2-agent engine round, ~25s

# 4. real smoke (6 empirical agents, 2 rounds, ~3-4 min)
make -f llm/Makefile townhall-smoke

# checkpoints: same run tag resumes silently — delete
#   outputs/llm_engine/townhall_*_checkpoint.json
# (or use a fresh SMOKE_RUN_TAG) to force a re-run.
```

Endpoint contract (unchanged): OpenAI path only (`/v1/chat/completions`),
model identifier is the full GGUF filename `gemma-4-E2B-it-Q8_0.gguf`,
local model file at `~/.cache/llama-models/gemma-4-E2B-it-GGUF/`.
GPU discipline: one llama-server at a time, `parallel=1` locally.

## Addendum 2026-07-20 — local Arm-B collection under the p6-cache signature

The role scoping above ("collection happens on the pod") is superseded:
the full Arm-B collection (390 runs = 130 seeds × 3 compositions, n=6, t=8)
runs **locally** under a pinned throughput config, IU-approved 2026-07-20.

**Collection server signature (supersedes V8's canonical-p1 conformance for
collection runs only):**

```
N_PARALLEL=6  CTX=24576 (= 4096 per slot)  CACHE_REUSE=256  UBATCH=512
+ canonical flags: --jinja --reasoning off --flash-attn on --cache-type-k/v q8_0
```

Evidence: `outputs/llm_engine/bench_local_config_cache/scoreboard.md`
(p6-cache: 48.9s macro round, 1.72 calls/s, 0 retries/nudges/fallbacks/
http-errors, ~2.9GB VRAM of 8GB). Client change riding along: 500-retries
now perturb the sampling seed and double the token budget (`llm/client.py`
`_perturb_retry`) — first attempts stay deterministic; only retries diverge.

Launch path (single source of truth): `llm/run_arm_b_local_collection.sh`
(`server` | `sweep` | `status`). Validator:
`pipeline/validate_llm_collection.py`. Pre-launch gauntlet results are
recorded in the session log (`ops/session-log.md`) for the collection date.
