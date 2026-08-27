# LLM Cognitive Engine for Agora Deliberation ABM

Replaces the mathematical cognitive engines (DeGroot, Bayesian, Bounded Confidence, etc.)
with an LLM-backed engine that reasons about arguments through natural-language deliberation.

## Architecture

The LLM operates as a **stochastic cognitive machine** — it doesn't approximate cognition
with equations, it *performs* cognition. Confirmation bias, open-mindedness, and group
dynamics are injected as personality framing in the system prompt, not applied as post-hoc
mathematical corrections.

```
Protocol Layer (agora/protocols.py)
    │
    ├─ voice()    ──→  AgenticLLMEngine  ──→  ToolCallHarness
    ├─ evaluate() ──→       │                     │
    └─ reflect()  ──→       │                     ├─ System prompt (repertoire, opinion, cognitive style)
                            │                     ├─ Tool calls: query_attacks, query_supports, submit_*
                            │                     └─ Agentic loop until terminal action
                            │
                            └─ Structural enforcement only: weight clamping [-1,1], valid cid
```

### Components

| File | Purpose |
|------|---------|
| `engine.py` | `AgenticLLMEngine(CognitiveEngine)` — drop-in for the protocol layer |
| `harness.py` | Multi-turn agentic loop with retry, nudge, and tool dispatch |
| `client.py` | Async llama-server client using the OpenAI-compatible chat API and structured tracing |
| `prompts.py` | System prompt builders with cognitive style injection |
| `tools.py` | Tool definitions: `query_attacks`, `query_supports`, `query_repertoire`, `update_weight`, `submit_voice`, `submit_influence`, `done_reflecting` |
| `scenario_loader.py` | Load scenarios from JSON or bridge to existing agora scenarios |
| `townhall/runner.py` | Full town-hall deliberation runner with empirical agent profiles |
| `Makefile` | All automation targets (run from project root: `make -f llm/Makefile <target>`) |

### Design Principle

| Mechanism | LLM decides | Structure enforces |
|-----------|-------------|--------------------|
| Voice selection | Which cid to voice | Fallback if invalid |
| Influence scoring | 0-100 persuasiveness score | Integer range 0-100 |
| Weight updates | Which weights, new values | Clamp to [-1, 1] |
| Confirmation bias | Prompt-injected personality | Nothing |
| Graph reasoning | Queries attack/support graph via tools | Nothing |

---

## IO Paths

### Inputs

| Source | Path | What it provides |
|--------|------|-----------------|
| Scenarios (JSON) | `llm/scenarios/*.json` | Consideration pools for local smoke tests |
| Scenarios (agora) | `agora/src/agora/scenarios.py` | Hand-crafted empirical pools |
| Empirical profiles | `polis-analysis/output/ising_profiles.json` | Real participant data |
| IRT theta scores | `polis-analysis/output/irt_ising_theta.json` | Agent initial beliefs |

### Outputs (canonical location: `outputs/llm_engine/` at project root)

| Output | Path | Notes |
|--------|------|-------|
| Townhall results JSON | `outputs/llm_engine/<run_tag>_<ts>.json` | One file per run |
| Live trace (streaming) | `outputs/llm_engine/townhall_<topic>_trace_live.json` | Overwritten each run |
| Final trace JSON | `outputs/llm_engine/townhall_<topic><tag>_trace_<ts>.json` | Complete call log |
| Checkpoint JSON | `outputs/llm_engine/townhall_<topic>_checkpoint.json` | Auto-resume state |
| Beta sweep results | `outputs/llm_engine/beta_local/<run_tag_prefix>/` | Structured by tag |
| Retry-loop logs | `outputs/llm_engine/logs/` | Supervisor logs |

> **Note:** Old runs (pre-standardization) wrote to `llm/traces/`. New LLM runs write
> to `outputs/llm_engine/`. Do not mix them in analysis.
> **Exception:** the rule-based comparable arm (`townhall/rule_based_runner.py`,
> driven by `pipeline/run_comparable_rule_based_n6_t8.py`) writes to
> `llm/traces/rule_based/` — 1200 regenerated traces as of Sprint 12 (2026-07-01).

---

## Quick Start

```bash
# From project root, on a fresh clone:
bash llm/setup.sh    # ensures deps in the canonical venv (agora/.venv),
                     # verifies imports, checks llama-server at :20434

# Launch the local backend (RTX 2060, canonical flags: parallel=1, ctx=4096,
# jinja tool-calling, reasoning off, flash-attn, q8 KV cache):
bash pipeline/runpod/scripts/start_llama_server_local.sh

# Or manage deps manually (canonical venv, per CLAUDE.md):
agora/.venv/bin/pip install aiohttp numpy scipy pytest mypy ruff
```

## Usage

```bash
make -f llm/Makefile help            # show all targets
make -f llm/Makefile health          # check server + GPU status
make -f llm/Makefile test            # minimal 2-agent, 1-round test
make -f llm/Makefile trace-view      # inspect latest trace
make -f llm/Makefile trace-view-agentic  # inspect latest agentic trace

# Smoke test (6 agents, 2 rounds, ~4 min on GPU)
make -f llm/Makefile townhall-smoke

# Full beta sweep (20 seeds × 3 compositions × 8 rounds, ~55 min on RTX 2060)
make -f llm/Makefile townhall-beta-local-sweep

# Cloud-backed sweep (sends inference to tunneled H100 pod)
make -f llm/Makefile townhall-beta-cloud-sweep

# Supervised retry loop (restarts on failure, auto-resumes from checkpoints)
make -f llm/Makefile townhall-beta-local-retry-loop
```

### From the experiment runner

```python
from agora.experiment import ExperimentConfig, Runner

config = ExperimentConfig(
    name="llm_test",
    seed=42,
    engine="llm",
    llm_model="gemma-4-E2B-it-Q8_0.gguf",
    llm_base_url="http://localhost:20434",
    n_agents=6,
    n_rounds=5,
    confirmation_bias=0.3,
)
runner = Runner()
result = runner.run(config)
```

---

## Inference backend

**We use llama-server (llama.cpp), not Ollama.** llama-server exposes both
`/api/chat` (Ollama-compatible) and `/v1/chat/completions` (OpenAI-compatible).
**Use the OpenAI path exclusively.**

The Ollama path has a null-content bug: when an assistant message is re-submitted
in a multi-turn tool-calling loop, llama-server rejects it with
`"All non-assistant messages must contain 'content'"` (400). Every
voice/evaluate/reflect call fails on retry 1 with this path.

Set in `client.py`:
```python
DEFAULT_API_FLAVOR = "openai"                      # -> /v1/chat/completions
DEFAULT_MODEL = "gemma-4-E2B-it-Q8_0.gguf"        # full filename, not Ollama tag
```

### Endpoint contract

| Environment | URL | Model identifier |
|-------------|-----|-----------------|
| Local GPU | `http://localhost:20434` | `gemma-4-E2B-it-Q8_0.gguf` |
| Cloud pod tunnel | `http://localhost:21435` | `gemma-4-E2B-it-Q8_0.gguf` |

The model identifier must match the served GGUF filename **exactly**. The Ollama
tag format (`gemma4:e2b`) only works with `/api/chat`, not `/v1/chat/completions`.

### Pod / tunnel gotchas

- The local checkout may contain `llm/gemma4-e2b-q4km.gguf` (old q4 artifact).
  On the pod, use the q8 model at `/workspace/models/gemma-4-E2B-it-Q8_0.gguf`.
- A pod can be `RUNNING` while `llama-server` is still down — restart the backend
  and re-check `/v1/models` before opening the tunnel.
- If `tunnel-up` says backend is unreachable even though the server is up:
  ```bash
  REMOTE_GGUF_NAME='gemma-4-E2B-it-Q8_0.gguf' BACKEND=llama ./pipeline/runpod/mpsa tunnel-up
  ```

### GPU discipline (local RTX 2060, 8 GB VRAM)

One llama-server process at a time. `intra_parallel=1` in all local configs.
Local runs are smoke tests and prompt inspection only. Data collection happens on
the pod (H100). Keep local TOML configs at N=3, T=3.

| Mode | Time/call | 275-call run |
|------|-----------|-------------|
| CPU (no CUDA) | ~130s | ~10 hours |
| CUDA, no thinking | ~0.8s | ~4 min |
| CUDA, thinking (1024 tok) | ~12s | ~55 min |

---

## Checkpoints and re-runs

The runner auto-resumes from checkpoints. If you change prompts and re-run,
you will **silently get the old results** unless you delete checkpoints first:

```bash
rm outputs/llm_engine/townhall_*_checkpoint.json
```

Checkpoints are keyed by run tag. A fresh sweep with the same tag will resume
instead of re-running. There is no `--force` flag.

---

## Traces

Every LLM call is logged to structured JSON. Each trace file contains:

```json
{
  "model": "gemma-4-E2B-it-Q8_0.gguf",
  "n_calls": 275,
  "total_elapsed_s": 547.0,
  "calls": [
    {
      "seq": 1,
      "hook": "voice",
      "agent_id": "A_00",
      "prompt": "...",
      "response_raw": "...",
      "response_parsed": {"tool_calls": [...]},
      "prompt_tokens": 326,
      "gen_tokens": 42,
      "elapsed_s": 1.2
    }
  ]
}
```

---

## Scenarios

### Built-in JSON

`llm/scenarios/carbon_tax.json` — 10 considerations (5 pro, 5 con), 8 attack edges, 2 support edges.
`llm/scenarios/minimum_wage_seattle_crossover.json` — default smoke scenario.

### Agora hand-crafted

```python
from llm.scenario_loader import from_agora_scenario
pool = from_agora_scenario("barabas_consensual")         # 15 considerations
pool = from_agora_scenario("barabas_non_consensual")     # 15 considerations
pool = from_agora_scenario("jackman_sniderman_symmetric")  # 14 considerations
```

---

## Agent composition taxonomy

Agents are drawn from a 3×3 grid:
- **Policy axis** (rows): Pro, Ambivalent, Con — overall opinion direction
- **Coherence axis** (cols): Coherent, Mixed, Ambivalent — precision of the argument set

Active compositions for the MPSA 2026 design:

| Composition | N | Description |
|-------------|---|-------------|
| `symmetric` | 10 | Balanced cells across the grid |
| `polarized` | 10 | Pro/Coherent vs Con/Coherent anchors with Ambivalent/Mixed center |
| `symmetric_n6` | 6 | Local smoke preset, symmetric |
| `polarized_n6` | 6 | Local smoke preset, polarized |
| `three_clusters_n6` | 6 | 2 Pro (Coherent, Mixed), 2 Con (Coherent, Mixed), 2 Ambivalent |

Agents start with only the considerations they actually voted on in the Polis survey
(mean ≈ 5.4 of 10). New considerations enter via `update_weight` during reflect —
detected in traces as `"_adopted": true`.

---

## Prompt architecture

There are two prompt families: `build_neutral_*` and `build_baseline_*`. Cognitive
differentiation comes from **epistemic stance injection** (how the agent processes
information), not from scripted rules in the task block. The persona prompt injects:

> "You value coherence in your beliefs — your views reflect real experience and
> reflection. When you encounter a new argument, you naturally notice whether it
> fits with what you already believe…"

This delegates resistance to the considerations vector structure, not to an explicit
instruction to resist. Do not add scripted resistance back to the task block — it is
tautological (can't distinguish framing from instruction).

The `reason` field was removed from the `UPDATE_WEIGHT` tool schema because the model
generates reason strings with unescaped double-quotes, breaking JSON serialization
(500 errors). Do not re-add it.

---

## TOML config and CLI precedence

`run_llm_sweep.py` loads TOML configs via `--config`. CLI args override TOML.
When no CLI arg is given, use the resolved variable (e.g., `conditions`, not
`args.conditions`) downstream — including in metrics output. If you add new output
sections, use the resolved variables or you'll get `TypeError: 'NoneType' is not
iterable` when config came from TOML.

---

## Model

**Gemma 4 E2B** (Google, April 2026) — serves via the q8 GGUF identifier
`gemma-4-E2B-it-Q8_0.gguf` on the OpenAI-compatible endpoint.

- Thinking mode available (bounded to 1024 tokens for tool-calling reliability)
- The model identifier must be the full `.gguf` filename, not an Ollama tag

---

## AWS Deployment (legacy)

Legacy deployment helper scripts remain in `llm/deploy/` but have not been
reconciled with the current llama-server contract. Treat them as historical.

```bash
make -f llm/Makefile aws-launch      # launch g4dn.xlarge (T4, ~$0.16/hr)
make -f llm/Makefile aws-ssh         # SSH into instance
make -f llm/Makefile aws-terminate   # shut down when done
```

Requires AWS CLI configured with GPU quota for `g4dn` instances in `us-east-1`.

## Sprint-15 ablation file family (at llm/ root, deliberately)

These stay at `llm/` root (not a subfolder) while the paused neutrality wave's
untracked state in `llm/traces/ablation/` references them; revisit after the
wave closes.

| File | Role |
|---|---|
| `prompt_variants.py` | Named prompt variants (control byte-identical to production; wave-1 tweaks + neutrality register levers) |
| `run_ablation_cell.py` | One cell = (variant × infra) over paired seeds; GPU guard with warmup |
| `score_ablation_cells.py` / `report_ablation.py` | Paired scoring vs control → LaTeX table + decision memo |
| `plot_ablation_deltas.py` / `plot_ablation_trajectories.py` / `analyze_ablation_reliability.py` | Analysis suite (forest plots, dynamics, failure phases) |
| `run_ablation_programme.sh` | Wave-1 programme (prompt/reasoning/quant cells + scoring) |
| `run_neutrality_ablation.sh` / `run_neutrality_r256_cell.sh` | Register-neutrality wave + reasoning-interaction cell |
| `run_newstack_validation.sh` | Refreshed-weights / coopmat2-binary / E4B validation |
| `run_ablation_analysis.sh` | Re-run the whole analysis suite on current scores |

## Scale-out collection (Sprint 16)

High-occupancy collection infrastructure — evidence in
`agora/analysis/sprint-15-ablation-prep/frontier-research-report.md` and the
sprint-16 plan. Real runs keep only ~2.7/6 slots busy; multiple concurrent run
lanes recover the idle capacity.

> ⚠ **Read `llm/SCALING-FINDINGS.md` before tuning concurrency.** The
> "~2.5 days / 434 tok/s at 24-way / ~3–5×" numbers are the short-prompt
> *probe*; the Sprint-16 P4a A/B on real runs measured **1.72× at 3 lanes,
> ~1.9–2.1× at 4 lanes** (per-run pace degrades ~1.69× under contention), so
> the 1200-run projection is **~6–7 days, not 2.5**. The lever is `--lanes`
> (real ceiling ~4; the tool defaults to 8 — override it), and `p12/49k` was
> retained over `p24/98k`. Keep `--reasoning off` for trace comparability.

```bash
# 1. server (24 slots x 4096 ctx, q8 KV, coopmat2 binary, refreshed weights)
bash llm/run_collection_server.sh 128        # or 256 (reasoning budget)

# 2. orchestrator: 8 lanes over the full grid, idempotent, pausable
python llm/run_collection_parallel.py --cell collection_main \
    --seeds 1-400 --lanes 8                  # 400 seeds x 3 comps = 1200 runs

# progress: llm/traces/logs/collection_main_progress.jsonl + status lines
# pause:    send SIGTERM (lanes finish in-flight runs, ~15 min bound)
# resume:   relaunch the same command (checkpoints replay in ~1s)
```

Gates built in: warmed bench >= 20 tok/s, VRAM < 6.5 GB, >= 10 GB disk free.
Pilot gate (sprint-16 exit): `llm/run_scaleout_pilot.sh` — >= 2.5x macro
speedup, zero failures, VRAM under ceiling.
