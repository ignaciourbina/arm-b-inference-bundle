# Arm-B inference bundle — SeaWulf

Local LLM inference for *Simulating Open Democracy* (dissertation chapter,
Ignacio Urbina, Stony Brook). This bundle runs the **Arm-B** deliberation
collection: `AgenticLLMEngine` through the `TownHall` protocol, on the
Polis-grounded Seattle minimum-wage scenario, 3 composition pools, n=6, t=8.

## Workflow commands

`make help` lists everything. The Makefile encodes the invariants below so they
never have to be remembered: `PYTHONSAFEPATH=1`, `module load slurm`, the v100
one-job limit, and the compression prerequisite.

| Command | Does |
|---|---|
| `make doctor` | preflight every invariant a job depends on; **run before submitting** |
| `make status` | slurm queue + on-disk status of every cell |
| `make sota` · `make pilot` | submit collections — **refuses** if a v100 job already exists |
| `make progress CELL=x` · `make log` · `make gpu` | monitor a running job |
| `make validate CELL=x` · `make score CELLS="a b"` | validator / paired scoring |
| `make scaling-report` · `make growth` | the two analysis reports |
| `make quota` · `make du` · `make gz-check` | storage, per `docs/STORAGE-POLICY.md` |

`make compress` is **gated on `make gz-check`** and will refuse until the six
trace readers are `.gz`-aware. That gate is the policy, executable.

## Read these first

| Doc | What it settles |
|---|---|
| [`BACKLOG.md`](BACKLOG.md) | **What is outstanding and why**, P0–P3. Read before starting anything. |
| [`docs/SOTA-CONFIG-MEMO.md`](docs/SOTA-CONFIG-MEMO.md) | **The recommended config**, with paired evidence for quantization / reasoning budget / cache-reuse / slots, the measured wall-clock scaling law, and the round-1 freeze result. |
| [`docs/STORAGE-POLICY.md`](docs/STORAGE-POLICY.md) | Disk quotas and the placement rule. **Traces are irreplaceable → home. Weights/images → scratch.** Never invert. |
| [`docs/CLUSTER_POLICY_JOBS.md`](docs/CLUSTER_POLICY_JOBS.md) | Verbatim SeaWulf job limits. **`v100` is 1 node / 1 job per user** — 2 GPUs is the hard ceiling. |
| [`docs/LARGER-MODEL-ASSESSMENT.md`](docs/LARGER-MODEL-ASSESSMENT.md) | Which larger Gemma to test and why (MoE active-params argument). |
| [`REPRODUCE.md`](REPRODUCE.md) | The bundle's own reproduction instructions. |
| [`llm/SCALING-FINDINGS.md`](llm/SCALING-FINDINGS.md) | Upstream throughput guidance. Wins over any contradicting code comment. |
| [`docs/CLAUDE-inherited-parent-repo.md`](docs/CLAUDE-inherited-parent-repo.md) | The parent repo's CLAUDE.md. Most of its "Hot start" points at files absent here. |

## Environment (non-obvious, all required)

```bash
export PYTHONPATH=.:agora/src
export PYTHONSAFEPATH=1        # MANDATORY — see gotcha 1
agora/.venv/bin/python ...     # the venv is a micromamba env on scratch (symlink)
```

- **Model weights**: `~/.cache/llama-models` is a **symlink to
  `/gpfs/scratch/$USER/models`**, so the repo's hardcoded paths resolve with no
  weights in home.
- **llama-server**: `~/src/llama.cpp/build/bin/llama-server` is a **shim** that
  execs Singularity. CentOS 7 (glibc 2.17) cannot run modern llama.cpp natively.
- Tests: `pytest llm/tests/` — **76 pass**. Run after any code change.
- Commit only when asked. Never push without confirmation.

## Gotchas that have already cost time

1. **`llm/types.py` shadows the stdlib `types` module.** Running any script
   inside `llm/` *by path* puts `llm/` on `sys.path[0]` and the interpreter dies
   during startup imports. `PYTHONSAFEPATH=1` fixes it; `python -m llm.x` avoids
   it. This breaks `llm/Makefile`'s retry-loop target on any Python 3.11.
2. **CUDA arch must be `-real`, not PTX.** The V100 driver is 550 (CUDA 12.4);
   stock llama.cpp images are newer and give V100 only `70-virtual`, so PTX JIT
   fails with *"the provided PTX was compiled with an unsupported toolchain"*.
   Build with `CMAKE_CUDA_ARCHITECTURES=70-real` (or `60-real;70-real` to
   include P100). Builds live in `/gpfs/scratch/$USER/src/llama.cpp/build*`.
3. **Reasoning flags — verified against b10644 `--help`, 2026-08-30.** Both
   `-rea, --reasoning [on|off|auto]` (env `LLAMA_ARG_REASONING`) and
   `--reasoning-budget N` (`0` = off, `-1` = unrestricted) exist. Also present
   and unused: `--reasoning-effort`, `--reasoning-preserve`,
   `--reasoning-format`. `slurm/start_server_reasoning.sh` passes `--reasoning
   on --reasoning-budget 256` and that is correct — do not "simplify" it away.
   *(An earlier version of this file claimed `--reasoning` did not exist. It
   does. Check `--help` before trusting a flag claim — and note the shim cannot
   run on a login node, so `--help` needs a GPU allocation: `srun --jobid=<id>
   -n1 llama-server --help`.)*
4. **`skip-existing` scans `BETA_TRACE_ROOT`, i.e. the parent dir.** Anything
   left under `llm/traces/beta_local/` silently suppresses re-runs regardless of
   folder name. Quarantine outside the tree.
5. **`--resume` replays the checkpoint**, so a *deterministic* failure can never
   be cleared by the retry pass. Delete the checkpoint and run fresh. The resume
   guard now refuses to continue across a changed model / temperature /
   prompt-variant.
6. **`v100` allocates whole nodes.** `--gres=gpu:1` still consumes all 28 CPUs
   and strands the second GPU. Always request `gpu:2` and shard across both.
7. **`df` lies on GPFS** — it reports fileset size, not your quota. Use
   `myquota`.

## Current state

- **LOCKED (2026-08-30): reasoning ON, budget 256. Temperature 1.0.** These are
  fixed properties of Arm B, like the model and the scenario — not open axes.
  Never propose a reasoning-off arm; never resume `abl_off_t10`. The
  reasoning-off cells (`seawulf_test_l4`) are the historical evidence that
  closed the axis. Rationale: `docs/SOTA-CONFIG-MEMO.md`.
- **`armb_sota_r256_t10`**: 300/300, validator **PASS** — reasoning b256,
  temp 1.0, Q8_0, control prompt. This is the SoTA collection.
- Ablation cells under `llm/traces/collection/` (`abl_*`, `sl_*`, `cd_*`,
  `sota_*`) back the memo's four axes.
- `polis-analysis/output/` holds the Ising profiles, re-derived from raw Polis
  data and validated against the methods appendix (see `PROVENANCE.md` there).

## Domain (unchanged from the parent repo)

- Opinion is emergent, never assigned: `o_i = clip(sum(w·d)/|R|, -1, 1)`.
  Weights are in **raw vote space**; the engine applies direction itself, so
  direction-aligning them double-applies it.
- 3 pools in `llm/townhall/compositions.py`: `polarized_n6`, `symmetric_n6`,
  `three_clusters_n6`.
- **Headline claim**: the LLM engine trends toward epistemic integration
  (higher alignment, more meta-consensus, lower dispersion); the rule engine
  toward argumentative circulation.
- **DRI is very noisy** (CV ~110%, per-run range −0.27 to +0.81). n=15 cells
  cannot support DRI conclusions; they can support calls/run and diversity.
