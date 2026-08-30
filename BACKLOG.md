# Backlog — Arm-B on SeaWulf

Ordered by priority, not by area. Each item says **why** it matters, because a
backlog that only says *what* gets re-litigated by whoever reads it next.

Frozen decisions live in `docs/SOTA-CONFIG-MEMO.md` and `CLAUDE.md` — check
there before proposing anything that looks like an open axis. Reasoning ON
(budget 256) and temperature 1.0 are **not** open axes.

Last updated 2026-08-30.

---

## P0 — in flight

### 1. Model-scale pilot (job 80089)
26B-A4B on GPU 0, E4B on GPU 1, 15 runs each (seeds 1–5 × 3 pools), paired by
seed against `armb_sota_r256_t10`. Running ~31 min, both cells clean at 0
failures. Preflight already answered the biggest unknown: **llama.cpp b10644
does support Gemma 4 MoE** (26.9 / 32.8 GB VRAM at 4 slots).

Report in this order when it lands — the first item is the go/no-go, and a
model that scores better on benchmarks but emits worse constrained JSON is
worse for this task:

1. **failure rate** vs the ~1% reasoning-on E2B baseline
2. peak prompt+gen tokens vs the 4096/slot ceiling (E2B peaked at 3833 = 94%,
   and a larger model's more verbose thinking may not fit)
3. measured s/run/GPU vs the ~1080 estimate in `LARGER-MODEL-ASSESSMENT.md`
4. paired quality deltas — `make score CELLS="ms_26B_a4b ms_E4B"`
5. growth pattern across E2B → E4B → 26B-A4B — `make growth`

> Note the log looks stalled while running: Python's stdout is block-buffered
> through the sbatch's `sed` pipe. Use `make progress CELL=...`, which reads the
> telemetry jsonl directly and bypasses the pipe.

### 2. Commit the pilot cells
`ms_26B_a4b` and `ms_E4B` are deliberately untracked until the job finishes —
committing mid-write would capture torn files.

---

## P1 — do next

### 3. `make provenance CELL=x` and `make compare CELLS="a b"`
Config fingerprint per cell: assert every run in a cell shares one config, and
report what differs between two cells.

**Why this is first.** The most expensive failures on this project were not
crashes but *invalid comparisons that looked valid*. One cell was collected with
seed 8 at p6 and the other 29 at p24 and nothing flagged it. Worse, the data
cannot currently answer the question: `ablation_infra` was added partway
through, so for the `sl_*`, `cd_*` and `vo_*` families the independent variable
— slot count, lane count, cache-reuse, GPU model — exists **only in the
directory name and the sbatch script**. And the two slot curves sit at different
temperatures (`sl_slots_*` at 0.3, `cd_slots_*` at 1.0) and must never be
pooled. The memo keeps this straight; the trace files do not.

Start from `config` + `config.ablation_infra` in any final; see
`slurm/cell_status.py` for the cell-walking pattern.

### 4. `make doc-check`
Mechanically verify the checkable claims in `CLAUDE.md` and `docs/`: referenced
paths exist, documented `llama-server` flags appear in `--help`, and every env
var an sbatch sets is actually read somewhere in the code.

**Why.** Three doc-vs-code errors surfaced in a single day: gotcha 3 asserted
`--reasoning` does not exist (it does — `-rea, --reasoning [on|off|auto]`),
`make score`'s upstream defaults pointed at two directories that do not exist,
and `COLLECT_MAX_VRAM_MB` was documented behaviour that had never been
implemented. **A wrong gotcha is worse than no gotcha** — it makes the next
agent "fix" a working script.

Note `--help` needs a GPU allocation; the shim cannot run on a login node
(`libcuda.so.1` missing). Use `srun --jobid=<running> -n1 llama-server --help`.

### 5. Estimator regression test
Pin `slurm/scaling_law_report.py` against the 300-run cell, whose true runtime
is independently known and which the current estimator matches to 0.4%.

**Why.** Makespan was computed wrongly twice — once by comparing an unnormalised
2-GPU cell against 1-GPU cells ("1.95× slower"), once by measuring first-to-last
completion, which undercounts badly at small n ("12% faster"). Both produced
confident, plausible, wrong numbers. This is the failure class whose output
reaches the dissertation, and one assertion prevents its return.

---

## P2 — should do

### 6. `docs/DECISIONS.md`
One file: locked decisions, dates, rationale. Currently scattered across four
docs. Reasoning ON; temperature 1.0; Q4 rejected; b128 rejected; 31B out of
scope; zip strictly dominated; `SOC595` not ours to use; traces stay in
backed-up home. A fresh agent should learn in one read what **not** to reopen.

### 7. Decide the compression question — it may already be answered
`STORAGE-POLICY.md` §7 plans Tier-1 per-file gzip (23×), blocked on adding
`.gz` support to six reader scripts. But git delta-compresses the same files at
**42×** (measured: 74.8 MB → 1.8 MB; whole repo 3.2 GB → 136 MB) while leaving
working copies as plain JSON that every reader already handles.

So the reader work may be unnecessary for anything committed. **Decide before
building it.** `make gz-check` currently blocks compression by design.

### 8. Backfill provenance into the older cells
Reconstruct `ablation_infra` for `sl_*`, `cd_*`, `vo_*` from the sbatch scripts
and directory names into a sidecar manifest per cell. Makes item 3 able to
answer for the whole corpus rather than only the newer cells. Do **not** rewrite
the trace files — write a sidecar.

### 9. `make smoke-gpu`
Short interactive allocation → start server → one round → tear down. Today the
only way to discover the stack is broken is to submit a 23-hour job.

### 10. `make weights-check`
Size + GGUF magic + recorded sha256 for every model. A job once launched against
a 1.4 GB fragment of a 25 GB file. `make pilot` gates on bytes now, but only for
the pilot and only for the 26B.

---

## P3 — decide, or someday

### 11. Repo visibility, then push  *(deferred by IU — not urgent)*
Branches cannot be private; GitHub visibility is per-repository. Either make
`arm-b-inference-bundle` private, or split public-code / private-data. Nothing
has been pushed. For eventual citation the traces want a DOI archive (Zenodo,
OSF, or SBU institutional) — GitHub is working history, not an archive.

### 12. Dead and stalled artifacts
- `llm/openrouter_adapter.py` — complete but unused; Arm C dropped when the
  project went local-inference-only. Remove or keep as a record.
- `abl_off_t10` — 0 finals, reasoning-off, abandoned when reasoning was locked.
  **Do not resume.** Delete or keep as a record of the closed axis.
- `lanesweep_solo` — 0 finals, stalled.

### 13. Rename `llm/types.py`
It shadows the stdlib `types` module and kills the interpreter during startup
imports whenever a script inside `llm/` is run by path. `PYTHONSAFEPATH=1` is a
workaround applied everywhere; renaming the module is the actual fix. Touches
every importer, so it needs a quiet moment and a full `make test`.

### 14. Untested levers from `LARGER-MODEL-ASSESSMENT.md`
- **Speculative decoding via the shipped MTP draft heads.** Sprint-16 found
  n-gram speculation does not stack at high occupancy, but a trained MTP draft
  head is a different mechanism. Even 1.5× would materially change every
  projection in the memo.
- **`dflash-` variants** exist for 26B-A4B and 31B and are unidentified. Worth
  knowing before assuming plain Q8 is the right file.

### 15. DRI at model scale needs n ≫ 15
DRI is too noisy to resolve growth-pattern questions at pilot size (CV ~110%;
SE ~0.05–0.09 at n=15 versus ~0.015 at n=300). The 15-run cells produce
non-monotonic curves and contradictory horizons for cells differing only in
temperature — that is noise, not mechanism. **Meta-consensus and argumentative
diversity are the indicators that can resolve model scale at n=15** (SE
0.002–0.008). If a DRI conclusion is wanted across model scale, it needs a cell
an order of magnitude larger, which is a GPU-days decision, not a quick add.
