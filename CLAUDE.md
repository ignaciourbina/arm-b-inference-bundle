# Project: Simulating Open Democracy

ABM of deliberative democracy. Dissertation chapter (Ignacio Urbina, Stony Brook).

## Hot start

0. **The paper's empirical claim is the LLM-vs-rule engine comparison**
   (`AgenticLLMEngine` vs `EmpiricalArgumentEngine`, `TownHall` protocol, Polis-
   grounded Seattle minimum-wage scenario, 3 composition pools). The classic
   five-cognitive-engine × five-protocol calibration scorecard
   (`benchmark-registry.json`, Sprints 04–11/13) was first-iteration grounding
   for the rule-based engine, not the paper's current results — don't let
   scorecard regressions there outrank Sprint-12/Arm-B work. See
   "What this project is" in `INVENTORY.md` for the full framing.
1. Read `INVENTORY.md` for full project state, scorecard, sprint history, and open items.
2. Read the latest sprint decision memo in `agora/analysis/sprint-{NN}-*/sprint-*-decision-memo.md`.
3. The formal model spec is `manuscript/theoretical-appendix-v3.tex`, not the code (v3 = v2 + the Sprint-11 nine-decision refactor; v2 and v1 are superseded).
4. For post-data-loss recovery state (recovered / still missing), read `RECOVERY-STATUS.md`.

## Non-obvious

- **Repository layout re-tiered 2026-08-26** (mpsa-prep → `pipeline/`; frozen tier
  consolidated under `archive/`) — see `INVENTORY.md` "Repository layout —
  canonical locations" for the 5-tier convention (trackers / pipelines /
  supporting / frozen / pending-cleanup). Rule of thumb: a pipeline's own
  tooling stays inside that pipeline; cross-cutting docs go to root `ops/`,
  never nested inside a pipeline folder. Before creating a new top-level
  folder or a cross-cutting doc, check that table first.

- The fast path is `agora/analysis/jit_engines.py`, not the production `src/agora/`. The sweep uses JIT by default.
- `run_factorial.py` lives at `agora/` root, not in `analysis/`.
- Sprint folders with plans and memos live in `agora/analysis/sprint-{NN}-{name}/`.
- Sprints 01-04 predate the sprint-folder convention; their history is only in git.
- Benchmark targets are in `agora/analysis/benchmark-registry.json` (not code).
- The venv is `agora/.venv/`, activate with `source agora/.venv/bin/activate`.
- `weight_correlation` was renamed to `repertoire_correlation` in Sprint 08.
- The paper's analysis pipeline is `pipeline/` (repo root; **renamed from `mpsa-prep/` 2026-08-26**, recovered 2026-06-30) — trajectory analysis, growth models, figures, validators, sweep configs, RunPod ops scripts (`pipeline/runpod/mpsa`). Historical docs still say `mpsa-prep/`.
- `archive/recovery/recovery-audit/` holds the tmp2/ recovery audit trail (disposition manifests + the 120-file dangling-blobs inventory). `RECOVERY-STATUS.md` is the top-level summary.
- Build the manuscript with **XeLaTeX/LuaLaTeX** (not pdflatex): `manuscript/preamble.tex` loads fontspec/MinionPro and fails loud under pdflatex.
- **`manuscript/final-manuscript.tex` is AUTO-GENERATED** — edit the section files, then `python manuscript/merge_manuscript.py` + xelatex. Never edit canonical directly (the next merge clobbers it); `split_manuscript.py` is the inverse if canonical ever drifts. The "theoretical appendix" merge source is `theoretical-appendix-manuscript.tex` (condensed), NOT the v3 spec.
- Trajectory-metric opinion is pool-normalized: `o_i = clip(sum(w·d)/M, -1, 1)` (`eq:v3-reported-opinion`, adopted 2026-07-02); the agents' internal opinion stays the repertoire mean. The pooled growth panel starts at round 1 (round 0 dropped for log-round).
- `tmp2-recovered/` (still at root, untracked, 0 tracked files) is a spent staging dump; its `dangling-blobs/` holds 13 unique preserve-backup blobs (1.2GB→303MB, sha-verified) pending external rsync backup — run `archive/recovery/recovery-audit/backup-preserve-blobs.sh <dest>`, verify, then move the dir to `archive/recovery/` (gate in TODO.md).
- The lost rule-based traces are regenerable: `pipeline/run_comparable_rule_based_n6_t8.py` rebuilds all 1200 comparable-rule traces into `llm/traces/rule_based/` in ~17s (Sprint 12). The 390 LLM traces still require GPU re-run.
- `meta_consensus_agreement` reads the co-vote-imputed `Agent.salience_prior` (dense `full_weights`) when present; opinion/variance/plurality/DRI read the sparse live repertoire (split representation, Sprint 12).

## Workflow preferences

- Use plan mode for anything non-trivial. Save plans to sprint folders on disk.
- When launching parallel agents, prefer Opus with extended thinking for research tasks.
- Sprint convention: `plan.md` at start, `sprint-{NN}-decision-memo.md` at close.
- Always run `pytest tests/ -x -q` after code changes. Baseline: 433 agora tests (+ 41 in `llm/tests/`, run from project root).
- For sweeps, use `--engine-path jit` (default). Production path is `--engine-path production`.
- Commit only when asked. Never push without confirmation.

## Domain

- Opinion `o_i = clip(sum(w * d) / |R|, -1, 1)` is emergent from weights, never assigned (agents' internal/deliberating opinion; the metric-facing "reported" opinion used in trajectory analysis is pool-normalized, see `eq:v3-reported-opinion` above).
- DRI (Deliberative Reasoning Index) is the primary calibration metric for the background scorecard; in the paper's current empirical section the analogous metric is `dri_orthogonal` / "epistemic alignment."
- `repertoire_correlation` (rho) controls MVN direction-biased repertoire init. rho=0 is uniform (backward compat), rho>0 creates pro/con typed agents with U(0,1) endorsement weights.
- Causal conditioning: `group_composition` is only causal in citizens_assembly and committee_plenary. `facilitation` is overridden in town_hall.

### Paper-current empirical design (Arm A / Arm B)

- **Arm A (rule):** `EmpiricalArgumentEngine` — graph-structured updating,
  calibrated to the Polis Seattle $15/hour minimum-wage conversation, run
  through the `TownHall` protocol.
- **Arm B (LLM):** `AgenticLLMEngine` — tool-constrained LLM cognition, same
  scenario/protocol.
- **3 composition pools:** `polarized_n6`, `symmetric_n6`, `three_clusters_n6`
  (n=6, t=8 rounds), defined in `llm/townhall/compositions.py`.
- **Headline result:** LLM engine trends toward epistemic integration (higher
  alignment, more meta-consensus, lower dispersion); rule engine trends toward
  argumentative circulation (more voiced diversity, cross-camp uptake, less
  consolidation). See `manuscript/results-section-outline.tex`.
