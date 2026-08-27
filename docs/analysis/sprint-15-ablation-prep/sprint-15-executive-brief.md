# Sprint 15 — Ablation Executive Brief

**Date:** 2026-08-20 · **Status:** results current as of the refill pass (reasoning
cells final at 15/15; prompt/quant cells still filling — see Caveats).
**Purpose:** choose the configuration for extending Arm B from 390 → 1200 runs
(~65 GPU h), on evidence rather than default.

---

## Bottom line

**Turn reasoning ON with a 256-token budget for the 1200-run extension.** It is
the only intervention that moves the headline results, and it moves *all* of
them in the desired direction — more argument diversity, higher epistemic
alignment, less repetition — while **costing fewer LLM calls, not more.** Prompt
tweaks are near-null at this power; Q4 quantization degrades reliability and buys
nothing.

---

## What was tested

Three ablation axes against the validated production control (seeds 1–5 × 3
compositions, n=6 agents, t=8 rounds, paired by seed; paired *t*-tests):

| Axis | Cells |
|---|---|
| **Prompt** | `anti-repetition`, `explicit-tradeoff`, `terse` (each changes one hook) |
| **Reasoning** | `reasoning_on_b256`, `reasoning_on_b512` (gemma "thinking" on, bounded) |
| **Quantization** | `quant_Q4_K_M` (Q4 vs the production Q8_0) |

Control = seeds 1–5 of the 390-run production collection (identical config).

---

## Results by axis

### Reasoning — the signal (both cells complete, 15/15)

`reasoning_on_b256` vs control, all significant:

| Outcome | Δ | p | Reading |
|---|---:|---:|---|
| distinct args / agent | **+0.41** | .001 | agents voice more distinct considerations |
| distinct args / room | **+0.80** | .003 | more of the argument space surfaced |
| top-cid share | **−0.065** | .004 | less concentration on one point (less repetition) |
| DRI (final) | **+0.19** | .019 | higher epistemic alignment |
| LLM calls / run | **−192** | <.001 | **cheaper** — reflect terminates cleanly, fewer retries |

`reasoning_on_b512` shows the same directions but weaker (distinct/agent +0.36,
top-cid −0.069, meta-consensus +0.014, calls −36). **The smaller 256 budget wins**
— the larger budget gains nothing and risks the 4096-token per-slot context.

The −192 calls/run is the surprise: reasoning does not just improve quality, it
*reduces* cost, because the thinking pass lets the model complete the REFLECT
phase in one shot instead of looping through failed tool-call retries.

### Prompt — near-null at this power

Only two marginal effects survive: `terse` raises distinct-args/agent (+0.27,
p=.039) and `explicit-tradeoff` nudges DRI (+0.16, p=.045). `anti-repetition`
shows nothing significant. At 5 seeds/composition these are weak evidence; the
prompt hooks are not where the leverage is.

### Quantization — degrades reliability, no measurable upside

`quant_Q4_K_M` produced **no significant deltas** — but that null is low-power,
not reassurance: the Q4 cell **failed 40% of its runs, every failure in the
REFLECT phase**, vs 0–20% for the Q8 cells. This is the pre-registered
quantization-fragility caveat, confirmed: heavier quantization breaks tool-call/
reflect termination in exactly the model's known weak spot. **Do not quantize to
Q4 for the main run.**

---

## Reliability (the failure finding)

| Cell | ok/total | fail % | failure phase |
|---|---|---:|---|
| reasoning_on_b256 | 15/15 | 0% | — |
| reasoning_on_b512 | 15/15 | 0% | — |
| prompt_terse | 15/15 | 0% | — |
| prompt_anti-repetition | 14/15 | ~7% | — |
| prompt_explicit-tradeoff | 12/15 | 20% | REFLECT |
| **quant_Q4_K_M** | **9/15** | **40%** | **REFLECT ×6** |

Failures are not random — they concentrate in REFLECT, the gemma-4-E2B
termination weakness, and Q4 amplifies it most.

---

## Recommendation for the 1200-run extension

1. **Reasoning: ON, budget 256.** Strongest and most consistent gains, and it
   lowers per-run cost — materially relevant at 1200 runs / ~65 GPU h.
2. **Prompt: keep the production (control) prompt.** No variant clears the bar
   convincingly; changing it would add an uncontrolled factor for no measured gain.
3. **Quantization: stay on Q8_0.** Q4 costs 40% of runs with no upside.

Net config: **production prompt + reasoning-on-256 + Q8_0.**

---

## Caveats

- **Power.** 5 seeds/composition detects large paired effects and nothing
  subtler. Prompt/quant nulls mean "not detected at this power," not "absent."
- **Refill in progress.** Reasoning cells are final (15/15); `explicit-tradeoff`
  (12/15) and `quant` (9/15) are still filling. The reasoning conclusion is
  stable; the prompt/quant numbers may firm up slightly. Re-run
  `bash llm/run_ablation_analysis.sh` after P5 re-scores.
- **Q4 provenance.** The Q8_0 is from `ggml-org/gemma-4-E2B-it-GGUF`; the Q4_K_M
  is from `unsloth/gemma-4-E2B-it-GGUF` (same base `google/gemma-4-e2b-it`, but a
  different quantizer). The reliability finding is robust to this; a fair Q4
  *quality* comparison would want the same quantizer.
- **One machine, one session.** Any driver/serving change between a cell and the
  main run invalidates the pairing.

---

## Artifacts

- Paired deltas (stats): `agora/analysis/sprint-15-ablation-prep/ablation-scores.json`
- Delta table (LaTeX): `pipeline/output/reports/ablation/ablation_deltas.tex`
- Delta forest plots: `pipeline/output/reports/ablation/ablation_deltas.pdf`
- Reliability: `.../ablation-reliability.md` + `ablation_reliability.pdf`
- Trajectories: `pipeline/output/reports/ablation/ablation_trajectories.pdf`
- Full decision memo (auto-filled numbers): `sprint-15-decision-memo.md`
- Regenerate everything: `bash llm/run_ablation_analysis.sh`
