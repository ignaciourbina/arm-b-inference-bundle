# Sprint 15 — Ablation reliability

Run completion, failure phase, and cost per cell. `fail%` counts runs
the driver could not complete; the phase is where they died.

| Cell | Axis | ok/total | fail% | fail phases | mean LLM calls/run | tok/s | min/run |
|---|---|---|---|---|---|---|---|
| `frontier_e2b_b128` | other | 14/15 | 7% | REFLECT×1 | 588 | 70.4 | 14.1 |
| `frontier_e2bqat_b256` | other | 13/15 | 13% | REFLECT×2 | 591 | 94.5 | 14.3 |
| `newstack_e2b_r0715` | other | 15/15 | 0% | — | 605 | 69.8 | 18.4 |
| `newstack_e4b_qat` | other | 15/15 | 0% | — | 647 | 64.7 | 27.5 |
| `prompt_anti-repetition` | prompt | 15/15 | 0% | — | 801 | 68.4 | 0.6 |
| `prompt_explicit-tradeoff` | prompt | 13/15 | 13% | REFLECT×2 | 818 | 68.2 | 0.6 |
| `prompt_neutral-persona` | prompt | 15/15 | 0% | — | 790 | 69.0 | 0.0 |
| `prompt_neutral-stance` | prompt | 10/15 | 33% | REFLECT×5 | 815 | 70.6 | 0.9 |
| `prompt_no-overlay` | prompt | 11/15 | 27% | REFLECT×4 | 814 | 65.9 | 2.1 |
| `prompt_terse` | prompt | 15/15 | 0% | — | 816 | 68.9 | 0.0 |
| `quant_Q4_K_M` | quant | 15/15 | 0% | — | 766 | 95.7 | 2.3 |
| `reasoning_on_b256` | reasoning | 15/15 | 0% | — | 609 | 70.5 | 0.0 |
| `reasoning_on_b512` | reasoning | 15/15 | 0% | — | 765 | 70.6 | 0.0 |

## Headline

- Q4 cell failure rate: **0%**; Q8 cells (prompt+reasoning) mean: **8%**.
- Q4 failures concentrate in the **—** phase — the Sprint-14 gemma-4-E2B reflect-termination weakness, amplified by heavier quantization.
- Read the paired-delta table (`ablation_deltas.tex`) knowing the Q4
  cell rests on fewer completed pairs; its nulls are low-power, not
  evidence of no effect.
