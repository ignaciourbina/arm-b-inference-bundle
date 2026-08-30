# Which larger Gemma to assess on this task — research prep

**Date:** 2026-08-30. Scope: pick the next model to run the Arm-B deliberation
task against, on SeaWulf V100s, local inference only.

**Bottom line: `gemma-4-26B-A4B-it` (Q8_0, ggml-org).** It is the only candidate
that buys a large quality jump without a proportional wall-clock penalty,
because its MoE routing decouples the two. Second choice `E4B` as a cheap
sanity step; `12B` is strictly dominated; `31B` does not fit.

---

## The Gemma 4 lineup

| Model | Total params | **Active params** | Q8_0 size | type |
|---|---:|---:|---:|---|
| **E2B** (current) | 5.0 B | **2.0 B** | 4.62 GB | MoE |
| E4B | 8.0 B | **4.0 B** | 7.47 GB | MoE |
| 12B | 12 B | **12 B** | 11.79 GB | dense |
| **26B-A4B** | 26 B | **3.8 B** | 25.01 GB | MoE |
| 31B | 31 B | **31 B** | 30.39 GB | dense |

All five are published by **ggml-org** with Q8_0 — the same publisher as our
current weights, so the quantizer stays constant (the confound that muddied the
Sprint-15 Q4 rejection). Every one also ships an **MTP draft model** for
speculative decoding (0.05–0.26 GB).

## Why active params are the number that matters here

Decode on this hardware is **memory-bandwidth-bound** — established repeatedly:
it is why slots hurt, why the lane ceiling is ~4, and why a 28-core CPU node is
1/13 of a V100. Bandwidth cost per token scales with **active** parameters,
while VRAM scales with **total** parameters. For MoE models those diverge
sharply.

Scaling from our measured baseline (E2B: 249 tok/s aggregate decode,
569 s/run/GPU, reasoning-on b256):

| Model | active | est. decode | est. s/run/GPU | 300 runs, 2 GPUs | VRAM (Q8 + KV) |
|---|---:|---:|---:|---:|---:|
| E2B | 2.0 B | 249 tok/s | 569 | **23.7 h** (measured) | ~6 GB |
| E4B | 4.0 B | ~125 | ~1140 | ~47 h | ~9 GB |
| **26B-A4B** | **3.8 B** | **~131** | **~1080** | **~45 h** | **~27 GB** |
| 12B | 12 B | ~42 | ~3400 | ~142 h | ~13 GB |
| 31B | 31 B | ~16 | ~8800 | ~367 h | **~32 GB — will not fit** |

**26B-A4B strictly dominates 12B**: it is a bigger, better model *and* ~3x
faster on this workload, because 3.8 B active beats 12 B active. That inversion
is the whole argument — on a bandwidth-bound box, picking by parameter count is
the wrong instinct.

**31B is out.** 30.4 GB of weights leaves no room for KV cache on a 32 GB V100.
Tensor-splitting across both GPUs would fit it but forfeits the
one-server-per-GPU sharding that gives us near-linear throughput, and at ~16
tok/s a 300-run collection is ~367 h. Not viable.

## Quality justification

Published Gemma 4 benchmarks (no E2B-specific figures are published, which is
itself a caveat):

| Model | MMLU-Pro | GPQA Diamond | AIME 2026 |
|---|---:|---:|---:|
| 31B dense | 85.2 | 84.3 | 89.2 |
| 26B MoE | 82.6 | 82.3 | — |

26B-A4B sits within ~3 points of the 31B flagship at ~1/8 the decode cost.

## The criterion that actually decides it — not benchmarks

The binding requirement for this task is **tool-call reliability under a
constrained schema**, not general reasoning. Every failure we have diagnosed is
a REFLECT tool-call failure:

* Q4_0 — malformed argument keys (`{'done=true,reasoning': ...}`), 33% failure
* b128 — valid calls but never reaching terminal `done_reflecting`, 13% failure
* reasoning-off Q8 — 20–30% failure
* reasoning-on Q8 b256 — ~1% failure

A model that scores better on GPQA but emits worse constrained JSON would be
worse here. **The pilot must measure failure rate first and quality second.**

## Risks to check before committing

1. **Architecture support.** Our llama.cpp is b10644 (2026-08-27). Confirm it
   implements Gemma 4 MoE routing for 26B-A4B — `--version` plus a 1-run smoke.
   A rebuild is cheap if not.
2. **Context headroom.** Peak prompt+gen on the 300-run set was 3,833 of 4,096
   per slot (94%). A larger model with more verbose thinking may exceed it,
   forcing a bigger `--ctx-size`; at 25 GB of weights the VRAM headroom for KV
   is much thinner than with E2B. **Measure peak tokens in the pilot.**
3. **Slot count.** p6 x 4096 = 24,576 ctx is affordable at 4.6 GB of weights;
   at 25 GB it may not be. Expect to reduce slots, which the axis-(d) curve
   says costs ~8% at p12 and ~20% at p24 — but that curve was measured on E2B
   and may not transfer.
4. **Speculative decoding is an untested lever.** Every model ships an MTP
   draft. Sprint-16 reported "ngram-spec does not stack at high occupancy", but
   that was n-gram speculation, not a trained MTP draft head — a different
   mechanism. If it delivers even 1.5x it materially changes these projections.
5. **`dflash-` variants** exist for 26B-A4B and 31B and are undocumented here;
   worth identifying before assuming the plain Q8 is the right file.

## Proposed pilot (cheap, decisive)

Run **15 runs** (seeds 1–5 x 3 pools) of `26B-A4B` at the SoTA config
(reasoning b256, temp 1.0, control prompt, Q8_0), paired by seed against the
300-run Arm-B set. ~4.5 h on 2 GPUs at the estimate above.

Report, in this order:
1. **failure rate** (the go/no-go)
2. peak prompt+gen tokens vs the 4,096 ceiling
3. measured s/run/GPU vs the ~1080 estimate
4. paired quality deltas (`llm/score_ablation_cells.py`)

Add an `E4B` cell on the second GPU for the same cost, giving a three-point
model-scale curve (E2B → E4B → 26B-A4B) rather than a single comparison.

**Framing:** this is a **new arm**, not a faster Arm B. Arm B is defined as
`gemma-4-E2B-it-Q8_0` and all Sprint-15 ablations are E2B-specific. A larger
model answers "do the deliberation findings hold as model capacity grows?" —
a model-scale robustness result, not a substitute for the existing collection.

## Sources

- Gemma 4 model overview — <https://ai.google.dev/gemma/docs/core>
- Gemma 4 model card — <https://ai.google.dev/gemma/docs/core/model_card_4>
- Gemma 4 Technical Report — <https://arxiv.org/pdf/2607.02770>
- `google/gemma-4-26B-A4B-it` — <https://huggingface.co/google/gemma-4-26B-A4B-it>
- GGUF weights — `ggml-org/gemma-4-{E2B,E4B,12B,26B-A4B,31B}-it-GGUF`
