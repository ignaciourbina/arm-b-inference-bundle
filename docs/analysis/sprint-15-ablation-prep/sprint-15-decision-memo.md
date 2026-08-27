# Sprint 15 — Ablation Decision Memo

**Status:** DRAFT — numbers auto-filled, judgement left to the author.

Purpose: decide the configuration for extending Arm B from 390 to 1200
runs (~65 GPU hours), on evidence rather than default.

## Cells collected

| Cell | Runs paired |
|---|---|
| `prompt_anti-repetition` | 15 |
| `prompt_explicit-tradeoff` | 13 |
| `prompt_terse` | 15 |
| `quant_Q4_K_M` | 15 |
| `reasoning_on_b256` | 15 |
| `reasoning_on_b512` | 15 |

## Axis 1 — prompt

### `prompt_anti-repetition`

- distinct args / agent: 1.442 → 1.571 (+0.129, p=0.109)
- top-cid share: 0.481 → 0.487 (+0.006, p=0.802)
- saturation @r1 (mechanism): 0.804 → 0.829 (+0.024, p=0.165)
- DRI final (headline): 0.130 → 0.211 (+0.080, p=0.213)
- meta-consensus final (headline): 0.680 → 0.681 (+0.001, p=0.936)

Reading: if voicing diversity moves while saturation does not, the
repetition documented in Sprint 14 was prompt-induced; if neither
moves, it is a property of the engine's state dynamics.

### `prompt_explicit-tradeoff`

- distinct args / agent: 1.436 → 1.462 (+0.026, p=0.613)
- top-cid share: 0.469 → 0.463 (-0.006, p=0.837)
- saturation @r1 (mechanism): 0.801 → 0.830 (+0.030, p=0.251)
- DRI final (headline): 0.126 → 0.302 (+0.176, p=0.024)
- meta-consensus final (headline): 0.681 → 0.672 (-0.009, p=0.424)

Reading: if voicing diversity moves while saturation does not, the
repetition documented in Sprint 14 was prompt-induced; if neither
moves, it is a property of the engine's state dynamics.

### `prompt_terse`

- distinct args / agent: 1.442 → 1.716 (+0.273, p=0.039)
- top-cid share: 0.481 → 0.456 (-0.025, p=0.307)
- saturation @r1 (mechanism): 0.804 → 0.824 (+0.020, p=0.257)
- DRI final (headline): 0.130 → 0.194 (+0.064, p=0.533)
- meta-consensus final (headline): 0.680 → 0.676 (-0.004, p=0.605)

Reading: if voicing diversity moves while saturation does not, the
repetition documented in Sprint 14 was prompt-induced; if neither
moves, it is a property of the engine's state dynamics.

## Axis 2a — reasoning

### `reasoning_on_b256`

- saturation @r1: 0.804 → 0.807 (+0.002, p=0.919)
- distinct args / agent: 1.442 → 1.853 (+0.411, p=0.001)
- DRI final: 0.130 → 0.316 (+0.186, p=0.019)
- LLM calls / run (cost proxy): 801.200 → 609.133 (-192.067, p=<.001)

### `reasoning_on_b512`

- saturation @r1: 0.804 → 0.813 (+0.008, p=0.607)
- distinct args / agent: 1.442 → 1.800 (+0.358, p=0.003)
- DRI final: 0.130 → 0.273 (+0.143, p=0.061)
- LLM calls / run (cost proxy): 801.200 → 765.267 (-35.933, p=0.038)

## Axis 2b — quantization

### `quant_Q4_K_M`

- DRI final: 0.130 → 0.150 (+0.020, p=0.635)
- meta-consensus final: 0.680 → 0.676 (-0.004, p=0.745)
- distinct args / agent: 1.442 → 1.433 (-0.009, p=0.852)

Watch tool-call reliability here specifically: the parse-500 class is
a formatting failure and is the most likely casualty of heavier
quantization (see Sprint 14 §on the llama-server parse-500 replay behaviour).

## Decision (author)

- [ ] Prompt configuration for the 1200-run extension: ______
- [ ] Reasoning on/off (and budget if on): ______
- [ ] Quantization: ______

### Caveats to carry

- Five seeds per composition is enough to detect large paired effects and
  nothing else. A null here is 'not detected at this power', not 'absent'.
- Cells were collected in one session on one machine; any driver or
  serving change between a cell and the main run invalidates the pairing.
- Table 10 provenance (Sprint 14) is unresolved for Arm A; nothing in this
  memo bears on that.
