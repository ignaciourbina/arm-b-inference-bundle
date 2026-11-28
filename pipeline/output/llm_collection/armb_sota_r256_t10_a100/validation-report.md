# Arm-B (AgenticLLMEngine) collection — validation report

Traces dir: `llm/traces/collection/armb_sota_r256_t10_a100`  ·  design: 200 seeds × 3 compositions (n=6, t=8, baseline, AgenticLLMEngine, gemma-4-E2B-it-Q8_0.gguf)

## Verdict: **PASS (partial)**

| Check | Result |
|---|---|
| A. Completeness | 300/600 finals; missing=300 (allowed: partial), extras=0, duplicates=0, strays=0; checkpoints=300, llm-traces=301 |
| B/C. Schema + physical validity | 0 file(s) with errors |
| D. Config uniformity | 1 distinct config signature(s) (must be 1) |
| E. LLM-call quality | 182133 calls, 0 errors, max prompt+gen=3622 tok (fail ≥4096), 0 tool-loop retries, 0 file(s) over fallback-share warn |
| Opinion movement (info) | mean of mean_abs_shift = 0.2688; 300/300 runs with movement > 0 |
| Log scan | `[warn] voice failed`×0; `[warn] eval failed`×0; `HookLoopError`×0; `chat got 500`×0; `no progress`×0 |

### Missing cells

polarized/s1, polarized/s2, polarized/s3, polarized/s4, polarized/s5, polarized/s6, polarized/s7, polarized/s8, polarized/s9, polarized/s10, polarized/s11, polarized/s12, polarized/s13, polarized/s14, polarized/s15, polarized/s16, polarized/s17, polarized/s18, polarized/s19, polarized/s20, polarized/s21, polarized/s22, polarized/s23, polarized/s24, polarized/s25, polarized/s26, polarized/s27, polarized/s28, polarized/s29, polarized/s30, polarized/s31, polarized/s32, polarized/s33, polarized/s34, polarized/s35, polarized/s36, polarized/s37, polarized/s38, polarized/s39, polarized/s40, polarized/s41, polarized/s42, polarized/s43, polarized/s44, polarized/s45, polarized/s46, polarized/s47, polarized/s48, polarized/s49, polarized/s50, polarized/s51, polarized/s52, polarized/s53, polarized/s54, polarized/s55, polarized/s56, polarized/s57, polarized/s58, polarized/s59, polarized/s60
... (300 total)

### Per-file warnings (non-blocking)

- `townhall_minimum_wage_seattle_armb_sota_r256_t10_a100_baseline_polarized_n6_s192_1795758862.json`
  - max prompt+gen tokens 3622 > 3600

### Uniform engine config (collection-wide)

```json
{
  "ablation_infra": "reasoning=on;budget=256;quant=Q8_0;temp=1.0;server=p6/24576;lanes=4;harness=strict;gpu=NVIDIA-A100-80GB-PCIe",
  "condition": "baseline",
  "confirmation_bias": 0.3,
  "harness_profile": "strict",
  "model": "gemma-4-E2B-it-Q8_0.gguf",
  "n_agents": 6,
  "n_rounds": 8,
  "prompt_variant": "control",
  "prompt_variant_note": "production BASELINE prompt profile (unchanged)",
  "reflect_decision_policy": "explicit_update_or_no_update_v2",
  "server_props": {
    "model_path": "~/.cache/llama-models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf",
    "n_ctx_slot": 4096,
    "total_slots": 6
  },
  "temperature": 1.0,
  "topic": "minimum_wage_seattle",
  "topic_description": "Should Seattle implement a $15/hour minimum wage?"
}
```

### Recorded server signature (external provenance)

```json
{
  "backend": "llama-server on SeaWulf 2x A100-80GB (arm-b-inference-bundle)",
  "n_parallel": 6,
  "ctx_size": 24576,
  "ctx_per_slot": 4096,
  "lanes": 4,
  "flags": "reasoning=on;budget=256;quant=Q8_0;temp=1.0;harness=strict"
}
```
