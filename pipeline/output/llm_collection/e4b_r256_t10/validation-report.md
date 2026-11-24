# Arm-B (AgenticLLMEngine) collection — validation report

Traces dir: `llm/traces/collection/e4b_r256_t10`  ·  design: 100 seeds × 3 compositions (n=6, t=8, baseline, AgenticLLMEngine, gemma-4-E4B-it-Q8_0.gguf)

## Verdict: **PASS**

| Check | Result |
|---|---|
| A. Completeness | 300/300 finals; missing=0, extras=0, duplicates=0, strays=0; checkpoints=300, llm-traces=301 |
| B/C. Schema + physical validity | 0 file(s) with errors |
| D. Config uniformity | 1 distinct config signature(s) (must be 1) |
| E. LLM-call quality | 144694 calls, 0 errors, max prompt+gen=3766 tok (fail ≥4096), 0 tool-loop retries, 0 file(s) over fallback-share warn |
| Opinion movement (info) | mean of mean_abs_shift = 0.1289; 300/300 runs with movement > 0 |

### Per-file warnings (non-blocking)

- `townhall_minimum_wage_seattle_e4b_r256_t10_baseline_symmetric_n6_s18_1794996582.json`
  - max prompt+gen tokens 3634 > 3600
- `townhall_minimum_wage_seattle_e4b_r256_t10_baseline_symmetric_n6_s24_1795005799.json`
  - max prompt+gen tokens 3671 > 3600
- `townhall_minimum_wage_seattle_e4b_r256_t10_baseline_symmetric_n6_s45_1795044275.json`
  - max prompt+gen tokens 3615 > 3600
- `townhall_minimum_wage_seattle_e4b_r256_t10_baseline_symmetric_n6_s52_1794966588.json`
  - max prompt+gen tokens 3601 > 3600
- `townhall_minimum_wage_seattle_e4b_r256_t10_baseline_three_clusters_n6_s6_1794974294.json`
  - max prompt+gen tokens 3719 > 3600
- `townhall_minimum_wage_seattle_e4b_r256_t10_baseline_three_clusters_n6_s76_1795011217.json`
  - max prompt+gen tokens 3766 > 3600

### Uniform engine config (collection-wide)

```json
{
  "ablation_infra": "reasoning=on;budget=256;quant=Q8_0;temp=1.0;harness=format;server=p6/24576;lanes=4;model=gemma-4-E4B-it-Q8_0.gguf;gpu=NVIDIA-A100-80GB-PCIe",
  "condition": "baseline",
  "confirmation_bias": 0.3,
  "harness_profile": "format",
  "model": "gemma-4-E4B-it-Q8_0.gguf",
  "n_agents": 6,
  "n_rounds": 8,
  "prompt_variant": "control",
  "prompt_variant_note": "production BASELINE prompt profile (unchanged)",
  "reflect_decision_policy": "explicit_update_or_no_update_v2",
  "server_props": {
    "model_path": "~/.cache/llama-models/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q8_0.gguf",
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
  "backend": "llama-server (Vulkan, RTX 2060 SUPER 8GB, local :20434)",
  "n_parallel": 6,
  "ctx_size": 24576,
  "ctx_per_slot": 4096,
  "cache_reuse": 256,
  "ubatch": 512,
  "flags": "--jinja --reasoning off --flash-attn on --cache-type-k/v q8_0"
}
```
