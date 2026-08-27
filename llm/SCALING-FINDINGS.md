# Throughput & Scale-Out Findings — READ BEFORE TUNING CONCURRENCY

Authoritative, **measured-on-real-runs** guidance for collection throughput.
This corrects the optimistic *probe* projections that still appear in some code
comments. If a comment and this file disagree, **this file wins** — it reflects
the Sprint-16 P4a A/B on real deliberation runs, not the short-prompt probe.

Full evidence: the sprint memos (in the source repo under
`agora/analysis/sprint-15-ablation-prep/` and
`agora/analysis/sprint-16-collection-scaleout/plan.md`; in the public bundle
under `docs/analysis/`).

---

## The one correction that matters

The frontier **probe** reported *"aggregate decode scales to ~434 tok/s at
24-way with no plateau"* → which suggested `--parallel 24` and *"~2.5 days for
the 1200-run collection"* and *"~3–5× faster."*

**Sprint-16 P4a measured this directly on real runs and it does not hold:**

| Config (real t=8 runs, p12/49k server) | Result |
|---|---|
| Solo (1 lane) | 729.5 s/run baseline |
| **3 lanes** | makespan 1270 s → 423 s/run effective → **1.72× macro** |
| **4 lanes** (projected/measured) | **≈ 1.9–2.1×** |
| Per-run pace under 3-way contention | **degrades ~1.69×** |
| 1200-run collection | **~6–7 days**, not 2.5 |

Why the probe overestimated: it used short shared-prefix prompts, so it never
hit the decode-bandwidth wall. On real 3–4k-token contexts, **decode bandwidth
binds much sooner** — so aggregate tok/s keeps climbing while per-run
wall-clock (what actually sets collection time) degrades. **Never infer
collection wall-clock from an aggregate tok/s curve.**

## How concurrency actually works here

- **The lever is `--lanes` in `run_collection_parallel.py`**, not server
  `--parallel`. Lanes are disjoint round-robin slices of the (composition ×
  seed) grid, each a sequential runner loop — no locking, `--resume`
  idempotent.
- **Total server load = `lanes × --run-parallel`** (within-run agent
  concurrency, default 6). To add lanes productively you must scale server
  slots to match, or the extra requests just queue (latency balloons, makespan
  barely moves).
- **Real-run lane ceiling ≈ 4.** `run_collection_parallel.py` *defaults to 8
  lanes* — that is above the measured ceiling; **override to `--lanes 4`.**
- A single run only keeps ~2.7–4.5 of 6 slots busy because agent steps within a
  round are sequential (voice → evaluate → reflect). Oversubscribing slots ~2:1
  with lanes is *intentional* — it fills those gaps.

## Server config

- **Tuned collection config is `p12/49k`** (12 slots, `CTX=49152` total =
  4096/slot), **retained over `p24/98k`** — the p24 penalty on whole runs was
  only ~8%; p12 wins on VRAM/simplicity. The "13 tok/s alarm" was momentary
  long-context batch timing, not sustained pace.
- q8 KV (f16 measured 0.52× — rejected). QAT-Q4 and ngram-spec rejected at high
  occupancy. coopmat2 source binary + refreshed (Jul-15) E2B Q8_0 weights.

## The comparability guardrail (do not violate)

- The pinned Arm-B collection is **`--reasoning off`**. `run_collection_server.sh`
  flips **`--reasoning on --reasoning-budget 128/256`** — that changes
  *generation semantics*, not just speed. **Traces collected reasoning-on are
  NOT comparable** to the pinned reasoning-off set. Do not mix them.
- **Slot count / ctx is throughput-only and comparability-safe** (same model,
  same sampling, reasoning off). So you can scale p6 → p12 → p24 *without*
  touching the reasoning flag — start the server via
  `pipeline/runpod/scripts/start_llama_server_local.sh` (which is reasoning-off)
  with `N_PARALLEL`/`CTX` overridden, **not** via `run_collection_server.sh`.
- Don't switch orchestrators mid-collection: the pinned
  `run_arm_b_local_collection.sh` sweep writes `llm/traces/beta_local/...`;
  `run_collection_parallel.py` writes `llm/traces/collection/<cell>/` with a
  different tag scheme. Pick one; scale by relaunching the *same* one with more
  `--lanes` (idempotent via `--resume`).

## Recommended reasoning-off scale-up (evidence-based)

```bash
# 1. more slots, reasoning STILL off (NOT run_collection_server.sh):
pkill -x llama-server; sleep 3
N_PARALLEL=12 CTX=49152 CACHE_REUSE=256 UBATCH=512 \
  bash pipeline/runpod/scripts/start_llama_server_local.sh

# 2. 4 lanes (override the default 8):
python llm/run_collection_parallel.py \
  --cell <collection_cell> --seeds <range> \
  --lanes 4 --run-parallel 6 --rounds 8 --agents 6
```

Expect **~1.9–2.1×** over single-lane. Measure makespan from the telemetry
JSONL (`llm/traces/logs/<cell>_progress.jsonl`); do not extrapolate from tok/s.
