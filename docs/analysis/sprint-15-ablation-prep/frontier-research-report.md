# Efficiency/Performance Frontier — Overnight Research Report

**Date:** 2026-08-26→27 (overnight, autonomous run) · **Status:** COMPLETE —
all phases collected, scored, and documented. TL;DR: the local 1200-run price
drops from ~370 to ~140 GPU-h (b128 + p12×2-concurrent + spec-at-p6; QAT and
f16-KV rejected on evidence), and a pilot-gated GPT-5-mini path (~$210, built
and offline-tested tonight) offers a 1–2-day alternative.

## Why this programme

The Sprint-15 ablation settled the *quality* configuration: refreshed E2B Q8_0
weights, coopmat2 Vulkan binary, **reasoning-on-256** (more argument diversity,
higher DRI, fewer LLM calls than reasoning-off). But quality costs wall-clock:
**18.4 min/run measured**, pricing the planned 1200-run extension at **~370
GPU-hours (~15 days)** on the RTX 2060 SUPER. This programme measures every
lever that could cut that price *without giving up the measured quality gains*,
so the extension decision (N vs time vs money) is made on numbers.

## Design

Two instrument classes:

1. **Bench matrix (Phase A)** — pure throughput, deliberation-shaped prompt
   (~420 tokens), single-stream + N-way concurrent aggregate tok/s
   (`llm/bench_frontier_probe.py`). Concurrent aggregate is the number that
   prices a collection run (the harness keeps up to 6 agent calls in flight).
   Configs: baseline (q8 KV, p6) · KV f16 · ngram self-speculation (lossless,
   so bench-only — no quality cell needed) · E2B QAT-Q4_0 weights · parallel-12.
2. **Paired quality/efficiency cells (Phase B)** — the two levers that could
   change *quality*, run as full 15-run cells under the forward config and
   scored against the `newstack_e2b_r0715` control (refreshed E2B Q8, b256):
   - **B1 `frontier_e2b_b128`** — reasoning budget 128 vs 256. If quality holds
     at half the thinking budget, run time drops materially at zero quality cost.
   - **B2 `frontier_e2bqat_b256`** — official QAT-Q4_0 weights (2.4 GB vs 4.7 GB)
     at b256. QAT claims near-lossless (KLD 0.0017); our earlier Q4 fragility
     finding was a *requant*, which QAT invalidates — but tool-call reliability
     transfer is untested anywhere. Our cell is the first data point.

Scoring: `llm/score_ablation_cells.py` paired by (composition, seed) —
voicing diversity, saturation, DRI, meta-consensus, opinion variance,
LLM calls/run — plus run-completion/failure-phase reliability.

## Results

### Phase A — bench matrix

| Config | single tok/s | concurrent aggregate tok/s | per-slot | vs baseline (agg) |
|---|---:|---:|---:|---:|
| baseline (q8 KV, p6) | 70.2 | 123.2 (6-way) | 20.5 | 1.00× |
| KV **f16** | 66.3 | **64.6** (6-way) | 10.8 | **0.52× — ELIMINATED** |
| **ngram self-spec** | 69.4 | **191.2** (6-way) | 31.9 | **1.55×** (lossless) |
| **E2B QAT-Q4_0** | **97.6** | **203.7** (6-way) | 34.0 | **1.65×** (quality gated on B2) |
| **parallel-12** | 66.0 | **254.3** (12-way) | 21.2 | **2.06×** (per-slot holds!) |

**Reading:**
- **f16 KV halves concurrent throughput** (memory-bandwidth bound at 6-way) —
  the q8 KV choice is not just VRAM thrift, it is the faster config. Eliminated.
- **ngram self-speculation: +55% aggregate, zero cost.** It is lossless by
  construction (draft-verify preserves the sampling distribution), so no
  quality cell is needed — adopt on bench evidence alone.
- **QAT-Q4_0 is the single biggest lever**: +39% single-stream, +65% aggregate
  (smaller weights = less memory traffic). Everything now rides on the B2
  quality/reliability cell.
- **parallel-12 scales**: per-slot throughput holds at 12 concurrent (21.2 vs
  20.5), so aggregate doubles. Collection could run **two runs concurrently**
  at p12 for ~2× macro throughput — VRAM permitting (measured fine at 49k ctx).
- The levers are **composable** (spec × QAT × p12 are independent mechanisms);
  a combo bench runs after Phase B frees the GPU. Naive stacking suggests up to
  ~3× aggregate, which with b128 (if B1 holds) could compress the 1200-run
  extension from ~370 GPU-h toward **~100 GPU-h** — from 15 days to ~4.

### Phase B1 — reasoning budget 128 (paired vs b256 control)

**Verdict: quality holds at half the thinking budget, 23% less wall-clock.**
14/15 runs (one REFLECT failure — within the normal transient class),
**14.1 min/run vs 18.4** for b256.

| Outcome | b256 (control) | b128 | Δ | p |
|---|---:|---:|---:|---:|
| distinct args / agent | 1.905 | 1.971 | +0.067 | .538 |
| DRI (final) | 0.190 | 0.355 | +0.165 | .118 |
| meta-consensus (final) | 0.701 | 0.692 | −0.009 | .244 |
| opinion var (final) | 0.240 | 0.168 | −0.072 | **.057** ⚠ |
| LLM calls / run | 604.9 | 587.5 | −17.4 | .298 |

Nothing significant at 14 pairs; DRI trends *up* if anything. **Watch-item:**
opinion variance trends lower (p=.057) — b128 may slightly accelerate
consolidation; worth 5 more paired seeds before adopting for the collection if
that outcome matters to the headline claim. Cost math if adopted: 18.4 → 14.1
min/run cuts the local 1200-run price from ~370 to **~282 GPU-h**, and
proportionally shrinks cloud reasoning-token spend.

### Phase B2 — QAT-Q4_0 weights (paired vs Q8 control)

**Verdict: throughput star, quality FAIL — rejected for the collection.**
13/15 runs (2 REFLECT failures vs 0 for Q8), 14.3 min/run, 94.5 tok/s bench.
But two **significant behavioral shifts** at n=13 pairs:

| Outcome | Q8 ctl | QAT-Q4 | Δ | p |
|---|---:|---:|---:|---:|
| top-cid share (repetition) | 0.422 | 0.485 | **+0.063** | **.034*** |
| opinion var (final) | 0.234 | 0.168 | **−0.066** | **.047*** |
| meta-consensus (final) | 0.702 | 0.672 | −0.030 | .125 |

Both significant shifts point the same way: **more repetition, more
consolidation** — precisely the direction that would *inflate* the paper's
epistemic-integration headline. Running the collection on QAT-Q4 would bake a
quantization artifact into the substantive claim. This is also the first
measured datapoint anywhere on QAT→tool-workload behavior transfer: KLD
near-losslessness (0.0017) does **not** guarantee behavioral neutrality in a
600-call agentic loop. Reject.

### Combo bench — do the surviving levers stack?

`Q8 + ngram-spec + parallel-12` (reasoning-on-256, 49k ctx, 3.65 GB VRAM):
**252.9 agg tok/s at 12-way — identical to plain p12 (254.3).**
Speculation's +55% (measured at p6) vanishes at 12-way: the GPU is already
saturated by batch parallelism, and draft verification competes for the same
compute. **Spec and p12 are alternative ~1.5–2× routes, not multiplicative.**

## Decision framework — filled

Quality gate first (paired nulls on headline outcomes + reliability parity),
then cheapest config wins:

| Lever | Quality gate | Throughput | Verdict |
|---|---|---|---|
| refreshed weights + coopmat2 binary | ✅ null (validated earlier) | ~= | **adopted** |
| KV f16 | (moot) | 0.52× | ❌ eliminated |
| **ngram self-spec** | lossless by construction | 1.55× at p6, ~1× at p12 | ✅ **adopt for p6 runs** |
| **reasoning b128** (vs b256) | ✅ null (⚠ opinion-var p=.057 watch-item) | 1.30× | ✅ **adopt, with a +5-seed check on opinion variance first if that outcome is headline-load-bearing** |
| QAT-Q4_0 | ❌ 2 significant shifts + 2 failures | 1.4× | ❌ rejected |
| **parallel-12, two concurrent runs** | no per-run behavior change (server-side only) | ~2× macro | ✅ **adopt** (two `run_ablation_cell` drivers on disjoint seed ranges) |

**Local frontier config:** refreshed E2B Q8 + coopmat2 + b128 + p12 with two
concurrent runs → **~7 min/run effective → 1200 runs ≈ 140 GPU-h (≈ 6 days)**,
down from ~370 GPU-h (15 days). Conservative fallback (keep b256): ≈ 184 GPU-h.

**Cloud frontier:** GPT-5-mini flex+cache ≈ **$210, 1–2 days** (pilot-gated;
adapter built and offline-tested tonight — see the OpenAI section above).

**The decision that remains (author's):** local-6-days-free vs cloud-2-days-
$210 vs hybrid (cloud for the collection, local for ablations). All three are
now evidence-priced; nothing further blocks the 1200-run extension.

## Parallel deliverable: OpenAI GPT-5-mini inference layer (IU-requested, done)

Built while the GPU cells ran. A third backend now sits behind the same
hexagonal port: `LLM_API_FLAVOR=openai-cloud` selects `OpenAICloudClient`
(`llm/openai_adapter.py`) — engine/harness untouched, local paths unaffected.

- **Research cycle** (3 web agents + synthesis): `docs/design/
  openai-gpt5-mini-adapter-design.md` — surface choice (Chat Completions),
  exact GPT-5-mini request shape, cost model (**~$210 for a full 1200-run
  collection on flex + prompt-cache**, vs ~370 GPU-hours locally), determinism
  posture, risk register.
- **IU's Study-3 production intelligence** folded in as an addendum — it
  reversed one decision (raw aiohttp instead of the SDK in the request path;
  Study 3 hit the SDK's event-loop failure under threads) and contributed the
  forced-by-name tool choice and dual-shape reasoning extraction.
- **Implementation**: param discipline for reasoning models, strict tools,
  Retry-After-honoring 429 backoff (fatal on quota errors), per-call cost
  ledger with cached/reasoning token capture, `OPENAI_BUDGET_USD` hard guard.
  **9 offline tests** (fake HTTP), llm suite 60 → 69 green.
- **Morning gate**: `OPENAI_API_KEY=... bash llm/run_openai_pilot.sh` — 3 runs,
  reports reasoning-tokens/call, cache-hit ratio, $/run vs model. Needs your
  project-scoped, spend-limited key; nothing spends money until you run it.

Strategic note: if the pilot validates, the cost frontier changes shape — the
1200-run extension becomes a **$-scale decision instead of a GPU-weeks
decision**, and the local card becomes the ablation/pilot instrument rather
than the collection workhorse. The cross-model comparison (gemma-4-E2B vs
GPT-5-mini as Arm-B cognition) also becomes a cheap robustness study.

## Context for the morning review

- The repo reorg (Stages 0–6) completed first — see the reorg commits and the
  new INVENTORY tier table. Stage 7 (tmp2 external backup) is gated in TODO.md
  on a physical backup destination.
- The paused neutrality wave (`llm/traces/ablation/NEUTRALITY-WAVE-PAUSED.txt`)
  remains paused by design — it pairs against the OLD stack and is resumable.
- Provenance: every cell stamps its infra string into each trace's config block.

---

# Correction addendum (2026-08-27, post-pilot)

The scale-out pilot (sprint-16 P4a) **revises this report's throughput
projections downward**. The bench-probe extrapolation ("434 tok/s aggregate ⇒
~2.5 days for 1200 runs") does not survive contact with real runs: on
full-length contexts, per-slot decode degrades much faster than the ~420-token
probe showed. Measured: **3 lanes → 1.72× macro** (solo 729.5 s vs 423 s/run
effective; per-run pace 1.69× slower under contention), zero failures.

Standing numbers: 4 lanes ≈ ~2× ⇒ 1200 runs ≈ **6–7 days local** (vs 12
sequential) — not 2.5. The cloud path's relative advantage grows accordingly.
The p24/98k "broken config" reading in the pilot's early telemetry was also
wrong — whole-run penalty is ~8%; the 13 tok/s lines were momentary
long-context batch timings. Full data: sprint-16 plan, Results section.
