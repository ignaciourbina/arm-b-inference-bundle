# Sprint 16 — Collection Scale-Out (plan)

**Date:** 2026-08-27 · **Goal:** implement the scaffolding to run the 1200-run
Arm-B extension at high GPU occupancy (~2.5 days local), exploiting the
Sprint-15/frontier findings. Prep only — the collection itself launches on
IU's go.

## Evidence base (frontier report, 2026-08-27)

- Real runs keep only **~2.7 of 6 server slots busy** (19.4 h slot-time /
  7.1 h wall) — the bottleneck is slot starvation, not the card.
- Aggregate decode scales to **434 tok/s at 24-way** (vs 70 single) with no
  plateau; prefill is 7% of GPU-busy (negligible).
- Adopted config: refreshed E2B Q8_0 + coopmat2 binary + q8 KV +
  reasoning-on (b128 pending its opinion-variance watch-item; b256 fallback).
  QAT-Q4 and f16-KV rejected on evidence; ngram-spec does not stack at high
  occupancy (omit).
- Per-run decode ≈ 73k tokens ⇒ at ~420 tok/s aggregate, 1200 runs ≈ ~60
  GPU-h ≈ **2.5 days**.

## Deliverables (all this sprint, in order)

1. **P1 — server launcher** `llm/run_collection_server.sh`: p24 / ctx 98304
   (4096 per slot) / q8 KV / coopmat2 binary / refreshed weights / reasoning
   budget as arg (128|256). Prints VRAM after load; refuses to start if the
   model or binary is missing.
2. **P2 — parallel collection orchestrator** `llm/run_collection_parallel.py`:
   - N **lanes** (default 8), each lane a sequential loop of runner
     subprocesses over its own slice of the (composition × seed) grid —
     disjoint by construction, no locking needed.
   - Per-run `--resume` (checkpoint short-circuit) so the whole collection is
     idempotent; a lane crash affects only its current run.
   - **Gate before work**: warmed bench ≥ threshold; VRAM headroom check;
     disk-space check on the trace volume.
   - **Telemetry**: single JSONL progress file (run start/finish/fail, wall
     seconds, lane id); periodic status line (runs done/total, runs/hour,
     ETA, GPU util snapshot).
   - **Pause/resume**: SIGTERM → lanes finish their in-flight run and exit
     (bounded ~15 min); a marker file documents resume command. Idempotent
     relaunch continues.
   - **Failure policy**: failed runs recorded and retried once at the end
     (fresh attempt, same seed); persistent failures listed in the summary.
3. **P3 — offline tests** `llm/tests/test_collection_parallel.py`: grid
   slicing (disjoint, complete, balanced), resume filtering, telemetry
   records, retry queue — subprocess layer stubbed.
4. **P4 — occupancy pilot (the gate for the 2.5-day claim)**: 6 short runs
   (t=4) sequential vs 6 runs across 6 lanes on the p24 server; report
   measured speedup, VRAM at 98k ctx, per-run slowdown. Success: ≥2.5×
   macro speedup, VRAM < 6.5 GB, zero failures. Numbers → frontier report
   addendum + this plan's results section.
5. **P5 — launch documentation**: exact collection command (with b128/b256
   choice left explicit), pause/resume procedure, expected timeline, in
   `llm/README.md` + HANDOFF update. Commit everything.

## Non-goals

- Launching the 1200-run collection (IU's call: local vs cloud vs hybrid).
- The OpenAI path (own pilot gate, needs IU's key).
- Touching engine/harness/client — this is pure orchestration around the
  existing runner; the paused neutrality wave and all traces stay untouched.

## Risks

| Risk | Mitigation |
|---|---|
| High-occupancy attention at real 3-4k ctx worse than probe | P4 measures real runs, not probes; scale lanes down if speedup < 2.5× |
| VRAM blowout at 98k ctx | P1 prints VRAM; P4 gate < 6.5 GB; fallback ctx 73728/p18 |
| 8 drivers thrash the 8-thread CPU during prefill bursts | lanes stagger their starts by 90 s; threads stay at 8 server-side |
| Runner timeouts at slower per-slot pace | runner client timeout is 600 s — per-call at ~18 tok/s × 2k tokens ≈ 110 s, ample margin |

---

## Results (filling as measured)

### P4a — quick A/B (2026-08-27, p12/49k, b128, t=4, telemetry-based)

- Solo run: **729.5 s**. 3 lanes: 3 runs, makespan **1,270 s** → 423 s/run
  effective → **macro speedup 1.72×** (per-run pace degraded 1.69× under
  contention — decode bandwidth binds sooner on real runs than the
  short-prompt sweep predicted).
- p24/98k config penalty re-measured: only ~8% on whole runs (788 vs 730 s) —
  the earlier "13 tok/s" alarm was momentary long-context batch timing, not
  sustained pace. p12/49k retained as collection config for its VRAM/simplicity.
- t=4 runs cost ~87% of t=8 runs → early rounds + fixed overhead dominate;
  the marginal round is cheap.
- Revised projection: 4 lanes ≈ 1.9–2.1× → 1200 runs ≈ **6–7 days local**
  (not 2.5 — the occupancy model overestimated by ignoring per-slot decode
  degradation on real contexts). Still ~2× better than sequential (12 days).

### P4b — 4-hour 4-lane pilot: RUNNING (33 × t=8 runs, cell `pilot_4lane`)

Measures sustained runs/hour, stability, and doubles as b128 watch-item data.

### P4a addendum — machinery validation + design finding

- **Scaffolding validated end-to-end on real runs**: gates (bench/VRAM/disk),
  disjoint lane slicing, per-lane telemetry, collection-grade traces, zero
  failures across both arms. Two self-caught defects (status-loop
  quantization; hardcoded server config) fixed and committed same-day.
- **Design finding — rounds are cheap at the margin**: a t=4 run costs ~87%
  of a t=8 run (729.5 vs ~845 s equivalent-config), i.e. early rounds + fixed
  per-run overhead dominate. Implication beyond throughput: extending
  trajectories (t=12/t=16) for richer growth-model estimation would cost far
  less than linear intuition suggests — a candidate for the extension design
  discussion, since the pooled log-round panel gains most from later rounds.
