# Arm-B SoTA operating point on SeaWulf — decision memo

**Status: COMPLETE.** All four axes measured, collection validated. Local
inference only (llama.cpp on cluster GPUs; no hosted APIs).

Objective function: **makespan per run (s/run/GPU)** — never aggregate tok/s.
Quality deltas from `llm/score_ablation_cells.py`, paired by composition+seed.

---

## Recommended config

```
model        gemma-4-E2B-it-Q8_0.gguf   (ggml-org)   <- NOT Q4 (axis a)
reasoning    ON, budget 256                          <- LOCKED, see below
temperature  1.0                                     <- LOCKED
prompt       control (production)
server       p6 / ctx 24576 / q8 KV / flash-attn     <- p6, not p12/p24 (axis d)
             cache-reuse: any value (no effect, axis c)
orchestrator run_collection_parallel.py --lanes 4 --run-parallel 6
sharding     one llama-server per GPU, disjoint seed slices
```

**Measured: 569 s/run/GPU** on the 300-run collection; ~1% model failure rate.

### Reasoning ON is a locked decision (2026-08-30), not a live trade-off

Everything below about reasoning-off is **the evidence that closed the axis**,
retained for the record. It is not an option under consideration. Do not
propose reasoning-off arms, and do not resume the stalled `abl_off_t10` cell.

Three independent lines converged:

1. **Reliability** — reasoning-off fails 20-30% of REFLECT tool calls across
   six cells; reasoning-on ~1%. Some failures are *deterministic* dead-ends
   `--resume` structurally cannot clear.
2. **The substantive finding depends on it.** Reasoning-off deliberation is a
   one-shot update: meta-consensus saturates by round 2 (+0.005 after r2, ~1
   SE) and the room collapses onto one consideration (distinct args 4.4 ->
   2.8, top-cid share 0.35 -> 0.54). Reasoning-on keeps integrating across all
   8 rounds (mc +0.022 after r2 = 11 SE; diversity holds ~3.4). **This holds at
   temp 0.3 and 1.0 alike**, so it is the reasoning capacity, not the sampling
   temperature. `make growth` reproduces the decomposition.
3. The ~1.5-1.7x wall-clock cost was weighed against those and accepted.

Temperature 1.0 is locked on separate, principled grounds: it samples the
model's actual next-token distribution, whereas a lower temperature sharpens
toward the mode and the agents would no longer be the cognition the study
claims to observe. Selecting temperature to maximise a headline metric would
be a researcher-degrees-of-freedom error.

### This is a quality purchase, not a speed win — stated plainly

Reasoning-off at the same server config runs **260 s/run/GPU**, i.e. 2.2x
faster. Adjusting for its ~25% failure rate gives an *effective* ~347 s/run
against reasoning-on's ~583, so reasoning-on still costs **~1.7x** net.

It is recommended anyway because of what it buys:
* **~25x better reliability** (~1% vs 20-30% failures across six cells), and
  reasoning-off failures include DETERMINISTIC dead-ends that `--resume` can
  never clear (a run needed hand-intervention; see the retry-policy note below).
* **Sprint-15's quality gains replicate here**: +0.420 distinct args/agent
  (p=.003) and -192.5 LLM calls/run (p<.001) versus their +0.41 / -192.07 --
  the calls figure agreeing to 0.4 of one call on different hardware.
* **Trajectories stay alive**: post-round-1 movement is 3.3x higher than
  reasoning-off (see the freeze section).

If pure wall-clock were the only objective, reasoning-OFF at p6/lanes-4 wins.
It is rejected on reliability and quality, not on speed.

---

## Measured wall-clock (established)

Per-GPU makespan, measured from orchestrator `collect_start` to last completion
(validated to 0.4% against a known job runtime).

| Config | n | s/run/GPU | failures |
|---|---:|---:|---:|
| reasoning off, temp 0.3 | 25 | **439** | 8 (~20–30%) |
| reasoning b256, temp 0.3 | 15 | 675 | 0 |
| reasoning b256, temp 1.0 | 15 | 690 | 0 |
| reasoning b256, temp 1.0 (300-run cell, p6) | **300** | **576** | 11 (3.5%)* |

\* 8 of the 11 were SIGTERM casualties at the self-resubmission handover, not
model failures; all were resumable from checkpoints. Underlying model failure
rate is ~1%.

**Reasoning-on costs ~1.3–1.5× wall-clock and buys ~20× better reliability.**
Reasoning-off failed 20–30% across six independent cells; reasoning-on ran the
full 300-run collection with ~1% underlying model failures (11 recorded, 8 of
them SIGTERM casualties at the chain handover). At 300+ runs the failure tax —
retries plus hand-intervention on deterministic dead-ends — closes most of the
apparent wall-clock gap.

**Collection validated:** 300/300, validator PASS, 1 config signature,
178,999 calls, 0 errors, 0 tool-loop retries. Peak prompt+gen was 3,833 of the
4,096 per-slot budget (94%) — independent support for SMALLER thinking budgets,
since b512 would risk mid-tool-call truncation.

**Temperature is nearly free** (+2% wall-clock) and is retained at 1.0 on
methodological grounds: it samples the model's actual next-token distribution,
where lower temperatures sharpen toward the mode.

### Scaling law (reasoning-OFF; see caveat)

| Lanes (p24) | s/run | | Slots (lanes=4) | s/run |
|---:|---:|---|---:|---:|
| 1 | 538 | | **6** | **260** |
| 2 | 464 | | 12 | 279 |
| 4 | 431 | | 24 | 390 |
| 6 | 286* | | 48 | 456 |

\* L=6 ran on a quieter node; a matched-conditions rerun gave L6=286 vs
L4=292 — i.e. **the lane ceiling is ~4**, and L=6's apparent advantage was an
artifact. Same-config replication (L4@p6: 260 vs 292) puts the **noise floor at
~12%**, so differences under ~15% are not meaningful.

**Caveat:** this curve is reasoning-OFF only. At reasoning-on the p6 advantage
shrinks from 1.34× to 1.11×, so the slot optimum is **generation-length
dependent**. Axis (d) re-measures it.

---

## (a) Quantization Q4_0 vs Q8_0 at reasoning-on — **REJECTED**

Cell `abl_q4_b256` (P100), Q4 from **ggml-org** — the same publisher as our Q8,
so the unsloth-vs-ggml-org confound in the Sprint-15 rejection is removed. Also
tested AT reasoning-on, where the REFLECT fragility was hypothesised to vanish.

| Outcome | Q8 | Q4 | delta | p |
|---|---:|---:|---:|---:|
| distinct args / agent | 1.833 | 2.103 | +0.269 | .012 |
| distinct args / room | 5.154 | 6.308 | +1.154 | <.001 |
| **LLM calls / run** | 610.5 | 754.9 | **+144.4** | **<.001** |

**Failure rate: 33.3%** (vs 3.5% for Q8/b256) — 9.5x worse. Failure mode from
`FAIL_*.log` is structural JSON corruption, e.g. the model emitting
`{'done=true,reasoning': ...}` — collapsing `{"done": true, "reasoning": ...}`
into one malformed key. Reasoning-on does NOT rescue it.

**Net cost, not saving:** to land the same number of COMPLETED runs, Q4 needs
~1.5x the attempts x 1.24x the calls ~= **1.86x the work**. That swamps the
~1.6x faster decode the bandwidth argument predicted.

**The quality column is uninterpretable.** With 33% of runs failing, the 13
scored runs are SURVIVORS — the ones where format held. The diversity deltas
are plausibly manufactured by that selection, and should not be cited as a Q4
advantage.

**Verdict: stay on Q8_0.** The hypothesis that Q4's fragility was a
quantizer/reasoning-off artifact is refuted with the confound removed.

## (b) Reasoning budget b128 vs b256 — **BETTER QUALITY, WORSE RELIABILITY**

Cell `abl_q8_b128` (P100), paired n=15 against the 300-run collection.

| Outcome | b256 | b128 | delta | p |
|---|---:|---:|---:|---:|
| distinct args / agent | 1.800 | 2.189 | **+0.389** | **.001** |
| distinct args / room | 5.133 | 5.867 | **+0.733** | **.003** |
| meta-consensus (final) | 0.691 | 0.715 | **+0.025** | **.019** |
| opinion var (final) | 0.255 | 0.187 | **-0.067** | **.011** |
| DRI (final) | 0.187 | 0.253 | +0.066 | .47 |
| LLM calls / run | 612.2 | 618.2 | +6.0 | .61 |

b128 tests better on four outcomes, two of which touch the integration claim
(meta-consensus up, dispersion down), and this is directionally consistent with
the Sprint-15 trend (b256 beat b512).

**Read this as a signal, not a finding.** `abl_q8_b128` is a 15-run benchmark
cell — a prolonged preflight whose job is failure rate, VRAM and wall-clock.
Substantive measures are the 300-run collection's business (see CLAUDE.md,
"Cell sizes are different instruments"). Two specific reasons not to bank the
quality result:

* **The pairing is weaker than it looks.** Pairing by seed is supposed to hold
  the trajectory fixed and vary only the treatment. Across different GPUs it
  does not: different kernels give different floating-point results, and at
  temperature 1.0 a different logit is a different sampled token. Same seed,
  different run. This is closer to an unpaired n=15 comparison than a paired
  one.
* Only the voicing metrics (distinct args) are low-variance enough to carry
  weight at this size; meta-consensus and opinion variance here are suggestive.

**Cost: 13.3% failure rate vs b256's 3.5%.** Failure mode differs from Q4's --
no malformed JSON; the model emits valid `update_weight` calls but exhausts the
128-token budget before reaching terminal `done_reflecting`. Better when it
finishes, finishes less often.

**Wall-clock NOT established.** This cell ran on P100 while the control ran on
V100, so no speed claim is made. A b256 cell on P100 would be needed for a fair
comparison. Flat LLM calls (+6, ns) suggests little cost difference.

**Verdict: unresolved, and not resolvable from this cell.** The reliability cost
is real and is measured on the right instrument — 13.3% vs 3.5% failures is
exactly what a 15-run benchmark cell is for. The quality gain is not: it would
need its own collection at n≈300 on matched hardware to become a finding. Since
the reliability cost alone is ~40 extra failed runs per 300, b128 is not adopted
without that evidence.

## (c) Prefix cache-reuse — **NULL, and the premise was wrong**

Measured AT reasoning-on, p6 / lanes 4, 9 runs per cell:

| cache-reuse | s/run/GPU |
|---:|---:|
| 0 | 585.7 |
| **256** | **573.3** |
| 1024 | 583.9 |
| 4096 | 593.7 |

Total spread **3.5%**, non-monotonic, far inside the ~12% noise floor.
**No detectable effect.** Quality unmoved (distinct/agent 1.78-1.83,
meta-consensus 0.710-0.732, calls 603-616), confirming it is throughput-only.

**Why the premise was wrong:** the 6.3:1 prefill:decode ratio is in TOKENS, not
TIME. At ~1000 tok/s prefill and ~250 tok/s decode, a run spends roughly 224 s
prefilling and 672 s decoding -- **decode is ~75% of wall-clock while being
~25% of tokens**. Prefix caching optimises the minority of the time budget, so
a null was the correct expectation. Keep 256 (harmless); do not tune it.

## (d) Slot curve at reasoning-on — **p6 CONFIRMED, penalty halves**

Measured AT reasoning-on, lanes 4, 9 runs per cell:

| slots | s/run/GPU | vs p6 |
|---:|---:|---:|
| **6** | **576.8** | — |
| 12 | 625.3 | +8% |
| 24 | 695.8 | **+20%** |

p6 remains optimal, so the collection config was right. But the **magnitude is
generation-length dependent, as suspected**: reasoning-OFF showed p24 **50%**
slower (390 vs 260); at reasoning-on it is **20%**. The curve keeps its shape
and flattens. Quality unmoved across all three (distinct/agent 1.80-1.93,
meta-consensus 0.714-0.716).

This corrects the earlier scaling law, which was measured entirely
reasoning-off and over-stated the slot penalty for the regime the collection
actually runs in.

---

## Round-1 freeze — RESOLVED (n=300 + n=30)

**Question:** is the concentration of opinion movement in round 1 genuine
dynamical settling, or saturation of the clipped 10-consideration metric?

**Answer: genuine settling, not a metric artifact — and reasoning-on
substantially unfreezes it.**

Two independent checks rule out measurement saturation:
1. **Clipping never fires.** 0.0% of agent-rounds have |raw opinion| > 1 in
   either cell, so `clip(·, -1, 1)` is never active.
2. **Weights track opinion.** `|Δweights|` follows `|Δopinion|` round by round;
   the underlying belief state genuinely stops changing.

Mean |Δopinion| per round:

| | r1 | r2 | r3 | r8 | post-r1 total | r1 share |
|---|---:|---:|---:|---:|---:|---:|
| reasoning-off, t0.3 (n=30, 180 traj) | 0.2080 | 0.0273 | 0.0068 | 0.0026 | 0.0625 | **77%** |
| reasoning-on b256, t1.0 (**n=300, 1800 traj**) | 0.1383 | 0.0512 | 0.0354 | **0.0216** | **0.2081** | **40%** |

**Post-round-1 movement is 3.3x higher under reasoning-on**, and round 1's
share falls from 77% to 40%.

The trajectory-level statistic is starker still:

| | median last-change round | frozen after r2 | clipped agent-rounds |
|---|---:|---:|---:|
| reasoning-off | **1** | **69%** | 0.00% |
| reasoning-on | **6** | **19%** | 0.00% |

Under reasoning-on the MEDIAN agent is still revising its beliefs at **round 6**,
and only 19% have settled by r2. Under reasoning-off the median agent stops at
round 1 and 69% are done by r2. Clipping is inactive in both (0.00%), so metric
saturation is excluded by direct measurement, not assumption.

### Implication for trajectory length

An earlier reading of the reasoning-OFF data suggested extending to t=12/t=16
would add "flat tail, not signal". **That does not hold at the SoTA config.**
Under reasoning-on agents are still moving at r8 (0.0188 ≈ 14% of the r1
magnitude, vs 1% for reasoning-off), so **t=8 may be truncating the dynamics**.
Combined with the Sprint-16 P4a finding that a t=4 run costs ~87% of a t=8 run,
extending trajectories is both cheap and likely to add real signal — but only
at reasoning-on.

**Recommendation: extend t beyond 8 for reasoning-on collections.** Movement at
r8 is still 0.0216 (~16% of the r1 magnitude) and the median agent is active to
r6, so t=8 truncates the dynamics. Combined with the Sprint-16 P4a finding that
a t=4 run costs ~87% of a t=8 run, longer trajectories are cheap and would add
real growth-model signal -- at reasoning-on only.

**Caveat:** the two cells differ in *both* reasoning and temperature, so this
isolates the config pair, not reasoning alone. A reasoning-off/temp-1.0 cell
would separate them.

---

## Fair use

Per `docs/CLUSTER_POLICY_JOBS.md`: `v100` is 1 node / 1 job per user, `p100` a
separate 1-job allowance. Throughout, at most one job on each was held — the
300-run collection on v100, the ablation pair on p100 — never exceeding 32
nodes or 100 queued jobs, and no multi-node CPU fleet.

---

## Projections at the recommended config

Regenerate with `PYTHONSAFEPATH=1 agora/.venv/bin/python slurm/scaling_law_report.py`.

At **569 s/run/GPU** (measured on the 300-run collection, p6 / lanes 4 /
reasoning b256 / temp 1.0):

| | 1 V100 node (2 GPUs) | note |
|---|---:|---|
| 390 runs | **30.8 h** | needs 2 chained jobs (24 h cap) |
| 1200 runs | **94.8 h** | ~4 chained jobs |

**The "4 GPU" column in the generated table is NOT attainable by one user.**
`v100` is 1 node / 1 job per user (`docs/CLUSTER_POLICY_JOBS.md`), so 2 GPUs is
the ceiling. `p100` adds a separate 1-job allowance (~1 V100-equivalent, needs
the `60-real` build), and a 32-node CPU fleet is ~2.4x the V100 node but costs
fairshare across all future jobs. Untested and likely the largest lever:
`a100-long` allows **2 jobs per user** on faster cards.

Both totals exceed the 24 h cap, so **self-resubmitting chunked jobs are
mandatory**. That mechanism is proven: the 300-run collection completed across
two chain links, resuming from per-round checkpoints, and the resume guard now
refuses to continue across a changed model / temperature / prompt variant.

### Retry-policy gap (open)

`--resume` replays the identical checkpoint state, so a DETERMINISTIC failure
can never be cleared by the orchestrator's retry pass -- observed on
`polarized_n6` seed 8, which failed 3/3 resumed attempts and then succeeded on
the FIRST checkpoint-free attempt. The retry policy should fall back to a
checkpoint-free restart after N resumed failures. Low priority at reasoning-on
(~1% failures), critical at reasoning-off (20-30%).
