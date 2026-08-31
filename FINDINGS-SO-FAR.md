# Findings so far — Arm-B on SeaWulf

Cumulative record, plain English. Every finding states its sample size and its
scope. Where a number cannot support a conclusion, that is said outright rather
than left for the reader to infer.

Last updated 2026-08-31. Numbers regenerated from the committed traces, not
quoted from memory.

---

## 0. What is being run, so the rest is readable

Six LLM agents deliberate for eight rounds about a $15/hour Seattle minimum
wage. Each round every agent voices one consideration, the other five rate how
much it moved them, and then each agent privately revises the weights it places
on ten considerations. Opinion is never assigned: it emerges as the
direction-weighted average of an agent's own weights.

The agents are initialised from real Polis survey respondents on this exact
issue, via an Ising/IRT model of their voting patterns. Three "composition
pools" set the panel's structure — polarized, symmetric, three-clusters.

The headline collection is 300 runs: 100 random seeds × 3 pools, one scenario,
178,999 recorded LLM calls. The seed redraws each agent's parameters, so there
are 300 distinct 6-agent panels, though only about 15 distinct parameter draws
per agent slot.

---

## 1. Reasoning changes the *shape* of deliberation, not just the score

**This is the substantive finding, and it is the best-supported thing here.**

With reasoning disabled, deliberation is effectively a one-shot update. 76.9%
of all opinion movement happens in round 1, the median agent stops revising
after round 1, and 69% of agents are frozen after round 2. The room converges
onto a single dominant argument: distinct considerations voiced falls from 4.37
to 2.83 across the eight rounds, and the share held by the single most-voiced
consideration climbs from 0.35 to 0.54.

With reasoning enabled at budget 256, deliberation sustains across all eight
rounds. Round 1 carries 39.9% of movement rather than 76.9%, the median agent
keeps revising until round 6, and only 19% are frozen after round 2.
Argumentative variety holds at about 3.4 distinct considerations instead of
collapsing to 2.8, and the top-argument share settles at 0.49 rather than 0.54.

Agreement on *reasons* keeps accumulating only when reasoning is on. Measured by
meta-consensus, the reasoning-off condition is finished by round 2 — it gains
+0.005 from round 2 to round 8, which is about one standard error, i.e. flat.
The reasoning-on condition gains +0.022 over the same span, about eleven
standard errors.

**This is the reasoning, not the temperature.** The pattern holds at temperature
0.3 and at temperature 1.0 alike, in separate cells. Comparing only the original
pair would have confounded the two, because that pair changed both at once.

Sample sizes: reasoning-off n=30 (180 agent trajectories), reasoning-on n=300
(1,800 trajectories).

---

## 2. The round-1 concentration is real settling, not a measurement artifact

The obvious worry was that opinions stop moving because they hit the ±1 clipping
boundary. They do not: clipping fires on **0.00%** of agent-rounds in both
conditions. The freeze is the agents settling, not the metric saturating.

---

## 3. What we almost concluded about bigger models was entirely our own fault

**This is the most important methodological finding in the project.**

Running an 8B model (E4B) through the same protocol, 20% of runs failed on first
attempt, against the 2B model's 3.5%. The natural reading — a more capable model
is worse at following our constrained tool-calling protocol — was wrong.

All three failures had one signature. The model replied `{"unchanged": true}`,
meaning "I am not revising anything." That is a correct answer, and it is
literally the argument object of the `no_update` tool. Our loop required that
same intent wrapped in a tool-call envelope, retried five times, and killed the
run. We had tuned prompts against the 2B model for months, and had tuned the
*scaffolding* to it without noticing.

The composition pattern corroborated the diagnosis before the fix was tried. All
three failures were in the `symmetric` pool and none in the other two, and
`symmetric` is where the 8B model emits the fewest terminal actions per run
(9.4, versus 14–15 elsewhere). The 2B model is flat at about 20 across all three
pools, because it was tuned to be.

**After one narrow fix — accept a bare argument object as its tool call — the 8B
model's first-attempt failure rate went from 20.0% to 0.0% over 15 runs.**

The general lesson: once a harness is tuned on one model, measurements of "model
quality" silently become measurements of harness fit.

---

## 4. There were two independent harness artifacts, not one

The 26B model's failures were a *different* mode, and the fix above does not
touch them. It exhausted the reflect round budget while still emitting valid
`update_weight` calls — the budget is `max(5, n_updates + 1)`, sized for how
many revisions the 2B model makes. A model that revises more considerations runs
out of turns before it can call `done_reflecting`.

It also took an HTTP 400, and the server said why: *"request (4121 tokens)
exceeds the available context size (4096 tokens)"*. Its context ceiling was set
from the 2B model's needs. At 6144 tokens per slot it peaked at 74% and the
error did not recur.

After fixing both, the 26B's first-attempt failure rate went from 13.3% to 6.7%.
Its one remaining failure is a 500 from llama.cpp — *"Failed to parse tool call
arguments as JSON"* — which is genuine malformed output, not a harness limit.

---

## 5. Only one of the two harness fixes is safe to apply retroactively

This distinction decides whether a bigger-model collection can be compared to
the existing one.

The bare-argument-object rule **provably never fires for the 2B model**. Across
all 179,605 calls of the 300-run collection, 640 had no parsed tool call and
**zero** of them would have been coerced. It cannot have changed that data.

The reflect-budget change **would** have changed it. Two of the 2B model's own
failures were "no terminal tool within budget", exactly the mode the extra slack
fixes.

The two are therefore separate settings. `format` carries only the safe rule and
is the minimal adaptation that makes the 8B model comparable to the existing
collection. `lenient` adds the unsafe one and is reserved for the 26B.

---

## 6. Speed predictions from parameter counts were wrong in both directions and in the wrong order

Decode on this hardware is memory-bandwidth-bound, so the standard prediction is
that cost scales with *active* parameters. That prediction failed.

| model | active params | predicted s/run/GPU | measured |
|---|---:|---:|---:|
| E2B (2B) | 2.0 B | — | **569** |
| E4B (8B) | 4.0 B | ~1140 | **785** |
| 26B-A4B | 3.8 B | ~1080 | **1463** |

The 8B model cost 1.4× the 2B model, not the predicted 2×. The 26B cost 2.6×,
not the predicted 1.9×. Worse, the theory got the *ordering* wrong: 26B-A4B has
fewer active parameters than E4B, so it should have been the faster of the two.
It was 1.9× slower.

Roughly 11% of the 26B's original figure was configuration, not model — giving
it back the 6 server slots it should have had (it ran with 4 against a workload
asking for 6 concurrent agents) cut its time from 1648 to 1463 s/run/GPU.

---

## 7. The measured configuration optimum

All measured on V100s, objective is makespan per run per GPU, never aggregate
tokens per second.

- **Reasoning on, budget 256.** Costs about 1.5× the wall-clock of reasoning-off
  and buys a roughly 25× reduction in failures (~1% versus 20–30%). Locked as a
  project decision; see finding 1 for why it is not a tunable.
- **Q8_0, not Q4_0.** Q4 failed 33% of runs with malformed argument keys and
  needed 144 more calls per run, so it is about 1.86× the work per completed run
  despite being a smaller model.
- **Temperature 1.0**, chosen because it samples the model's actual next-token
  distribution. A lower temperature sharpens toward the mode, so the agents
  would not be the cognition the study claims to observe. Choosing it to
  maximise a headline metric would have been a researcher-degrees-of-freedom
  error.
- **6 server slots.** Measured 260 s/run at p6 versus 390 at p24 with reasoning
  off. The penalty is generation-length dependent: at reasoning-on, p24's
  disadvantage halves from 50% to 20%.
- **4 concurrent lanes.** L6 measured 286.0 s/run against L4's 292.2 under
  matched conditions — a 2% difference, so the ceiling is real and L4 is
  adequate.
- **Prefix cache-reuse has no effect at any setting** (0, 256, 1024, 4096). The
  premise was wrong: prefill dominates *token* counts but decode is about 75% of
  wall-clock at 25% of the tokens.

Result: **569 s/run/GPU**, 300 runs in ~24 GPU-hours, validator PASS, 0
collection errors.

---

## 8. Deliberation outcomes do not depend on how much the agents explore

Agents may optionally query the argument graph (`query_supports`,
`query_attacks`) before judging. These are elective calls chosen by the model's
weights, so they are behavior, not overhead.

Within the 300-run collection, that behavior varies substantially — 32.2% to
44.6% of tool calls, median 38.2% — and predicts essentially nothing:

| outcome | correlation | 95% interval |
|---|---:|:---:|
| reason-sharing (DRI) | −0.067 | [−0.179, 0.046] |
| meta-consensus | 0.055 | [−0.058, 0.168] |
| opinion variance | 0.002 | [−0.112, 0.115] |

These intervals rule out moderate and large effects; they do not rule out small
ones. **The scope limit matters**: this is variation *within* the 2B model's
observed range. The 8B model sits near 20%, entirely outside it, so this result
does not extrapolate there.

---

## 9. The reason-sharing metric is too noisy for small cells, and this is quantified

DRI has a coefficient of variation around 110% per run, with a per-run range of
−0.27 to +0.81. Its standard error is about 0.015 at n=300 and 0.05–0.09 at
n=15.

The practical consequence was observed directly: two 15-run cells differing only
in temperature produced DRI curves that wander non-monotonically and give
contradictory answers about when reason-sharing peaks. That is noise, not
mechanism.

**Meta-consensus and argumentative diversity are the substantive indicators that
n=15 can carry** (standard errors 0.002–0.008). DRI is not.

---

## 10. Cell sizes are different instruments

A ~15-run cell measures the machine: makespan per run, failure rate, VRAM, peak
context, and whether a configuration survives a long run. A ~300-run collection
measures the phenomenon.

Reporting a substantive difference from a 15-run cell is a category error rather
than a power problem, and enlarging a pilot "to get significance" is the wrong
response because it is a different instrument.

Applying this caught a real overstatement in our own configuration memo, which
had claimed a smaller reasoning budget was "significantly better on four
outcomes" from a 15-run cell. That claim had a second defect: the cell ran on
P100 hardware against a V100 control, and at temperature 1.0 different kernels
give different floating point, hence different sampled tokens. Pairing by seed
across different hardware does not hold the trajectory fixed, so it was closer
to an unpaired comparison than a paired one.

---

## 11. Several of our own measurements were the bug

- **Makespan was computed wrongly twice.** Once by comparing an unnormalised
  2-GPU cell against 1-GPU cells ("1.95× slower"), once by measuring first-to-
  last completion, which undercounts badly when the first few runs finish
  together ("12% faster"). Both produced confident, plausible, wrong numbers.
  The correct method anchors on the collection-start marker and normalises per
  GPU.
- **A memory check read the wrong GPU.** `nvidia-smi` is an NVML tool and
  ignores `CUDA_VISIBLE_DEVICES`, so a job sharded across two GPUs read GPU 0's
  memory for both shards and aborted a model that was using 5.6 GB because a
  different model on the other card was using 26.9 GB.
- **A documented environment variable was never implemented.** Jobs setting
  `COLLECT_MAX_VRAM_MB` believed they were raising a limit that was hardcoded.
- **A status display cried "stalled" at a healthy job**, because it used a fixed
  30-minute silence threshold while telemetry only fires at run boundaries and a
  26B run takes longer than that.
- **A documented "gotcha" was itself false.** The project notes claimed
  `--reasoning on|off` does not exist in llama.cpp. It does. A wrong note is
  worse than no note, because it invites someone to "fix" working code.

---

## 12. Comparisons between cells were sometimes structurally impossible

Automated provenance checking found that the two slot-curve families share **no
composition-and-seed pairs at all** — they used disjoint seed ranges. A "paired"
comparison across them was never possible, independently of the fact that they
also ran at different temperatures.

Six cells are unbalanced, with seeds missing from failed runs. Two of them, used
as a matched pair to settle the lane-count question, are missing *different*
seeds, so that comparison ran over slightly different seed sets.

Every cell is internally uniform on model, temperature, slots and prompt — the
one known instance of a split cell was created later, by editing the harness
while a collection was in flight, and was caught automatically.

---

## 13. Storage: version control compresses this data better than the tools built for it

The traces compress dramatically because they are near-identical JSON. Measured
on one cell: per-file gzip 23.3×, single zip 23.3×, `tar.zst -19` 67.4×. **Zip is
strictly dominated** — same ratio as gzip while giving up per-file access.

Git beats the planned approach: it delta-compresses across files at **42×**, so
3.2 GB of traces occupy 136 MB of repository while the working copies stay
readable plain JSON with no reader changes. The gzip-support work planned for
the six reader scripts may be unnecessary for anything committed.

---

## 14. The platform facts that cost time to learn

- This cluster's OS is too old for any modern LLM runtime; containers are
  mandatory, not a preference.
- The GPU driver cannot compile newer CUDA intermediate code, so llama.cpp must
  be built targeting the V100's native architecture. Stock images fail at
  runtime.
- Requesting one GPU on a `v100` node consumes all 28 CPUs and strands the
  second GPU, so every job should request both and shard across them.
- `df` reports fileset size on this filesystem, not your quota. Only `myquota`
  is authoritative.
- A file named `types.py` inside the package shadows the standard library and
  kills the interpreter during startup; `PYTHONSAFEPATH=1` is required
  everywhere.
- Memory sizing from a single measurement was 3.5% pessimistic because fixed
  overhead was attributed to per-token cost. Two points give 30 KB/token of
  cache and ~26.4 GB fixed for the 26B, meaning context was never the binding
  constraint it appeared to be.

---

## 15. What is NOT yet established

- **Whether the deliberation findings hold at larger model scale.** The pilots
  measured cost and reliability only. Answering the substantive question
  requires a full collection per model, about 33 hours for the 8B model.
- **Whether the 8B model's 0% failure rate survives the minimal fix.** It was
  measured with both harness adaptations enabled, though its failures were all
  the type the safe one addresses.
- **Why the 26B model makes far fewer accepted weight changes per run** — 9.9,
  against 32.2 for the 2B model and 36.5 for the 8B — while completing all 240
  evaluations. Observed at n=15; large enough to matter if it survives.
- **Whether elective exploration matters below the 2B model's observed range.**
  The null in finding 8 does not extend to the 8B model's ~20%.
- **Whether speculative decoding via the shipped draft models changes any of the
  cost projections.** Untested.
