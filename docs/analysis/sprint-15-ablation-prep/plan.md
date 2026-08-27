# Sprint 15 — Pre-Collection Ablation Programme (24 h GPU budget)

**Purpose.** Before committing ~65 h of GPU to extending Arm B from 390 to 1200
runs, establish (a) how sensitive the results are to prompt wording, and (b)
what the inference stack itself contributes — reasoning on/off, and quantization
depth. Cheap ablations first, big run second.

**Status of this plan: IMMUTABLE once committed.** All points below are to be
completed in order. Deviations are recorded as addenda, not edits.

---

## P0 — BLOCKING: the GPU is currently unusable

Measured this session, not assumed:

| | July collection | Now |
|---|---|---|
| eval speed | 43.25 tok/s | **3.75 tok/s** |
| per-call latency (76-tok gen) | ~1.8 s | **22.9 s** |
| Vulkan/GPU lines in server log | present | **none** |

Root cause: NVIDIA userspace libraries are at **580.173.02**
(`libnvidia-glcore.so.580.173.02`, NVML 580.173) while the loaded kernel module
is **580.159.03** (`/proc/driver/nvidia/version`). `nvidia-smi` fails with
"Driver/library version mismatch". Vulkan resolves through the same userspace
stack, so device enumeration fails and llama.cpp **silently falls back to CPU** —
no error, just 11.5× slower.

**Consequence for the budget.** At CPU speed a single n=6/t=8 run costs ~102 min
instead of ~8.9 min. The whole 24 h budget would buy ~14 runs — less than one
ablation cell. **Every GPU item below is gated on this fix.**

**Fix (requires the user; I will not reboot the machine):**
```bash
sudo systemctl isolate multi-user.target   # drop X so the modules are free
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia && sudo systemctl isolate graphical.target
# simplest reliable alternative: sudo reboot
```
Verification (P0 exit criterion): `nvidia-smi` reports the GPU, and a restarted
`llama-server` logs Vulkan device selection and benches **> 30 tok/s** eval.

---

## P1 — Instrumentation and knobs (no GPU required; do this while P0 is pending)

The ablations need three things the stack does not currently expose.

1. **Prompt-variant selection.** `ToolCallHarness.prompt_builder` is a dataclass
   field (`llm/harness.py:517`, `:728`) defaulting to `BASELINE_PROMPT_BUILDER`,
   and `PromptProfile` (`llm/prompts.py:233`) is a frozen dataclass of ~15
   wording knobs. But `llm/townhall/runner.py` constructs `BaselineHarness()`
   with no way to pass a profile. Add a `--prompt-variant` flag resolving a
   named registry of profiles.
2. **Config capture in the trace.** The runner writes `config` with model,
   confirmation_bias and reflect policy but records **nothing about prompts,
   reasoning or quantization**. Ablation traces would be indistinguishable
   after the fact. Add these fields.
3. **Ablation driver + scorer.** One script to run a named cell (variant ×
   pools × seeds), and one to score cells against the control on the metric
   families already used, plus the voicing-repetition statistics from
   Sprint 14 (which are the most sensitive to prompt wording).

**Control cell is free.** Seeds 1–5 of the existing validated 390-run
collection are the control; they need no GPU time. All ablation cells use the
**same seeds 1–5 × 3 pools = 15 runs** so comparisons are paired.

---

## P2 — Axis 1: prompt ablation (3 variants × 15 runs ≈ 6.7 h)

Baseline per-run cost 8.9 min (measured July, p6-cache, n=6/t=8).

| # | Variant | One-line change | Hypothesis |
|---|---|---|---|
| A | `control` | production BASELINE profile | — (existing traces, 0 h) |
| B | `anti-repetition` | voice task line asks the agent not to repeat its previous argument unless it still best reflects its view | Sprint 14 found LLM agents voice only ~1.5 distinct arguments; this tests whether that is prompt-induced or state-induced |
| C | `explicit-tradeoff` | reflect task lines require naming the strongest opposing consideration before updating | tests whether cross-camp uptake (which *declines* in Arm B) is recoverable by prompt |
| D | `terse` | strip cognitive-style and attack-context blocks | tests how much of the behaviour is carried by prompt scaffolding vs the engine |

Primary outcomes: distinct-arguments-per-agent and top-cid share (Sprint 14
table), then the seven trajectory metrics. Report paired differences vs control
on the same seeds.

## P3 — Axis 2a: reasoning (≈ 6 h, cost to be calibrated first)

`--reasoning [on|off|auto]` and **`--reasoning-budget N`** are both available,
so thinking can be *bounded* rather than all-or-nothing. Constraint: per-slot
context is **4096 tokens** and deliberation prompts already run 1.8–2.9 k, so an
unbounded budget risks context exhaustion and the tool-call parse-500 failure
mode documented in Sprint 14.

1. **Calibrate** (15 min): per-call latency and tool-call success at
   `--reasoning on` with budget ∈ {256, 512, unbounded}. Abort any setting whose
   tool-call success < 95 %.
2. Run **two** cells at the two cheapest viable budgets, 15 runs each, reducing
   to 2 seeds × 3 pools if calibration shows > 3× cost.

Primary outcomes: does thinking change trajectories, or only cost? Specifically
whether the early weight saturation (~80 % at |w| ≥ 0.99 after round 1) relaxes.

## P4 — Axis 2b: quantization (≈ 2 h)

`llama-quantize` is available locally and there is 245 G free.

1. Produce `Q4_K_M` from the existing Q8 file (requantization — document that it
   is Q8→Q4, not F16→Q4, which is a mild quality caveat).
2. One cell, 15 runs. Q4 should be *faster* than Q8, so this is the cheapest axis.

Primary outcomes: do trajectories shift, and does tool-call reliability degrade
(the parse-500 class is a formatting failure and is the most likely casualty of
heavier quantization).

## P5 — Analysis and report (≈ 1 h, no GPU)

Paired per-seed comparison of every cell against control; a generated LaTeX
table in the Sprint-14 style; a decision memo recommending the configuration for
the 1200-run extension.

---

## Budget

| Phase | GPU hours |
|---|---|
| P2 prompt (3 cells) | 6.7 |
| P3 reasoning (2 cells + calibration) | ~6 |
| P4 quantization (1 cell + quantize) | ~2 |
| Contingency / re-runs | ~4 |
| **Total** | **~19 h of 24 h** |

Margin is deliberate: Sprint 14 measured ~1 supervisor-absorbed crash per 1–2
runs before the truncation fix, and thermal/VRAM monitoring is currently
unavailable because `nvidia-smi` is broken by the same driver mismatch as P0.

## Exit criteria

1. P0 verified: GPU restored, > 30 tok/s.
2. P1 committed: prompt-variant flag, config capture, driver, scorer.
3. P2–P4 cells collected and validated.
4. P5 memo names the configuration for the 1200-run extension, with paired
   evidence for each choice.

---

## Addendum 1 (2026-08-15) — P0 root cause confirmed; fix requires the user

Diagnosis completed:

| | version |
|---|---|
| on-disk module `/lib/modules/6.14.0-36-generic/updates/dkms/nvidia.ko.zst` | **580.173.02** |
| loaded module `/proc/driver/nvidia/version` | **580.159.03** |
| userspace `libnvidia-gl-580`, NVML | 580.173.02 |

`nvidia-driver-580-open 580.173.02` is installed; the machine has not rebooted
since, so the **stale 580.159.03 module is still resident**. The on-disk module
already matches userspace, so no package work is needed — only a module reload.

**Cannot be applied by the agent:** `sudo` requires a password (not
passwordless), and `nvidia_drm` (12 users) / `nvidia_modeset` (16 users) are
held by the display server, so unloading would kill the desktop session.

**User action:** `sudo reboot` (simplest), or drop to multi-user, reload the
modules, and return to graphical.

**P0 exit check:** `nvidia-smi` reports the GPU, `/proc/driver/nvidia/version`
reads 580.173.02, and a restarted llama-server benches > 30 tok/s.

## Addendum 2 (2026-08-15) — P1 complete and validated without a GPU

All P1 items are committed (0e8beea) and independently verified:

* **Variant registry validated by unit test** (`llm/tests/test_prompt_variants.py`,
  9 new tests, suite now 50). The load-bearing property is pinned: `control`
  renders **byte-identical** voice/evaluate/reflect prompts to
  `BASELINE_PROMPT_BUILDER`, so a control cell is directly comparable with the
  production 390-run collection. Each other variant is asserted to change
  exactly its own hook and nothing else — anti-repetition touches only voice,
  explicit-tradeoff only reflect, terse shortens all three and drops the
  coherence overlay.
* **CPU-fallback guard verified live**: the cell driver benched 3.3 tok/s and
  refused to start, which is the intended P0 protection.
* **Scorer self-test**: scoring the control against itself returns exactly
  0.000 on all eight outcomes.

A CPU smoke run of the full runner raised `HookLoopError` (empty completion
after 5 attempts). Investigated rather than assumed: it used a degenerate
2-agent/no-composition configuration on the CPU-fallback backend, and the
prompt-construction tests above show all four variants render valid prompts.
Recorded as an artefact of that configuration, not a defect in the variants;
it will be re-checked on the first GPU cell.

**Still blocked at P0.** The tunnel for remote access was started
(`make tunnel-asus-desktop`, detached) but requires a GitHub device-code login
that only the user can perform. Nothing in P2–P5 can begin until the stale
NVIDIA module is replaced and the backend benches > 30 tok/s.
