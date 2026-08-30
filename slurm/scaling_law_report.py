#!/usr/bin/env python3
"""Measured wall-clock scaling law for Arm-B on SeaWulf V100s.

Objective function is MAKESPAN PER RUN (s/run) -- never aggregate tok/s.
(SCALING-FINDINGS.md: "never infer collection wall-clock from an aggregate
tok/s curve".)

Makespan is read from the orchestrator telemetry rather than job walltime, so
server startup and queue time are excluded. Each cell is normalised to
*per-GPU* s/run: a cell sharded across 2 GPUs completes twice the work per
second of wall-clock, and comparing an unnormalised 2-GPU cell against a 1-GPU
cell is exactly the mistake that inverts the reasoning-cost conclusion.
"""
import json, os, sys, glob
from collections import OrderedDict

LOGS = "llm/traces/logs"

def cell_stats(cell, n_gpus=1):
    p = f"{LOGS}/{cell}_progress.jsonl"
    if not os.path.exists(p):
        return None
    ts = []
    launches = []
    fails = retries = 0
    for line in open(p):
        try: d = json.loads(line)
        except Exception: continue
        e = d.get("event")
        if e in ("run_done", "retry_done"): ts.append(d["ts"])
        elif e == "collect_start": launches.append(d["ts"])

        elif e == "run_fail": fails += 1
        elif e == "retry_fail": retries += 1
    if len(ts) < 2: return None
    # Measure from the first run START, not the first completion. Using
    # first-to-last completion undercounts badly at small n: with L lanes the
    # first L runs finish together, so span/n reads ~1/L of the true pace until
    # n >> L. That bias is what made an 8-run cell look 1.6x faster than it is.
    # A cell's telemetry can span several orchestrator launches (cancellations,
    # restarts) with idle gaps that are not work. `collect_start` marks each
    # launch, so score the launch that produced the most completed runs and
    # measure from ITS start marker. Measuring first-to-last completion instead
    # undercounts at small n (with L lanes the first L finish together), which
    # is what made an 8-run cell look 1.6x faster than it really was.
    if launches:
        bounds = sorted(launches) + [float("inf")]
        best_n, best = -1, None
        for lo, hi in zip(bounds, bounds[1:]):
            got = [t for t in ts if lo <= t < hi]
            if len(got) > best_n: best_n, best = len(got), (lo, got)
        t0, got = best
        n = len(got)
        if n < 2: return None
        span = max(got) - t0
    else:
        n = len(ts); span = max(ts) - min(ts)
        if n < 2: return None

    return dict(n=n, span=span,
                per_run=span / n,                    # as-run (all GPUs)
                per_run_gpu=(span / n) * n_gpus,     # normalised to one GPU
                runs_h=3600 * n / span,
                fails=fails, persistent=retries)

# (label, cell, n_gpus, axis)
CELLS = [
    ("reasoning off, temp 0.3  (baseline)", "seawulf_test_l4",  2, "reasoning"),
    ("reasoning b256, temp 1.0 (300-run)",  "armb_sota_r256_t10", 2, "reasoning"),
    ("reasoning b256, temp 0.3",            "sota_r256_t03",    1, "reasoning"),
    ("reasoning b256, temp 1.0",            "sota_r256_t10",    1, "reasoning"),
]
for L in (1, 2, 4, 6):
    CELLS.append((f"lanes={L}  (p24, off, 0.3)", f"sl_lanes_L{L}", 1, "lanes"))
for P in (6, 12, 24, 48):
    CELLS.append((f"slots p={P}  (lanes=4, off, 0.3)", f"sl_slots_p{P}", 1, "slots"))
CELLS.append(("L=6 @ p6  (predicted optimum)", "vo_L6_p6", 1, "verify"))
CELLS.append(("L=4 @ p6  (control)",           "vo_L4_p6", 1, "verify"))
# reasoning-ON axes (the regime the collection actually runs in)
for CR in (0, 256, 1024, 4096):
    CELLS.append((f"cache-reuse {CR}  (p6, b256, t1.0)", f"cd_cache_{CR}", 1, "cache [reasoning-ON]"))
for P in (6, 12, 24):
    CELLS.append((f"slots p={P}  (lanes=4, b256, t1.0)", f"cd_slots_p{P}", 1, "slots [reasoning-ON]"))
CELLS.append(("Q4_0 @ b256, t1.0 (P100)", "abl_q4_b256", 1, "quant"))
CELLS.append(("b128 @ Q8, t1.0 (P100)",   "abl_q8_b128", 1, "budget"))

rows = OrderedDict()
for label, cell, g, axis in CELLS:
    s = cell_stats(cell, g)
    if s: rows[label] = (s, axis, cell)

if not rows:
    print("no completed cells yet"); sys.exit(0)

print("=" * 96)
print("MEASURED WALL-CLOCK SCALING LAW  —  objective: makespan per run (s/run)")
print("=" * 96)
cur = None
for label, (s, axis, cell) in rows.items():
    if axis != cur:
        cur = axis
        print(f"\n--- axis: {axis} ---")
        print(f"  {'config':<38} {'n':>3} {'s/run(GPU)':>11} {'runs/h/GPU':>11} {'fail':>5}")
    print(f"  {label:<38} {s['n']:>3} {s['per_run_gpu']:>11.1f} {3600/s['per_run_gpu']:>11.1f} {s['fails']:>5}")

# fastest config + projections
best = min(rows.items(), key=lambda kv: kv[1][0]["per_run_gpu"])
bl, (bs, _, _) = best
print("\n" + "=" * 96)
print(f"FASTEST MEASURED CONFIG: {bl}  ->  {bs['per_run_gpu']:.1f} s/run/GPU")
print("=" * 96)
print(f"\n{'config':<38} {'390 @2GPU':>10} {'390 @4GPU':>10} {'1200 @2GPU':>11} {'1200 @4GPU':>11}")
for label, (s, axis, cell) in rows.items():
    r = s["per_run_gpu"]
    f = lambda N, g: f"{N * r / g / 3600:.1f} h"
    print(f"{label:<38} {f(390,2):>10} {f(390,4):>10} {f(1200,2):>11} {f(1200,4):>11}")
print("\nnote: 24 h scheduler cap -> anything above ~24 h needs chunked, "
      "self-resubmitting jobs (--resume replays completed runs in ~1 s).")
