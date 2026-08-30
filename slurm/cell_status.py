#!/usr/bin/env python3
"""Per-cell collection status: what is on disk, what the telemetry says.

Deliberately does NOT compute makespan-per-run. That estimator is subtle (it
must anchor on `collect_start`, not first completion, and must normalise by GPU
count) and it already lives, validated, in `slurm/scaling_law_report.py` --
use `make scaling-report`. What this reports instead are direct measurements:

  finals/ckpt/traces  file counts on disk
  disk                bytes under the cell dir
  fail                run_fail + retry_fail events
  state               running / stalled / complete / idle, from the last event
  med run             median per-run wall_s (single-run latency, NOT makespan)
  rate                completions/hour over the trailing window, when running

Usage: cell_status.py [cell ...]     (default: every cell under collection/)
"""
import json, os, sys, time, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLL = os.path.join(ROOT, "llm/traces/collection")
LOGS = os.path.join(ROOT, "llm/traces/logs")
# Telemetry is emitted only at run boundaries, so silence between run_start and
# run_done is normal and its length scales with the model. A fixed threshold
# false-alarms on big models: at 26B a single run outlasts 30 minutes, which had
# the pilot reading "stalled" while both servers were demonstrably busy. So the
# threshold follows the cell's own observed pace, and a cell that has not
# finished a run yet is reported as `warming` rather than judged at all -- from
# telemetry alone there is no way to tell a slow first run from a hung one.
STALL_FLOOR_S = 3600    # never call a cell stalled inside an hour of silence
STALL_MULT = 3          # ... or 3x its own median run, whichever is longer
TRAIL = 20              # completions in the trailing window used for the rate


def disk_and_counts(cell):
    d = os.path.join(COLL, cell)
    finals = ckpt = traces = 0
    size = 0
    for dirpath, _, names in os.walk(d):
        for nm in names:
            try:
                size += os.path.getsize(os.path.join(dirpath, nm))
            except OSError:
                pass
            if not nm.startswith("townhall_") or not nm.endswith(".json"):
                continue
            if nm.endswith("_checkpoint.json"):
                ckpt += 1
            elif "_trace_" in nm:
                traces += 1
            else:
                finals += 1
    return finals, ckpt, traces, size


def telemetry(cell):
    p = os.path.join(LOGS, f"{cell}_progress.jsonl")
    if not os.path.exists(p):
        return None
    done_ts, walls, fails, last_ts, ended = [], [], 0, 0.0, False
    with open(p) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            ts = float(d.get("ts", 0) or 0)
            last_ts = max(last_ts, ts)
            e = d.get("event")
            if e in ("run_done", "retry_done"):
                done_ts.append(ts)
                if d.get("wall_s"):
                    walls.append(float(d["wall_s"]))
            elif e in ("run_fail", "retry_fail"):
                fails += 1
            elif e == "collect_end":
                ended = True
    return dict(done_ts=sorted(done_ts), walls=sorted(walls),
                fails=fails, last_ts=last_ts, ended=ended)


def human(n):
    for u in ("B", "K", "M", "G", "T"):
        if n < 1024 or u == "T":
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024


def main():
    cells = sys.argv[1:]
    if not cells:
        cells = sorted(os.path.basename(p) for p in glob.glob(f"{COLL}/*")
                       if os.path.isdir(p))
    if not cells:
        print("no cells under llm/traces/collection/")
        return 0
    now = time.time()
    print(f"{'cell':<24}{'finals':>7}{'ckpt':>6}{'traces':>7}{'disk':>8}"
          f"{'fail':>6}  {'state':<9}{'med run':>9}{'rate/h':>8}  last event")
    print("-" * 104)
    tf = td = 0
    for c in cells:
        if not os.path.isdir(os.path.join(COLL, c)):
            print(f"{c:<24}  (no such cell)")
            continue
        finals, ckpt, traces, size = disk_and_counts(c)
        tf += finals
        td += size
        t = telemetry(c)
        state, med, rate, last = "-", "", "", ""
        if t:
            age = now - t["last_ts"]
            w = t["walls"]
            if w:
                med = f"{w[len(w) // 2]:.0f}s"
            limit = max(STALL_FLOOR_S, STALL_MULT * w[len(w) // 2]) if w else STALL_FLOOR_S
            if t["ended"]:
                state = "complete"
            elif age < limit:
                state = "RUNNING" if w else "warming"
            elif not w:
                state = "warming?"      # nothing has finished; cannot judge pace
            else:
                state = "stalled"
            if state == "RUNNING" and len(t["done_ts"]) >= 2:
                win = t["done_ts"][-TRAIL:]
                span = win[-1] - win[0]
                if span > 0:
                    rate = f"{3600 * (len(win) - 1) / span:.1f}"
            last = time.strftime("%m-%d %H:%M", time.localtime(t["last_ts"]))
            if age > 60:
                last += f" ({age / 3600:.1f}h ago)" if age > 3600 else f" ({age / 60:.0f}m ago)"
        print(f"{c:<24}{finals:>7}{ckpt:>6}{traces:>7}{human(size):>8}"
              f"{(t['fails'] if t else 0):>6}  {state:<9}{med:>9}{rate:>8}  {last}")
    print("-" * 104)
    print(f"{'TOTAL':<24}{tf:>7}{'':>6}{'':>7}{human(td):>8}")
    print("\nmed run = single-run latency, not makespan. For s/run/GPU use `make scaling-report`.")
    print("warming = started, no run finished yet; a trailing `?` means the pace is")
    print("unknown, so check the job and server logs rather than trusting this column.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
