#!/usr/bin/env python3
"""Parallel collection orchestrator — N lanes of runner subprocesses.

Why: real runs keep only ~2.7 of the server's slots busy (frontier report
2026-08-27); the server batches to ~434 tok/s aggregate at 24-way. Running
many runs CONCURRENTLY against one p24 server (see run_collection_server.sh)
recovers the idle slots: ~2.5 days for the 1200-run collection instead of ~15.

Design (sprint-16 plan):
  * The (composition x seed) grid is sliced round-robin into N disjoint lanes;
    each lane runs its slice sequentially via the standard runner subprocess.
    Disjoint by construction — no locking, no shared state between lanes.
  * Every run uses --resume, so relaunching this orchestrator (crash, pause,
    reboot) replays completed runs from checkpoints in ~1s and continues.
  * SIGTERM/SIGINT => graceful pause: lanes finish their in-flight run, then
    stop (bounded by one run's duration). A PAUSED marker documents resume.
  * Failed runs are queued and retried once at the end; persistent failures
    are listed in the final summary and the telemetry JSONL.
  * Telemetry: one JSONL (llm/traces/logs/<cell>_progress.jsonl) with a line
    per run event, plus a status line every --status-interval seconds.

Usage:
    python llm/run_collection_parallel.py --cell collection_main \\
        --seeds 1-200 --lanes 8 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
COMPOSITIONS = ["polarized_n6", "symmetric_n6", "three_clusters_n6"]
MIN_TOKS_PER_S = 20.0          # warmed-bench gate (CPU fallback signature ~3-11)
MIN_FREE_DISK_GB = 10.0        # trace volume safety
MAX_VRAM_MB = 6500             # sprint-16 plan gate
DEFAULT_LANE_STAGGER_S = 90    # avoid synchronized prefill bursts

_pause = threading.Event()


def parse_seeds(spec: str) -> list[int]:
    """'1-200' or '1,2,5' or '1-50,101-150' -> sorted unique ints."""
    seeds: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.update(range(int(lo), int(hi) + 1))
        elif part:
            seeds.add(int(part))
    return sorted(seeds)


def build_grid(seeds: list[int], compositions: list[str]) -> list[tuple[str, int]]:
    """Composition-major grid; every (comp, seed) exactly once."""
    return [(c, s) for s in seeds for c in compositions]


def slice_lanes(grid: list[tuple[str, int]], lanes: int) -> list[list[tuple[str, int]]]:
    """Round-robin slicing: disjoint, complete, balanced within 1 item."""
    return [grid[i::lanes] for i in range(lanes)]


def completed_runs(out_dir: Path) -> set[tuple[str, int]]:
    """(comp, seed) pairs that already have a FINAL trace in out_dir."""
    import re
    done: set[tuple[str, int]] = set()
    if not out_dir.exists():
        return done
    rx = re.compile(r"_(polarized|symmetric|three_clusters)_n6_s(\d+)_\d+\.json$")
    for p in out_dir.iterdir():
        if "_trace_" in p.name or "checkpoint" in p.name:
            continue
        m = rx.search(p.name)
        if m:
            done.add((f"{m.group(1)}_n6", int(m.group(2))))
    return done


def bench_gate(base_url: str) -> float:
    """Warmed single-completion bench (mirrors run_ablation_cell's guard)."""
    def one(max_tokens: int) -> tuple[float, int]:
        body = {"model": "gate", "messages": [
            {"role": "user", "content": "In two sentences, why might a minimum wage raise prices?"}],
            "temperature": 0.3, "max_tokens": max_tokens}
        req = urllib.request.Request(f"{base_url}/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.load(resp)
        return time.monotonic() - t0, int((data.get("usage") or {}).get("completion_tokens", 0))
    one(16)  # warm the graph
    dt, gen = one(80)
    return gen / dt if dt > 0 else 0.0


def vram_mb() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def gpu_util() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "n/a"


class Telemetry:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def emit(self, **event) -> None:
        event["ts"] = round(time.time(), 1)
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(event) + "\n")


def run_one(cell: str, comp: str, seed: int, out_dir: Path, rounds: int,
            agents: int, run_parallel: int) -> int:
    tag = f"{cell}_baseline_{comp}_s{seed}"
    cmd = [sys.executable, "-m", "llm.townhall.runner",
           "--topic", "minimum_wage_seattle",
           "--scenario-path", str(BASE_DIR / "llm/scenarios/minimum_wage_seattle_crossover.json"),
           "--agents", str(agents), "--rounds", str(rounds),
           "--seed", str(seed), "--condition", "baseline",
           "--empirical-init",
           "--profiles-path", str(BASE_DIR / "polis-analysis/output/ising_profiles.json"),
           "--theta-path", str(BASE_DIR / "polis-analysis/output/irt_ising_theta.json"),
           "--composition", comp, "--parallel", str(run_parallel),
           "--prompt-variant", "control",
           "--run-tag", tag, "--output-dir", str(out_dir), "--resume"]
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", f"{BASE_DIR}:{BASE_DIR / 'agora' / 'src'}")
    proc = subprocess.run(cmd, cwd=str(BASE_DIR), env=env,
                          capture_output=True, text=True)
    return proc.returncode


def lane_worker(lane_id: int, work: list[tuple[str, int]], cell: str,
                out_dir: Path, rounds: int, agents: int, run_parallel: int,
                telemetry: Telemetry, failures: list[tuple[str, int]],
                fail_lock: threading.Lock, stagger_s: int) -> None:
    time.sleep(lane_id * stagger_s)
    for comp, seed in work:
        if _pause.is_set():
            telemetry.emit(event="lane_paused", lane=lane_id)
            return
        t0 = time.monotonic()
        telemetry.emit(event="run_start", lane=lane_id, comp=comp, seed=seed)
        rc = run_one(cell, comp, seed, out_dir, rounds, agents, run_parallel)
        wall = round(time.monotonic() - t0, 1)
        if rc == 0:
            telemetry.emit(event="run_done", lane=lane_id, comp=comp,
                           seed=seed, wall_s=wall)
        else:
            telemetry.emit(event="run_fail", lane=lane_id, comp=comp,
                           seed=seed, wall_s=wall, rc=rc)
            with fail_lock:
                failures.append((comp, seed))
    telemetry.emit(event="lane_done", lane=lane_id)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True, help="Collection name (output dir).")
    ap.add_argument("--seeds", required=True, help="e.g. '1-200' or '1-50,101-150'")
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--run-parallel", type=int, default=6,
                    help="Within-run agent concurrency (per lane).")
    ap.add_argument("--compositions", nargs="+", default=COMPOSITIONS)
    ap.add_argument("--base-url", default="http://localhost:20434")
    ap.add_argument("--out-root", type=Path, default=BASE_DIR / "llm/traces/collection")
    ap.add_argument("--status-interval", type=int, default=300)
    ap.add_argument("--lane-stagger", type=int, default=DEFAULT_LANE_STAGGER_S)
    ap.add_argument("--allow-slow", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    grid = build_grid(seeds, args.compositions)
    out_dir = args.out_root / args.cell
    out_dir.mkdir(parents=True, exist_ok=True)

    done = completed_runs(out_dir)
    todo = [(c, s) for (c, s) in grid if (c, s) not in done]
    print(f"[collect] {args.cell}: grid={len(grid)} done={len(done)} todo={len(todo)} "
          f"lanes={args.lanes}")
    if not todo:
        print("[collect] nothing to do.")
        return 0
    if args.dry_run:
        lanes = slice_lanes(todo, args.lanes)
        for i, w in enumerate(lanes):
            print(f"  lane {i}: {len(w)} runs, first={w[0] if w else '-'}")
        return 0

    # ---- gates ----
    free_gb = shutil.disk_usage(out_dir).free / 1e9
    if free_gb < MIN_FREE_DISK_GB:
        print(f"[abort] only {free_gb:.1f} GB free on trace volume")
        return 2
    try:
        speed = bench_gate(args.base_url)
    except Exception as exc:
        print(f"[abort] backend unreachable: {exc}")
        return 2
    print(f"[collect] bench gate: {speed:.1f} tok/s "
          f"({'OK' if speed >= MIN_TOKS_PER_S else 'CPU-FALLBACK SIGNATURE'})")
    if speed < MIN_TOKS_PER_S and not args.allow_slow:
        return 3
    mb = vram_mb()
    print(f"[collect] VRAM: {mb} MB (gate < {MAX_VRAM_MB})")
    if mb > MAX_VRAM_MB:
        print("[abort] VRAM above gate — lower --parallel/ctx on the server")
        return 4

    telemetry = Telemetry(BASE_DIR / f"llm/traces/logs/{args.cell}_progress.jsonl")
    telemetry.emit(event="collect_start", grid=len(grid), done=len(done),
                   todo=len(todo), lanes=args.lanes, bench_tok_s=round(speed, 1))

    def handle_stop(signum, frame):
        print(f"\n[collect] signal {signum}: pausing after in-flight runs "
              f"(bounded by one run, ~15 min) ...")
        _pause.set()
        (out_dir / "PAUSED.txt").write_text(
            f"paused by signal {signum} at {time.ctime()}; relaunch the same "
            f"command to resume (completed runs replay from checkpoints).\n")
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    failures: list[tuple[str, int]] = []
    fail_lock = threading.Lock()
    lanes = slice_lanes(todo, args.lanes)
    threads = [threading.Thread(
        target=lane_worker,
        args=(i, w, args.cell, out_dir, args.rounds, args.agents,
              args.run_parallel, telemetry, failures, fail_lock,
              args.lane_stagger),
        daemon=True)
        for i, w in enumerate(lanes) if w]
    t_start = time.monotonic()
    for t in threads:
        t.start()

    last_status = time.monotonic()
    while any(t.is_alive() for t in threads):
        time.sleep(15)  # fine tick so completion is noticed promptly
        if time.monotonic() - last_status < args.status_interval:
            continue
        last_status = time.monotonic()
        n_done = len(completed_runs(out_dir)) - len(done)
        hrs = (time.monotonic() - t_start) / 3600
        rate = n_done / hrs if hrs > 0 else 0
        eta_h = (len(todo) - n_done) / rate if rate > 0 else float("inf")
        print(f"[status] {n_done}/{len(todo)} new runs | {rate:.1f} runs/h | "
              f"eta {eta_h:.1f} h | gpu {gpu_util()} | vram {vram_mb()} MB",
              flush=True)
    for t in threads:
        t.join()

    # ---- one retry pass for failures (fresh attempt, same seed) ----
    if failures and not _pause.is_set():
        print(f"[collect] retrying {len(failures)} failed runs once ...")
        still_failed = []
        for comp, seed in failures:
            rc = run_one(args.cell, comp, seed, out_dir, args.rounds,
                         args.agents, args.run_parallel)
            telemetry.emit(event="retry_done" if rc == 0 else "retry_fail",
                           comp=comp, seed=seed, rc=rc)
            if rc != 0:
                still_failed.append((comp, seed))
        failures = still_failed

    total_done = len(completed_runs(out_dir))
    hrs = (time.monotonic() - t_start) / 3600
    telemetry.emit(event="collect_end", done_total=total_done,
                   failures=len(failures), hours=round(hrs, 2),
                   paused=_pause.is_set())
    print(f"[collect] {'PAUSED' if _pause.is_set() else 'DONE'}: "
          f"{total_done}/{len(grid)} complete, {len(failures)} persistent "
          f"failures, {hrs:.1f} h")
    if failures:
        print(f"[collect] persistent failures: {failures}")
    return 0 if not failures and not _pause.is_set() else 1


if __name__ == "__main__":
    raise SystemExit(main())
