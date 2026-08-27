#!/usr/bin/env python3
"""Run one Sprint-15 ablation cell: a named configuration over paired seeds.

A "cell" is (prompt variant x server config) evaluated on the SAME seeds and
compositions as every other cell, so comparisons are paired per seed rather
than between independent samples. The control cell is free: seeds 1-5 of the
validated 390-run production collection already exist and were produced by the
identical configuration this script calls `control`.

The script does three things the plain runner does not:

  * refuses to start unless the backend is actually on GPU. The Sprint-15
    P0 finding was an 11.5x silent slowdown caused by llama.cpp falling back to
    CPU after a driver upgrade, with no error raised. A cell collected on CPU
    would be scientifically fine but would blow the entire budget, so the guard
    is a hard precondition, overridable only with --allow-slow.
  * stamps LLM_ABLATION_INFRA into the environment so each trace records the
    server flags it was produced under (the runner writes this into config).
  * writes each cell to its own directory, leaving the production collection
    untouched.

Usage:
    python llm/run_ablation_cell.py --cell prompt_anti-repetition \
        --prompt-variant anti-repetition --infra "reasoning=off,quant=Q8_0"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ROOT = BASE_DIR / "llm/traces/ablation"
COMPOSITIONS = ["polarized_n6", "symmetric_n6", "three_clusters_n6"]

# Below this the backend is not on the GPU. Measured: Vulkan ~43 tok/s,
# CPU fallback ~3.75 tok/s (Sprint-15 P0).
MIN_EVAL_TOKS_PER_S = 20.0


def _one_completion(base_url: str, max_tokens: int, timeout: float) -> tuple[float, int]:
    """(wall_seconds, completion_tokens) for one short chat completion."""
    body = {
        "model": "gemma-4-E2B-it-Q8_0.gguf",
        "messages": [{"role": "user", "content":
                      "In two sentences, why might a minimum wage raise prices?"}],
        "temperature": 0.3, "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    dt = time.monotonic() - t0
    return dt, int((data.get("usage") or {}).get("completion_tokens", 0))


def bench_backend(base_url: str, timeout: float = 300.0) -> float:
    """Steady-state generation speed in tok/s.

    The first request after a server (re)start pays a one-off graph-compile /
    prompt-eval warmup (~4s cold on Vulkan) that makes a naive gen/wall-clock
    measurement read ~10 tok/s even when the GPU sustains ~67 tok/s. Since the
    programme restarts the server at every phase boundary, we discard a warmup
    call and measure the second one — otherwise the cold false-negative would
    trip the CPU-fallback guard and abort each phase.
    """
    _one_completion(base_url, max_tokens=16, timeout=timeout)  # warm the graph
    dt, gen = _one_completion(base_url, max_tokens=80, timeout=timeout)
    return gen / dt if dt > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", required=True, help="Cell name (output directory).")
    ap.add_argument("--prompt-variant", default="control")
    ap.add_argument("--infra", default="",
                    help="Free-text description of the server flags in force, "
                         "stamped into every trace (e.g. 'reasoning=on:256,quant=Q4_K_M').")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--compositions", nargs="+", default=COMPOSITIONS)
    ap.add_argument("--agents", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--base-url", default="http://localhost:20434")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--allow-slow", action="store_true",
                    help="Proceed even if the backend benches below the GPU threshold.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_root / args.cell
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cell] {args.cell}")
    print(f"[cell] prompt-variant={args.prompt_variant}  infra='{args.infra}'")
    print(f"[cell] {len(args.seeds)} seeds x {len(args.compositions)} compositions "
          f"= {len(args.seeds) * len(args.compositions)} runs -> {out_dir}")

    try:
        speed = bench_backend(args.base_url)
    except Exception as exc:
        print(f"[abort] backend unreachable at {args.base_url}: {exc}")
        return 2
    on_gpu = speed >= MIN_EVAL_TOKS_PER_S
    print(f"[cell] backend benched at {speed:.1f} tok/s "
          f"({'GPU' if on_gpu else 'CPU FALLBACK'})")
    if not on_gpu and not args.allow_slow:
        print(f"[abort] backend is below {MIN_EVAL_TOKS_PER_S:.0f} tok/s — this is the "
              "Sprint-15 P0 CPU-fallback signature. Fix the driver (see the sprint "
              "plan) or pass --allow-slow to proceed anyway.")
        return 3
    if args.dry_run:
        print("[cell] dry run; nothing executed")
        return 0

    env = dict(os.environ)
    env["LLM_ABLATION_INFRA"] = args.infra
    env.setdefault("PYTHONPATH", f"{BASE_DIR}:{BASE_DIR / 'agora' / 'src'}")

    started = time.monotonic()
    done = failed = 0
    total = len(args.seeds) * len(args.compositions)
    for comp in args.compositions:
        for seed in args.seeds:
            tag = f"{args.cell}_baseline_{comp}_s{seed}"
            cmd = [sys.executable, "-m", "llm.townhall.runner",
                   "--topic", "minimum_wage_seattle",
                   "--scenario-path", str(BASE_DIR / "llm/scenarios/minimum_wage_seattle_crossover.json"),
                   "--agents", str(args.agents), "--rounds", str(args.rounds),
                   "--seed", str(seed), "--condition", "baseline",
                   "--empirical-init",
                   "--profiles-path", str(BASE_DIR / "polis-analysis/output/ising_profiles.json"),
                   "--theta-path", str(BASE_DIR / "polis-analysis/output/irt_ising_theta.json"),
                   "--composition", comp, "--parallel", str(args.parallel),
                   "--prompt-variant", args.prompt_variant,
                   "--run-tag", tag, "--output-dir", str(out_dir), "--resume"]
            t0 = time.monotonic()
            proc = subprocess.run(cmd, cwd=str(BASE_DIR), env=env,
                                  capture_output=True, text=True)
            if proc.returncode == 0:
                done += 1
            else:
                failed += 1
                print(f"  [fail rc={proc.returncode}] {tag}: "
                      f"{proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ''}")
            elapsed = time.monotonic() - started
            rate = elapsed / max(done + failed, 1)
            print(f"  [{done + failed}/{total}] {tag} "
                  f"({time.monotonic() - t0:.0f}s)  eta {rate * (total - done - failed) / 60:.0f} min",
                  flush=True)

    mins = (time.monotonic() - started) / 60
    print(f"\n[cell] {args.cell}: {done} ok, {failed} failed, {mins:.1f} min "
          f"({mins / max(total, 1):.1f} min/run)")
    (out_dir / "CELL.json").write_text(json.dumps({
        "cell": args.cell, "prompt_variant": args.prompt_variant,
        "infra": args.infra, "seeds": args.seeds, "compositions": args.compositions,
        "agents": args.agents, "rounds": args.rounds,
        "backend_tok_per_s": round(speed, 2),
        "runs_ok": done, "runs_failed": failed, "minutes": round(mins, 1),
    }, indent=1))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
