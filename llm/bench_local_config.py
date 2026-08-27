#!/usr/bin/env python3
"""Local-inference configuration sweep: find the throughput-optimal
(server flags x harness concurrency) configuration on the local GPU while
guarding tool-call quality.

For each config this script:
  1. restarts llama-server via the canonical launch script with env overrides
     (N_PARALLEL / CTX / UBATCH), waits for health;
  2. runs a MICRO benchmark: K concurrent /v1/chat/completions requests with
     usage accounting -> generation tokens/sec at the configured concurrency;
  3. runs a MACRO benchmark: one real townhall round (6 empirical agents,
     fixed seed, runner --parallel matched to server slots) -> wall time,
     LLM calls, prompt/gen token totals, per-call latency, VRAM;
  4. extracts QUALITY guards from the trace: parse-failure / retry / fallback
     markers (a fast config that breaks tool calling loses).

Results -> JSON + markdown scoreboard. The canonical server is relaunched
with default flags at the end.

Usage:
    source agora/.venv/bin/activate
    python llm/bench_local_config.py [--configs p1 p2 ...] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LAUNCH = BASE_DIR / "pipeline/runpod/scripts/start_llama_server_local.sh"
BASE_URL = "http://localhost:20434"

# name -> (server env overrides, runner --parallel, micro concurrency)
CONFIGS: dict[str, dict] = {
    "p1-baseline": {"env": {"N_PARALLEL": "1", "CTX": "4096"},  "run_par": 1, "micro": 1},
    "p2":          {"env": {"N_PARALLEL": "2", "CTX": "8192"},  "run_par": 2, "micro": 2},
    "p3":          {"env": {"N_PARALLEL": "3", "CTX": "12288"}, "run_par": 3, "micro": 3},
    "p4":          {"env": {"N_PARALLEL": "4", "CTX": "16384"}, "run_par": 4, "micro": 4},
    "p6":          {"env": {"N_PARALLEL": "6", "CTX": "24576"}, "run_par": 6, "micro": 6},
    "p4-ub1024":   {"env": {"N_PARALLEL": "4", "CTX": "16384", "UBATCH": "1024"},
                    "run_par": 4, "micro": 4},
    # prompt-prefix KV reuse: deliberation prompts are ~95% shared-prefix
    # prefill, so --cache-reuse should cut evaluate-phase latency hard.
    "p1-cache":    {"env": {"N_PARALLEL": "1", "CTX": "4096", "CACHE_REUSE": "256"},
                    "run_par": 1, "micro": 1},
    "p2-cache":    {"env": {"N_PARALLEL": "2", "CTX": "8192", "CACHE_REUSE": "256"},
                    "run_par": 2, "micro": 2},
    "p4-cache":    {"env": {"N_PARALLEL": "4", "CTX": "16384", "CACHE_REUSE": "256"},
                    "run_par": 4, "micro": 4},
    "p6-cache":    {"env": {"N_PARALLEL": "6", "CTX": "24576", "CACHE_REUSE": "256"},
                    "run_par": 6, "micro": 6},
}

MICRO_PROMPT = (
    "You are agent A_03 in a town-hall deliberation about a $15/hour minimum "
    "wage. Rate the persuasiveness of this argument from 0 to 100 and reply "
    "with only the integer. Argument: raising the wage floor reduces employee "
    "turnover, which lowers hiring costs for small businesses."
)


def http_json(path: str, payload: dict | None = None, timeout: float = 180.0):
    req = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def server_pids() -> list[int]:
    out = subprocess.run(["pgrep", "-f", "llama-server --model"],
                         capture_output=True, text=True)
    return [int(x) for x in out.stdout.split()]


def stop_server() -> None:
    for pid in server_pids():
        os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if not server_pids():
            return
        time.sleep(0.5)
    for pid in server_pids():
        os.kill(pid, signal.SIGKILL)
    time.sleep(1)


def start_server(env_overrides: dict[str, str], log_path: Path) -> None:
    stop_server()
    env = {**os.environ, **env_overrides}
    with open(log_path, "w") as log:
        subprocess.Popen(["bash", str(LAUNCH)], env=env,
                         stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            http_json("/v1/models")
            return
        except Exception:
            time.sleep(1.0)
    raise RuntimeError(f"server failed to come up; see {log_path}")


def vram_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    return int(out.stdout.strip().splitlines()[0])


def micro_bench(concurrency: int, n_requests: int = 12) -> dict:
    payload = {
        "model": "gemma-4-E2B-it-Q8_0.gguf",
        "messages": [{"role": "user", "content": MICRO_PROMPT}],
        "max_tokens": 64, "temperature": 0.7,
    }

    def one(_):
        t0 = time.monotonic()
        resp = http_json("/v1/chat/completions", payload)
        dt = time.monotonic() - t0
        u = resp.get("usage", {})
        return dt, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)

    one(0)  # warm-up, uncounted
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as tp:
        results = list(tp.map(one, range(n_requests)))
    wall = time.monotonic() - t0
    gen = sum(r[2] for r in results)
    return {
        "wall_s": round(wall, 2),
        "req_per_s": round(n_requests / wall, 2),
        "gen_tok_per_s": round(gen / wall, 1),
        "mean_latency_s": round(statistics.mean(r[0] for r in results), 2),
    }


def macro_bench(name: str, run_par: int, out_dir: Path) -> dict:
    run_dir = out_dir / f"macro_{name}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    env = {**os.environ, "PYTHONPATH": ".:agora/src",
           "LLM_BASE_URL": BASE_URL, "LLM_API_FLAVOR": "openai",
           "LLM_MODEL": "gemma-4-E2B-it-Q8_0.gguf"}
    cmd = [
        str(BASE_DIR / "agora/.venv/bin/python"), "-m", "llm.townhall.runner",
        "--topic", "minimum_wage_seattle",
        "--scenario-path", "llm/scenarios/minimum_wage_seattle_crossover.json",
        "--agents", "6", "--rounds", "1", "--seed", "42",
        "--condition", "baseline", "--empirical-init",
        "--profiles-path", "polis-analysis/output/ising_profiles.json",
        "--theta-path", "polis-analysis/output/irt_ising_theta.json",
        "--composition", "symmetric_n6",
        "--parallel", str(run_par),
        "--run-tag", f"bench_{name}",
        "--output-dir", str(run_dir),
    ]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=BASE_DIR, env=env,
                          capture_output=True, text=True, timeout=1200)
    wall = time.monotonic() - t0
    stdout = proc.stdout + proc.stderr

    traces = sorted(run_dir.glob("townhall_*_trace_*.json"))
    metrics: dict = {"wall_s": round(wall, 1), "exit": proc.returncode}
    if proc.returncode != 0:
        metrics["error_tail"] = stdout[-400:]
        return metrics
    if traces:
        t = json.loads(traces[-1].read_text())
        calls = t.get("calls", [])
        lat = [c["elapsed_s"] for c in calls if "elapsed_s" in c]
        metrics.update({
            "n_calls": t.get("n_calls", len(calls)),
            "calls_per_s": round(t.get("n_calls", len(calls)) / wall, 2),
            "prompt_tok": sum(c.get("prompt_tokens", 0) for c in calls),
            "gen_tok": sum(c.get("gen_tokens", 0) for c in calls),
            "median_call_s": round(statistics.median(lat), 2) if lat else None,
            "p95_call_s": round(sorted(lat)[int(0.95 * len(lat))], 2) if lat else None,
        })
        metrics["gen_tok_per_s"] = round(metrics["gen_tok"] / wall, 1)
    # quality guards from runner output + trace
    low = stdout.lower()
    metrics["quality"] = {
        "retries": low.count("retry"),
        "nudges": low.count("nudge"),
        "fallbacks": low.count("fallback"),
        "http_errors": low.count("http error") + low.count(" 500 ") + low.count(" 400 "),
    }
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS),
                    choices=list(CONFIGS))
    ap.add_argument("--out-dir", type=Path,
                    default=BASE_DIR / "outputs/llm_engine/bench_local_config")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for name in args.configs:
        cfg = CONFIGS[name]
        print(f"\n=== {name}: server {cfg['env']} | runner --parallel {cfg['run_par']} ===",
              flush=True)
        start_server(cfg["env"], args.out_dir / f"server_{name}.log")
        time.sleep(2)
        row = {"config": name, **cfg["env"], "run_par": cfg["run_par"]}
        try:
            row["micro"] = micro_bench(cfg["micro"])
            print(f"  micro: {row['micro']}", flush=True)
            row["macro"] = macro_bench(name, cfg["run_par"], args.out_dir)
            print(f"  macro: wall={row['macro'].get('wall_s')}s "
                  f"calls={row['macro'].get('n_calls')} "
                  f"quality={row['macro'].get('quality')}", flush=True)
            row["vram_mib"] = vram_mib()
        except Exception as e:  # keep sweeping; record the failure
            row["error"] = repr(e)
            print(f"  ERROR: {e!r}", flush=True)
        rows.append(row)
        (args.out_dir / "results.json").write_text(json.dumps(rows, indent=1))

    # restore canonical server
    print("\nrestoring canonical server config...", flush=True)
    start_server({}, args.out_dir / "server_canonical_restore.log")

    # scoreboard
    lines = ["# Local config sweep — scoreboard", "",
             "| config | slots | runner par | micro gen tok/s | macro wall (1 round) | calls/s | macro gen tok/s | p95 call | VRAM MiB | retries/nudges/fallbacks/http |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m, M, q = r.get("micro", {}), r.get("macro", {}), r.get("macro", {}).get("quality", {})
        lines.append(
            f"| {r['config']} | {r.get('N_PARALLEL')} | {r.get('run_par')} "
            f"| {m.get('gen_tok_per_s', '—')} | {M.get('wall_s', '—')}s "
            f"| {M.get('calls_per_s', '—')} | {M.get('gen_tok_per_s', '—')} "
            f"| {M.get('p95_call_s', '—')}s | {r.get('vram_mib', '—')} "
            f"| {q.get('retries', '—')}/{q.get('nudges', '—')}/{q.get('fallbacks', '—')}/{q.get('http_errors', '—')} |")
    (args.out_dir / "scoreboard.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out_dir}/results.json and scoreboard.md")


if __name__ == "__main__":
    main()
