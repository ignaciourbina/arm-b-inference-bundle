#!/usr/bin/env python3
"""Throughput probe for one running llama-server config (frontier bench).

Measures, against the live server at --base-url:
  * single-stream generation tok/s (mean of --reps completions, after warmup)
  * N-way concurrent aggregate tok/s (all N slots firing simultaneously) —
    the number that actually prices a collection run, since the harness keeps
    up to n_agents calls in flight.
  * time-to-first-batch wall clock for the concurrent volley.

The prompt is deliberation-shaped (~600 tokens of framing + repertoire-like
lines) rather than a toy one-liner, so prefill and KV behavior resemble the
real workload. Emits one JSON line to stdout (append it to a results file).

Usage:
    python llm/bench_frontier_probe.py --label base --concurrent 6
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
import urllib.request

PROMPT = (
    "You are a participant in a structured public deliberation about a city "
    "minimum wage increase. Your position score is +0.42 on a scale from -1 "
    "(oppose) to +1 (support). The statements you currently support include: "
    + " ".join(
        f"S{i}: A ${10+i}/hour floor affects {'small employers' if i % 2 else 'take-home pay'} "
        f"through channel {i} with strength 0.{60+i}." for i in range(1, 13)
    )
    + " Considering the strongest argument on the other side, write a short "
    "reasoned reply (4-6 sentences) explaining which single consideration you "
    "would voice next and why it best represents your current view."
)


def one_call(base_url: str, max_tokens: int, timeout: float) -> tuple[float, int, int]:
    body = {
        "model": "bench",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    dt = time.monotonic() - t0
    usage = data.get("usage") or {}
    return dt, int(usage.get("completion_tokens", 0)), int(usage.get("prompt_tokens", 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--base-url", default="http://localhost:20434")
    ap.add_argument("--concurrent", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    # warmup (graph compile + slot prime)
    one_call(args.base_url, 16, args.timeout)
    one_call(args.base_url, 16, args.timeout)

    # single-stream
    singles = [one_call(args.base_url, args.max_tokens, args.timeout) for _ in range(args.reps)]
    single_toks = sum(g for _, g, _ in singles) / sum(dt for dt, _, _ in singles)
    prompt_tokens = singles[0][2]

    # concurrent volley
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=args.concurrent) as ex:
        results = list(ex.map(
            lambda _: one_call(args.base_url, args.max_tokens, args.timeout),
            range(args.concurrent),
        ))
    wall = time.monotonic() - t0
    agg_tokens = sum(g for _, g, _ in results)

    print(json.dumps({
        "label": args.label,
        "prompt_tokens": prompt_tokens,
        "single_stream_tok_s": round(single_toks, 2),
        "concurrent_n": args.concurrent,
        "concurrent_wall_s": round(wall, 2),
        "concurrent_agg_tok_s": round(agg_tokens / wall, 2),
        "per_slot_effective_tok_s": round(agg_tokens / wall / args.concurrent, 2),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
