#!/usr/bin/env python3
"""Benchmark LLM client: sequential vs parallel throughput.

Usage: python -m llm.benchmark
"""

import asyncio
import time

from .client import LLMClient

# Minimal plain prompt for throughput benchmarking.
TEST_PROMPT = (
    "Evaluate this policy claim in one short JSON object. "
    "Claim: Carbon taxes reduce emissions. "
    "Counterargument: Carbon taxes burden low-income households."
)


async def bench_sequential(client: LLMClient, n: int = 8) -> float:
    """Run n requests sequentially, return total seconds."""
    t0 = time.monotonic()
    for _ in range(n):
        await client.generate(TEST_PROMPT)
    return time.monotonic() - t0


async def bench_parallel(client: LLMClient, n: int = 8) -> float:
    """Run n requests in parallel, return total seconds."""
    t0 = time.monotonic()
    await client.generate_batch([TEST_PROMPT] * n)
    return time.monotonic() - t0


async def main() -> None:
    client = LLMClient(max_concurrent=4, max_tokens=16, temperature=0.1)

    ok = await client.health()
    if not ok:
        print("ERROR: LLM server not reachable at", client.base_url)
        return

    print(f"Model: {client.model}")
    print(f"Parallel slots: {client.max_concurrent}")
    print(f"Test prompt tokens: ~{len(TEST_PROMPT.split())}")
    print()

    # Warmup
    print("Warming up...", flush=True)
    await client.generate(TEST_PROMPT)

    n = 8
    print(f"\nSequential ({n} requests)...", flush=True)
    t_seq = await bench_sequential(client, n)
    print(f"  Total: {t_seq:.1f}s | Per request: {t_seq/n:.1f}s")

    print(f"\nParallel ({n} requests, {client.max_concurrent} concurrent)...", flush=True)
    t_par = await bench_parallel(client, n)
    print(f"  Total: {t_par:.1f}s | Per request: {t_par/n:.1f}s")

    speedup = t_seq / t_par if t_par > 0 else 0
    print(f"\nSpeedup: {speedup:.1f}x")
    print(f"Effective throughput: {n/t_par:.2f} req/s (parallel) vs {n/t_seq:.2f} req/s (sequential)")

    # Estimate for a deliberation round
    agents = 20
    calls_per_agent = 3  # voice + evaluate + reflect
    total_calls = agents * calls_per_agent
    est_parallel = total_calls / (n / t_par)
    est_sequential = total_calls / (n / t_seq)
    print(f"\nEstimated round time ({agents} agents, {calls_per_agent} calls each):")
    print(f"  Sequential: {est_sequential:.0f}s ({est_sequential/60:.1f} min)")
    print(f"  Parallel:   {est_parallel:.0f}s ({est_parallel/60:.1f} min)")


if __name__ == "__main__":
    asyncio.run(main())
