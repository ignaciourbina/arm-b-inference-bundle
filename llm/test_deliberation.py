#!/usr/bin/env python3
"""Integration test: run a mini deliberation round with the LLM engine.

Usage: PYTHONPATH=.:agora/src llm/.venv/bin/python -m llm.test_deliberation
"""

import asyncio
import sys
import time
from pathlib import Path

import numpy as np

from llm.influence_scale import format_influence_likert

from agora.agents import Agent, AgentParams  # type: ignore[import-untyped]
from llm.client import LLMClient
from llm.engine import AgenticLLMEngine
from llm.scenario_loader import load_builtin


async def main() -> None:
    client = LLMClient(max_tokens=128, temperature=0.2)
    if not await client.health():
        print("ERROR: LLM server not reachable")
        sys.exit(1)

    engine = AgenticLLMEngine(client=client)
    pool = load_builtin("carbon_tax")
    rng = np.random.default_rng(42)

    # Create 4 agents with different repertoires and stances
    agents = []
    for i in range(4):
        params = AgentParams(open_mindedness=rng.uniform(0.3, 0.8))
        agent = Agent(id=f"A_{i:02d}", params=params)
        cids = pool.sample_repertoire(6, rng)
        for cid in cids:
            c = pool.get(cid)
            # Initial weight: slight random bias around direction
            agent.weights[cid] = float(np.clip(
                c.direction * rng.uniform(0.2, 0.8),
                -1.0, 1.0,
            ))
        agents.append(agent)
        print(f"Agent {i}: opinion={agent.opinion(pool):+.3f}, "
              f"repertoire={list(agent.weights.keys())}")

    print("\n--- Deliberation Round ---\n")

    # Phase 1: Voice
    print("VOICING:")
    voiced = []
    t0 = time.monotonic()
    for i, agent in enumerate(agents):
        cid = engine.voice(agent, pool, rng)
        if cid is None:
            print(f"  Agent {i} skips voice (no supported statements)")
            continue
        c = pool.get(cid)
        print(f"  Agent {i} voices {cid}: \"{c.label}\" (d={int(c.direction):+d})")
        voiced.append((i, cid, agent.opinion(pool)))
    t_voice = time.monotonic() - t0
    print(f"  [{t_voice:.1f}s]\n")

    # Phase 2: Evaluate (each agent evaluates each other's voiced argument)
    print("EVALUATING:")
    all_updates: dict[int, list[tuple[str, float, float]]] = {i: [] for i in range(len(agents))}
    t0 = time.monotonic()
    for speaker_idx, cid, speaker_op in voiced:
        for listener_idx, listener in enumerate(agents):
            if listener_idx == speaker_idx:
                continue
            score = engine.evaluate(listener, cid, speaker_op, pool, rng)
            all_updates[listener_idx].append((cid, speaker_op, score))
            print(f"  Agent {listener_idx} hears {cid} from Agent {speaker_idx}: "
                f"persuasiveness={format_influence_likert(score)}")
    t_eval = time.monotonic() - t0
    print(f"  [{t_eval:.1f}s]\n")

    # Phase 3: Reflect
    print("REFLECTING:")
    t0 = time.monotonic()
    for i, agent in enumerate(agents):
        old_opinion = agent.opinion(pool)
        old_weights = dict(agent.weights)
        engine.reflect(agent, all_updates[i], pool, rng)
        new_opinion = agent.opinion(pool)
        changed = {c: (old_weights[c], agent.weights[c])
                   for c in agent.weights
                   if abs(old_weights.get(c, 0) - agent.weights[c]) > 0.001}
        print(f"  Agent {i}: opinion {old_opinion:+.3f} -> {new_opinion:+.3f} "
              f"(delta={new_opinion - old_opinion:+.3f})")
        if changed:
            for c, (old, new) in changed.items():
                print(f"    {c}: {old:+.3f} -> {new:+.3f}")
    t_reflect = time.monotonic() - t0
    print(f"  [{t_reflect:.1f}s]\n")

    total = t_voice + t_eval + t_reflect
    total_calls = len(agents) + len(agents) * (len(agents) - 1) + len(agents)
    print(f"TOTAL: {total:.1f}s for {total_calls} LLM calls "
          f"({total/total_calls:.1f}s/call avg)")

    # Save trace
    trace_path = Path(__file__).parent / "traces"
    trace_file = client.save_trace(
        trace_path / f"deliberation_{int(time.time())}.json"
    )
    print(f"\nTrace saved: {trace_file}")


if __name__ == "__main__":
    asyncio.run(main())
