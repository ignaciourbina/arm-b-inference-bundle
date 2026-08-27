#!/usr/bin/env python3
"""Overnight deliberation run: realistic scenario, multiple rounds.

6 agents, 5 rounds, carbon tax scenario (10 considerations, 8 attacks).
Saves per-round snapshots and full trace.

Usage:
    PYTHONPATH=.:agora/src llm/.venv/bin/python llm/tests/overnight_run.py
"""

import json
import time
from pathlib import Path

import numpy as np

from agora.agents import Agent, AgentParams  # type: ignore[import-untyped]
from agora.considerations import ArgumentPool  # type: ignore[import-untyped]
from llm.client import LLMClient
from llm.engine import AgenticLLMEngine
from llm.influence_scale import format_influence_likert
from llm.scenario_loader import load_builtin

TRACES_DIR = Path(__file__).resolve().parent.parent / "traces"

N_AGENTS = 6
N_ROUNDS = 5
SEED = 42


def build_agents(pool: ArgumentPool, rng: np.random.Generator) -> list[Agent]:
    agents = []
    for i in range(N_AGENTS):
        params = AgentParams(
            open_mindedness=float(rng.uniform(0.2, 0.8)),
            prior_precision=float(rng.uniform(0.5, 3.0)),
        )
        agent = Agent(id=f"A_{i:02d}", params=params)
        cids = pool.sample_repertoire(6, rng)
        for cid in cids:
            c = pool.get(cid)
            agent.weights[cid] = float(np.clip(
                c.direction * rng.uniform(0.2, 0.8), -1.0, 1.0,
            ))
        agents.append(agent)
    return agents


def snapshot_agents(agents: list[Agent], pool: ArgumentPool) -> list[dict]:
    return [
        {
            "id": a.id,
            "opinion": round(a.opinion(pool), 4),
            "weights": {c: round(w, 4) for c, w in a.weights.items()},
            "open_mindedness": round(a.params.open_mindedness, 3),
        }
        for a in agents
    ]


def run_round(
    round_num: int,
    agents: list[Agent],
    pool: ArgumentPool,
    engine: AgenticLLMEngine,
    rng: np.random.Generator,
) -> dict:
    """Run one deliberation round: voice → evaluate → reflect."""
    print(f"\n{'='*60}")
    print(f"  ROUND {round_num}")
    print(f"{'='*60}")

    round_data: dict = {"round": round_num, "voice": [], "evaluate": [], "reflect": {}}
    t0 = time.monotonic()

    # --- VOICE ---
    print("\n  VOICE:")
    voiced: list[tuple[Agent, str]] = []
    for a in agents:
        cid = engine.voice(a, pool, rng)
        if cid is None:
            print(f"    {a.id} skips voice")
            continue
        c = pool.get(cid)
        print(f"    {a.id} (op={a.opinion(pool):+.3f}) voices {cid}: \"{c.label[:50]}...\"")
        voiced.append((a, cid))
        round_data["voice"].append({"agent": a.id, "cid": cid})

    # --- EVALUATE ---
    print("\n  EVALUATE:")
    updates: dict[str, list[tuple[str, float, float]]] = {a.id: [] for a in agents}
    for speaker, cid in voiced:
        for listener in agents:
            if listener.id == speaker.id:
                continue
            score = engine.evaluate(listener, cid, speaker.opinion(pool), pool, rng)
            updates[listener.id].append((cid, speaker.opinion(pool), score))
            print(
                f"    {listener.id} hears {cid} from {speaker.id} -> "
                f"pers={format_influence_likert(score)}"
            )
            round_data["evaluate"].append({
                "listener": listener.id, "speaker": speaker.id,
                "cid": cid, "influence_likert": score,
            })

    # --- REFLECT ---
    print("\n  REFLECT:")
    for a in agents:
        old_op = a.opinion(pool)
        old_w = dict(a.weights)
        engine.reflect(a, updates[a.id], pool, rng)
        new_op = a.opinion(pool)
        delta = new_op - old_op
        changed = sum(1 for c in a.weights if abs(old_w.get(c, 0) - a.weights[c]) > 0.001)
        print(f"    {a.id}: {old_op:+.3f} -> {new_op:+.3f} (delta={delta:+.3f}, {changed} weights changed)")
        round_data["reflect"][a.id] = {
            "opinion_before": round(old_op, 4),
            "opinion_after": round(new_op, 4),
            "weights_changed": changed,
        }

    elapsed = time.monotonic() - t0
    round_data["elapsed_s"] = round(elapsed, 1)
    print(f"\n  Round {round_num} complete: {elapsed:.0f}s")
    return round_data


def main() -> None:
    print(f"Overnight run: {N_AGENTS} agents, {N_ROUNDS} rounds, carbon_tax scenario")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    client = LLMClient(max_tokens=512, temperature=0.3, timeout=300.0)
    engine = AgenticLLMEngine(client=client, confirmation_bias=0.3)
    pool = load_builtin("carbon_tax")
    rng = np.random.default_rng(SEED)
    agents = build_agents(pool, rng)

    results: dict = {
        "config": {
            "n_agents": N_AGENTS,
            "n_rounds": N_ROUNDS,
            "scenario": "carbon_tax",
            "seed": SEED,
            "model": client.model,
            "confirmation_bias": engine.confirmation_bias,
        },
        "snapshots": [snapshot_agents(agents, pool)],
        "rounds": [],
    }

    print("\nInitial opinions:")
    for a in agents:
        print(f"  {a.id}: {a.opinion(pool):+.3f} (om={a.params.open_mindedness:.2f})")

    total_t0 = time.monotonic()
    for r in range(1, N_ROUNDS + 1):
        round_data = run_round(r, agents, pool, engine, rng)
        results["rounds"].append(round_data)
        results["snapshots"].append(snapshot_agents(agents, pool))

    total_elapsed = time.monotonic() - total_t0

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"LLM calls: {client._seq}")
    print(f"Avg per call: {total_elapsed/client._seq:.1f}s" if client._seq else "")
    print("\nFinal opinions:")
    initial = results["snapshots"][0]
    for i, a in enumerate(agents):
        init_op = initial[i]["opinion"]
        final_op = a.opinion(pool)
        print(f"  {a.id}: {init_op:+.3f} -> {final_op:+.3f} (shift={final_op-init_op:+.3f})")

    results["total_elapsed_s"] = round(total_elapsed, 1)
    results["total_llm_calls"] = client._seq

    # Save
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    trace_path = TRACES_DIR / f"overnight_trace_{ts}.json"
    client.save_trace(trace_path)
    print(f"\nTrace:   {trace_path}")

    results_path = TRACES_DIR / f"overnight_results_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {results_path}")
    print(f"\nFinished: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
