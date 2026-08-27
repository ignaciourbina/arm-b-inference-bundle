#!/usr/bin/env python3
"""Minimal integration test: 2 agents, 1 round, 3 considerations.

Uses the AgenticLLMEngine with tool calling.
Saves trace + results JSON to llm/traces/.

Usage:
    PYTHONPATH=.:agora/src llm/.venv/bin/python llm/tests/minimal_round.py
"""

import json
import time
from pathlib import Path

import numpy as np

from llm.influence_scale import format_influence_likert

from agora.agents import Agent, AgentParams  # type: ignore[import-untyped]
from agora.considerations import ArgumentPool, AttackGraph, Consideration  # type: ignore[import-untyped]
from llm.client import LLMClient
from llm.engine import AgenticLLMEngine

TRACES_DIR = Path(__file__).resolve().parent.parent / "traces"


def build_pool() -> ArgumentPool:
    pool = ArgumentPool()
    pool.add(Consideration("C_01", "Carbon pricing reduces emissions effectively", 1.0, 0.9))
    pool.add(Consideration("C_02", "Carbon taxes burden low-income households", -1.0, 0.85))
    pool.add(Consideration("C_03", "Revenue funds green transition", 1.0, 0.7))
    pool.attack_graph = AttackGraph()
    pool.attack_graph.add_attack("C_03", "C_02", 0.7)
    return pool


def build_agents() -> list[Agent]:
    a0 = Agent(id="A_00", params=AgentParams(open_mindedness=0.5))
    a0.weights = {"C_01": 0.7, "C_02": -0.3, "C_03": 0.4}

    a1 = Agent(id="A_01", params=AgentParams(open_mindedness=0.6))
    a1.weights = {"C_01": -0.2, "C_02": 0.6, "C_03": -0.1}

    return [a0, a1]


def run() -> None:
    client = LLMClient(max_tokens=512, temperature=0.2, timeout=300.0)
    engine = AgenticLLMEngine(client=client)
    pool = build_pool()
    agents = build_agents()
    rng = np.random.default_rng(42)

    results: dict = {
        "engine": "AgenticLLMEngine",
        "agents_initial": {
            a.id: {"weights": dict(a.weights), "opinion": round(a.opinion(pool), 4)}
            for a in agents
        },
        "voice": [],
        "evaluate": [],
        "reflect": {},
    }

    t0 = time.monotonic()

    # --- VOICE ---
    print("=== VOICE ===")
    voiced: list[tuple[Agent, str]] = []
    for a in agents:
        cid = engine.voice(a, pool, rng)
        if cid is None:
            print(f"  {a.id} skips voice")
            continue
        c = pool.get(cid)
        print(f"  {a.id} voices {cid}: \"{c.label}\"")
        voiced.append((a, cid))
        results["voice"].append({"agent": a.id, "cid": cid, "label": c.label})

    # --- EVALUATE ---
    print("\n=== EVALUATE ===")
    updates: dict[str, list[tuple[str, float, float]]] = {a.id: [] for a in agents}
    for speaker, cid in voiced:
        for listener in agents:
            if listener.id == speaker.id:
                continue
            score = engine.evaluate(listener, cid, speaker.opinion(pool), pool, rng)
            updates[listener.id].append((cid, speaker.opinion(pool), score))
            print(f"  {listener.id} hears {cid} -> persuasiveness={format_influence_likert(score)}")
            results["evaluate"].append({
                "listener": listener.id,
                "speaker": speaker.id,
                "cid": cid,
                "influence_likert": score,
            })

    # --- REFLECT ---
    print("\n=== REFLECT ===")
    for a in agents:
        old_w = dict(a.weights)
        old_op = a.opinion(pool)
        engine.reflect(a, updates[a.id], pool, rng)
        new_op = a.opinion(pool)
        deltas = {
            c: {"old": round(old_w[c], 4), "new": round(a.weights[c], 4)}
            for c in a.weights if abs(old_w[c] - a.weights[c]) > 0.001
        }
        print(f"  {a.id}: opinion {old_op:+.3f} -> {new_op:+.3f}")
        for c, d in deltas.items():
            print(f"    {c}: {d['old']:+.4f} -> {d['new']:+.4f}")
        results["reflect"][a.id] = {
            "opinion_before": round(old_op, 4),
            "opinion_after": round(new_op, 4),
            "weight_deltas": deltas,
        }

    elapsed = time.monotonic() - t0
    results["elapsed_s"] = round(elapsed, 1)
    results["llm_calls"] = client._seq
    print(f"\nDONE: {elapsed:.1f}s, {client._seq} LLM calls")

    # --- SAVE ---
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    trace_path = TRACES_DIR / f"agentic_trace_{ts}.json"
    client.save_trace(trace_path)
    print(f"Trace:   {trace_path}")

    results_path = TRACES_DIR / f"agentic_results_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    run()
