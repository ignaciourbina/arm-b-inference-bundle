#!/usr/bin/env python3
"""Town hall deliberation with real debate-gpt-x profiles.

10 agents instantiated from real participant data deliberate on a policy
topic over multiple rounds. Rich history tracking produces a complete
JSON record of the deliberation.

Usage:
    PYTHONPATH=.:agora/src llm/.venv/bin/python -m llm.townhall.runner \
        --topic minimum_wage_seattle --agents 10 --rounds 3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from agora.agents import Agent  # type: ignore[import-untyped]
from agora.considerations import ArgumentPool  # type: ignore[import-untyped]
from llm.client import LLMClient
from llm.engine import AgenticLLMEngine
from llm.harness import BaselineHarness, ToolCallHarness, _voice_user_msg
from llm.influence_scale import (
    INFLUENCE_LIKERT_FALLBACK,
    format_influence_likert,
)
from llm.scenario_loader import load_builtin
from llm.townhall.compositions import COMPOSITION_NAMES, resolve_composition
from llm.townhall.data_loader import build_agents
from llm.townhall.history import (
    AgentSnapshot,
    EvaluateEvent,
    ReflectEvent,
    RoundRecord,
    TownHallRecord,
    VoiceEvent,
)

TRACES_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "llm_engine"

# Topic descriptions for prompt context.
TOPIC_DESCRIPTIONS = {
    "minimum_wage_seattle": "Should Seattle implement a $15/hour minimum wage?",
}


def _variant_note(name: str) -> str:
    """One-line description of a prompt variant, for the trace config block."""
    try:
        from llm.prompt_variants import VARIANT_NOTES
        return VARIANT_NOTES.get(name, "")
    except Exception:
        return ""


def _server_props(base_url: str) -> dict:
    """Capture what the serving backend is actually running.

    Recorded per run so an ablation cell is self-describing: the model file and
    per-slot context are read from the server rather than inferred from the
    launch script, which may not be what is running.
    """
    try:
        import urllib.request

        with urllib.request.urlopen(f"{base_url}/props", timeout=5) as resp:
            props = json.load(resp)
        gen = props.get("default_generation_settings", {}) or {}
        return {
            "model_path": props.get("model_path"),
            "total_slots": props.get("total_slots"),
            "n_ctx_slot": gen.get("n_ctx"),
        }
    except Exception:
        return {}


def snapshot_agents(agents: list[Agent], pool: ArgumentPool) -> list[AgentSnapshot]:
    """Capture current agent state."""
    return [
        {
            "id": a.id,
            "opinion": round(a.opinion(pool), 4),
            "weights": {c: round(w, 4) for c, w in a.weights.items()},
        }
        for a in agents
    ]


def run_round(
    round_num: int,
    agents: list[Agent],
    pool: ArgumentPool,
    engine: AgenticLLMEngine,
    rng: np.random.Generator,
    llm_calls_before: int,
    parallel: int = 1,
) -> RoundRecord:
    """Run one deliberation round: voice → evaluate → reflect.

    With `parallel > 1`, each phase dispatches its per-agent (or per-pair)
    LLM calls via a ThreadPoolExecutor. The configured LLM backend is
    expected to handle the requested concurrent requests.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"  ROUND {round_num}  (intra-round parallel={parallel})", flush=True)
    print(f"{'='*60}", flush=True)

    record = RoundRecord(round_num=round_num)
    t0 = time.monotonic()

    # --- VOICE ---
    print("\n  VOICE PHASE:", flush=True)
    t_voice = time.monotonic()

    def _safe_voice(a):
        try:
            return engine.voice(a, pool, rng)
        except Exception as e:
            print(f"    [warn] voice failed for {a.id}: {e}; using fallback", flush=True)
            if a.weights:
                # fallback: strongest endorsed item (w>0) with highest persuasiveness
                candidates = [c for c, w in a.weights.items() if w > 0]
                if candidates:
                    return max(candidates, key=lambda c: a.weights[c] * pool.get(c).persuasiveness)
                return None
            return pool.all_ids()[0]

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as tp:
            voice_futures = [tp.submit(_safe_voice, a) for a in agents]
            cids = [f.result() for f in voice_futures]
    else:
        cids = [_safe_voice(a) for a in agents]

    voiced: list[tuple[Agent, str]] = []
    for a, cid in zip(agents, cids):
        op = a.opinion(pool)
        if cid is None:
            print(f"    {a.id} (op={op:+.3f}) skips voice.", flush=True)
            for line in _voice_user_msg(a, pool).splitlines():
                print(f"      {line}", flush=True)
            continue
        c = pool.get(cid)
        voiced.append((a, cid))
        record.voices.append(VoiceEvent(
            agent_id=a.id, cid=cid, label=c.label, agent_opinion=round(op, 4),
        ))
        print(f"    {a.id} (op={op:+.3f}) voices {cid}: \"{c.label[:55]}...\"", flush=True)
    print(f"  voice phase: {time.monotonic()-t_voice:.1f}s", flush=True)

    # --- EVALUATE ---
    print("\n  EVALUATE PHASE:", flush=True)
    t_eval = time.monotonic()
    pending: dict[str, list[tuple[str, float, float]]] = {a.id: [] for a in agents}

    # Build (speaker, cid, listener, sp_op) pairs
    pairs: list[tuple[Agent, str, Agent, float]] = []
    for speaker, cid in voiced:
        sp_op = speaker.opinion(pool)
        for listener in agents:
            if listener.id == speaker.id:
                continue
            pairs.append((speaker, cid, listener, sp_op))

    total_evals = len(pairs)

    last_eval_printed = 0

    def _print_eval_progress(
        i: int,
        total: int,
        speaker: Agent,
        cid: str,
        listener: Agent,
        score: float,
        *,
        interval: int,
    ) -> None:
        nonlocal last_eval_printed
        if i != 1 and i != total and i % interval != 0:
            return

        if i > last_eval_printed + 1:
            print("    ...", flush=True)

        elapsed = time.monotonic() - t_eval
        eta = (elapsed / i) * (total - i) if i > 0 else 0
        print(f"    [{i}/{total}] "
              f"{listener.id} hears {cid} from {speaker.id} -> pers={format_influence_likert(score)} "
              f"(ETA: {eta:.0f}s)", flush=True)
        last_eval_printed = i

    def _do_eval(args):
        speaker, cid, listener, sp_op = args
        try:
            score = engine.evaluate(listener, cid, sp_op, pool, rng)
        except Exception as e:
            print(
                f"    [warn] eval failed for {listener.id}/{cid}: {e}; "
                f"using {format_influence_likert(INFLUENCE_LIKERT_FALLBACK)}",
                flush=True,
            )
            score = INFLUENCE_LIKERT_FALLBACK
        return (speaker, cid, listener, sp_op, score)

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as tp:
            eval_futures = [tp.submit(_do_eval, p) for p in pairs]
            for i, fut in enumerate(eval_futures, 1):
                speaker, cid, listener, sp_op, score = fut.result()
                pending[listener.id].append((cid, sp_op, score))
                record.evaluations.append(EvaluateEvent(
                    listener_id=listener.id,
                    speaker_id=speaker.id,
                    cid=cid,
                    influence_likert=int(score),
                    listener_opinion=round(listener.opinion(pool), 4),
                ))
                _print_eval_progress(
                    i, total_evals, speaker, cid, listener, score, interval=20,
                )
    else:
        for i, p in enumerate(pairs, 1):
            speaker, cid, listener, sp_op, score = _do_eval(p)
            pending[listener.id].append((cid, sp_op, score))
            record.evaluations.append(EvaluateEvent(
                listener_id=listener.id,
                speaker_id=speaker.id,
                cid=cid,
                influence_likert=int(score),
                listener_opinion=round(listener.opinion(pool), 4),
            ))
            _print_eval_progress(
                i, total_evals, speaker, cid, listener, score, interval=10,
            )
    print(f"  eval phase: {time.monotonic()-t_eval:.1f}s", flush=True)

    # --- REFLECT ---
    print("\n  REFLECT PHASE:", flush=True)
    t_ref = time.monotonic()
    old_opinions = {a.id: a.opinion(pool) for a in agents}
    old_weights = {a.id: dict(a.weights) for a in agents}

    def _do_reflect(a: Agent) -> None:
        engine.reflect(a, pending[a.id], pool, rng)

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as tp:
            list(tp.map(_do_reflect, agents))
    else:
        for a in agents:
            _do_reflect(a)

    for a in agents:
        old_op = old_opinions[a.id]
        old_w = old_weights[a.id]
        new_op = a.opinion(pool)

        deltas = {}
        changed = 0
        for c in sorted(set(old_w) | set(a.weights)):
            d = a.weights.get(c, 0.0) - old_w.get(c, 0.0)
            if abs(d) > 0.001:
                deltas[c] = round(d, 4)
                changed += 1

        print(f"    {a.id}: {old_op:+.3f} -> {new_op:+.3f} "
              f"(delta={new_op - old_op:+.3f}, {changed} weights changed)", flush=True)

        record.reflections.append(ReflectEvent(
            agent_id=a.id,
            opinion_before=round(old_op, 4),
            opinion_after=round(new_op, 4),
            weights_changed=changed,
            weight_deltas=deltas,
        ))
    print(f"  reflect phase: {time.monotonic()-t_ref:.1f}s", flush=True)

    elapsed = time.monotonic() - t0
    record.elapsed_s = round(elapsed, 1)
    record.llm_calls = engine.client._seq - llm_calls_before
    print(f"\n  Round {round_num} complete: {elapsed:.0f}s ({record.llm_calls} LLM calls)", flush=True)
    return record


def run_townhall(
    topic: str = "minimum_wage_seattle",
    n_agents: int = 10,
    n_rounds: int = 3,
    seed: int = 42,
    confirmation_bias: float = 0.3,
    output_dir: Path | None = None,
    scenario_path: Path | None = None,
    empirical_init: bool = False,
    profiles_path: Path | None = None,
    theta_path: Path | None = None,
    run_tag: str | None = None,
    parallel: int = 1,
    condition: str = "baseline",
    composition: str | None = None,
    resume: bool = False,
    prompt_variant: str = "control",
) -> TownHallRecord:
    """Execute a complete town hall deliberation.

    Parameters
    ----------
    scenario_path
        Optional explicit scenario JSON (overrides `topic` for loading).
        Use for external scenarios like the Polis crossover scenario.
    empirical_init
        If True, build agents via build_empirical_agents (Ising-profile
        weights + coherence-derived AgentParams + profile scaffolding).
        If False, use stance-mapped init via build_agents.
    profiles_path, theta_path
        Passed to build_empirical_agents when empirical_init=True.
    run_tag
        Optional suffix for output filenames (e.g., "cb060_s42") so
        multi-run sweeps don't overwrite each other.
    """
    from llm.townhall.data_loader import build_empirical_agents

    if output_dir is None:
        output_dir = TRACES_DIR

    topic_desc = TOPIC_DESCRIPTIONS.get(topic, topic)
    print("Town Hall Deliberation")
    print(f"  Topic: {topic_desc}")
    print(f"  Agents: {n_agents}, Rounds: {n_rounds}, Seed: {seed}")
    print(f"  CB: {confirmation_bias}, empirical_init: {empirical_init}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Load scenario
    if scenario_path is not None:
        from llm.scenario_loader import load_scenario
        pool = load_scenario(scenario_path)
    else:
        pool = load_builtin(topic)
    print(f"  Scenario: {len(pool.considerations)} considerations, "
          f"{len(pool.attack_graph.attacks)} attacks, "
          f"{len(pool.attack_graph.supports)} supports")

    # 2. Build agents
    if empirical_init:
        comp_dict = resolve_composition(composition, n_agents=n_agents)
        agents, profiles = build_empirical_agents(
            pool,
            n=n_agents,
            seed=seed,
            profiles_path=profiles_path,
            theta_path=theta_path,
            composition=comp_dict,
        )
        print(f"  Profiles: {n_agents} agents from Ising stratification "
              f"(composition={composition or 'empirical-default'}, "
              f"voting pattern = profile-derived weights)")
    else:
        agents, profiles = build_agents(pool, topic, n_agents, seed)
        print(f"  Profiles loaded: "
              f"{sum(1 for p in profiles if p['stance']=='Pro')} Pro, "
              f"{sum(1 for p in profiles if p['stance']=='Con')} Con")

    # 3. Build engine with condition-specific harness
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"townhall_{topic}_trace_live.json"
    # LLM backend is env-configurable so the same runner can drive
    # OpenAI-compatible backends such as llama-server or vLLM.
    # LLM_BASE_URL / LLM_MODEL / LLM_API_FLAVOR override LLMClient defaults.
    client_kwargs: dict = {
        "max_tokens": 512,
        "temperature": 0.3,
        "timeout": 600.0,
        "trace_path": trace_path,
    }
    if os.environ.get("LLM_BASE_URL"):
        client_kwargs["base_url"] = os.environ["LLM_BASE_URL"]
    if os.environ.get("LLM_MODEL"):
        client_kwargs["model"] = os.environ["LLM_MODEL"]
    if os.environ.get("LLM_API_FLAVOR"):
        client_kwargs["api_flavor"] = os.environ["LLM_API_FLAVOR"]
    if client_kwargs.get("api_flavor") == "openai-cloud":
        # Lazy import: the cloud adapter (api.openai.com, GPT-5 mini) is only
        # loaded when selected, so local backends never depend on it. It strips
        # sampling params GPT-5 rejects (temperature stays harmless here) and
        # enforces its own budget guard. See docs/design/
        # openai-gpt5-mini-adapter-design.md.
        from llm.openai_adapter import OpenAICloudClient
        client_kwargs.pop("base_url", None)  # cloud URL is fixed in the adapter
        client = OpenAICloudClient(**client_kwargs)
    else:
        client = LLMClient(**client_kwargs)
    harness: ToolCallHarness
    if condition == "baseline":
        harness_kwargs: dict = {
            "client": client,
            "confirmation_bias": confirmation_bias,
            "topic_description": topic_desc,
        }
        if prompt_variant != "control":
            # "control" deliberately falls through to BaselineHarness' own
            # default builder, so a control run is byte-identical to production.
            from llm.prompt_variants import resolve_variant
            harness_kwargs["prompt_builder"] = resolve_variant(prompt_variant)
        harness = BaselineHarness(**harness_kwargs)
        print(f"  Condition: BASELINE  (prompt variant: {prompt_variant})")
    else:
        raise ValueError(f"Unknown condition '{condition}'; expected baseline")
    engine = AgenticLLMEngine(client=client, confirmation_bias=confirmation_bias)
    engine._harness = harness

    # 4. Initialize or resume record
    tag_infix = f"_{run_tag}" if run_tag else ""
    checkpoint_path = output_dir / f"townhall_{topic}{tag_infix}_checkpoint.json"
    start_round = 1
    resumed = False
    if resume and checkpoint_path.exists():
        record = TownHallRecord.from_json(checkpoint_path)
        # Config sanity check — refuse to resume across incompatible configs.
        ck = record.config
        mismatches = [
            (k, v, ck.get(k)) for k, v in (
                ("topic", topic), ("n_agents", n_agents),
                ("n_rounds", n_rounds), ("seed", seed),
            ) if ck.get(k) != v
        ]
        if mismatches:
            raise ValueError(
                f"Cannot resume {checkpoint_path.name}: config mismatch: "
                + ", ".join(f"{k} want={w} got={g}" for k, w, g in mismatches)
            )
        # Restore agent weights from the last snapshot.
        last_snap = record.snapshots[-1] if record.snapshots else []
        snap_by_id = {s["id"]: s for s in last_snap}
        for a in agents:
            if a.id in snap_by_id:
                a.weights = {c: float(w) for c, w in snap_by_id[a.id]["weights"].items()}
        start_round = len(record.rounds) + 1
        resumed = True
        print(f"  [RESUME] Loaded checkpoint with {len(record.rounds)} completed rounds; "
              f"continuing from round {start_round}.")
    else:
        record = TownHallRecord(
            config={
                "topic": topic,
                "topic_description": topic_desc,
                "n_agents": n_agents,
                "n_rounds": n_rounds,
                "seed": seed,
                "condition": condition,
                "composition": composition,
                "model": client.model,
                "confirmation_bias": confirmation_bias,
                "reflect_decision_policy": "explicit_update_or_no_update_v2",
                # Ablation provenance (Sprint 15). Without these a cell's traces
                # are indistinguishable from production after the fact.
                "prompt_variant": prompt_variant,
                "prompt_variant_note": _variant_note(prompt_variant),
                # What the SERVER actually ran, read from it rather than assumed;
                # LLM_ABLATION_INFRA is stamped by the ablation driver with the
                # launch flags it used (reasoning / quantization).
                "server_props": _server_props(client.base_url),
                "ablation_infra": os.environ.get("LLM_ABLATION_INFRA", ""),
            },
            profiles=profiles,
            scenario={
                "considerations": [
                    {"id": c.id, "label": c.label, "direction": c.direction,
                     "persuasiveness": c.persuasiveness}
                    for c in pool.considerations.values()
                ],
            },
        )
        record.add_snapshot(snapshot_agents(agents, pool))

    # 5. Print initial state
    print(f"\n{'='*60}")
    print("  INITIAL STATE" + (" (restored from checkpoint)" if resumed else ""))
    print(f"{'='*60}")
    for a, p in zip(agents, profiles):
        print(
            f"  {a.id}: op={a.opinion(pool):+.3f} "
            f"cell={p.get('arm_a_agent_id','?')} "
            f"theta={p.get('ising_latent_theta')} "
            f"precision={p.get('prior_precision'):.2f}"
        )

    # 6. Run rounds
    total_t0 = time.monotonic()
    if start_round > n_rounds:
        print(f"  [RESUME] All {n_rounds} rounds already in checkpoint; skipping to save.")
    for r in range(start_round, n_rounds + 1):
        calls_before = client._seq
        round_record = run_round(r, agents, pool, engine, np.random.default_rng(seed + r), calls_before, parallel=parallel)
        record.rounds.append(round_record)
        record.add_snapshot(snapshot_agents(agents, pool))
        # Flush after each round so data survives crashes
        record.compute_summary()
        record.to_json(checkpoint_path)
        print(f"  [checkpoint saved: {checkpoint_path}]")

    total_elapsed = time.monotonic() - total_t0

    # 7. Summary
    record.compute_summary()
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  LLM calls: {client._seq}")
    if client._seq:
        print(f"  Avg per call: {total_elapsed/client._seq:.1f}s")
    print(f"  Sign flips: {record.summary.get('sign_flips', 0)}")
    print(f"  Mean |shift|: {record.summary.get('mean_abs_shift', 0):.4f}")
    print()

    initial = record.snapshots[0]
    final = record.snapshots[-1]
    for i_state, f_state in zip(initial, final):
        shift = f_state["opinion"] - i_state["opinion"]
        print(f"  {i_state['id']}: {i_state['opinion']:+.3f} -> {f_state['opinion']:+.3f} "
              f"(shift={shift:+.3f})")

    # 8. Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    tag = f"_{run_tag}" if run_tag else ""

    results_path = output_dir / f"townhall_{topic}{tag}_{ts}.json"
    record.to_json(results_path)
    print(f"\n  Results: {results_path}")

    trace_path = output_dir / f"townhall_{topic}{tag}_trace_{ts}.json"
    client.save_trace(trace_path)
    print(f"  Trace:   {trace_path}")
    print(f"\n  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Town hall deliberation with real profiles")
    parser.add_argument("--topic", default="minimum_wage_seattle")
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confirmation-bias", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--scenario-path", type=Path, default=None,
        help="External scenario JSON (overrides --topic for loading).",
    )
    parser.add_argument(
        "--empirical-init", action="store_true",
        help="Build agents via Ising-profile weights + profile scaffolding.",
    )
    parser.add_argument(
        "--profiles-path", type=Path, default=None,
        help="ising_profiles.json path (used with --empirical-init).",
    )
    parser.add_argument(
        "--theta-path", type=Path, default=None,
        help="irt_ising_theta.json path (used with --empirical-init).",
    )
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument(
        "--parallel", type=int, default=int(os.environ.get("TH_PARALLEL", "1")),
           help="Intra-round LLM concurrency. The configured backend should "
               "support at least this many concurrent requests.",
    )
    parser.add_argument(
           "--condition", type=str, default="baseline",
            choices=["baseline"],
            help="LLM prompt condition. 'baseline' = baseline framing.",
    )
    parser.add_argument(
        "--composition", type=str, default=None,
           choices=list(COMPOSITION_NAMES),
           help="Composition preset. Use the 10-agent sweep presets for collection and the "
               "*_n6 presets for quick local smoke runs. Default = empirical proportions "
               "from largest-remainder allocation (no composition override).",
    )
    parser.add_argument(
        "--prompt-variant", type=str, default="control",
        help="Named prompt variant from llm/prompt_variants.py (ablation axis 1). "
             "'control' is byte-identical to the production baseline.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="If a checkpoint exists for this run-tag, load it and continue "
             "from the next unfinished round (config must match).",
    )
    args = parser.parse_args()

    run_townhall(
        topic=args.topic,
        n_agents=args.agents,
        n_rounds=args.rounds,
        seed=args.seed,
        confirmation_bias=args.confirmation_bias,
        output_dir=args.output_dir,
        scenario_path=args.scenario_path,
        empirical_init=args.empirical_init,
        profiles_path=args.profiles_path,
        theta_path=args.theta_path,
        run_tag=args.run_tag,
        parallel=args.parallel,
        condition=args.condition,
        composition=args.composition,
        resume=args.resume,
        prompt_variant=args.prompt_variant,
    )


if __name__ == "__main__":
    main()
