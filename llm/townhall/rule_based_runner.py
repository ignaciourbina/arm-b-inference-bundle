#!/usr/bin/env python3
"""Rule-based Town Hall runner with TownHallRecord-compatible output."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from agora.agents import Agent  # type: ignore[import-untyped]
from agora.considerations import ArgumentPool  # type: ignore[import-untyped]
from agora.engines import EmpiricalArgumentEngine  # type: ignore[import-untyped]
from llm.scenario_loader import load_builtin
from llm.townhall.compositions import COMPOSITION_NAMES, resolve_composition
from llm.townhall.data_loader import build_empirical_agents
from llm.townhall.history import (
    AgentSnapshot,
    EvaluateEvent,
    ReflectEvent,
    RoundRecord,
    TownHallRecord,
    VoiceEvent,
)

TRACES_DIR = Path(__file__).resolve().parent.parent / "traces" / "rule_based"
DEFAULT_MINIMUM_WAGE_SCENARIO = (
    Path(__file__).resolve().parent.parent / "scenarios" / "minimum_wage_seattle_crossover.json"
)
DEFAULT_ISING_PROFILES_PATH = (
    Path(__file__).resolve().parent.parent.parent / "polis-analysis" / "output" / "ising_profiles.json"
)
DEFAULT_IRT_THETA_PATH = (
    Path(__file__).resolve().parent.parent.parent / "polis-analysis" / "output" / "irt_ising_theta.json"
)

TOPIC_DESCRIPTIONS = {
    "minimum_wage_seattle": "Should Seattle implement a $15/hour minimum wage?",
}


def snapshot_agents(agents: list[Agent], pool: ArgumentPool) -> list[AgentSnapshot]:
    return [
        {
            "id": a.id,
            "opinion": round(a.opinion(pool), 4),
            "weights": {c: round(w, 4) for c, w in a.weights.items()},
        }
        for a in agents
    ]


def _influence_to_likert(influence: float) -> int:
    return int(round(float(np.clip((influence / 1.5) * 100.0, 0.0, 100.0))))


def run_round(
    round_num: int,
    agents: list[Agent],
    pool: ArgumentPool,
    engine: EmpiricalArgumentEngine,
    rng: np.random.Generator,
) -> RoundRecord:
    print(f"\n{'='*60}", flush=True)
    print(f"  ROUND {round_num}", flush=True)
    print(f"{'='*60}", flush=True)

    record = RoundRecord(round_num=round_num)
    t0 = time.monotonic()

    print("\n  VOICE PHASE:", flush=True)
    t_voice = time.monotonic()
    voiced: list[tuple[Agent, str]] = []
    for agent in agents:
        cid = engine.voice(agent, pool, rng)
        op = agent.opinion(pool)
        consideration = pool.get(cid)
        voiced.append((agent, cid))
        record.voices.append(VoiceEvent(
            agent_id=agent.id,
            cid=cid,
            label=consideration.label,
            agent_opinion=round(op, 4),
        ))
        print(
            f"    {agent.id} (op={op:+.3f}) voices {cid}: \"{consideration.label[:55]}...\"",
            flush=True,
        )
    print(f"  voice phase: {time.monotonic()-t_voice:.1f}s", flush=True)

    print("\n  EVALUATE PHASE:", flush=True)
    t_eval = time.monotonic()
    pending: dict[str, list[tuple[str, float, float]]] = {a.id: [] for a in agents}
    total_evals = len(agents) * max(len(agents) - 1, 0)
    completed = 0

    for speaker, cid in voiced:
        speaker_opinion = speaker.opinion(pool)
        for listener in agents:
            if listener.id == speaker.id:
                continue
            influence = engine.evaluate(listener, cid, speaker_opinion, pool, rng)
            likert = _influence_to_likert(influence)
            pending[listener.id].append((cid, speaker_opinion, influence))
            record.evaluations.append(EvaluateEvent(
                listener_id=listener.id,
                speaker_id=speaker.id,
                cid=cid,
                influence_likert=likert,
                listener_opinion=round(listener.opinion(pool), 4),
            ))
            completed += 1
            if completed == 1 or completed == total_evals or completed % 20 == 0:
                print(
                    f"    [{completed}/{total_evals}] {listener.id} hears {cid} from {speaker.id} -> pers={likert}",
                    flush=True,
                )
    print(f"  eval phase: {time.monotonic()-t_eval:.1f}s", flush=True)

    print("\n  REFLECT PHASE:", flush=True)
    t_reflect = time.monotonic()
    old_opinions = {a.id: a.opinion(pool) for a in agents}
    old_weights = {a.id: dict(a.weights) for a in agents}

    for agent in agents:
        engine.reflect(agent, pending[agent.id], pool, rng)

    for agent in agents:
        old_opinion = old_opinions[agent.id]
        old_weight_map = old_weights[agent.id]
        new_opinion = agent.opinion(pool)
        deltas: dict[str, float] = {}
        changed = 0
        for cid in sorted(set(old_weight_map) | set(agent.weights)):
            delta = agent.weights.get(cid, 0.0) - old_weight_map.get(cid, 0.0)
            if abs(delta) > 0.001:
                deltas[cid] = round(delta, 4)
                changed += 1

        print(
            f"    {agent.id}: {old_opinion:+.3f} -> {new_opinion:+.3f} "
            f"(delta={new_opinion - old_opinion:+.3f}, {changed} weights changed)",
            flush=True,
        )
        record.reflections.append(ReflectEvent(
            agent_id=agent.id,
            opinion_before=round(old_opinion, 4),
            opinion_after=round(new_opinion, 4),
            weights_changed=changed,
            weight_deltas=deltas,
        ))
    print(f"  reflect phase: {time.monotonic()-t_reflect:.1f}s", flush=True)

    elapsed = time.monotonic() - t0
    record.elapsed_s = round(elapsed, 1)
    record.llm_calls = 0
    print(f"\n  Round {round_num} complete: {elapsed:.0f}s (0 LLM calls)", flush=True)
    return record


def run_townhall(
    topic: str = "minimum_wage_seattle",
    n_agents: int = 10,
    n_rounds: int = 8,
    seed: int = 42,
    output_dir: Path | None = None,
    scenario_path: Path | None = None,
    profiles_path: Path | None = None,
    theta_path: Path | None = None,
    run_tag: str | None = None,
    condition: str = "baseline",
    composition: str | None = None,
    voicing_mode: str = "weight_prop",
    supported_only_voice: bool = True,
    confirmation_bias: float = 0.0,
    base_lr: float = 0.20,
    precision_power: float = 1.0,
    apply_congruence_gate: bool = False,
    reflect_mode: str = "explicit_stance",
    p_pro_base: float = 0.28,
    p_counter_base: float = 0.12,
    resume: bool = False,
) -> TownHallRecord:
    if output_dir is None:
        output_dir = TRACES_DIR

    topic_desc = TOPIC_DESCRIPTIONS.get(topic, topic)
    print("Rule-Based Town Hall Deliberation")
    print(f"  Topic: {topic_desc}")
    print(f"  Agents: {n_agents}, Rounds: {n_rounds}, Seed: {seed}")
    print(
        f"  voicing_mode: {voicing_mode}, supported_only_voice: {supported_only_voice}, "
        f"confirmation_bias: {confirmation_bias}, base_lr: {base_lr}, "
        f"precision_power: {precision_power}, apply_congruence_gate: {apply_congruence_gate}, "
        f"reflect_mode: {reflect_mode}, p_pro_base: {p_pro_base}, p_counter_base: {p_counter_base}"
    )
    deprecated_modes: list[str] = []
    if not supported_only_voice:
        deprecated_modes.append("allow-opposed-voice")
    if reflect_mode != "explicit_stance":
        deprecated_modes.append(f"reflect_mode={reflect_mode}")
    if apply_congruence_gate:
        deprecated_modes.append("congruence-gate")
    if deprecated_modes:
        print(
            "  [DEPRECATED] Using legacy rule-based comparison settings: "
            + ", ".join(deprecated_modes),
            flush=True,
        )
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    if scenario_path is None and topic == "minimum_wage_seattle" and DEFAULT_MINIMUM_WAGE_SCENARIO.exists():
        scenario_path = DEFAULT_MINIMUM_WAGE_SCENARIO
    if profiles_path is None and DEFAULT_ISING_PROFILES_PATH.exists():
        profiles_path = DEFAULT_ISING_PROFILES_PATH
    if theta_path is None and DEFAULT_IRT_THETA_PATH.exists():
        theta_path = DEFAULT_IRT_THETA_PATH

    if scenario_path is not None:
        from llm.scenario_loader import load_scenario

        pool = load_scenario(scenario_path)
    else:
        pool = load_builtin(topic)
    print(
        f"  Scenario: {len(pool.considerations)} considerations, "
        f"{len(pool.attack_graph.attacks)} attacks, "
        f"{len(pool.attack_graph.supports)} supports"
    )

    comp_dict = resolve_composition(composition, n_agents=n_agents)

    agents, profiles = build_empirical_agents(
        pool,
        n=n_agents,
        seed=seed,
        profiles_path=profiles_path,
        theta_path=theta_path,
        composition=comp_dict,
    )
    print(
        f"  Profiles: {n_agents} agents from Ising stratification "
        f"(composition={composition or 'empirical-default'}, "
        f"voting pattern = profile-derived weights)"
    )

    engine = EmpiricalArgumentEngine(
        voicing_mode=voicing_mode,
        supported_only_voice=supported_only_voice,
        confirmation_bias=confirmation_bias,
        base_lr=base_lr,
        precision_power=precision_power,
        apply_congruence_gate=apply_congruence_gate,
        reflect_mode=reflect_mode,
        p_pro_base=p_pro_base,
        p_counter_base=p_counter_base,
    )
    if reflect_mode == "explicit_stance":
        reflect_decision_policy = "explicit_statement_stance_v1"
    elif apply_congruence_gate:
        reflect_decision_policy = "congruence_gated_proportional_update_v1"
    else:
        reflect_decision_policy = "ungated_directional_delta_v1"

    output_dir.mkdir(parents=True, exist_ok=True)
    tag_infix = f"_{run_tag}" if run_tag else ""
    checkpoint_path = output_dir / f"townhall_{topic}{tag_infix}_checkpoint.json"
    start_round = 1
    resumed = False

    if resume and checkpoint_path.exists():
        record = TownHallRecord.from_json(checkpoint_path)
        ck = record.config
        mismatches = [
            (key, wanted, ck.get(key))
            for key, wanted in (
                ("topic", topic),
                ("n_agents", n_agents),
                ("n_rounds", n_rounds),
                ("seed", seed),
                ("composition", composition),
            )
            if ck.get(key) != wanted
        ]
        if mismatches:
            raise ValueError(
                f"Cannot resume {checkpoint_path.name}: config mismatch: "
                + ", ".join(f"{key} want={wanted} got={got}" for key, wanted, got in mismatches)
            )
        last_snapshot = record.snapshots[-1] if record.snapshots else []
        by_id = {snap["id"]: snap for snap in last_snapshot}
        for agent in agents:
            if agent.id in by_id:
                agent.weights = {
                    cid: float(weight)
                    for cid, weight in by_id[agent.id]["weights"].items()
                }
        start_round = len(record.rounds) + 1
        resumed = True
        print(
            f"  [RESUME] Loaded checkpoint with {len(record.rounds)} completed rounds; "
            f"continuing from round {start_round}."
        )
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
                "arm": "rule_based",
                "engine": "EmpiricalArgumentEngine",
                "voicing_mode": voicing_mode,
                "supported_only_voice": supported_only_voice,
                "confirmation_bias": confirmation_bias,
                "base_lr": base_lr,
                "precision_power": precision_power,
                "apply_congruence_gate": apply_congruence_gate,
                "reflect_mode": reflect_mode,
                "p_pro_base": p_pro_base,
                "p_counter_base": p_counter_base,
                "reflect_decision_policy": reflect_decision_policy,
            },
            profiles=profiles,
            scenario={
                "considerations": [
                    {
                        "id": consideration.id,
                        "label": consideration.label,
                        "direction": consideration.direction,
                        "persuasiveness": consideration.persuasiveness,
                    }
                    for consideration in pool.considerations.values()
                ],
            },
        )
        record.add_snapshot(snapshot_agents(agents, pool))

    print(f"\n{'='*60}")
    print("  INITIAL STATE" + (" (restored from checkpoint)" if resumed else ""))
    print(f"{'='*60}")
    for agent, profile in zip(agents, profiles):
        print(
            f"  {agent.id}: op={agent.opinion(pool):+.3f} "
            f"cell={profile.get('arm_a_agent_id', '?')} "
            f"theta={profile.get('ising_latent_theta')} "
            f"precision={profile.get('prior_precision'):.2f}"
        )

    total_t0 = time.monotonic()
    if start_round > n_rounds:
        print(f"  [RESUME] All {n_rounds} rounds already in checkpoint; skipping to save.")
    for round_num in range(start_round, n_rounds + 1):
        round_record = run_round(
            round_num,
            agents,
            pool,
            engine,
            np.random.default_rng(seed + round_num),
        )
        record.rounds.append(round_record)
        record.add_snapshot(snapshot_agents(agents, pool))
        record.compute_summary()
        record.to_json(checkpoint_path)
        print(f"  [checkpoint saved: {checkpoint_path}]")

    total_elapsed = time.monotonic() - total_t0
    record.compute_summary()
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print("  LLM calls: 0")
    print(f"  Sign flips: {record.summary.get('sign_flips', 0)}")
    print(f"  Mean |shift|: {record.summary.get('mean_abs_shift', 0):.4f}")
    print()

    initial = record.snapshots[0]
    final = record.snapshots[-1]
    for initial_state, final_state in zip(initial, final):
        shift = final_state["opinion"] - initial_state["opinion"]
        print(
            f"  {initial_state['id']}: {initial_state['opinion']:+.3f} -> {final_state['opinion']:+.3f} "
            f"(shift={shift:+.3f})"
        )

    ts = int(time.time())
    tag = f"_{run_tag}" if run_tag else ""
    results_path = output_dir / f"townhall_{topic}{tag}_{ts}.json"
    record.to_json(results_path)
    print(f"\n  Results: {results_path}")
    print(f"\n  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based Town hall deliberation")
    parser.add_argument("--topic", default="minimum_wage_seattle")
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--scenario-path",
        type=Path,
        default=None,
        help="External scenario JSON (overrides --topic for loading).",
    )
    parser.add_argument(
        "--profiles-path",
        type=Path,
        default=None,
        help="ising_profiles.json path for empirical initialization.",
    )
    parser.add_argument(
        "--theta-path",
        type=Path,
        default=None,
        help="irt_ising_theta.json path for empirical initialization.",
    )
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument(
        "--condition",
        type=str,
        default="baseline",
        choices=["baseline"],
        help="Compatibility field kept aligned with the LLM TownHall schema.",
    )
    parser.add_argument(
        "--composition",
        type=str,
        default=None,
        choices=list(COMPOSITION_NAMES),
        help="Composition preset shared with the LLM TownHall runner.",
    )
    parser.add_argument(
        "--voicing-mode",
        type=str,
        default="weight_prop",
        choices=["impact_prop", "argmax", "weight_prop"],
        help="Rule-based voice selection rule.",
    )
    parser.add_argument(
        "--supported-only-voice",
        dest="supported_only_voice",
        action="store_true",
        help="Canonical default. Restrict voicing to currently supported statements, with attack fallback when none are supported.",
    )
    parser.add_argument(
        "--allow-opposed-voice",
        dest="supported_only_voice",
        action="store_false",
        help="Deprecated legacy mode: allow voicing from the full signed repertoire by score magnitude.",
    )
    parser.add_argument("--confirmation-bias", type=float, default=0.0)
    parser.add_argument(
        "--base-lr",
        type=float,
        default=0.20,
        help="EmpiricalArgumentEngine base learning rate before precision scaling.",
    )
    parser.add_argument(
        "--precision-power",
        type=float,
        default=1.0,
        help="Exponent on prior precision in the reflect update denominator.",
    )
    parser.add_argument(
        "--no-congruence-gate",
        dest="apply_congruence_gate",
        action="store_false",
        help="Canonical default. Disable the empirical reflect congruence gate.",
    )
    parser.add_argument(
        "--congruence-gate",
        dest="apply_congruence_gate",
        action="store_true",
        help="Deprecated legacy mode: restore the congruence gate in reflect.",
    )
    parser.add_argument(
        "--reflect-mode",
        type=str,
        default="explicit_stance",
        choices=["directional_delta", "explicit_stance"],
        help="Reflection rule. Canonical default is explicit statement-level stance updates; directional_delta is deprecated legacy behavior.",
    )
    parser.add_argument(
        "--p-pro-base",
        type=float,
        default=0.28,
        help="Base gate probability multiplier for pro-attitudinal arguments.",
    )
    parser.add_argument(
        "--p-counter-base",
        type=float,
        default=0.12,
        help="Base gate probability multiplier for counter-attitudinal arguments.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If a checkpoint exists for this run-tag, continue from the next unfinished round.",
    )
    parser.set_defaults(supported_only_voice=True, apply_congruence_gate=False)
    args = parser.parse_args()

    run_townhall(
        topic=args.topic,
        n_agents=args.agents,
        n_rounds=args.rounds,
        seed=args.seed,
        output_dir=args.output_dir,
        scenario_path=args.scenario_path,
        profiles_path=args.profiles_path,
        theta_path=args.theta_path,
        run_tag=args.run_tag,
        condition=args.condition,
        composition=args.composition,
        voicing_mode=args.voicing_mode,
        supported_only_voice=args.supported_only_voice,
        confirmation_bias=args.confirmation_bias,
        base_lr=args.base_lr,
        precision_power=args.precision_power,
        apply_congruence_gate=args.apply_congruence_gate,
        reflect_mode=args.reflect_mode,
        p_pro_base=args.p_pro_base,
        p_counter_base=args.p_counter_base,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()