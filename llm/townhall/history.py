"""Rich deliberation history tracking for town hall simulations.

Records per-event detail: who voiced what, who was persuaded, how opinions
shifted. Serializes to a single JSON file for analysis and replay.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypeAlias, TypedDict, cast

from llm.types import ObjectMap

AgentWeights: TypeAlias = dict[str, float]


def _as_str(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _as_float(value: object, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _as_int(value: object, default: int = 0) -> int:
    return value if isinstance(value, int) else default


def _as_likert(value: object, default: int = 0) -> int:
    if not isinstance(value, int):
        return default
    if not 0 <= value <= 100:
        return default
    return value


def _as_weight_deltas(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {
        key: float(delta)
        for key, delta in value.items()
        if isinstance(key, str) and isinstance(delta, (int, float))
    }


class AgentSnapshot(TypedDict):
    id: str
    opinion: float
    weights: AgentWeights


@dataclass
class VoiceEvent:
    """Record of one agent voicing a consideration."""

    agent_id: str
    cid: str
    label: str
    agent_opinion: float


@dataclass
class EvaluateEvent:
    """Record of one agent evaluating an argument."""

    listener_id: str
    speaker_id: str
    cid: str
    influence_likert: int
    listener_opinion: float


@dataclass
class ReflectEvent:
    """Record of one agent's reflection outcome."""

    agent_id: str
    opinion_before: float
    opinion_after: float
    weights_changed: int
    weight_deltas: dict[str, float]


@dataclass
class RoundRecord:
    """Complete record of one deliberation round."""

    round_num: int
    voices: list[VoiceEvent] = field(default_factory=list)
    evaluations: list[EvaluateEvent] = field(default_factory=list)
    reflections: list[ReflectEvent] = field(default_factory=list)
    elapsed_s: float = 0.0
    llm_calls: int = 0


@dataclass
class TownHallRecord:
    """Complete deliberation record for serialization."""

    config: ObjectMap = field(default_factory=dict)
    profiles: list[ObjectMap] = field(default_factory=list)
    scenario: ObjectMap = field(default_factory=dict)
    rounds: list[RoundRecord] = field(default_factory=list)
    snapshots: list[list[AgentSnapshot]] = field(default_factory=list)
    opinion_trajectories: dict[str, list[float]] = field(default_factory=dict)
    summary: ObjectMap = field(default_factory=dict)

    def add_snapshot(self, agents_state: list[AgentSnapshot]) -> None:
        """Record a snapshot and update opinion trajectories."""
        self.snapshots.append(agents_state)
        for state in agents_state:
            aid = state["id"]
            if aid not in self.opinion_trajectories:
                self.opinion_trajectories[aid] = []
            self.opinion_trajectories[aid].append(state["opinion"])

    def compute_summary(self) -> ObjectMap:
        """Compute summary statistics from recorded data."""
        if len(self.snapshots) < 2:
            return {}

        initial = {s["id"]: s["opinion"] for s in self.snapshots[0]}
        final = {s["id"]: s["opinion"] for s in self.snapshots[-1]}

        shifts = {
            aid: final[aid] - initial[aid]
            for aid in initial if aid in final
        }
        abs_shifts = [abs(v) for v in shifts.values()]

        total_evals = sum(len(r.evaluations) for r in self.rounds)
        total_reflects = sum(
            sum(re.weights_changed for re in r.reflections)
            for r in self.rounds
        )
        flip_count = sum(
            1 for aid in shifts
            if initial[aid] * final[aid] < 0  # sign change
        )

        self.summary = {
            "n_agents": len(initial),
            "n_rounds": len(self.rounds),
            "total_elapsed_s": round(sum(r.elapsed_s for r in self.rounds), 1),
            "total_llm_calls": sum(r.llm_calls for r in self.rounds),
            "mean_abs_shift": round(sum(abs_shifts) / len(abs_shifts), 4) if abs_shifts else 0,
            "max_abs_shift": round(max(abs_shifts), 4) if abs_shifts else 0,
            "sign_flips": flip_count,
            "total_weight_changes": total_reflects,
            "total_evaluations": total_evals,
            "shifts": {aid: round(v, 4) for aid, v in shifts.items()},
        }
        return self.summary

    def to_json(self, path: Path) -> None:
        """Serialize full record to JSON."""
        data = {
            "config": self.config,
            "profiles": self.profiles,
            "scenario": self.scenario,
            "rounds": [asdict(r) for r in self.rounds],
            "snapshots": self.snapshots,
            "opinion_trajectories": self.opinion_trajectories,
            "summary": self.summary,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: Path) -> "TownHallRecord":
        """Reconstruct a record from a checkpoint JSON (used for resume)."""
        with open(path) as f:
            data = cast(ObjectMap, json.load(f))
        rec = cls(
            config=cast(ObjectMap, data.get("config", {})),
            profiles=cast(list[ObjectMap], data.get("profiles", [])),
            scenario=cast(ObjectMap, data.get("scenario", {})),
            snapshots=cast(list[list[AgentSnapshot]], data.get("snapshots", [])),
            opinion_trajectories=cast(dict[str, list[float]], data.get("opinion_trajectories", {})),
            summary=cast(ObjectMap, data.get("summary", {})),
        )
        for rd in cast(list[ObjectMap], data.get("rounds", [])):
            rec.rounds.append(RoundRecord(
                round_num=cast(int, rd["round_num"]),
                voices=[
                    VoiceEvent(
                        agent_id=_as_str(v.get("agent_id")),
                        cid=_as_str(v.get("cid")),
                        label=_as_str(v.get("label")),
                        agent_opinion=_as_float(v.get("agent_opinion")),
                    )
                    for v in cast(list[ObjectMap], rd.get("voices", []))
                ],
                evaluations=[
                    EvaluateEvent(
                        listener_id=_as_str(e.get("listener_id")),
                        speaker_id=_as_str(e.get("speaker_id")),
                        cid=_as_str(e.get("cid")),
                        influence_likert=_as_likert(e.get("influence_likert")),
                        listener_opinion=_as_float(e.get("listener_opinion")),
                    )
                    for e in cast(list[ObjectMap], rd.get("evaluations", []))
                ],
                reflections=[
                    ReflectEvent(
                        agent_id=_as_str(r.get("agent_id")),
                        opinion_before=_as_float(r.get("opinion_before")),
                        opinion_after=_as_float(r.get("opinion_after")),
                        weights_changed=_as_int(r.get("weights_changed")),
                        weight_deltas=_as_weight_deltas(r.get("weight_deltas")),
                    )
                    for r in cast(list[ObjectMap], rd.get("reflections", []))
                ],
                elapsed_s=_as_float(rd.get("elapsed_s", 0.0)),
                llm_calls=_as_int(rd.get("llm_calls", 0)),
            ))
        return rec
