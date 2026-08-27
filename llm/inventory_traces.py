#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any


JsonMap = dict[str, Any]


def _discover_result_files(traces_root: Path) -> list[Path]:
    result_files: list[Path] = []
    for path in sorted(traces_root.rglob("townhall_*.json")):
        name = path.name
        if path.parent.name == "inventory":
            continue
        if name == "townhall_inventory.json":
            continue
        if "_trace_" in name:
            continue
        if name.endswith("_checkpoint.json"):
            continue
        if name.endswith("_trace_live.json"):
            continue
        result_files.append(path)
    return result_files


def _load_json(path: Path) -> JsonMap:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")
    return data


def _count_trace_tool_calls(trace_data: JsonMap) -> int:
    total = 0
    calls = trace_data.get("calls", [])
    if not isinstance(calls, list):
        return 0

    for call in calls:
        if not isinstance(call, dict):
            continue
        tool_calls_made = call.get("tool_calls_made")
        if isinstance(tool_calls_made, list):
            total += len(tool_calls_made)
            continue

        response_parsed = call.get("response_parsed")
        if not isinstance(response_parsed, dict):
            continue
        tool_calls = response_parsed.get("tool_calls")
        if isinstance(tool_calls, list):
            total += len(tool_calls)
    return total


def _trace_path_for_result(result_path: Path) -> Path | None:
    match = re.search(r"_(\d+)$", result_path.stem)
    if match is None:
        return None
    timestamp = match.group(1)
    prefix = result_path.stem[: -(len(timestamp) + 1)]
    return result_path.with_name(f"{prefix}_trace_{timestamp}.json")


def _extract_run_tag(
    *,
    stem: str,
    topic: str,
    condition: str,
    composition: str,
    seed: int,
    timestamp: int | None,
) -> str | None:
    prefix = f"townhall_{topic}_"
    suffix = f"_{condition}_{composition}_s{seed}"
    if timestamp is not None:
        suffix = f"{suffix}_{timestamp}"
    if stem.startswith(prefix) and stem.endswith(suffix):
        start = len(prefix)
        end = len(stem) - len(suffix)
        return stem[start:end] or None
    return None


def _infer_condition_and_composition(stem: str) -> tuple[str | None, str | None]:
    patterns = [
        r"(?:^|_)(?P<condition>baseline|neutral|persona)_(?P<composition>[a-z0-9_]+)_s\d+(?:_\d+)?$",
        r"(?:^|_)(?P<condition>baseline|neutral|persona)_(?P<composition>[a-z0-9_]+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match is not None:
            return match.group("condition"), match.group("composition")
    return None, None


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _mean(values: list[int | float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _build_run_record(result_path: Path, workspace_root: Path) -> JsonMap:
    result_data = _load_json(result_path)
    config = result_data.get("config", {})
    summary = result_data.get("summary", {})
    rounds = result_data.get("rounds", [])

    if not isinstance(config, dict):
        config = {}
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(rounds, list):
        rounds = []

    topic = config.get("topic") if isinstance(config.get("topic"), str) else None
    condition = config.get("condition") if isinstance(config.get("condition"), str) else None
    composition = config.get("composition") if isinstance(config.get("composition"), str) else None
    n_agents = config.get("n_agents") if isinstance(config.get("n_agents"), int) else None
    n_rounds = config.get("n_rounds") if isinstance(config.get("n_rounds"), int) else None
    seed = config.get("seed") if isinstance(config.get("seed"), int) else None

    timestamp_match = re.search(r"_(\d+)$", result_path.stem)
    timestamp = int(timestamp_match.group(1)) if timestamp_match is not None else None

    if condition is None or composition is None:
        inferred_condition, inferred_composition = _infer_condition_and_composition(result_path.stem)
        if condition is None:
            condition = inferred_condition
        if composition is None:
            composition = inferred_composition

    trace_path = _trace_path_for_result(result_path)
    trace_exists = trace_path is not None and trace_path.exists()
    trace_data = _load_json(trace_path) if trace_exists and trace_path is not None else None

    llm_calls_total = None
    tool_calls_total = None
    if isinstance(trace_data, dict):
        llm_calls_total = trace_data.get("n_calls") if isinstance(trace_data.get("n_calls"), int) else None
        tool_calls_total = _count_trace_tool_calls(trace_data)
    elif isinstance(summary.get("total_llm_calls"), int):
        llm_calls_total = summary.get("total_llm_calls")

    completed_rounds = len(rounds)
    status = "complete" if n_rounds is not None and completed_rounds == n_rounds else "partial"

    run_tag = None
    if all(value is not None for value in (topic, condition, composition, seed)):
        run_tag = _extract_run_tag(
            stem=result_path.stem,
            topic=str(topic),
            condition=str(condition),
            composition=str(composition),
            seed=int(seed),
            timestamp=timestamp,
        )

    record: JsonMap = {
        "run_id": result_path.stem,
        "run_tag": run_tag,
        "topic": topic,
        "condition": condition,
        "composition": composition,
        "seed": seed,
        "n_agents": n_agents,
        "n_rounds": n_rounds,
        "completed_rounds": completed_rounds,
        "status": status,
        "tool_calls_total": tool_calls_total,
        "llm_calls_total": llm_calls_total,
        "result_file": _relative_path(result_path, workspace_root),
        "trace_file": _relative_path(trace_path, workspace_root) if trace_exists and trace_path is not None else None,
        "timestamp": timestamp,
    }

    if isinstance(summary.get("total_elapsed_s"), (int, float)):
        record["total_elapsed_s"] = float(summary["total_elapsed_s"])
    if isinstance(summary.get("sign_flips"), int):
        record["sign_flips"] = summary["sign_flips"]
    if isinstance(summary.get("mean_abs_shift"), (int, float)):
        record["mean_abs_shift"] = float(summary["mean_abs_shift"])
    if isinstance(summary.get("total_evaluations"), int):
        record["total_evaluations"] = summary["total_evaluations"]
    if isinstance(summary.get("total_weight_changes"), int):
        record["total_weight_changes"] = summary["total_weight_changes"]

    return record


def _build_breakdown(run_records: list[JsonMap]) -> list[JsonMap]:
    grouped: dict[tuple[Any, Any, Any, Any], list[JsonMap]] = defaultdict(list)
    for record in run_records:
        key = (
            record.get("condition"),
            record.get("n_agents"),
            record.get("composition"),
            record.get("n_rounds"),
        )
        grouped[key].append(record)

    breakdown: list[JsonMap] = []
    for key in sorted(grouped, key=lambda item: tuple("" if value is None else str(value) for value in item)):
        records = grouped[key]
        tool_call_values = [value for value in (r.get("tool_calls_total") for r in records) if isinstance(value, int)]
        llm_call_values = [value for value in (r.get("llm_calls_total") for r in records) if isinstance(value, int)]
        seeds = sorted({value for value in (r.get("seed") for r in records) if isinstance(value, int)})
        breakdown.append(
            {
                "condition": key[0],
                "n_agents": key[1],
                "composition": key[2],
                "n_rounds": key[3],
                "run_count": len(records),
                "complete_runs": sum(1 for record in records if record.get("status") == "complete"),
                "partial_runs": sum(1 for record in records if record.get("status") != "complete"),
                "seeds": seeds,
                "tool_calls_total_sum": sum(tool_call_values),
                "tool_calls_total_mean": _mean(tool_call_values),
                "llm_calls_total_sum": sum(llm_call_values),
                "llm_calls_total_mean": _mean(llm_call_values),
            }
        )
    return breakdown


def build_inventory(traces_root: Path, workspace_root: Path) -> JsonMap:
    run_records = [_build_run_record(path, workspace_root) for path in _discover_result_files(traces_root)]
    run_records.sort(
        key=lambda record: (
            "" if record.get("topic") is None else str(record.get("topic")),
            "" if record.get("run_tag") is None else str(record.get("run_tag")),
            "" if record.get("condition") is None else str(record.get("condition")),
            "" if record.get("composition") is None else str(record.get("composition")),
            -1 if not isinstance(record.get("seed"), int) else int(record.get("seed")),
            -1 if not isinstance(record.get("timestamp"), int) else int(record.get("timestamp")),
        )
    )

    llm_call_values = [value for value in (record.get("llm_calls_total") for record in run_records) if isinstance(value, int)]
    tool_call_values = [value for value in (record.get("tool_calls_total") for record in run_records) if isinstance(value, int)]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "traces_root": str(traces_root),
        "run_count": len(run_records),
        "summary": {
            "complete_runs": sum(1 for record in run_records if record.get("status") == "complete"),
            "partial_runs": sum(1 for record in run_records if record.get("status") != "complete"),
            "tool_calls_total_sum": sum(tool_call_values),
            "llm_calls_total_sum": sum(llm_call_values),
        },
        "breakdown": _build_breakdown(run_records),
        "runs": run_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory townhall result and trace files.")
    parser.add_argument(
        "--traces-root",
        default="llm/traces",
        help="Root directory to scan for townhall result and trace files.",
    )
    parser.add_argument(
        "--output",
        default="llm/traces/inventory/townhall_inventory.json",
        help="Path for the JSON inventory artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path.cwd()
    traces_root = (workspace_root / args.traces_root).resolve()
    output_path = (workspace_root / args.output).resolve()

    inventory = build_inventory(traces_root, workspace_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(inventory, handle, indent=2)
        handle.write("\n")

    print(f"Wrote inventory for {inventory['run_count']} runs to {output_path}")


if __name__ == "__main__":
    main()