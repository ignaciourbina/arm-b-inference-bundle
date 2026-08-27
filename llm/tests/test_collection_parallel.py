"""Offline tests for the parallel collection orchestrator (sprint-16).

The subprocess layer is stubbed; these guard the pure logic the 2.5-day
collection rests on: seed parsing, grid construction, disjoint/complete/
balanced lane slicing, resume filtering, and telemetry structure.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm.run_collection_parallel import (
    Telemetry,
    build_grid,
    completed_runs,
    parse_seeds,
    slice_lanes,
)

COMPS = ["polarized_n6", "symmetric_n6", "three_clusters_n6"]


def test_parse_seeds_ranges_lists_and_dedup():
    assert parse_seeds("1-5") == [1, 2, 3, 4, 5]
    assert parse_seeds("1,3,5") == [1, 3, 5]
    assert parse_seeds("1-3,3-5") == [1, 2, 3, 4, 5]
    assert parse_seeds("10-12, 1") == [1, 10, 11, 12]


def test_grid_is_complete_and_unique():
    grid = build_grid([1, 2], COMPS)
    assert len(grid) == 6
    assert len(set(grid)) == 6
    assert ("symmetric_n6", 2) in grid


def test_lane_slicing_disjoint_complete_balanced():
    grid = build_grid(list(range(1, 101)), COMPS)  # 300 runs
    lanes = slice_lanes(grid, 8)
    flat = [item for lane in lanes for item in lane]
    assert sorted(flat) == sorted(grid)              # complete
    assert len(set(flat)) == len(grid)               # disjoint
    sizes = [len(lane) for lane in lanes]
    assert max(sizes) - min(sizes) <= 1              # balanced


def test_lane_slicing_more_lanes_than_work():
    lanes = slice_lanes(build_grid([1], COMPS), 8)   # 3 runs, 8 lanes
    assert sum(len(l) for l in lanes) == 3
    assert sum(1 for l in lanes if l) == 3           # 3 non-empty lanes


def test_completed_runs_filters_finals_only(tmp_path: Path):
    (tmp_path / "townhall_x_cell_baseline_polarized_n6_s7_123.json").touch()
    (tmp_path / "townhall_x_cell_baseline_symmetric_n6_s7_checkpoint.json").touch()
    (tmp_path / "townhall_x_cell_trace_three_clusters_n6_s7_1._trace_.json").touch()
    (tmp_path / "CELL.json").touch()
    done = completed_runs(tmp_path)
    assert done == {("polarized_n6", 7)}


def test_resume_filtering_excludes_done():
    grid = build_grid([1, 2], COMPS)
    done = {("polarized_n6", 1), ("symmetric_n6", 2)}
    todo = [(c, s) for (c, s) in grid if (c, s) not in done]
    assert len(todo) == 4
    assert ("polarized_n6", 1) not in todo


def test_telemetry_appends_json_lines(tmp_path: Path):
    t = Telemetry(tmp_path / "progress.jsonl")
    t.emit(event="run_start", lane=0, comp="polarized_n6", seed=1)
    t.emit(event="run_done", lane=0, comp="polarized_n6", seed=1, wall_s=42.0)
    lines = [json.loads(l) for l in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert [l["event"] for l in lines] == ["run_start", "run_done"]
    assert all("ts" in l for l in lines)
