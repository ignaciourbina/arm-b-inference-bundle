"""I/O utilities: CSV export, per-round history traces, and structured logging.

Version : 0.4.2
Module  : agora.io
Spec    : manuscript/theoretical-appendix.tex §9 (output pipeline)

Overview
--------
    results_to_csv  — write scalar experiment results (one row per run)
    history_to_csv  — write per-agent per-round traces with opinion,
                      consideration weights (JSON-encoded), and DRI.
                      Uses an internal _dri_from_snapshots() that mirrors
                      DRI.compute() but operates on AgentSnapshot objects.
    setup_logging   — JSON-formatted structured logger for experiment runs.

Changelog
---------
0.4.2  3e0d09d  Review cleanup, DRI docstrings
0.4.2  c5de059  [US-002] DRI trace computation
0.4.0  e3b2d53  Restore from calibration branch (baseline)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr

from agora.history import SimulationHistory


def results_to_csv(results: list[dict[str, Any]], path: str | Path) -> None:
    """Write experiment results to CSV, including only scalar values.

    Non-scalar fields (e.g., SimulationHistory) are silently skipped.
    """
    if not results:
        return

    scalar_keys = [
        k for k in results[0]
        if isinstance(results[0][k], (int, float, str, bool))
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in scalar_keys})


def _dri_from_snapshots(
    agent_snapshots: tuple[Any, ...],
) -> float:
    """Compute DRI from AgentSnapshot objects without needing Agent/Pool.

    Projects weight vectors to the union of all agent repertoires
    (zero-filling missing considerations).
    """
    if len(agent_snapshots) < 2:
        return 0.0

    cosine_sims: list[float] = []
    opinion_sims: list[float] = []

    all_cids = sorted(set().union(*(s.repertoire for s in agent_snapshots)))
    if not all_cids:
        return 0.0

    for i in range(len(agent_snapshots)):
        a = agent_snapshots[i]
        for j in range(i + 1, len(agent_snapshots)):
            b = agent_snapshots[j]

            v_a = np.array([a.weights.get(cid, 0.0) for cid in all_cids])
            v_b = np.array([b.weights.get(cid, 0.0) for cid in all_cids])

            n1 = float(np.linalg.norm(v_a))
            n2 = float(np.linalg.norm(v_b))
            cos = float(np.dot(v_a, v_b) / (n1 * n2)) if n1 > 0 and n2 > 0 else 0.0
            op_sim = 1.0 - abs(a.opinion - b.opinion)

            cosine_sims.append(cos)
            opinion_sims.append(op_sim)

    if len(cosine_sims) < 2:
        return 0.0

    eps = 1e-12
    std_cos = float(np.std(cosine_sims))
    std_op = float(np.std(opinion_sims))
    if std_cos < eps and std_op < eps:
        return 1.0
    if std_cos < eps or std_op < eps:
        return 0.0

    r, _ = pearsonr(cosine_sims, opinion_sims)
    return float(r)


def history_to_csv(
    results: list[dict[str, Any]],
    path: str | Path,
) -> None:
    """Write per-agent per-round history to CSV.

    Columns: run_id, agent_id, round, opinion, consideration_weights, dri.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "run_id", "agent_id", "round", "opinion",
        "consideration_weights", "dri",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            run_id = result["name"]
            history: SimulationHistory | None = result.get("history")
            if history is None:
                continue

            for snap in history.snapshots:
                dri = _dri_from_snapshots(snap.agent_snapshots)
                for agent_snap in snap.agent_snapshots:
                    writer.writerow({
                        "run_id": run_id,
                        "agent_id": agent_snap.agent_id,
                        "round": snap.round_num,
                        "opinion": agent_snap.opinion,
                        "consideration_weights": json.dumps(agent_snap.weights),
                        "dri": dri,
                    })


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure structured logging for agora experiments."""
    logger = logging.getLogger("agora")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s",'
            ' "logger": "%(name)s", "message": "%(message)s"}'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
