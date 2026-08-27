"""Agentic LLM cognitive engine for Agora deliberation.

Drop-in replacement for mathematical engines. The LLM IS the cognitive
engine — it reasons about arguments, exhibits biases naturally, and
updates beliefs through tool-calling loops. Only structural constraints
(weight clamping, interface conformance) are enforced mechanically.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from agora.agents import Agent  # type: ignore[import-untyped]
from agora.considerations import ArgumentPool  # type: ignore[import-untyped]
from agora.engines import CognitiveEngine  # type: ignore[import-untyped]

from .client import LLMClient
from .harness import ToolCallHarness
from .influence_scale import INFLUENCE_LIKERT_MAX, INFLUENCE_LIKERT_MIN

if TYPE_CHECKING:
    from numpy.random import Generator

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from sync context, handling existing loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


@dataclass
class AgenticLLMEngine(CognitiveEngine):
    """Cognitive engine backed by a local LLM with tool calling.

    The LLM operates in an agentic loop: it receives context via system
    prompt (repertoire, opinion, cognitive style) and makes tool calls
    to query the argumentation graph and commit actions.

    Confirmation bias, facilitation, and group dynamics are injected as
    prompt context — the LLM exhibits them naturally through reasoning,
    not through post-hoc mathematical corrections.

    Only structural invariants are enforced:
    - Weight clamping to [-1, 1]
    - Valid cid validation
    - Persuasiveness scores constrained to integer 0-100 ratings
    """

    client: LLMClient = field(default_factory=LLMClient)
    confirmation_bias: float = 0.0
    graph_aware_eval: bool = False
    graph_aware_propagation: bool = False
    propagation_rate: float = 0.1
    group_building_level: int = 3
    facilitated: bool = False
    _harness: ToolCallHarness = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._harness = ToolCallHarness(
            client=self.client,
            confirmation_bias=self.confirmation_bias,
            facilitated=self.facilitated,
            group_building_level=self.group_building_level,
        )

    def voice(
        self, agent: Agent, pool: ArgumentPool, rng: Generator
    ) -> str | None:
        if not agent.weights:
            ids = pool.all_ids()
            return str(rng.choice(ids))

        result = _run_async(self._harness.voice(agent, pool))
        if result.get("skip") is True:
            return None
        cid = result.get("cid", "")
        if cid and cid in pool.considerations:
            return cid

        raise RuntimeError(f"{agent.id}/voice: harness returned invalid cid {cid!r}")

    def evaluate(
        self,
        agent: Agent,
        cid: str,
        speaker_opinion: float,
        pool: ArgumentPool,
        rng: Generator,
    ) -> int:
        result = _run_async(
            self._harness.evaluate(agent, cid, speaker_opinion, pool)
        )
        score = result.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise RuntimeError(f"{agent.id}/evaluate: harness returned invalid score {score!r}")
        score_float = float(score)
        score_int = int(score_float)
        if (
            score_float != float(score_int)
            or not INFLUENCE_LIKERT_MIN <= score_int <= INFLUENCE_LIKERT_MAX
        ):
            raise RuntimeError(f"{agent.id}/evaluate: harness returned invalid score {score!r}")
        return score_int

    def reflect(
        self,
        agent: Agent,
        round_updates: list[tuple[str, float, float]],
        pool: ArgumentPool,
        rng: Generator,
    ) -> None:
        if not round_updates:
            return

        result = _run_async(
            self._harness.reflect(agent, round_updates, pool)
        )
        for upd in result.get("updates", []):
            cid = upd.get("cid", "")
            if upd.get("_removed") is True:
                agent.weights.pop(cid, None)
                continue
            new_w = upd.get("new_weight")
            if new_w == 0:
                agent.weights.pop(cid, None)
                continue
            if not isinstance(new_w, (int, float)):
                weight = upd.get("weight")
                stance = upd.get("stance")
                if stance == "endorse":
                    sign = 1
                elif stance == "reject":
                    sign = -1
                else:
                    sign = None
                if isinstance(weight, (int, float)) and sign in (-1, 1):
                    new_w = float(weight) * sign
            if cid in agent.weights and isinstance(new_w, (int, float)):
                agent.weights[cid] = float(np.clip(new_w, -1.0, 1.0))
        agent.clamp_weights()
