"""Sprint-15 ablation prompt variants.

These guard the property the whole ablation programme rests on: `control` must
be byte-identical to the production builder, and every other variant must
differ from it in exactly the one way it claims to. A variant that silently
drifted from production would make its cell incomparable with the existing
390-run collection, and the error would only surface after the GPU time was
spent.
"""

from __future__ import annotations

import numpy as np
import pytest

from agora.agents import AgentPopulation
from llm.prompt_variants import VARIANT_NAMES, resolve_variant
from llm.prompts import BASELINE_PROMPT_BUILDER, PromptContext
from llm.scenario_loader import load_scenario

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCENARIO = REPO / "llm/scenarios/minimum_wage_seattle_crossover.json"
PROFILES = REPO / "polis-analysis/output/ising_profiles.json"
THETA = REPO / "polis-analysis/output/irt_ising_theta.json"

pytestmark = pytest.mark.skipif(
    not (SCENARIO.exists() and PROFILES.exists()),
    reason="scenario/profile artifacts not present",
)


@pytest.fixture(scope="module")
def fixture_agent_pool():
    pool = load_scenario(SCENARIO)
    agents = AgentPopulation(pool).from_ising_profiles(
        str(PROFILES), n=6, rng=np.random.default_rng(1), theta_path=str(THETA)
    )
    ctx = PromptContext(
        topic_description="Should Seattle implement a $15/hour minimum wage?",
        confirmation_bias=0.3,
    )
    return agents[0], pool, ctx


def _build(builder, agent, pool, ctx):
    return (
        builder.build_voice(agent, pool, context=ctx),
        builder.build_evaluate(agent, pool.all_ids()[0], 0.3, pool, context=ctx),
        builder.build_reflect(agent, [], pool, context=ctx),
    )


def test_control_is_byte_identical_to_production(fixture_agent_pool):
    agent, pool, ctx = fixture_agent_pool
    assert _build(resolve_variant("control"), agent, pool, ctx) == _build(
        BASELINE_PROMPT_BUILDER, agent, pool, ctx
    )


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_every_variant_builds_usable_prompts(name, fixture_agent_pool):
    agent, pool, ctx = fixture_agent_pool
    voice, evaluate, reflect = _build(resolve_variant(name), agent, pool, ctx)
    assert "submit_voice" in voice
    assert "submit_influence" in evaluate
    assert "update_weight" in reflect
    for text in (voice, evaluate, reflect):
        assert len(text) > 200, f"{name} produced a suspiciously short prompt"


def test_anti_repetition_changes_only_voice(fixture_agent_pool):
    agent, pool, ctx = fixture_agent_pool
    c_voice, c_eval, c_reflect = _build(resolve_variant("control"), agent, pool, ctx)
    v_voice, v_eval, v_reflect = _build(resolve_variant("anti-repetition"), agent, pool, ctx)
    assert "prefer a different one" in v_voice
    assert "prefer a different one" not in c_voice
    assert (v_eval, v_reflect) == (c_eval, c_reflect)


def test_explicit_tradeoff_changes_only_reflect(fixture_agent_pool):
    agent, pool, ctx = fixture_agent_pool
    c_voice, c_eval, c_reflect = _build(resolve_variant("control"), agent, pool, ctx)
    v_voice, v_eval, v_reflect = _build(resolve_variant("explicit-tradeoff"), agent, pool, ctx)
    assert "cuts AGAINST" in v_reflect
    assert "cuts AGAINST" not in c_reflect
    assert (v_voice, v_eval) == (c_voice, c_eval)


def test_terse_drops_overlay_and_shortens_every_hook(fixture_agent_pool):
    agent, pool, ctx = fixture_agent_pool
    control = _build(resolve_variant("control"), agent, pool, ctx)
    terse = _build(resolve_variant("terse"), agent, pool, ctx)
    assert "value coherence" in control[0]
    assert "value coherence" not in terse[0]
    for t, c in zip(terse, control):
        assert len(t) < len(c)


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown prompt variant"):
        resolve_variant("does-not-exist")


# --- neutrality wave: each cell changes ONLY its intended register lever ------


def test_neutral_persona_strips_roleplay_keeps_stance_and_tasks(fixture_agent_pool):
    agent, pool, ctx = fixture_agent_pool
    control = _build(resolve_variant("control"), agent, pool, ctx)
    variant = _build(resolve_variant("neutral-persona"), agent, pool, ctx)
    for text in variant:
        assert "real Seattle resident" not in text
        assert "neutral analyst" not in text
        assert "actual convictions" not in text
    # stance label register is NOT this cell's lever — verbal label survives
    assert any(w in variant[0] for w in ("SUPPORT", "OPPOSE"))
    # info retained: topic still present; task/instruction blocks unchanged
    assert ctx.topic_description in variant[0]
    for v, c in zip(variant, control):
        assert v.split("TASK:")[-1] == c.split("TASK:")[-1]


def test_neutral_stance_strips_labels_keeps_persona(fixture_agent_pool):
    agent, pool, ctx = fixture_agent_pool
    variant = _build(resolve_variant("neutral-stance"), agent, pool, ctx)
    # the charged verbal opinion label is gone from the intros...
    for text in variant:
        assert "STRONGLY SUPPORT" not in text and "STRONGLY OPPOSE" not in text
        assert "WEAKLY SUPPORT" not in text and "WEAKLY OPPOSE" not in text
    # ...but the persona register is untouched
    assert "real Seattle resident" in variant[0]
    assert "neutral analyst" in variant[0]


def test_no_overlay_keeps_attack_context_unlike_terse(fixture_agent_pool):
    agent, pool, ctx = fixture_agent_pool
    builder = resolve_variant("no-overlay")
    assert builder.features.include_attack_context is True
    assert resolve_variant("terse").features.include_attack_context is False
    voice, evaluate, reflect = _build(builder, agent, pool, ctx)
    for text in (voice, evaluate, reflect):
        assert "value coherence" not in text
        assert "do not revise them lightly" not in text
    # intros are NOT this cell's lever
    assert "real Seattle resident" in voice


def test_no_gradualism_changes_only_reflect_task(fixture_agent_pool):
    agent, pool, ctx = fixture_agent_pool
    c_voice, c_eval, c_reflect = _build(resolve_variant("control"), agent, pool, ctx)
    v_voice, v_eval, v_reflect = _build(resolve_variant("no-gradualism"), agent, pool, ctx)
    assert (v_voice, v_eval) == (c_voice, c_eval)
    assert "Most rounds: 0-3 total calls" in c_reflect
    assert "Most rounds: 0-3 total calls" not in v_reflect
    assert "strengthen it slightly" not in v_reflect
    assert "nudge it toward zero" not in v_reflect
    # the tool contract itself is intact
    assert "update_weight(cid, weight, stance)" in v_reflect
    assert "done_reflecting" in v_reflect


def test_neutral_full_neutralizes_every_register_lever(fixture_agent_pool):
    agent, pool, ctx = fixture_agent_pool
    voice, evaluate, reflect = _build(resolve_variant("neutral-full"), agent, pool, ctx)
    for text in (voice, evaluate, reflect):
        assert "real Seattle resident" not in text
        assert "neutral analyst" not in text
        assert "value coherence" not in text
        assert "STRONGLY SUPPORT" not in text and "STRONGLY OPPOSE" not in text
    assert "Most rounds: 0-3 total calls" not in reflect
    # information held fixed: topic, opinion score, tool contract all present
    assert ctx.topic_description in voice
    assert "update_weight(cid, weight, stance)" in reflect
    builder = resolve_variant("neutral-full")
    assert builder.features.include_attack_context is True  # info levers untouched
