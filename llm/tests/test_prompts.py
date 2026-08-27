from agora.agents import Agent, AgentParams
from agora.considerations import ArgumentPool, Consideration
from llm.prompts import (
    BASELINE_PROMPT_BUILDER,
    build_baseline_evaluate_prompt,
    build_baseline_reflect_prompt,
    build_baseline_voice_prompt,
)


def _build_pool() -> ArgumentPool:
    pool = ArgumentPool()
    pool.add(Consideration(id="C_01", label="Pro statement", direction=1.0, persuasiveness=0.8))
    pool.add(Consideration(id="C_02", label="Con statement", direction=-1.0, persuasiveness=0.6))
    pool.add(Consideration(id="C_03", label="Unsure statement", direction=1.0, persuasiveness=0.4))
    return pool


def test_baseline_reflect_prompt_uses_support_disagree_buckets() -> None:
    pool = _build_pool()
    agent = Agent(
        id="A1",
        params=AgentParams(),
        weights={"C_01": 0.9, "C_02": -0.7, "C_03": 0.02},
    )

    prompt = build_baseline_reflect_prompt(
        agent=agent,
        round_updates=[("C_01", 0.5, 100), ("C_02", -0.4, 50)],
        pool=pool,
        topic_description="Test policy",
    )

    assert "STATEMENTS YOU CURRENTLY SUPPORT:" in prompt
    assert "STATEMENTS YOU CURRENTLY DISAGREE WITH:" in prompt
    assert "STATEMENTS YOU FEEL UNSURE ABOUT:" in prompt
    assert "strength=0.900 dir=+1 PRO" in prompt
    assert "strength=0.700 dir=-1 CON" in prompt
    assert "persuasiveness=0.80" not in prompt
    assert "persuasiveness=0.60" not in prompt
    assert "your_view=DISAGREE, your_strength=0.700" in prompt
    assert "persuasiveness_you_rated=100/100 (min=0, max=100)" in prompt
    assert "persuasiveness_you_rated=50/100 (min=0, max=100)" in prompt
    assert "TOOL CALL FORMAT FOR update_weight:" in prompt
    assert "update_weight(cid, weight, stance)" in prompt
    assert "weight=0 removes the statement from your repertoire" in prompt
    assert 'stance="endorse" means you SUPPORT/ACCEPT the statement itself' in prompt
    assert 'stance="reject" means you OPPOSE/DISAGREE WITH the statement itself' in prompt
    assert "For CON statements, endorse means you agree with the CON argument" in prompt
    assert "You value coherence in your beliefs" in prompt
    assert "your_w=" not in prompt
    assert "w=+" not in prompt
    assert "w=-" not in prompt
    assert " x=1, " not in prompt
    assert " x=-1, " not in prompt
    assert "new_weight" not in prompt


def test_baseline_reflect_prompt_makes_new_argument_speaker_bounds_explicit() -> None:
    pool = _build_pool()
    agent = Agent(
        id="A1",
        params=AgentParams(),
        weights={"C_01": 0.9, "C_02": -0.7},
    )

    prompt = build_baseline_reflect_prompt(
        agent=agent,
        round_updates=[("C_03", 0.5, 75)],
        pool=pool,
        topic_description="Test policy",
    )

    assert "speaker_op=+0.500 (min=-1, max=+1)" in prompt
    assert "persuasiveness_you_rated=75/100 (min=0, max=100)" in prompt


def test_baseline_voice_prompt_lists_only_supported_statements() -> None:
    pool = _build_pool()
    agent = Agent(
        id="A1",
        params=AgentParams(),
        weights={"C_01": 0.9, "C_02": -0.7},
    )

    prompt = build_baseline_voice_prompt(
        agent=agent,
        pool=pool,
        topic_description="Test policy",
    )

    assert "STATEMENTS YOU CURRENTLY SUPPORT (pick ONE of these to voice):" in prompt
    assert "C_01 [strength=0.900 dir=+1 PRO]" in prompt
    assert "C_02 [strength=0.700 dir=-1 CON]" not in prompt
    assert "Do not voice a statement you currently disagree with." in prompt
    assert "You value coherence in your beliefs" in prompt


def test_baseline_stance_label_splits_weak_and_strong_branches() -> None:
    assert BASELINE_PROMPT_BUILDER._stance_label(0.6) == "STRONGLY SUPPORT"
    assert BASELINE_PROMPT_BUILDER._stance_label(0.2) == "WEAKLY SUPPORT"
    assert BASELINE_PROMPT_BUILDER._stance_label(-0.2) == "WEAKLY OPPOSE"
    assert BASELINE_PROMPT_BUILDER._stance_label(-0.6) == "STRONGLY OPPOSE"


