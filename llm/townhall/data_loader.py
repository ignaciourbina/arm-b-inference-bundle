"""Load debate-gpt-x user profiles and build agents for town hall simulation.

Data source: datasets/debate-gpt-x/data/processing/processed_data/users_df.json
Format: pandas dict-of-dicts {column_name: {user_id: value}}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np
from numpy.random import Generator

from agora.agents import Agent, AgentParams  # type: ignore[import-untyped]
from agora.considerations import ArgumentPool  # type: ignore[import-untyped]
from llm.types import ObjectMap

PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJ_ROOT / "datasets" / "debate-gpt-x" / "data" / "processing"

DEMO_COLS = [
    "education", "ethnicity", "gender", "income",
    "party", "political_ideology", "religious_ideology",
]

# Political ideology → open_mindedness mapping
_IDEOLOGY_OPENNESS = {
    "Moderate": 0.7,
    "Undecided": 0.75,
    "Apathetic": 0.6,
    "Liberal": 0.5,
    "Conservative": 0.45,
    "Libertarian": 0.5,
    "Progressive": 0.5,
    "Socialist": 0.45,
    "Communist": 0.4,
    "Anarchist": 0.4,
    "Green": 0.55,
}

UserProfile: TypeAlias = ObjectMap
UserTable: TypeAlias = dict[str, UserProfile]
ColumnarUserTable: TypeAlias = dict[str, dict[str, object]]


def _profile_str(profile: UserProfile, key: str) -> str:
    value = profile.get(key)
    return value if isinstance(value, str) else ""


def _profile_count(profile: UserProfile, key: str) -> int:
    value = profile.get(key)
    return value if isinstance(value, int) else 0


def load_users(path: Path | None = None) -> UserTable:
    """Load users_df.json and transpose to {user_id: {field: value}}.

    The raw file is pandas dict-of-dicts: {column: {user_id: value}}.
    We transpose to per-user dicts for easier access.
    """
    if path is None:
        path = DATA_DIR / "processed_data" / "users_df.json"
    if not Path(path).exists():
        raise FileNotFoundError(
            f"debate-gpt-x user table not found at {path}. The dataset is not "
            "checked into this repo (datasets/debate-gpt-x/ does not exist); "
            "pass an explicit path= to load_users(). The empirical Polis path "
            "(build_empirical_agents + --profiles-path) does not use this loader."
        )

    with open(path) as f:
        raw = cast(ColumnarUserTable, json.load(f))

    columns = list(raw.keys())
    # Get all user IDs from the first column
    user_ids = list(raw[columns[0]].keys())

    users: UserTable = {}
    for uid in user_ids:
        users[uid] = {col: raw[col].get(uid) for col in columns}

    return users


def select_participants(
    users: UserTable,
    topic: str,
    n: int = 10,
    seed: int = 42,
) -> list[UserProfile]:
    """Select n users with complete demographics and a Pro/Con stance on topic.

    Uses stratified sampling proportional to stance distribution, with
    deterministic seeding for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Filter to users with stance + complete demographics
    eligible = []
    for uid, profile in users.items():
        stance = _profile_str(profile, topic)
        if stance not in ("Pro", "Con"):
            continue
        if not all(profile.get(col) is not None for col in DEMO_COLS):
            continue
        eligible.append({**profile, "user_id": uid, "stance": stance})

    pro = [u for u in eligible if u["stance"] == "Pro"]
    con = [u for u in eligible if u["stance"] == "Con"]

    # Stratified allocation proportional to distribution
    pro_ratio = len(pro) / (len(pro) + len(con))
    n_pro = max(1, min(n - 1, round(n * pro_ratio)))
    n_con = n - n_pro

    # Shuffle and select
    rng.shuffle(pro)
    rng.shuffle(con)
    selected = pro[:n_pro] + con[:n_con]
    rng.shuffle(selected)

    return selected


def build_persona(profile: UserProfile) -> str:
    """Convert raw demographics into a natural language persona description."""
    parts = []

    gender = _profile_str(profile, "gender")
    if gender:
        parts.append(f"You identify as {gender.lower()}.")

    education = _profile_str(profile, "education")
    if education:
        parts.append(f"Your education level is {education.lower()}.")

    ethnicity = _profile_str(profile, "ethnicity")
    if ethnicity:
        parts.append(f"You identify as {ethnicity}.")

    party = _profile_str(profile, "party")
    if party:
        parts.append(f"You are affiliated with the {party}.")

    ideology = _profile_str(profile, "political_ideology")
    if ideology:
        parts.append(f"Your political views are {ideology.lower()}.")

    religion = _profile_str(profile, "religious_ideology")
    if religion:
        parts.append(f"Your religious background is {religion}.")

    income = _profile_str(profile, "income")
    if income:
        parts.append(f"Your income range is {income}.")

    stance = _profile_str(profile, "stance")
    if stance == "Pro":
        parts.append("You generally support national health care.")
    elif stance == "Con":
        parts.append("You generally oppose national health care.")

    return " ".join(parts)


def map_stance_to_weights(
    profile: UserProfile,
    pool: ArgumentPool,
    rng: Generator,
) -> dict[str, float]:
    """Map a user's Pro/Con stance to initial consideration weights.

    Pro users get positive weights on pro-direction considerations and
    smaller negative weights on con-direction ones. Con users mirror this.
    """
    stance = _profile_str(profile, "stance") or "Pro"
    sign = 1.0 if stance == "Pro" else -1.0
    weights = {}

    for cid, c in pool.considerations.items():
        # Aligned considerations get stronger weights
        if c.direction * sign > 0:
            mag = float(rng.uniform(0.3, 0.8))
        else:
            mag = float(rng.uniform(0.1, 0.35))

        # Weight sign follows consideration direction × stance alignment
        weights[cid] = float(np.clip(c.direction * sign * mag, -1.0, 1.0))

    return weights


def map_demographics_to_params(profile: UserProfile) -> AgentParams:
    """Map real demographics to AgentParams.

    Moderates/undecided get higher open_mindedness. Users with more debate
    experience get higher prior_precision.
    """
    ideology = _profile_str(profile, "political_ideology")
    om = _IDEOLOGY_OPENNESS.get(ideology, 0.5)

    # Debate experience → prior precision
    n_debates = _profile_count(profile, "number_of_all_debates")
    n_voted = _profile_count(profile, "number_of_voted_debates")
    experience = n_debates + n_voted
    # Scale: 0 debates → 1.0, 100+ → 3.0
    pp = min(1.0 + experience / 50.0, 3.0)

    return AgentParams(
        open_mindedness=om,
        prior_precision=pp,
    )


def _select_any_participants(
    users: UserTable,
    n: int,
    seed: int,
) -> list[UserProfile]:
    """Sample n users with complete demographics, ignoring topic stance.

    Used when agent beliefs are imported from an external source (Ising
    profiles) and we just need demographic scaffolding for the LLM prompt.
    """
    rng = np.random.default_rng(seed)
    eligible = []
    for uid, profile in users.items():
        if not all(profile.get(col) is not None for col in DEMO_COLS):
            continue
        eligible.append({**profile, "user_id": uid, "stance": "neutral"})
    rng.shuffle(eligible)
    return eligible[:n]


def build_empirical_agents(
    pool: ArgumentPool,
    n: int = 20,
    seed: int = 42,
    profiles_path: Path | None = None,
    theta_path: Path | None = None,
    composition: dict[tuple[str, str], int] | None = None,
    precision_exponent: float = 0.5,
    data_dir: Path | None = None,
) -> tuple[list[Agent], list[UserProfile]]:
    """Build agents with Ising-profile beliefs only.

    Mirrors `AgentPopulation.from_ising_profiles` from Arm A: same
    stratified sampling over the 3x3 (policy, coherence) joint
    distribution, same weights, same coherence-derived AgentParams.

    The LLM persona IS the voting pattern. No synthetic demographics
    are attached (we don't have authentic topic-specific demographics
    for these Polis participants, so fabricating them with debate-gpt-x
    would be inauthentic).

    Returns (agents, profiles) where profiles carries just stratification
    metadata (policy/coherence cell, latent theta) for later analysis.
    """
    from agora.agents import AgentPopulation

    popgen = AgentPopulation(pool=pool)
    rng_agents = np.random.default_rng(seed)
    empirical_agents = popgen.from_ising_profiles(
        profiles_path,
        n=n,
        rng=rng_agents,
        theta_path=theta_path,
        precision_exponent=precision_exponent,
        composition=composition,
    )

    # Rename ids to the TH_NN convention the LLM runner expects, keep
    # only Polis-derived metadata for cross-referencing.
    agents: list[Agent] = []
    profiles: list[UserProfile] = []
    for i, a in enumerate(empirical_agents):
        a_new = Agent(
            id=f"TH_{i:02d}",
            params=a.params,
            weights=dict(a.weights),
            latent_theta=a.latent_theta,
            salience_prior=(dict(a.salience_prior) if a.salience_prior is not None else None),
        )
        agents.append(a_new)
        profiles.append({
            "arm_a_agent_id": a.id,
            "ising_latent_theta": a.latent_theta,
            "prior_precision": a.params.prior_precision,
            "open_mindedness": a.params.open_mindedness,
            # Co-vote-imputed dense salience prior (full_weights). Threaded into
            # the TownHallRecord so the trajectory analysis can reconstruct the
            # effective salience map for meta_consensus_agreement (eq:v3-meta-consensus).
            "salience_prior": (dict(a.salience_prior) if a.salience_prior is not None else None),
        })
    return agents, profiles


def build_agents(
    pool: ArgumentPool,
    topic: str = "national_health_care",
    n: int = 10,
    seed: int = 42,
    data_dir: Path | None = None,
) -> tuple[list[Agent], list[UserProfile]]:
    """Build agents from real debate-gpt-x user profiles.

    Returns (agents, profiles) where each profile dict contains the raw
    demographics plus 'persona' (natural language description) and 'stance'.
    """
    if data_dir is not None:
        users_path = data_dir / "processed_data" / "users_df.json"
    else:
        users_path = None

    users = load_users(users_path)
    participants = select_participants(users, topic, n, seed)
    rng = np.random.default_rng(seed + 1000)

    agents: list[Agent] = []
    profiles: list[UserProfile] = []

    for i, profile in enumerate(participants):
        persona = build_persona(profile)
        params = map_demographics_to_params(profile)
        weights = map_stance_to_weights(profile, pool, rng)

        agent = Agent(id=f"TH_{i:02d}", params=params)
        for cid, w in weights.items():
            agent.weights[cid] = w

        agents.append(agent)
        profiles.append({
            **{col: profile.get(col) for col in DEMO_COLS},
            "user_id": profile["user_id"],
            "stance": profile["stance"],
            "persona": persona,
        })

    return agents, profiles
