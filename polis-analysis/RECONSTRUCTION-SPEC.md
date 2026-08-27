# Polis → Scenario + Ising Profiles — Reconstruction Spec

> **Status:** ✅ pipeline reconstructed & validated (2026-06-29). All three
> artifacts regenerated to `polis-analysis/output/` and validated **exactly**
> against the manuscript (Q stats, 15/19 edges, 108 profiles, both 3×3 splits).
> Scripts: `build_covote.py` → `build_profiles.py` → `build_irt.py` →
> `build_pipeline.py` (consolidated generator).
>
> **Remaining for end-to-end manuscript replication:**
> 1. Reconstruct `AgentPopulation.from_ising_profiles` (lost loader method).
> 2. Overlay tmp2's 4 manuscript agora files (`considerations.py`, `engines.py`,
>    `experiment.py`, `scenarios.py`) onto canonical `agora/` so the manuscript
>    `Consideration` (persuasiveness/irt) + `EmpiricalArgumentEngine` are present.
> 3. Wire paths + run Arm A (rule-based, no GPU) over N=6/T=8/3-compositions/20-seeds.
> 4. **Refinement:** `irt_a` (2PL discrimination) collapsed to a constant under the
>    bounded fit; `irt_b` and `latent_theta` (corr 0.97 w/ policy) are fine.
>    `irt_a`/`irt_b` are optional fallback fields; persuasiveness (primary) is exact.
> **Why this exists:** the original `mpsa-prep/polis-analysis/` pipeline that
> produced `minimum_wage_seattle_crossover.json`, `ising_profiles.json`, and
> `irt_ising_theta.json` was lost with the deleted `tmp2/` working tree and is
> **not** recoverable from git, the `ralph` repo, or any stash. This document
> reconstructs the method from the **final manuscript methods appendix**
> (§"Semantic and Ising Construction of Considerations"), which was verified
> byte-identical between `tmp2-recovered/manuscript/final-manuscript.tex` and the
> pre-corruption PDF `final-manuscript (29).pdf`.

The **raw Polis data** *is* recovered and verified: `datasets/polis-openData/15-per-hour-seattle/`
(from `github.com/compdemocracy/openData`, commit `3be5785`, 339 participants,
30 moderated statements, topic `$15/hour`, `pol.is/2demo`).

---

## Outputs to regenerate

| File | Consumed by | Schema |
|------|-------------|--------|
| `minimum_wage_seattle_crossover.json` | both arms (`load_scenario`) | considerations[] + attacks[] + supports[] |
| `ising_profiles.json` | `AgentPopulation.from_ising_profiles` | 108 engaged participant profiles |
| `irt_ising_theta.json` | `from_ising_profiles` (latent_theta) | {participant_id: theta} |

### Scenario JSON schema (from `llm/scenario_loader.py::load_scenario`)
```json
{
  "considerations": [
    {"id": "C_01", "label": "...", "direction": 1.0, "persuasiveness": 0.817,
     "irt_a": <float|null>, "irt_b": <float|null>}
  ],
  "attacks":  [{"attacker": "C_01", "target": "C_03", "strength": 0.457}],
  "supports": [{"supporter": "C_01", "target": "C_02", "strength": 0.645}]
}
```

### Profile schema (inferred from `from_ising_profiles` + `agent-population-taxonomy.md`)
Each engaged profile needs at minimum: participant id, full 10-item `weights`
{C_01..C_10: w_pi}, `policy_score` (r_p), `coherence` (z-standardized −energy),
and the derived `policy_cell` / `coherence_cell` used for 3×3 stratification.

---

## The pipeline (7 stages)

### Stage 0 — Preprocessing
- Source: `comments.csv` (statement text, `moderated`, `comment-id`),
  `participants-votes.csv` (per-participant vote on each comment-id; cells are
  `1`=agree, `-1`=disagree, `0`=pass, blank=missing).
- Keep the **30 moderated** statements (`moderated == 1`).
- Eligibility: a statement needs **≥ 34 directional votes** (~10% of 339).
  This drops exactly **1** → **29 eligible** statements.
- Vote coding: agree `+1`, disagree `−1`, pass `0`. **Pass and missing are
  excluded** from all co-vote calculations (absent, not zero).

### Stage 1 — Co-vote association matrix *Q* (the "Ising coupling matrix")
For each eligible statement pair *(i,j)*, Yule's *Q* on the 2×2 table:
```
Q_ij = (A_ij − D_ij) / (A_ij + D_ij)
```
- `A_ij` = # participants voting the **same** direction on both (agree-agree or disagree-disagree)
- `D_ij` = # voting **opposite** directions
- Compute **only when ≥ 10** participants cast directional votes on **both**.

**VALIDATION (off-diagonal, 29-item set):** mean `+0.099`, sd `0.339`,
range `[−0.714, +0.803]`, **59.4%** positive, **37.2%** negative.

### Stage 2 — Selection to 10 considerations
> The original used `e5-large-v2` embeddings + hierarchical agglomerative
> clustering (average linkage, cosine, K=10), then per-cluster representative by
> **separation power**. That clustering script is lost. **But the final pool is
> fully published (manuscript Table 5), so we pin the 10 selected Polis IDs
> directly** and re-derive everything downstream from raw votes. This is faithful
> to the manuscript's *outputs* without needing the lost e5 clusterer.

Separation power (for reference / validation of Sep. column):
```
s_i = max_{C' ≠ C(i)} (1/|C'|) Σ_{j ∈ C'} |Q_ij|
```

**The 10 selected considerations (Table 5) — content directions are hand-coded:**

| ID | Polis | Dir | Clus | Sep | Cross | Pers |
|----|-------|-----|------|-----|-------|------|
| C_01 | 48 | Pro (+1) | 1 | .742 | .471 | .817 |
| C_02 | 11 | Pro (+1) | 2 | .571 | .541 | .900 |
| C_03 | 32 | Con (−1) | 3 | .571 | .032 | .300 |
| C_04 | 8  | Pro (+1) | 4 | .690 | .314 | .633 |
| C_05 | 45 | Pro (+1) | 5 | .739 | .400 | .734 |
| C_06 | 46 | Con (−1) | 6 | .657 | .130 | .416 |
| C_07 | 36 | Con (−1) | 7 | .458 | .429 | .768 |
| C_08 | 39 | Pro (+1) | 8 | .431 | .500 | .852 |
| C_09 | 0  | Con (−1) | 9 | .234 | .414 | .750 |
| C_10 | 20 | Con (−1) | 10 | .556 | .061 | .333 |

5 Pro, 5 Con. (Statement text in Table 5 / `comments.csv`.)

### Stage 3 — Support & attack edges (from *Q* among the 10)
Add an edge when `|Q_ij| ≥ 0.2` **and** ≥10 joint directional voters.
- `Q_ij < 0` → **attack** (stored **symmetric**)
- `Q_ij > 0` → **support** (stored **directed**, but source is symmetric *Q*)
- Strength: `ω_ij = 0.2 + 0.6·min(|Q_ij|, 1)`

**VALIDATION:** reproduce manuscript Table 6 exactly (15 attacks, 19 supports;
e.g. Attack C_01–C_03 = .457, Support C_01–C_02 = .645). Inverse check:
`|Q_ij| = (ω_ij − 0.2) / 0.6`.

### Stage 4 — IRT θ (1-D, 2-parameter logistic)
Fit a 1-D 2PL IRT on the **directional vote matrix** → per-participant latent
`θ_p` and per-item discrimination `irt_a` / difficulty `irt_b`.
- Sign/orientation convention: higher θ = more **pro-policy** (so that
  "opposing side" = θ<0 for pro items, θ>0 for con items).
- Output `irt_ising_theta.json = {participant_id: θ_p}`.

### Stage 5 — Persuasiveness via cross-over endorsement
For consideration *c*, opposing side = θ<0 (pro items) or θ>0 (con items):
```
cross_c = P(agree on c | θ on opposing side)
```
Then **min-max rescale** the 10 cross-over rates to **[0.3, 0.9]** → `persuasiveness`.

**VALIDATION:** reproduce the Cross + Pers columns of Table 5
(min cross .032→0.300 for C_03; max cross .541→0.900 for C_02).

### Stage 6 — Ising profile construction (mean-field)
On the **10 selected** items. Per consideration external field (from mean
directional vote `v̄_i`, clipped off ±1):
```
h_i = atanh(v̄_i)
```
For participant *p* with directionally-voted set `O_p`, impute each missing/passed
item *i* by mean-field expectation:
```
m_pi = tanh( h_i + Σ_{j ∈ O_p} Q_ij · v_pj )
```
Profile weight `w_pi` = observed vote if available, else `m_pi`.
**Inclusion:** participant kept iff **≥ 3 directional votes** among the 10.

**VALIDATION:** **108** engaged profiles out of 339; mean **5.40** observed
(median 5, range 3–10); **54.0%** observed / **46.0%** imputed.

### Stage 7 — Per-profile summary scores → 3×3 cells
- **Policy score:** `r_p = (1/10) Σ_i d_i · w_pi`  (`d_i` = content direction)
- **Coherence:** negative Ising energy, then **z-standardized over the 108 pool**:
  ```
  E_p = − Σ_{i<j} Q_ij · w_pi · w_pj
  coherence_p = zscore(−E_p)   # higher = more aligned with co-vote structure
  ```
- **Cells** (`briefs/agent-population-taxonomy.md`):
  - Policy: Pro `r_p > +0.1`, Con `r_p < −0.1`, else Ambivalent
  - Coherence: Coherent `z > +1`, Mixed `−1 ≤ z ≤ +1`, Ambivalent `z < −1`

**VALIDATION:** policy split **69 pro (63.9%) / 31 con (28.7%) / 8 amb (7.4%)**;
coherence split **22 coherent (20.4%) / 59 mixed (54.6%) / 27 amb (25.0%)**.

---

## Downstream: `AgentPopulation.from_ising_profiles` (also lost — reconstruct)

Signature (from `llm/townhall/data_loader.py::build_empirical_agents`):
```python
from_ising_profiles(profiles_path, n, rng, theta_path,
                    precision_exponent=0.5, composition=None) -> list[Agent]
```
Behavior:
- Load 108 profiles; assign each to its (policy, coherence) joint cell.
- Stratified-sample `n` agents to match `composition` (dict
  `{(policy, coherence): count}`); presets in `llm/townhall/compositions.py`
  (`symmetric_n6`, `polarized_n6`, `three_clusters_n6`).
- Each agent: `weights` = profile `w_pi`; `latent_theta` = θ from theta_path;
  `AgentParams` derived from coherence cell (Coherent → high `prior_precision`,
  low `open_mindedness`; Ambivalent → inverse), scaled by `precision_exponent`.

---

## Run config the manuscript used (for end-to-end replication)
- N=6 agents, T=8 rounds, baseline condition, 3 compositions
  (`symmetric_n6`, `polarized_n6`, `three_clusters_n6`).
- 20 seeds: `326 380 366 386 327 318 383 308 334 310 309 332 335 356 323 307 325 368 361 320`.
- Arm A: `EmpiricalArgumentEngine` (rule-based, no GPU). Arm B: LLM (Gemma-4-E2B via llama-server).

## Open reconstruction risks
1. **IRT orientation** — sign convention must yield the published cross-over /
   persuasiveness; validate against Table 5 Pers column to lock it.
2. **`from_ising_profiles` AgentParams formula** — exact `prior_precision` /
   `open_mindedness` mapping from coherence is only described qualitatively;
   `precision_exponent=0.5` is the known default. May need calibration.
3. **e5 clustering** is not reproduced; we pin the published 10-item pool instead.
