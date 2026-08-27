# Prompt-Neutrality Ablation — Design

**Date:** 2026-08-20 · **Motivation:** IU, reviewing the prompt constructor as a
substantive reviewer would.

## The question

The Arm-B headline (epistemic integration: rising alignment, rising
meta-consensus, falling dispersion) is produced under a prompt whose *register*
is deliberately non-neutral: the agent is told it is a real resident with real
convictions, that it values coherence and does not revise its views lightly,
its stance is rendered in charged capitalized labels, and the update
instruction is wrapped in magnitude-dampening adverbs. A reasonable reviewer
will ask: **would an information-equivalent but rhetorically neutral prompt
produce the same pattern?** If yes, the result is about the engine; if no, part
of the result is priming carried by the wording.

## Scope discipline (what is and is not a lever)

**Levers are TEXT ONLY.** The information content, the tool contract, and the
prompt structure are held fixed in every cell:

- the attack-context block stays ON (it is part of the engine's information
  design, not prompt rhetoric);
- speaker opinion scores stay visible;
- repertoire content, ordering, and numeric precision are unchanged;
- the voice/evaluate/reflect structure and all tool-call instructions keep
  their meaning — only priming wording changes.

Structural/informational ablations (no-attack-graph, speaker-blind, shuffled
repertoire, qualitative-only numbers) were considered and **rejected** for this
wave: they change what the agent knows, which is a different model, not a
cleaner prompt. (They remain available as robustness checks if a reviewer asks,
but they are not part of the neutrality question.)

## The four text levers

| Lever | Production text (control) | Neutral rendering | Priming hypothesis |
|---|---|---|---|
| **persona** | "You are a *real Seattle resident*… your voting pattern captures your *actual convictions* — you are speaking AS this person, *not as a neutral analyst*" | "You are a participant in a discussion of… Your ratings of the statements are listed below." | Roleplay-as-conviction primes identity defense → stance rigidity, less cross-camp movement |
| **conviction overlay** | "You *value coherence* in your beliefs…" + "You take your current views seriously and *do not revise them lightly*" | (removed; nothing substituted) | Explicitly primes resistance to updating → consolidation, weight saturation |
| **stance label** | "Your stance: **STRONGLY SUPPORT** this policy (opinion +0.734…)" — caps, forced-choice, no neutral band | "Your position score: +0.734 on −1 (oppose) to +1 (support)" — number only | Charged verbal labels prime camp identity beyond the number → polarization maintenance |
| **gradualism** | "strengthen it *slightly*", "*nudge* it toward zero", "Most rounds: **0-3** total calls" | "strengthen it", "move it toward zero", "You may call update_weight any number of times." | Magnitude adverbs and the 0-3 anchor directly shape the trajectory's step size |

## Cells

Five cells, each vs the same production control (seeds 1–5 × 3 pools, paired):

| Cell | persona | overlay | stance | gradualism | Isolates |
|---|---|---|---|---|---|
| `neutral-persona` | neutral | prod | prod | prod | roleplay register |
| `no-overlay` | prod | removed | prod | prod | conviction priming (distinct from `terse`, which also removed attack context) |
| `neutral-stance` | prod | prod | neutral | prod | label charge |
| `no-gradualism` | prod | prod | prod | neutral | magnitude anchoring |
| `neutral-full` | neutral | removed | neutral | neutral | **the maximally neutral register — the key cell** |

Decision rule:

- If `neutral-full` reproduces the headline pattern (integration signature
  within CI of control), the result is robust to prompt register — report that
  as a robustness check in the paper.
- If `neutral-full` diverges, the four single-lever cells attribute *which*
  wording carries the effect, and the paper's claim must be conditioned
  accordingly.

## Outcomes

Same paired scorer as Sprint-15 (`llm/score_ablation_cells.py`): the voicing
set (distinct args, top-cid share), saturation @r1 (mechanism), and the
headline endpoints (DRI, meta-consensus, opinion variance). Watch especially:

- `no-overlay` and `no-gradualism` → saturation and opinion variance (do
  weights still pin to ±1 by round 1 without the stability/gradualism text?);
- `neutral-persona` / `neutral-stance` → cross-camp movement and dispersion.

## Protocol

- Server config: the production collection signature (reasoning **off**, Q8_0,
  p6-cache) so every cell pairs against the existing 390-run control. A second
  wave under reasoning-on-256 (the Sprint-15 recommended config) can re-run
  `neutral-full` later — but neutrality-vs-control must be established on the
  configuration the control was collected under.
- 5 cells × 15 runs ≈ 75 runs ≈ 2.5–3 h at observed GPU pace.
- Implementation: `llm/prompt_variants.py` (`NeutralRegisterBuilder` +
  registry); guards in `llm/tests/test_prompt_variants.py` (control stays
  byte-identical; every cell changes only its intended lever).
- Launch: `bash llm/run_neutrality_ablation.sh` (server + 5 cells + P5 scoring).

## Threats and answers

- *"Removing text shortens the prompt — length is a confound."* True of any
  removal cell; `neutral-persona` and `neutral-stance` are near-length-neutral
  substitutions, and the `terse` result from wave 1 (near-null with a much
  larger removal) bounds the pure-length effect.
- *"Five seeds is low power."* Same power as wave 1: detects large priming
  effects only. A null is "no large register effect," which is exactly the
  robustness statement the paper needs.
- *"The neutral wording is itself a register."* Yes — neutrality here means
  descriptive/administrative register with identical information, not a claim
  of zero framing. The contrast is production-vs-descriptive, stated as such.
