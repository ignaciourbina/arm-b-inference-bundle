#!/usr/bin/env bash
# GPT-5-mini pilot: the go/no-go gate before any real-money collection.
#
# Runs a small paired slice (3 seeds x 1 composition, n=6, t=8) through the
# cloud adapter and reports the three numbers the design doc says decide
# everything (docs/design/openai-gpt5-mini-adapter-design.md §4, risks 1-3):
#   1. reasoning_tokens per call  (the cost wildcard at 'minimal' effort)
#   2. cached_tokens ratio        (should exceed ~85% after warm-up)
#   3. tool-call validity rate    (strict mode should make retries ~zero)
# plus actual $ spend from the cost ledger vs the cost-model prediction.
#
# Prereqs: export OPENAI_API_KEY (project-scoped, spend-limited key).
# Cost ceiling: OPENAI_BUDGET_USD (default 5 for the pilot) aborts overspend.
#
#   OPENAI_API_KEY=sk-... bash llm/run_openai_pilot.sh
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
PY=agora/.venv/bin/python
: "${OPENAI_API_KEY:?export OPENAI_API_KEY first (project-scoped key)}"
export LLM_API_FLAVOR=openai-cloud
export OPENAI_BUDGET_USD="${OPENAI_BUDGET_USD:-5}"
OUT=llm/traces/openai_pilot; mkdir -p "$OUT"

for seed in 1 2 3; do
  $PY -m llm.townhall.runner \
    --topic minimum_wage_seattle \
    --scenario-path llm/scenarios/minimum_wage_seattle_crossover.json \
    --agents 6 --rounds 8 --seed "$seed" --condition baseline \
    --empirical-init \
    --profiles-path polis-analysis/output/ising_profiles.json \
    --theta-path polis-analysis/output/irt_ising_theta.json \
    --composition polarized_n6 --parallel 6 \
    --run-tag "openai_pilot_polarized_n6_s${seed}" \
    --output-dir "$OUT" --resume
done

echo; echo "######## PILOT REPORT ########"
$PY - <<'PYEOF'
import json, glob
entries = []
for f in glob.glob("llm/traces/openai_pilot/*.cost.jsonl"):
    entries += [json.loads(l) for l in open(f)]
if not entries:
    print("no ledger entries found — did the runs execute?")
    raise SystemExit(1)
n = len(entries)
tot = sum(e["cost_usd"] for e in entries)
rt = [e["reasoning_tokens"] for e in entries]
cache_eligible = [e for e in entries if e["prompt_tokens"] > 0]
ratio = sum(e["cached_tokens"] for e in cache_eligible) / max(
    sum(e["prompt_tokens"] for e in cache_eligible), 1)
print(f"calls: {n}")
print(f"total spend: ${tot:.3f}  (${tot/3:.3f}/run; design-doc flex+cache target ~$0.18/run)")
print(f"reasoning tokens/call: mean {sum(rt)/n:.0f}  max {max(rt)}  (model assumed ~150)")
print(f"cache-hit ratio: {100*ratio:.1f}%  (target >= 85%)")
print()
print("GO criteria: $/run <= ~2x model; reasoning mean <= ~300; cache >= 85%;")
print("plus scorer comparison vs local control (run score_ablation_cells.py")
print("with --ablation-root llm/traces --cells openai_pilot after 15+ runs).")
PYEOF
