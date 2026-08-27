#!/usr/bin/env bash
# New-stack validation: quick wins + E4B trial, before committing the 1200-run
# extension. Replaces finishing the neutrality wave on a stack we may abandon
# (that wave is paused, checkpointed, resumable on the frozen old stack —
# see llm/traces/ablation/NEUTRALITY-WAVE-PAUSED.txt).
#
# Phases (each isolates one factor):
#   A  llama-bench A/B: Homebrew binary vs coopmat2 source build, SAME old
#      weights -> the binary effect (tests the 290 tok/s prompt-eval anomaly).
#   B  checksum old vs refreshed (Jul-15) GGUF -> proves the weights differ.
#   C  cell newstack_e2b_r0715: refreshed E2B Q8 + best binary + reasoning-256,
#      control prompt, 15 runs. Scored vs old reasoning_on_b256 cell ->
#      the quick-wins (weights+binary) effect under the forward config.
#   D  cell newstack_e4b_qat: E4B QAT-Q4_0, same everything -> the model-tier
#      effect, scored vs C.
#
#   bash llm/run_newstack_validation.sh
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
PY=agora/.venv/bin/python
LOGS=llm/traces/logs; mkdir -p "$LOGS"
SPRINT=agora/analysis/sprint-15-ablation-prep

OLD_BIN=$(command -v llama-server)
NEW_BIN=~/src/llama.cpp/build/bin/llama-server
NEW_BENCH=~/src/llama.cpp/build/bin/llama-bench
OLD_E2B=~/.cache/llama-models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf
NEW_E2B=~/.cache/llama-models/gemma-4-E2B-it-GGUF-r0715/gemma-4-E2B-it-Q8_0.gguf
E4B=~/.cache/llama-models/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_0.gguf
REPORT="$SPRINT/newstack-validation.md"

echo "# New-stack validation — $(date +%F)" > "$REPORT"

# ---------------------------------------------------------------- A: binary A/B
echo; echo "######## A: binary A/B (same old weights) ########"
pkill -x llama-server 2>/dev/null; sleep 3
echo -e "\n## A. Binary A/B (old weights, pp2048/tg128, fa on)\n" >> "$REPORT"
OLD_BENCH_DIR=$(dirname "$OLD_BIN")/../Cellar/llama.cpp
OLD_LB=$(find /home/linuxbrew/.linuxbrew/Cellar/llama.cpp -name llama-bench 2>/dev/null | head -1)
for pair in "homebrew:$OLD_LB" "coopmat2:$NEW_BENCH"; do
  name="${pair%%:*}"; bin="${pair#*:}"
  if [ -x "$bin" ]; then
    echo "--- $name ---"
    out=$("$bin" -m "$OLD_E2B" -p 2048 -n 128 -fa 1 2>/dev/null | grep -E "pp2048|tg128")
    echo "$out"
    echo -e "### $name\n\`\`\`\n$out\n\`\`\`" >> "$REPORT"
  else
    echo "[warn] $name bench binary missing: $bin" | tee -a "$REPORT"
  fi
done

# --------------------------------------------------------------- B: weight diff
echo; echo "######## B: weights actually differ? ########"
OLD_SHA=$(sha256sum "$OLD_E2B" | cut -c1-16)
NEW_SHA=$(sha256sum "$NEW_E2B" | cut -c1-16)
echo "old(Apr-21)=$OLD_SHA  refreshed(r0715)=$NEW_SHA"
echo -e "\n## B. Weights\nold(Apr-21) sha256[:16] = $OLD_SHA\nrefreshed   sha256[:16] = $NEW_SHA\ndiffer: $([ "$OLD_SHA" != "$NEW_SHA" ] && echo YES || echo 'NO — refresh claim FALSE for this file')" >> "$REPORT"

start_server() {  # start_server <binary> <model>
  pkill -x llama-server 2>/dev/null; sleep 3
  nohup "$1" --model "$2" --host 127.0.0.1 --port 20434 \
    --ctx-size 24576 --parallel 6 --batch-size 2048 --ubatch-size 512 \
    --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --jinja --threads 8 --cache-reuse 256 --cont-batching \
    --reasoning on --reasoning-budget 256 \
    > "$LOGS/newstack_server.log" 2>&1 &
  for _ in $(seq 1 40); do sleep 3; curl -sf localhost:20434/v1/models >/dev/null && return 0; done
  echo "[abort] server failed: $1 + $(basename "$2")"; return 1
}

# ------------------------------------------------- C: refreshed E2B, new binary
echo; echo "######## C: newstack_e2b_r0715 (refreshed weights + coopmat2 binary + r256) ########"
start_server "$NEW_BIN" "$NEW_E2B" || start_server "$OLD_BIN" "$NEW_E2B" || exit 1
$PY llm/run_ablation_cell.py --cell newstack_e2b_r0715 --prompt-variant control \
  --infra "reasoning=on:256,quant=Q8_0,weights=r0715,binary=coopmat2" \
  2>&1 | tee -a "$LOGS/ablation_newstack_e2b_r0715.log"

# ------------------------------------------------------------- D: E4B QAT trial
echo; echo "######## D: newstack_e4b_qat (E4B QAT-Q4_0, same stack) ########"
start_server "$NEW_BIN" "$E4B" || start_server "$OLD_BIN" "$E4B" || exit 1
$PY llm/run_ablation_cell.py --cell newstack_e4b_qat --prompt-variant control \
  --infra "reasoning=on:256,quant=QAT-Q4_0,model=E4B,weights=r0715era,binary=coopmat2" \
  2>&1 | tee -a "$LOGS/ablation_newstack_e4b_qat.log"

# ------------------------------------------------------------------- E: scoring
echo; echo "######## E: scoring ########"
echo -e "\n## C vs old reasoning_on_b256 (quick-wins effect)\n" >> "$REPORT"
$PY llm/score_ablation_cells.py --cells newstack_e2b_r0715 \
  --control-dir llm/traces/ablation/reasoning_on_b256 \
  --out "$SPRINT/newstack-e2b-scores.json" 2>&1 | tee -a "$REPORT"
echo -e "\n## D vs C (model-tier effect, same stack)\n" >> "$REPORT"
$PY llm/score_ablation_cells.py --cells newstack_e4b_qat \
  --control-dir llm/traces/ablation/newstack_e2b_r0715 \
  --out "$SPRINT/newstack-e4b-scores.json" 2>&1 | tee -a "$REPORT"

pkill -x llama-server 2>/dev/null
echo; echo "[done] report: $REPORT"
