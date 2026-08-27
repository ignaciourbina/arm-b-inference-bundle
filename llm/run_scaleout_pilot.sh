#!/usr/bin/env bash
# Sprint-16 P4: occupancy pilot — the sprint's exit condition.
#
# Two arms on the SAME p24 server, disjoint fresh seeds, short runs (t=4):
#   arm S: 6 runs sequential (1 lane)   — today's modus operandi
#   arm P: 6 runs across 6 lanes        — the scale-out claim
# Success (plan): P/S macro speedup >= 2.5x, VRAM < 6.5 GB, zero failures.
#
#   bash llm/run_scaleout_pilot.sh
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
PY=agora/.venv/bin/python
OUT=agora/analysis/sprint-16-collection-scaleout/pilot-results.md

bash llm/run_collection_server.sh 128 || exit 1

echo "######## ARM S: sequential (6 runs, 1 lane) ########"
T0=$(date +%s)
$PY llm/run_collection_parallel.py --cell scaleout_pilot_seq \
  --seeds 9001-9002 --lanes 1 --rounds 4 --lane-stagger 0 \
  --status-interval 120
SEQ_S=$(( $(date +%s) - T0 ))

echo "######## ARM P: parallel (6 runs, 6 lanes) ########"
T0=$(date +%s)
$PY llm/run_collection_parallel.py --cell scaleout_pilot_par \
  --seeds 9003-9004 --lanes 6 --rounds 4 --lane-stagger 15 \
  --status-interval 120
PAR_S=$(( $(date +%s) - T0 ))

VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo -1)
SEQ_FAIL=$(grep -c '"event": "run_fail"' llm/traces/logs/scaleout_pilot_seq_progress.jsonl 2>/dev/null || echo 0)
PAR_FAIL=$(grep -c '"event": "run_fail"' llm/traces/logs/scaleout_pilot_par_progress.jsonl 2>/dev/null || echo 0)

SPEEDUP=$($PY -c "print(f'{$SEQ_S/max($PAR_S,1):.2f}')")
cat > "$OUT" <<EOF
# Sprint-16 P4 pilot results ($(date +%F))

| Arm | runs | wall | failures |
|---|---|---|---|
| S sequential (1 lane) | 6 (t=4) | ${SEQ_S}s ($((SEQ_S/60)) min) | ${SEQ_FAIL} |
| P parallel (6 lanes)  | 6 (t=4) | ${PAR_S}s ($((PAR_S/60)) min) | ${PAR_FAIL} |

**Macro speedup: ${SPEEDUP}x** (gate: >= 2.5x)
**VRAM at p24/98k ctx under load-tail: ${VRAM} MB** (gate: < 6500)
**Failures: S=${SEQ_FAIL} P=${PAR_FAIL}** (gate: 0)

Verdict: $( $PY -c "print('PASS' if $SPEEDUP >= 2.5 and $VRAM < 6500 and $SEQ_FAIL+$PAR_FAIL == 0 else 'FAIL — see gates above')" )
EOF
cat "$OUT"
pkill -x llama-server 2>/dev/null
echo "[pilot] done; server stopped"
