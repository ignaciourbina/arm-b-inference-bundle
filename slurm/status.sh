#!/bin/bash
# Arm-B test-collection status on SeaWulf.
cd "$HOME/projects/arm-b-inference-bundle"
source /etc/profile.d/modules.sh 2>/dev/null; module load slurm 2>/dev/null
PREFIX="${ARM_B_PREFIX:-seawulf_test_p6cache}"
DIR="llm/traces/beta_local/$PREFIX"
TARGET="${TARGET:-30}"
n=$(find "$DIR" -name 'townhall_*.json' ! -name '*_checkpoint.json' ! -name '*_trace_*' 2>/dev/null | wc -l)
echo "finals: $n/$TARGET   ($DIR)"
echo "--- slurm ---"; squeue -u "$USER" -o "%.8i %.14j %.8T %.10M %.10l %R"
JOB=$(squeue -u "$USER" -h -o "%i" -n armb-test30 | head -1)
[ -n "$JOB" ] && { echo "--- gpu ---"; srun --jobid="$JOB" nvidia-smi \
  --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | head -1; }
echo "--- per composition ---"
for c in symmetric_n6 polarized_n6 three_clusters_n6; do
  k=$(find "$DIR" -name "townhall_*_${c}_s*.json" ! -name '*_checkpoint.json' ! -name '*_trace_*' 2>/dev/null | wc -l)
  echo "  $c: $k/10"
done
echo "--- validate (when complete) ---"
echo "  PYTHONSAFEPATH=1 PYTHONPATH=.:agora/src agora/.venv/bin/python \\"
echo "    pipeline/validate_llm_collection.py --traces-dir $DIR --prefix $PREFIX --reps 10"
