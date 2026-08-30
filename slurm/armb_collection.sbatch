#!/bin/bash
#SBATCH --job-name=armb-test30
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/scratch/%u/armb_test30_%j.log

# Arm-B TEST collection: 10 seeds x 3 compositions = 30 runs, n=6, t=8.
# Pinned p6-cache server signature. Distinct run-tag prefix so these never
# collide with the real 390-run arm_b_local_p6cache set.
cd "$HOME/projects/arm-b-inference-bundle"
export ARM_B_PREFIX=seawulf_test_p6cache
export SEEDS="1 2 3 4 5 6 7 8 9 10"
exec slurm/armb_gpu.sh bash -c 'bash llm/run_arm_b_local_collection.sh sweep'
