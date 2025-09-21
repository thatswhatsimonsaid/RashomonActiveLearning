#!/bin/bash
#SBATCH --job-name=Bar7_M7
#SBATCH --partition=short
#SBATCH --array=0-49
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/slurm_logs/Bar7/out/M7_S%a.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/slurm_logs/Bar7/error/M7_S%a.err
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000M
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd /mnt/beegfs/homes/simondn/RashomonActiveLearning

echo "Running Bar7_M7 | Seed (Task ID): $SLURM_ARRAY_TASK_ID"


/mnt/beegfs/homes/simondn/RashomonActiveLearning/.RAL/bin/python -m src.utils.run_experiment \
    --dataset Bar7 \
    --model TreeFarms \
    --selector QBC \
    --seed $SLURM_ARRAY_TASK_ID \
    --method_number 7 \
    regularization=0.01 rashomon_threshold=0.01 use_unique_trees=False

