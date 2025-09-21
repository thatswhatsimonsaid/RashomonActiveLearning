#!/bin/bash
#SBATCH --job-name=COMPAS_M6
#SBATCH --partition=long
#SBATCH --array=0-49
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/slurm_logs/COMPAS/out/M6_S%a.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/slurm_logs/COMPAS/error/M6_S%a.err
#SBATCH --time=2-11:59:00
#SBATCH --mem-per-cpu=30000M
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd /mnt/beegfs/homes/simondn/RashomonActiveLearning

echo "Running COMPAS_M6 | Seed (Task ID): $SLURM_ARRAY_TASK_ID"


/mnt/beegfs/homes/simondn/RashomonActiveLearning/.RAL/bin/python -m src.utils.run_experiment \
    --dataset COMPAS \
    --model TreeFarms \
    --selector QBC \
    --seed $SLURM_ARRAY_TASK_ID \
    --method_number 6 \
    regularization=0.01 rashomon_threshold=0.02 use_unique_trees=True

