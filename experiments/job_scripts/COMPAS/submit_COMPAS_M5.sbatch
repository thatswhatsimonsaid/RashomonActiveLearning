#!/bin/bash
#SBATCH --job-name=COMPAS_M5
#SBATCH --partition=short
#SBATCH --array=0-49
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/slurm_logs/COMPAS/out/M5_S%a.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/slurm_logs/COMPAS/error/M5_S%a.err
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000M
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd /mnt/beegfs/homes/simondn/RashomonActiveLearning

echo "Running COMPAS_M5 | Seed (Task ID): $SLURM_ARRAY_TASK_ID"


/mnt/beegfs/homes/simondn/RashomonActiveLearning/.RAL/bin/python -m src.utils.run_experiment \
    --dataset COMPAS \
    --model GPC \
    --selector BALD \
    --seed $SLURM_ARRAY_TASK_ID \
    --method_number 5 \
    n_ensemble_samples=100

