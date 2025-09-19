#!/bin/bash
#SBATCH --job-name=MONK1_M4
#SBATCH --partition=short
#SBATCH --array=0-49
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/job_logs/MONK1/out/M4_S%a.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/job_logs/MONK1/error/M4_S%a.err
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd /mnt/beegfs/homes/simondn/RashomonActiveLearning

echo "Running MONK1_M4 | Seed (Task ID): $SLURM_ARRAY_TASK_ID"


/mnt/beegfs/homes/simondn/RashomonActiveLearning/.RAL/bin/python -m src.utils.run_experiment \
    --dataset MONK1 \
    --model BNN \
    --selector BALD \
    --seed $SLURM_ARRAY_TASK_ID \
    --method_number 4 \
    epochs=50 n_ensemble_samples=100

