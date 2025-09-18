#!/bin/bash
#SBATCH --job-name=MONK1_TreeFarms_QBC_UNREAL
#SBATCH --partition=short
#SBATCH --array=0-49
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/job_logs/MONK1/out/MONK1_TreeFarms_QBC_UNREAL_%a.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/job_logs/MONK1/error/MONK1_TreeFarms_QBC_UNREAL_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd /mnt/beegfs/homes/simondn/RashomonActiveLearning
echo "Running MONK1_TreeFarms_QBC_UNREAL | Seed (Task ID): $SLURM_ARRAY_TASK_ID"

/mnt/beegfs/homes/simondn/RashomonActiveLearning/RAL/bin/python -m src.utils.run_experiment \
    --dataset MONK1 \
    --model TreeFarms \
    --selector QBC \
    --seed $SLURM_ARRAY_TASK_ID \
    regularization=0.01 rashomon_threshold=0.025 use_unique_trees=True

