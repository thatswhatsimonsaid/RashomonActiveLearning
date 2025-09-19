#!/bin/bash
#SBATCH --job-name=Iris_M8
#SBATCH --partition=short
#SBATCH --array=0-49
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/job_logs/Iris/out/M8_S%a.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/job_logs/Iris/error/M8_S%a.err
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd /mnt/beegfs/homes/simondn/RashomonActiveLearning

echo "Running Iris_M8 | Seed (Task ID): $SLURM_ARRAY_TASK_ID"


/mnt/beegfs/homes/simondn/RashomonActiveLearning/.RAL/bin/python -m src.utils.run_experiment \
    --dataset Iris \
    --model RandomForest \
    --selector QBC \
    --seed $SLURM_ARRAY_TASK_ID \
    --method_number 8 \
    n_estimators=100 use_unique_trees=False

