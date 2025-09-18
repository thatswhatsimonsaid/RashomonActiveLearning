#!/bin/bash
#SBATCH --job-name=MONK1_GPC_Passive
#SBATCH --partition=short
#SBATCH --array=0-49
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/job_logs/MONK1/out/MONK1_GPC_Passive_%a.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/job_logs/MONK1/error/MONK1_GPC_Passive_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd /mnt/beegfs/homes/simondn/RashomonActiveLearning
echo "Running MONK1_GPC_Passive | Seed (Task ID): $SLURM_ARRAY_TASK_ID"

/mnt/beegfs/homes/simondn/RashomonActiveLearning/RAL/bin/python -m src.utils.run_experiment \
    --dataset MONK1 \
    --model GPC \
    --selector Passive \
    --seed $SLURM_ARRAY_TASK_ID \
    

