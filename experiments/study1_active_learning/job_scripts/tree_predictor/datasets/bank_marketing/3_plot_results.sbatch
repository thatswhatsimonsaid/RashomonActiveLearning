#!/bin/bash
#SBATCH --job-name=Plot_bank_marketing
#SBATCH --partition=short
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study1_active_learning/slurm_logs/tree_predictor/bank_marketing/out/Plot.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study1_active_learning/slurm_logs/tree_predictor/bank_marketing/error/Plot.err
#SBATCH --time=00:20:00
#SBATCH --mem=2G
cd /mnt/beegfs/homes/simondn/RashomonActiveLearning
source .RAL/bin/activate
python src/utils/plot_results.py --dataset "bank_marketing" --study_dir "study1_active_learning/tree_predictor" --no-legend

