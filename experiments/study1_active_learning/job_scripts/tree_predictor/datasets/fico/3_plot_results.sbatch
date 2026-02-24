#!/bin/bash
#SBATCH --job-name=Plot_fico
#SBATCH --partition=short
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study1_active_learning/slurm_logs/tree_predictor/fico/out/Plot.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study1_active_learning/slurm_logs/tree_predictor/fico/error/Plot.err
#SBATCH --time=00:20:00
#SBATCH --mem=2G
cd /mnt/beegfs/homes/simondn/RashomonActiveLearning
source .RAL/bin/activate
python src/utils/plot_results.py --dataset "fico" --study_dir "study1_active_learning/tree_predictor" --no-legend

