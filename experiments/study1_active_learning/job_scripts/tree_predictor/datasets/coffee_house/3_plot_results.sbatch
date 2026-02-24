#!/bin/bash
#SBATCH --job-name=Plot_coffee_house
#SBATCH --partition=short
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study1_active_learning/slurm_logs/tree_predictor/coffee_house/out/Plot.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study1_active_learning/slurm_logs/tree_predictor/coffee_house/error/Plot.err
#SBATCH --time=00:20:00
#SBATCH --mem=2G
cd /mnt/beegfs/homes/simondn/RashomonActiveLearning
source .RAL/bin/activate
python src/utils/plot_results.py --dataset "coffee_house" --study_dir "study1_active_learning/tree_predictor" --no-legend

