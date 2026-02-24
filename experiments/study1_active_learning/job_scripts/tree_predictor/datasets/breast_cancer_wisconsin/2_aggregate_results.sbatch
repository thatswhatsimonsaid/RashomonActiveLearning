#!/bin/bash
#SBATCH --job-name=Agg_breast_cancer_wisconsin
#SBATCH --partition=short
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study1_active_learning/slurm_logs/tree_predictor/breast_cancer_wisconsin/out/Agg.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study1_active_learning/slurm_logs/tree_predictor/breast_cancer_wisconsin/error/Agg.err
#SBATCH --time=00:20:00
#SBATCH --mem=2G
cd /mnt/beegfs/homes/simondn/RashomonActiveLearning
module load Python/3.10.8-GCCcore-12.2.0
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
python src/utils/aggregate_results.py --dataset "breast_cancer_wisconsin" --study_dir "study1_active_learning/tree_predictor"
