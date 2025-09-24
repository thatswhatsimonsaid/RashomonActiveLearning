#!/bin/bash

# --- SLURM Configuration for BreastCancer ---
#SBATCH --job-name=entropy_BreastCancer
#SBATCH --partition=largemem
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=100GB
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/logs/entropy_BreastCancer_%j.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/logs/entropy_BreastCancer_%j.err

# --- Job Steps ---
# DEFINE THE FULL PATH TO YOUR PROJECT DIRECTORY
PROJECT_ROOT="/mnt/beegfs/homes/simondn/RashomonActiveLearning"

echo "--- Setting up job environment ---"
cd $PROJECT_ROOT
source .RAL/bin/activate

echo "================================================="
echo "STARTING ANALYSIS FOR DATASET: BreastCancer"
echo "================================================="

# STEP 1: Generate the universe data (the expensive part)
echo "[1/2] Generating universe data..."
python -m experiments.entropy_estimation_study generate \
    --dataset "BreastCancer" \
    --n_universe 1000000 \
    --seed 42

# STEP 2: Analyze the universe data and plot (the fast part)
echo "[2/2] Analyzing universe data and plotting..."
python -m experiments.entropy_estimation_study analyze \
    --dataset "BreastCancer" \
    --n_estimate 1000 \
    --seed 42

echo "Analysis for BreastCancer is finished."
