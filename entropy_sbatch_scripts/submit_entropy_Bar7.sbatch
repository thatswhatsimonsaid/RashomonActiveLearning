#!/bin/bash

# --- SLURM Configuration for Bar7 ---
#SBATCH --job-name=entropy_Bar7
#SBATCH --partition=largemem
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=100GB
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/logs/entropy_Bar7_%j.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/logs/entropy_Bar7_%j.err

# --- Job Steps ---
# DEFINE THE FULL PATH TO YOUR PROJECT DIRECTORY
PROJECT_ROOT="/mnt/beegfs/homes/simondn/RashomonActiveLearning"

echo "--- Setting up job environment ---"
cd $PROJECT_ROOT
source .RAL/bin/activate

echo "================================================="
echo "STARTING ANALYSIS FOR DATASET: Bar7"
echo "================================================="

# STEP 1: Generate the universe data (the expensive part)
echo "[1/2] Generating universe data..."
python -m experiments.entropy_estimation_study generate \
    --dataset "Bar7" \
    --n_universe 1000000 \
    --seed 42

# STEP 2: Analyze the universe data and plot (the fast part)
echo "[2/2] Analyzing universe data and plotting..."
python -m experiments.entropy_estimation_study analyze \
    --dataset "Bar7" \
    --n_estimate 1000 \
    --seed 42

echo "Analysis for Bar7 is finished."
