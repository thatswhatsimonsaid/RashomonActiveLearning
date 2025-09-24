#!/bin/bash

# --- SLURM Configuration ---
#SBATCH --job-name=entropy_study
#SBATCH --partition=largemem
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30000MB      # Requesting significant memory for the large RF
#SBATCH --cpus-per-task=16       # Request multiple CPUs for parallel training (n_jobs=-1)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=simondn@uw.edu
#SBATCH --output=logs/entropy_study_%j.out
#SBATCH --error=logs/entropy_study_%j.err

# --- Job Steps ---

# Create necessary directories
mkdir -p logs results/entropy_study

echo "--- Setting up job environment ---"
cd $SLURM_SUBMIT_DIR
source .RAL/bin/activate

# Define simulation parameters
N_UNIVERSE=1000000
N_ESTIMATE=1000
SEED=42

# Loop through each dataset and run the full analysis
DATA_DIR="src/data/processed"
for pkl_file in "$DATA_DIR"/*.pkl; do
    DATASET_NAME=$(basename "$pkl_file" .pkl)
    
    echo ""
    echo "================================================="
    echo "STARTING ANALYSIS FOR DATASET: $DATASET_NAME"
    echo "================================================="
    
    # STEP 1: Generate the universe data (the expensive part)
    echo "[1/2] Generating universe data..."
    python -m experiments.entropy_estimation_study generate \
        --dataset "$DATASET_NAME" \
        --n_universe $N_UNIVERSE \
        --seed $SEED
    
    # STEP 2: Analyze the universe data and plot (the fast part)
    echo "[2/2] Analyzing universe data and plotting..."
    python -m experiments.entropy_estimation_study analyze \
        --dataset "$DATASET_NAME" \
        --n_estimate $N_ESTIMATE \
        --seed $SEED
        
    echo "Analysis for $DATASET_NAME is finished."
done

echo ""
echo "--- All datasets processed successfully! ---"