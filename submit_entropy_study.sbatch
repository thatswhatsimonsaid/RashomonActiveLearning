#!/bin/bash

# --- SLURM Configuration ---
#SBATCH --job-name=entropy_estimation_study
#SBATCH --partition=largemem             # Using a partition from your master_config.py
#SBATCH --time=11:59:00               # 2 hours should be sufficient for several datasets
#SBATCH --mem-per-cpu=150GB          # 16GB memory, as TreeFarms can be memory-intensive
#SBATCH --mail-type=ALL               # Email notifications for start, end, and failure
#SBATCH --mail-user=simondn@uw.edu    # Using the email from your config
#SBATCH --output=logs/entropy_study_%j.out # Standard output log file
#SBATCH --error=logs/entropy_study_%j.err  # Standard error log file

# --- Job Steps ---

# Create a logs directory if it doesn't exist
mkdir -p logs

echo "--- Setting up job environment ---"
# Navigate to the directory where the job was submitted from
cd $SLURM_SUBMIT_DIR

# Activate your Python virtual environment
source .RAL/bin/activate

echo "--- Starting Entropy Estimation Study for All Datasets ---"

# Define the directory where your processed datasets are located
DATA_DIR="src/data/processed"

# Loop through each .pkl file in the data directory
for pkl_file in "$DATA_DIR"/*.pkl; do
    # Extract the dataset name from the file path
    DATASET_NAME=$(basename "$pkl_file" .pkl)
    
    echo "" # Add a blank line for readability in the log
    echo "================================================="
    echo "Running study for dataset: $DATASET_NAME"
    echo "================================================="
    
    # Run the Python script as a module for the current dataset
    python -m experiments.entropy_estimation_study --dataset "$DATASET_NAME"
done

echo ""
echo "--- All datasets processed successfully! ---"
