### SUMMARY ###
"""
Generator script for the entropy estimation study.

This script finds all available datasets and creates a dedicated .sbatch file
for each one, allowing them to be run as separate jobs on the SLURM cluster.
"""

### LIBRARIES ###
import stat
from pathlib import Path

# --- Configuration ---
SIMULATION_CONFIG = {
    "n_universe": 1000000,
    "n_estimate": 1000,
    "seed": 42,
}

SLURM_CONFIG = {
    "partition": "compute",
    "time": "11:59:00",
    "mem_per_cpu": "30GB",
    "cpus_per_task": 1,
    "mail_type": "ALL",
    "mail_user": "simondn@uw.edu",
}

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "entropy_sbatch_scripts"
LOG_DIR = PROJECT_ROOT / "logs"


def create_sbatch_file(dataset_name: str):
    """Generates a dedicated sbatch script for a single dataset."""
    
    sbatch_content = f"""#!/bin/bash

# --- SLURM Configuration for {dataset_name} ---
#SBATCH --job-name=entropy_{dataset_name}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --time={SLURM_CONFIG['time']}
#SBATCH --mem-per-cpu={SLURM_CONFIG['mem_per_cpu']}
#SBATCH --cpus-per-task={SLURM_CONFIG['cpus_per_task']}
#SBATCH --mail-type={SLURM_CONFIG['mail_type']}
#SBATCH --mail-user={SLURM_CONFIG['mail_user']}
#SBATCH --output={LOG_DIR}/entropy_{dataset_name}_%j.out
#SBATCH --error={LOG_DIR}/entropy_{dataset_name}_%j.err

# --- Job Steps ---
# DEFINE THE FULL PATH TO YOUR PROJECT DIRECTORY
PROJECT_ROOT="{PROJECT_ROOT}"

echo "--- Setting up job environment ---"
cd $PROJECT_ROOT
source .RAL/bin/activate

echo "================================================="
echo "STARTING ANALYSIS FOR DATASET: {dataset_name}"
echo "================================================="

# STEP 1: Generate the universe data (the expensive part)
echo "[1/2] Generating universe data..."
python -m experiments.entropy_estimation_study generate \\
    --dataset "{dataset_name}" \\
    --n_universe {SIMULATION_CONFIG['n_universe']} \\
    --seed {SIMULATION_CONFIG['seed']}

# STEP 2: Analyze the universe data and plot (the fast part)
echo "[2/2] Analyzing universe data and plotting..."
python -m experiments.entropy_estimation_study analyze \\
    --dataset "{dataset_name}" \\
    --n_estimate {SIMULATION_CONFIG['n_estimate']} \\
    --seed {SIMULATION_CONFIG['seed']}

echo "Analysis for {dataset_name} is finished."
"""
    sbatch_path = OUTPUT_DIR / f"submit_entropy_{dataset_name}.sbatch"
    with open(sbatch_path, 'w') as f:
        f.write(sbatch_content)
    
    sbatch_path.chmod(sbatch_path.stat().st_mode | stat.S_IEXEC)
    print(f"  -> Generated script: {sbatch_path.name}")


if __name__ == "__main__":
    print("--- Generating sbatch files for Entropy Estimation Study ---")
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    
    datasets = sorted([f.stem for f in DATA_DIR.glob("*.pkl")])
    if not datasets:
        print(f"Warning: No .pkl files found in {DATA_DIR}. No sbatch files will be generated.")
    
    for dataset in datasets:
        create_sbatch_file(dataset)
        
    run_all_content = "#!/bin/bash\n\n# Submits all generated entropy study jobs\nfor file in ./submit_entropy_*.sbatch; do\n    sbatch \"$file\"\ndone\n"
    run_all_path = OUTPUT_DIR / "run_all_entropy_studies.sh"
    with open(run_all_path, 'w') as f:
        f.write(run_all_content)
    run_all_path.chmod(run_all_path.stat().st_mode | stat.S_IEXEC)
    print(f"\n  -> Generated helper script: {run_all_path.name}")
    
    print("\n--- Generation complete ---")