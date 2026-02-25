### LIBRARIES ###
import stat
from pathlib import Path

### INPUT PARAMETERS ###
N_REPLICATIONS = 50
RASHOMON_MULTIPLIERS = [1.1, 10.0, 50.0, 100.0, 500.0] 

SIMULATION_CONFIG = {
    "n_train": 20,        
    "n_candidate": 500,
    "n_committee": 10,
    "max_depth": 5,
    "noise": 0.1,
    "beta": 200.0         
}

SLURM_CONFIG = {
    "partition": "short",
    "time": "11:59:00",
    "mem_per_cpu": "30GB",
    "mail_type": "ALL",
    "mail_user": "simondn@uw.edu",
}

# --- Paths ---
STUDY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_DIR.parent.parent.parent
WORKFLOWS_DIR = STUDY_DIR / "job_workflows"

def generate_workflow_scripts(multiplier: float):
    """Generates all .sbatch and helper .sh scripts for a single Rashomon Multiplier."""
    
    mult_str = str(multiplier).replace('.', '_')
    output_dir = WORKFLOWS_DIR / f"mult_{mult_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = output_dir / "logs"
    (logs_dir / "out").mkdir(parents=True, exist_ok=True)
    (logs_dir / "error").mkdir(parents=True, exist_ok=True)

    # --- 1. Create the main .sbatch array script ---
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=S3_ENT_{mult_str}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --time={SLURM_CONFIG['time']}
#SBATCH --mem-per-cpu={SLURM_CONFIG['mem_per_cpu']}
#SBATCH --array=1-{N_REPLICATIONS}
#SBATCH --output={logs_dir}/out/seed_%a.out
#SBATCH --error={logs_dir}/error/seed_%a.err

# 1. Use absolute path for the project root
PROJECT_ROOT="{PROJECT_ROOT}"
cd $PROJECT_ROOT

# 2. Use absolute path for the virtual environment
source "$PROJECT_ROOT/.RAL/bin/activate"

# 3. Force the current directory into the Python Path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "--- Running Entropy Study | Multiplier: {multiplier}, Seed: $SLURM_ARRAY_TASK_ID ---"

# 4. Use 'python' (now pointing to .RAL) to run the module
python -m experiments.study3_entropy_validation.entropy_estimation_study run \\
    --mode run \\
    --n_train {SIMULATION_CONFIG['n_train']} \\
    --n_candidate {SIMULATION_CONFIG['n_candidate']} \\
    --n_committee {SIMULATION_CONFIG['n_committee']} \\
    --max_depth {SIMULATION_CONFIG['max_depth']} \\
    --multiplier {multiplier} \\
    --beta {SIMULATION_CONFIG['beta']} \\
    --noise {SIMULATION_CONFIG['noise']} \\
    --seed $SLURM_ARRAY_TASK_ID
"""
    sbatch_path = output_dir / "submit_array.sbatch"
    with open(sbatch_path, 'w') as f: f.write(sbatch_content)

    # --- 2. Create Helper Scripts ---
    (output_dir / "1_run_all.sh").write_text("#!/bin/bash\nsbatch submit_array.sbatch\n")

    plot_content = f"""#!/bin/bash
PROJECT_ROOT="{PROJECT_ROOT}"
cd $PROJECT_ROOT
source .RAL/bin/activate
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
python -m experiments.study3_entropy_validation.entropy_estimation_study plot --mode plot
"""
    (output_dir / "2_generate_plots.sh").write_text(plot_content)

    cleanup_content = f"""#!/bin/bash
echo "Cleaning logs and raw data for multiplier {multiplier}..."
rm -rf "{output_dir}/logs"
echo "Cleanup complete."
"""
    (output_dir / "3_cleanup.sh").write_text(cleanup_content)
    
    for sh_file in output_dir.glob("*.sh"):
        sh_file.chmod(sh_file.stat().st_mode | stat.S_IEXEC)
    
    print(f"  -> Generated workflow for multiplier {multiplier}")

if __name__ == "__main__":
    print("--- Generating Workflows for Entropy Study Sweep ---")
    for multiplier in RASHOMON_MULTIPLIERS:
        generate_workflow_scripts(multiplier)
    print(f"\n--- Generation complete. Workflows in: {WORKFLOWS_DIR.relative_to(PROJECT_ROOT)}")