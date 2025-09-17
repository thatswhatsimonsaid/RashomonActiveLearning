### LIBRARIES ###
import os
import stat
from pathlib import Path
from master_config import EXPERIMENT_CONFIGS, N_REPLICATIONS, SLURM_CONFIG

### PATHS ###
LOG_DIR = Path("experiments/slurm_logs")
SRC_DIR = Path("src")
DATA_DIR = SRC_DIR / "data" / "processed"
SBATCH_DIR = Path("experiments/sbatch_files")

### FUNCTIONS ###
def create_sbatch_file(dataset_name: str, config: dict, project_root: Path):
    """Generates a dedicated sbatch script for a single experiment configuration."""
    
    ## CONFIG ##
    model_name = config["model"]
    selector_name = config["selector"]
    params = config["params"]

    ## NAME JOB ##
    job_name = f"{dataset_name}_{model_name}_{selector_name}"
    if model_name == "TreeFarms" and selector_name == "QBC":
        if params.get("use_unique_trees"):
            job_name += "_UNREAL"
        else:
            job_name += "_DUREAL"

    ## SBATCH CONTENT ##
    params_str = " ".join([f"{k}={v}" for k, v in params.items()])
    python_executable = project_root / "RAL/bin/python"
    
    python_command = f"""
{python_executable} -m src.utils.run_experiment \\
    --dataset {dataset_name} \\
    --model {model_name} \\
    --selector {selector_name} \\
    --seed $SLURM_ARRAY_TASK_ID \\
    {params_str}
"""

    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --array=0-{N_REPLICATIONS - 1}
#SBATCH --output={LOG_DIR}/out/{job_name}_%a.out
#SBATCH --error={LOG_DIR}/error/{job_name}_%a.err
#SBATCH --time={SLURM_CONFIG['time']}
#SBATCH --mem-per-cpu={SLURM_CONFIG['mem_per_cpu']}
#SBATCH --mail-type={SLURM_CONFIG['mail_type']}
#SBATCH --mail-user={SLURM_CONFIG['mail_user']}

cd {project_root}

echo "Running {job_name} | Seed (Task ID): $SLURM_ARRAY_TASK_ID"

{python_command}
"""
    
    sbatch_path = SBATCH_DIR / f"submit_{job_name}.sbatch"
    with open(sbatch_path, 'w') as f:
        f.write(sbatch_content)
    
    os.chmod(sbatch_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    print(f"  -> Generated script for {job_name}")

#### MAIN ###
if __name__ == "__main__":
    PROJECT_ROOT = Path.cwd().resolve()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "out").mkdir(exist_ok=True)
    (LOG_DIR / "error").mkdir(exist_ok=True)
    SBATCH_DIR.mkdir(exist_ok=True)
    
    datasets = sorted([f.stem for f in DATA_DIR.glob("*.pkl")])
    
    print("--- Generating sbatch job array files ---")
    for dataset in datasets:
        for exp_config in EXPERIMENT_CONFIGS:
            create_sbatch_file(dataset, config=exp_config, project_root=PROJECT_ROOT)
    print("--- Done ---")