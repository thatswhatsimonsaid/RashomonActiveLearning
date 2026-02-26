# Rashomon Ensembled Active Learning (REAL)

This repository contains the complete experimental framework for research on Rashomon Ensembled Active Learning (REAL). The platform is designed to automate large-scale Active Learning (AL) simulations on SLURM-based HPC clusters, specifically focusing on ensembling over the Rashomon Set (the set of all near-optimal, structurally distinct models).

## Abstract

Active learning reduces labeling costs by selecting samples that maximize information gain. A dominant framework, Query-by-Committee (QBC), typically relies on \textit{synthetic diversity} by inducing model disagreement through random feature subsetting or data blinding. While this approximates one notion of epistemic uncertainty, it sacrifices direct characterization of the version space to do so. We propose the complementary approach: \textit{Rashomon Ensembled Active Learning} (\textit{REAL}), which constructs a committee by exhaustively enumerating the Rashomon Set of all near-optimal models. To address functional redundancy within this set, we adopt a PAC-Bayesian framework using a Gibbs posterior to weight committee members by their empirical risk. Leveraging recent algorithmic advances, we exactly enumerate this set for the class of sparse decision trees. Across synthetic and real-world benchmarks, REAL outperforms randomized ensembles, particularly in moderately noisy environments where it strategically leverages expanded version-space diversity to achieve faster convergence.

## Table of Contents

- [Setup](#setup)
- [The 10-Method Benchmark Suite](#the-10-method-benchmark-suite)
- [Configuration & Execution](#configuration--execution)
- [Analysis Pipeline](#analysis-pipeline)
- [Project Structure](#project-structure)

---

## Setup

1. **Environment:** Create and activate the virtual environment:
```bash
python3 -m venv .RAL_CL
source .RAL_CL/bin/activate
pip install -r requirements.txt
```

2. **Backend:** Ensure the `pysortd` C++ backend is compiled and accessible in your `PYTHONPATH`.

---

## The 10-Method Benchmark Suite

The `master_config.py` defines 10 distinct selection strategies (M1–M10). The framework separates the **Predictor** (fixed at depth 5) from the **Selectors** (committees of depth 3):

| ID  | Category    | Selector Model       | Description                                  |
|-----|-------------|----------------------|----------------------------------------------|
| M1  | Baseline    | RandomForest         | Random Sampling                              |
| M6  | Baseline    | GreedyTree           | Classic Uncertainty Sampling                 |
| M7  | Baseline    | GreedyTree           | Hamming Diversity (Coreset)                  |
| M2  | QBC-RF      | RandomForest         | QBC with 3 features per split                |
| M3  | QBC-RF      | RandomForest         | QBC with `sqrt` features per split           |
| M4  | QBC-RF      | RandomForest         | QBC with all features per split              |
| M9  | Weighted RF | BMARandomForest      | Bayesian Model Averaging (Sqrt features)     |
| M10 | Weighted RF | BMARandomForest      | Bayesian Model Averaging (Full features)     |
| M5  | UNREAL      | PySORTDWrapper       | UNREAL (Uniform): $\beta=0$                  |
| M8  | UNREAL      | PySORTDWrapper       | BREAL (Bayesian): Calibrated $\beta$         |

---

## Configuration & Execution

### 1. Global Parameters (`master_config.py`)

- `N_REPLICATIONS`: Default is 25 seeds per method.
- `PREDICTION_PARAMS`: Used for the final evaluation model (Depth 5, Reg 0.001).
- `SELECTION_PARAMS`: Defines the committee characteristics (Depth 3, Max 10,000 trees).
- `beta: "calibrated"`: Triggers the Gibbs posterior weighting logic for M8, M9, and M10.

### 2. Generate and Submit

Generate the job scripts from the project root:
```bash
python experiments/generate_sbatch_arrays.py
```

Navigate to a dataset directory and submit:
```bash
cd experiments/job_scripts/compas
sbatch 1_run_all.sh
```

---

## Cluster Configuration

Before submitting jobs, configure the SLURM and experiment settings in two places:

### 1. SLURM Job Settings (`experiments/generate_sbatch_arrays.py`)

Edit the header variables at the top of the script to match your cluster's resources:
```bash
PARTITION="your_partition"   # e.g., "gpu", "compute", "short"
MEMORY="16G"                 # Memory per job (e.g., "8G", "32G", "64G")
TIME="12:00:00"              # Wall time limit (HH:MM:SS)
```

These values are injected into every generated `.sh` script as SLURM directives:
```bash
#SBATCH --partition=your_partition
#SBATCH --mem=16G
#SBATCH --time=12:00:00
```

### 2. Experiment Settings (`experiments/master_config.py`)

The central control panel for all AL simulation parameters:

| Parameter | Default | Description |
|---|---|---|
| `N_REPLICATIONS` | `25` | Number of random seeds per method |
| `DATASETS` | `[...]` | List of dataset names to benchmark |
| `PREDICTION_PARAMS` | `depth=5, reg=0.001` | Final evaluator model settings |
| `SELECTION_PARAMS` | `depth=3, max_trees=10000` | Committee/selector model settings |
| `beta` | `"calibrated"` | Weighting scheme for M8–M10 (`0` for uniform, `"calibrated"` for Gibbs) |
| `STUDY_DIR` | `"tree_predictor"` | Output subdirectory under `results/` |

### Quick Start Checklist

- [ ] Set `PARTITION`, `MEMORY`, and `TIME` in `generate_sbatch_arrays.py`
- [ ] Set `N_REPLICATIONS` and `DATASETS` in `master_config.py`
- [ ] Run `python experiments/generate_sbatch_arrays.py` to generate job scripts
- [ ] Submit with `sbatch experiments/job_scripts/<dataset>/1_run_all.sh`

## Analysis Pipeline

### Aggregation (The Fair Comparison Rule)

After Slurm jobs finish, run the aggregation script. It automatically finds the intersection of seeds that completed across all 10 methods to ensure a statistically valid "within-seed" comparison:
```bash
python src/utils/aggregate_results.py --dataset compas --study_dir tree_predictor
```

### Table & Plot Generation

- **Runtime Table:** Generates `RuntimeTable.tex` with median execution times.
- **Heatmaps:** Generates Efficiency Ratio plots (using the `0.01` pad fix) relative to the BREAL baseline.
- **Metric Grids:** Generates a 4x5 grid of PNGs for Accuracy, ECS, Oracle Agreement, and Tree Edit Distance.

---

## Project Structure

- `experiments/master_config.py`: The central control panel.
- `src/utils/`: Core logic including `learning_procedure.py` and `models.py`.
- `results/study1_active_learning/`: Top-level directory for all benchmark outputs.
- `.RAL_CL/`: Local virtual environment directory.
---