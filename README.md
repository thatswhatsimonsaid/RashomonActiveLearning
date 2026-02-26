# Rashomon Active Learning (RAL)

## Abstract

Active learning reduces labeling costs by selecting samples that maximize information gain. A dominant framework, Query-by-Committee (QBC), typically relies on \textit{synthetic diversity} by inducing model disagreement through random feature subsetting or data blinding. While this approximates one notion of epistemic uncertainty, it sacrifices direct characterization of the version space to do so. We propose the complementary approach: \textit{Rashomon Ensembled Active Learning} (\textit{REAL}) which constructs a committee by exhaustively enumerating the Rashomon Set of all near-optimal models. To address functional redundancy within this set, we adopt a PAC-Bayesian framework using a Gibbs posterior to weight committee members by their empirical risk. Leveraging recent algorithmic advances, we exactly enumerate this set for the class of sparse decision trees. Across synthetic and real-world benchmarks, REAL outperforms randomized ensembles, particularly in moderately noisy environments where it strategically leverages expanded version-space diversity to achieve faster convergence.

---

## Preliminary Results

Manuscript plots are found in 
```
results/study1_active_learning/PLOTS
```
and tables in 
```
results/study1_active_learning/Tables
```


| File | Description |
|------|-------------|
| `accuracy_history.png` & `f1_history.png` | Performance traces tracking predictive growth over iterations |
| `tree_edit_distance_history.png` | Structural convergence toward the "Oracle" (ground-truth) tree |
| `rashomon_size_history.png` & `committee_size_history.png` | The size of the Rashomon set and committee respecticely |
| `accuracy_variance.png` | Stability: variance across the 25 experimental seeds |

---

## Setup

This project requires **Python 3.10+**. The use of a virtual environment is strongly recommended.

### Create and Activate Environment

```bash
# Using venv
python3 -m venv .RAL_CL
source .RAL_CL/bin/activate
```

### Install Requirements

```bash
pip install -r requirements.txt
```

> **Note:** This project relies on `pysortd`. Ensure C++ build tools are available on your system or cluster for the solver components.

---


## Automated Workflow on an HPC Cluster

The project features a **"Smart Launcher"** system to manage large-scale simulations on a SLURM cluster. Follow the steps below from the project root.

### Step 1 — Preprocess Data
From the root `RashomonActiveLearning/` directory, fetch UCI/TreeFarms datasets and generate synthetic XOR/Parity stress tests. Output is saved to `src/data/`.

```bash
python utils/preprocess_data.py
```

### Step 2 — Generate SBATCH Jobs
Navigate to the study directory and run the master job factory. It reads `master_config.py` to generate thousands of `.sbatch` array jobs and master management scripts.

```bash
cd RashomonActiveLearning/experiments/study1_active_learning
python generate_sbatch_arrays.py
```

### Step 3 — Launch & Monitor Jobs
Navigate to the generated job scripts directory and run the orchestration scripts in order.

```bash
cd job_scripts/tree_predictor
```

| Script | Role |
|--------|------|
| `0a_ignite.sh` | **The Entry Point.** Starts the `1_smart_run.sh` governor in the background. |
| `1_smart_run.sh` | **The "Smart Launcher."** Monitors the cluster queue and staggers job submissions (up to a 1,800-task limit) to ensure high throughput without crashing the scheduler. |
| `2_global_aggregate.sh` | Aggregation trigger. Invokes `src/utils/aggregate_results.py` to compile the 25 independent seeds into statistical summaries once a dataset finishes. |
| `3a_global_plot.sh` | Triggers `src/utils/plot_results.py` to generate the final visual performance traces. |
| `3b_collect_plots.sh` | Gathers plots from the deep directory structure into a centralized `results/.../PLOTS/` folder. |

```bash
bash 0a_ignite.sh        # Entry point — launches the smart governor
bash 2_global_aggregate.sh   # Aggregate seeds once jobs complete
bash 3a_global_plot.sh       # Generate performance trace plots
bash 3b_collect_plots.sh     # Collect plots into centralized folder
```

---

## Directory Structure

```
src/utils/          # Core utilities package
src/data/           # Storage for processed .pkl datasets
experiments/        # Cluster orchestration
results/            # Output directory for raw seeds, aggregated data, and plots
```

### `src/utils/` — Core Research Package

| File | Description |
|------|-------------|
| `models.py` | Wrappers for Scikit-Learn and PySORTD; includes C++ serialization/unpickling fixes |
| `query_strategies.py` | Implementation of AL strategies (UNREAL, QBC, Coreset, Uncertainty) |
| `learning_procedure.py` | The simulation engine orchestrating the training and selection loop |
| `calibration.py` | Three-stage LOO-CV calibration (Hyperparameters → ε expansion → β tuning) |
| `tree_utils.py` | Converters and metrics for Tree Edit Distance (TED) |

### `experiments/` — Cluster Orchestration

| File | Description |
|------|-------------|
| `master_config.py` | Single source of truth for method definitions (M1–M9) and SLURM config |
| `generate_sbatch.arrays.py` | Job factory and master script generator |

---

## Code Overview

### Simulation Engine (`src/utils/`)

- **`run_experiment.py`** — The primary job wrapper. Coordinates dataset loading, calibration, model initialization, and the final AL loop.
- **`learning_procedure.py`** — Handles the iterative cycle:
  > Train Predictor & Selector → Evaluate Metrics → Select Query via Strategy → Update Pool
- **`calibration.py`** — Automatically tunes parameters on initial "pilot" data:
  - Finds the best `max_depth` and regularization
  - Expands the Rashomon threshold until a minimum set size is reached
  - Calibrates the Gibbs temperature (β) to align model entropy with prediction errors

### Selector Strategies (`src/utils/query_strategies.py`)

| Method | Name | Description |
|--------|------|-------------|
| M1 | `PassiveSelector` | Random sampling baseline |
| M2–M4 | `QBCSelector` | Query-By-Committee using Weighted Vote Entropy; applied to Random Forests with varying feature constraints |
| M5 | `UNREAL` | Our proposed strategy. Uses the PySORTD Rashomon set to calculate entropy over the space of near-optimal trees |
| M6 | `UncertaintySelector` | Traditional entropy sampling using a single greedy decision tree |
| M7 | `HammingDiversitySelector` | Coreset strategy maximizing the minimum Hamming distance to the labeled set |

### Analysis & Reporting (`src/utils/`)

| Script | Description |
|--------|-------------|
| `generate_label_efficiency.py` | Computes $N_{rel}$, measuring labels required relative to Random Sampling to reach performance milestones (70%, 80%, 90%) |
| `generate_auc_heatmaps.py` | Generates heatmaps of AUC ratios with budget truncation (90%), focusing on early-stage learning efficiency |
| `generate_runtime_table.py` | Compiles computational cost data into publication-ready LaTeX tables |
| `generate_dataset_table.py` | Summarizes dataset properties, including Rashomon Set Size and Effective Committee Size (ECS)
| `check_counts.py` | Diagnostic tool providing a real-time grid of experiment completion status |