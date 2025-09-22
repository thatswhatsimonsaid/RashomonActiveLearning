# Unique Rashomon Ensembled Active Learning (UNREAL)

🚧 This repository is currently under construction. 🚧

This repository contains the complete source code and experimental framework for the research paper on UNique Rashomon Ensembled Active Learning (UNREAL). The framework is designed for running, managing, and analyzing a large number of active learning simulations on a SLURM-based high-performance computing (HPC) cluster.

## Abstract

[NeurIPS Paper/Presentation](https://neurips.cc/virtual/2024/98966)

Active learning is based on selecting informative data points to enhance model predictions often using uncertainty as a selection criterion. However, when ensemble models such as random forests are used, there is a risk of the ensemble containing models with poor predictive accuracy or duplicates with the same interpretation. To address these challenges, we develop a novel approach called *UNique Rashomon Ensembled Active Learning (UNREAL)* to only ensemble the distinct set of near-optimal models called the Rashomon set. By ensembling over the Rashomon set, our method accounts for noise by capturing uncertainty across diverse yet plausible explanations, thereby improving the robustness of the query selection in the active learning procedure. We extensively evaluate *UNREAL* against current active learning procedures on five benchmark datasets. We demonstrate how taking a Rashomon approach can improve not only the accuracy and rate of convergence of the active learning procedure but can also lead to improved interpretability compared to traditional approaches. 

## Table of Contents

* [Project Overview](#project-overview)
* [Setup](#setup)
* [Workflow: How to Run Experiments](#workflow-how-to-run-experiments)
* [Project Structure](#project-structure)
* [Citing](#citing)

## Project Overview

This project is built around a highly automated and reproducible workflow. The entire experimental process, from job generation to final plotting, is controlled by a central configuration file. A generator script reads this configuration and automatically creates all the necessary SLURM job arrays and helper scripts for each dataset.

[Image of a flowchart of the project workflow]

The workflow is designed to be executed sequentially through a series of numbered shell scripts, which handle running simulations, aggregating results, generating plots, and cleaning up.

## Setup

This project uses Python 3.9 and manages dependencies through a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/thatswhatsimonsaid/RashomonActiveLearning.git](https://github.com/thatswhatsimonsaid/RashomonActiveLearning.git)
    cd RashomonActiveLearning
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv .RAL
    source .RAL/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Workflow: How to Run Experiments

The entire experimental pipeline is designed to be run from the command line.

### Step 1: Configure Your Experiments

All experiment parameters are defined in a single file: `experiments/master_config.py`. Before running, you should edit this file to:

* Set your SLURM cluster settings (`SLURM_CONFIG`).
* Define the number of simulation runs per method (`N_REPLICATIONS`).
* Define the list of experiments (`EXPERIMENT_CONFIGS`), specifying the model, selector, and any parameters for each method.

### Step 2: Generate the Scripts

Once your configuration is set, run the generator script from the project's root directory. This will automatically find all your datasets and create a dedicated subdirectory for each one inside `experiments/job_scripts/`.

```bash
python experiments/generate_sbatch_arrays.py
```

### Step 3: Execute the Workflow

Navigate into the newly created directory for the dataset you wish to run (e.g., `Iris`).

```bash
cd experiments/job_scripts/Iris
````

Inside, you will find a set of numbered helper scripts. Run them in order:

1.  **`./1_run_all.sh`**: Submits all SLURM job arrays to the cluster to run the simulations.
2.  **`./2_aggregate_results.sh`**: After jobs are complete, this script checks for missing results and then compiles all raw `.pkl` outputs into analysis-ready `.csv` files in the `results/<dataset>/aggregated/` directory.
3.  **`./3_plot_results.sh`**: Generates trace plots for each metric (e.g., accuracy, F1-score) and saves them as `.png` files in the `results/images/<dataset>/` directory.