### Summary ###
import numpy as np

### SLURM Configuration ###
SLURM_CONFIG = {
    "partition": "short",
    "time": "7:59:00",
    "mem_per_cpu": "30G",
    "mail_type": "FAIL",
    "mail_user": 'simondn@uw.edu'
}

### GLOBAL PARAMETERS ###
N_REPLICATIONS = 25
TASK_TYPE = "classification"
DATASETS = [
    'monk2',                 
    'Parity_8bit_Noise_26',  
    'Parity_8bit_Noise_16',  
    'Parity_8bit_Noise_06',
    'Parity_8bit_Noise_00',
    'Ablation_Bimodal_Study',
    ]

# PREDICTION PARAMETERS #
PREDICTION_PARAMS = {
    "max_depth": 5, 
    "regularization": 0.1,
    "time_limit": 30,
}


# SELECTION PARAMS #
SELECTION_PARAMS = {
    "max_depth": 3,
    "regularization": 0.1, 
    "time_limit": 60,
    "max_num_trees": 100000, 
    "beta": 100.0, 
}

# RF BASE PARAMS #
RF_SELECTION_PARAMS = {
    "n_estimators": 100,
    "use_unique_trees": False,
    "time_limit": 30,
    "max_depth": 3
}

### STUDIES ###
STUDIES = [
    {
        "name": "ABS",   
        "predictor": "PySORTDWrapper",
        "params": PREDICTION_PARAMS  
    }
]

### Selection Methods (The Epsilon Sweep) ###
BASE_SELECTORS = [
    # 1. Random Baseline #
    {
        "selector_model": "RandomForest", 
        "selector": "Random", 
        "fixed_threshold": 0.0,
        "params": SELECTION_PARAMS
    },
    
    # 2. RF Baseline (Standard Sqrt) #
    {
        "selector_model": "RandomForest", 
        "selector": "QBC", 
        "fixed_threshold": 0.0,
        "params": {**RF_SELECTION_PARAMS, "max_features": "sqrt"}
    },

    # --- UNREAL (Uniform Weights) Sweep ---
    {
        "selector_model": "PySORTDWrapper", 
        "selector": "QBC", 
        "fixed_threshold": 0.05,
        "label_suffix": "U0.05",
        "params": {**SELECTION_PARAMS, "beta": 0.0} 
    },
    {
        "selector_model": "PySORTDWrapper", 
        "selector": "QBC", 
        "fixed_threshold": 0.20,
        "label_suffix": "U0.20",
        "params": {**SELECTION_PARAMS, "beta": 0.0} 
    },
    {
        "selector_model": "PySORTDWrapper", 
        "selector": "QBC", 
        "fixed_threshold": 0.50,
        "label_suffix": "U0.50",
        "params": {**SELECTION_PARAMS, "beta": 0.0} 
    },
    {
        "selector_model": "PySORTDWrapper", 
        "selector": "QBC", 
        "fixed_threshold": 1.00,
        "label_suffix": "U1.00",
        "params": {**SELECTION_PARAMS, "beta": 0.0} 
    },

    # --- BREAL (Gibbs Weighted) Sweep ---
    {
        "selector_model": "PySORTDWrapper", 
        "selector": "QBC", 
        "fixed_threshold": 0.05,
        "label_suffix": "B0.05",
        "params": {**SELECTION_PARAMS, "beta": 100.0} 
    },
    {
        "selector_model": "PySORTDWrapper", 
        "selector": "QBC", 
        "fixed_threshold": 0.20,
        "label_suffix": "B0.20",
        "params": {**SELECTION_PARAMS, "beta": 100.0} 
    },
    {
        "selector_model": "PySORTDWrapper", 
        "selector": "QBC", 
        "fixed_threshold": 0.50,
        "label_suffix": "B0.50",
        "params": {**SELECTION_PARAMS, "beta": 100.0} 
    },
    {
        "selector_model": "PySORTDWrapper", 
        "selector": "QBC", 
        "fixed_threshold": 1.00,
        "label_suffix": "B1.00",
        "params": {**SELECTION_PARAMS, "beta": 100.0} 
    }
]