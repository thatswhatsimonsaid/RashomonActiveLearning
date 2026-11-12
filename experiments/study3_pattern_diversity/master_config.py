### Summary ###
"""
Synthetic-only study for Pattern Disambiguation vs UNREAL / QBC baselines.

Criterion: select the unlabeled point that maximizes the number of distinct
classification patterns among Rashomon trees when each tree's prediction at
that point is appended to its labeled-set prediction vector.
"""

### SLURM Configuration ###
SLURM_CONFIG = {
    "partition": "short",
    "time": "11:59:00",
    "mem_per_cpu": "30G",
    "mail_type": "FAIL",
    "mail_user": "simondn@uw.edu",
}

### GLOBAL PARAMETERS ###
N_REPLICATIONS = 25

# Synthetic XOR / Phi / Alpha stress tests only (no UCI / TreeFarms).
DATASETS = [
    "Synthetic_XOR_Baseline",
    "Synthetic_XOR_Phi_05",
    "Synthetic_XOR_Phi_10",
    "Synthetic_XOR_Phi_25",
    "Synthetic_XOR_Phi_45",
    "Synthetic_XOR_Alpha_25",
    "Synthetic_XOR_Alpha_50",
    "Synthetic_XOR_Alpha_75",
    "Synthetic_XOR_Alpha_100",
]

PREDICTION_PARAMS = {
    "max_depth": 5,
    "regularization": 0.001,
    "time_limit": 30,
}

SELECTION_PARAMS = {
    "max_depth": 3,
    "regularization": 0.001,
    "time_limit": 30,
    "max_num_trees": 10000,
    "beta": 0.0,
}

RF_SELECTION_PARAMS = {
    "n_estimators": 100,
    "use_unique_trees": False,
    "time_limit": 30,
}

### STUDIES ###
STUDIES = [
    {
        "name": "tree_predictor",
        "predictor": "PySORTDWrapper",
        "params": PREDICTION_PARAMS,
    }
]

### Selection Methods ###
BASE_SELECTORS = [
    # M1: Random
    {
        "selector_model": "RandomForest",
        "selector": "Random",
        "fixed_threshold": 0.0,
        "params": SELECTION_PARAMS,
    },
    # M2: RF-QBC (sqrt features)
    {
        "selector_model": "RandomForest",
        "selector": "QBC",
        "fixed_threshold": 0.0,
        "params": {**RF_SELECTION_PARAMS, "max_features": "sqrt"},
    },
    # M3: UNREAL uniform (vote entropy on Rashomon set)
    {
        "selector_model": "PySORTDWrapper",
        "selector": "QBC",
        "fixed_threshold": 0.0,
        "params": {**SELECTION_PARAMS, "beta": 0.0, "use_unique_trees": False},
    },
    # M4: UNREAL on unique patterns (vote entropy after pattern collapse)
    {
        "selector_model": "PySORTDWrapper",
        "selector": "QBC",
        "fixed_threshold": 0.0,
        "params": {**SELECTION_PARAMS, "beta": 0.0, "use_unique_trees": True},
    },
    # M5: Pattern disambiguation — maximize |distinct extended patterns|
    {
        "selector_model": "PySORTDWrapper",
        "selector": "PatternDisambiguation",
        "fixed_threshold": 0.0,
        "params": {**SELECTION_PARAMS, "beta": 0.0, "score_mode": "count"},
    },
    # M6: Pattern disambiguation — maximize effective pattern count exp(H)
    {
        "selector_model": "PySORTDWrapper",
        "selector": "PatternDisambiguation",
        "fixed_threshold": 0.0,
        "params": {**SELECTION_PARAMS, "beta": 0.0, "score_mode": "effective"},
    },
    # M7: Pattern disambiguation with calibrated Gibbs weights
    {
        "selector_model": "PySORTDWrapper",
        "selector": "PatternDisambiguation",
        "fixed_threshold": 0.0,
        "params": {
            **SELECTION_PARAMS,
            "beta": "calibrated",
            "score_mode": "count",
        },
    },
]
