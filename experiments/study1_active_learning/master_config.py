### SLURM Configuration ###
SLURM_CONFIG = {
    "partition": "short",
    "time": "11:59:00",
    "mem_per_cpu": "30G",
    "mail_user": "simondn@uw.edu",
    "mail_type": "FAIL"
}

### GLOBAL PARAMETERS ###
N_REPLICATIONS = 25

# PREDICTION PARAMETERS #
# Nota Bene: These will be overridden by calibrate.py!!!! but eval_depth stays 5
PREDICTION_PARAMS = {
    "max_depth": 5, 
    "regularization": 0.001,
    "time_limit": 30,
}

# SELECTION PARAMS #
SELECTION_PARAMS = {
    "max_depth": 3,
    "regularization": 0.001, 
    "time_limit": 30,
    "max_num_trees": 10000, 
    "beta": 0.0,  # Default to 0.0 (Uniform)
}

# RF BASE PARAMS #
RF_SELECTION_PARAMS = {
    "n_estimators": 100,
    "use_unique_trees": False,
    "time_limit": 30
}

### STUDIES ###
STUDIES = [
    {
        "name": "tree_predictor",   
        "predictor": "PySORTDWrapper",
        "params": PREDICTION_PARAMS  
    }
]

### Selection Methods ###
BASE_SELECTORS = [
    # 1. Random #
    {
        "selector_model": "RandomForest", 
        "selector": "Random", 
        "fixed_threshold": 0.0,
        "params": SELECTION_PARAMS
    },
    
    # 2. RF (Restricted / 3 Features) #
    {
        "selector_model": "RandomForest", 
        "selector": "QBC", 
        "fixed_threshold": 0.0,
        "params": {**RF_SELECTION_PARAMS, "max_features": 3}
    },
    
    # 3. RF (Standard / Sqrt) #
    {
        "selector_model": "RandomForest", 
        "selector": "QBC", 
        "fixed_threshold": 0.0,
        "params": {**RF_SELECTION_PARAMS, "max_features": "sqrt"}
    },

    # 4. RF (Full / All Features) #
    {
        "selector_model": "RandomForest", 
        "selector": "QBC", 
        "fixed_threshold": 0.0,
        "params": {**RF_SELECTION_PARAMS, "max_features": 1.0}
    },

    # 5. UNREAL (Uniform Weights / beta=0)
    {
        "selector_model": "PySORTDWrapper", 
        "selector": "QBC", 
        "fixed_threshold": 0.0,
        "params": {**SELECTION_PARAMS, "beta": 0.0} 
    },

    # 6. Classic Uncertainty Sampling (Greedy Tree) #
    {
        "selector_model": "GreedyTree", 
        "selector": "Uncertainty", 
        "fixed_threshold": 0.0,
        "params": SELECTION_PARAMS 
    },

    # 7. Coreset #
    {
    "selector_model": "GreedyTree",
    "selector": "HammingDiversity", 
    "fixed_threshold": 0.0,
    "params": {}
    },

    # 8. UNREAL- Weighted
    {
        "selector_model": "PySORTDWrapper", 
        "selector": "QBC", 
        "fixed_threshold": 0.0,  
        "params": {**SELECTION_PARAMS, "beta": "calibrated"} 
    },

    # 9. RF Weighted
    {
        "selector_model": "BMARandomForest", 
        "selector": "QBC", 
        "fixed_threshold": 0.0,
        "params": {
            **RF_SELECTION_PARAMS, 
            "max_depth": SELECTION_PARAMS["max_depth"], 
            "max_features": "sqrt",                     
            "beta": "calibrated"                        
        }
    },

    # 10. RF Weighted
    {
        "selector_model": "BMARandomForest", 
        "selector": "QBC", 
        "fixed_threshold": 0.0,
        "params": {
            **RF_SELECTION_PARAMS, 
            "max_depth": SELECTION_PARAMS["max_depth"], 
            "max_features": 1.0,                        
            "beta": "calibrated"                        
        }
    }
]