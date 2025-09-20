# --- SLURM Configuration ---
SLURM_CONFIG = {
    "partition": "short",
    "time": "11:59:00",
    "mem_per_cpu": "30000M",
    "mail_user": "simondn@uw.edu",
    "mail_type": "FAIL"
}

### Replications ###
N_REPLICATIONS = 50

### Experiment Configurations ###
EXPERIMENT_CONFIGS = [
    ### 1. Passive and Random Forest ###
    {"model": "RandomForest", 
     "selector": "Passive", 
     "params": {"n_estimators": 100}
    },

    ### 2. Passive and GPC ###
    {"model": "GPC", 
     "selector": "Passive",
     "params": {}
    },

    ### 3. Passive and BNN ###
    {"model": "BNN", 
     "selector": "Passive",
     "params": {"epochs": 50}
    },

    ### 4. BALD and BNN ###
    {"model": "BNN", 
     "selector": "BALD",
     "params": {"epochs": 50, "n_ensemble_samples": 100}
    },

    ### 5. BALD and GPC ###
    {"model": "GPC", 
     "selector": "BALD",
     "params": {"n_ensemble_samples": 100}
    },

    ### 6. UNREAL ###
    {"model": "TreeFarms", 
     "selector": "QBC",
     "params": {"regularization": 0.01, "rashomon_threshold": 0.025, "use_unique_trees": True}
    },

    ### 7. DUREAL ###
    {"model": "TreeFarms", 
     "selector": "QBC",
     "params": {"regularization": 0.01, "rashomon_threshold": 0.025, "use_unique_trees": False}
    },

    ### 8. RF and QBC ###
    {"model": "RandomForest", 
     "selector": "QBC",
    "params": {"n_estimators": 100, "use_unique_trees": False}
    },
]