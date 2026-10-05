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
TASK_TYPE = "regression"
# SORTD keeps a tree when its objective is at most (1 + ε) times optimal.
RASHOMON_THRESHOLD = 0.10

DATASETS = [
    "airfoil",
    "airquality",
    "enb-cool",
    "enb-heat",
    "optical",
    "real-estate",
    "seoul-bike",
    "servo",
    "sync",
    "yacht",
    "Synthetic_Regression_Baseline",
    "Synthetic_Regression_Alpha_25",
    "Synthetic_Regression_Alpha_50",
    "Synthetic_Regression_Alpha_75",
    "Synthetic_Regression_Alpha_100",
    "Synthetic_Regression_Phi_05",
    "Synthetic_Regression_Phi_10",
    "Synthetic_Regression_Phi_25",
    "Synthetic_Regression_Phi_45",
]

PREDICTION_PARAMS = {
    "max_depth": 3,
    "regularization": 0.01,
    "time_limit": 30,
}

SELECTION_PARAMS = {
    "max_depth": 3,
    "regularization": 0.01,
    "time_limit": 30,
    "max_num_trees": 100,
    "beta": 0.0,
}

RF_SELECTION_PARAMS = {
    "n_estimators": 100,
    "time_limit": 30,
}

### STUDIES ###
STUDIES = [
    {
        "name": "regression_predictor",
        "predictor": "PySORTDRegressor",
        "params": PREDICTION_PARAMS,
    }
]

### Selection Methods ###
BASE_SELECTORS = [
    {
        "selector_model": "RandomForestRegressor",
        "selector": "Random",
        "params": SELECTION_PARAMS,
    },
    {
        "selector_model": "RandomForestRegressor",
        "selector": "RegressionQBC",
        "params": {**RF_SELECTION_PARAMS, "max_features": 3, "beta": 0.0},
    },
    {
        "selector_model": "RandomForestRegressor",
        "selector": "RegressionQBC",
        "params": {**RF_SELECTION_PARAMS, "max_features": "sqrt", "beta": 0.0},
    },
    {
        "selector_model": "RandomForestRegressor",
        "selector": "RegressionQBC",
        "params": {**RF_SELECTION_PARAMS, "max_features": 1.0, "beta": 0.0},
    },
    {
        "selector_model": "PySORTDRegressor",
        "selector": "RegressionQBC",
        "params": {**SELECTION_PARAMS, "beta": 0.0},
    },
    {
        "selector_model": "GreedyRegressionTree",
        "selector": "RegressionUncertainty",
        "params": SELECTION_PARAMS,
    },
    {
        "selector_model": "GreedyRegressionTree",
        "selector": "HammingDiversity",
        "params": {},
    },
    {
        "selector_model": "PySORTDRegressor",
        "selector": "RegressionQBC",
        "params": {**SELECTION_PARAMS, "beta": "calibrated"},
    },
    {
        "selector_model": "RandomForestRegressor",
        "selector": "RegressionQBC",
        "params": {
            **RF_SELECTION_PARAMS,
            "max_depth": SELECTION_PARAMS["max_depth"],
            "max_features": "sqrt",
            "beta": "calibrated",
        },
    },
    {
        "selector_model": "RandomForestRegressor",
        "selector": "RegressionQBC",
        "params": {
            **RF_SELECTION_PARAMS,
            "max_depth": SELECTION_PARAMS["max_depth"],
            "max_features": 1.0,
            "beta": "calibrated",
        },
    },
]
