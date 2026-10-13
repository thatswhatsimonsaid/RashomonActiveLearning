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
# Running plus waiting tasks for this user. The smart launcher will not
# submit another dataset once the queue is at this size.
MAX_QUEUED_JOBS = 1000
TASK_TYPE = "regression"
# SORTD keeps a tree when its objective is at most (1 + ε) times optimal.
RASHOMON_THRESHOLD = 0.10

# seoul-bike is left out. Labeling its full pool is about 7,000 PySORTD refits.
DATASETS = [
    "airfoil",
    "airquality",
    "enb-cool",
    "enb-heat",
    "optical",
    "real-estate",
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

# Wu, Lin, and Huang (2019) benchmarks that are not in the PySORTD folder.
# concrete-cs is the 28-day strength from the slump test, not the 1,030-row
# concrete table. CPS wages omit the race column.
ALR_DATASETS = [
    "concrete-slump",
    "concrete-flow",
    "concrete-cs",
    "autompg",
    "no2",
    "pm10",
    "cps",
    "concrete",
]

DATASETS = DATASETS + ALR_DATASETS

# This wave labels each pool to the end. airfoil is about 1,200 refits, so it waits.
# ALR_DATASETS is the next wave. It is not in SUBMIT_DATASETS until that wave is launched.
SUBMIT_DATASETS = [
    "airquality",
    "servo",
    "yacht",
    "real-estate",
    "Synthetic_Regression_Baseline",
    "Synthetic_Regression_Alpha_25",
    "Synthetic_Regression_Alpha_50",
    "Synthetic_Regression_Alpha_75",
    "Synthetic_Regression_Alpha_100",
    "Synthetic_Regression_Phi_05",
    "Synthetic_Regression_Phi_10",
    "Synthetic_Regression_Phi_25",
    "Synthetic_Regression_Phi_45",
    "sync",
    "optical",
    "enb-cool",
    "enb-heat",
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
    "max_num_trees": 10000,
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
    {
        "selector_model": "GreedyRegressionTree",
        "selector": "GSx",
        "params": {"max_depth": 3},
    },
    {
        "selector_model": "GreedyRegressionTree",
        "selector": "GSy",
        "params": {"max_depth": 3},
    },
    {
        "selector_model": "GreedyRegressionTree",
        "selector": "iGS",
        "params": {"max_depth": 3},
    },
]
