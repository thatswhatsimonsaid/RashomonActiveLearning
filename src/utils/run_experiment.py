### Summary ###
"""
Main entry point for running active learning simulations.
"""

### Libraries ###
import argparse
import pickle
from pathlib import Path
from typing import Dict, Any
import warnings
from src.utils.data_handler import load_data, split_test_pool, get_random_initial_indices
from src.utils.calibration import calibrate_hyperparameters
from src.utils.models import (
    ModelWrapper, 
    RandomForestWrapper, 
    GOSDTWrapper,
    PySORTDWrapper,
    LogisticRegressionWrapper,
    GreedyDecisionTreeWrapper,
    BMARandomForestWrapper
)
from src.utils.query_strategies import (
    Selector, 
    PassiveSelector, 
    QBCSelector,
    UncertaintySelector,
    HammingDiversitySelector,
    ModelChangeSelector
)
from src.utils.learning_procedure import SimulationConfig, run_learning_procedure

# Filter warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")

### Registeries ###
SELECTOR_MODEL_REGISTRY = {
    "PySORTDWrapper": PySORTDWrapper, 
    "RandomForest": RandomForestWrapper,
    "LogisticRegression": LogisticRegressionWrapper,
    "GreedyTree": GreedyDecisionTreeWrapper,
    "BMARandomForest": BMARandomForestWrapper
}

PREDICTOR_MODEL_REGISTRY = {
    "PySORTDWrapper": PySORTDWrapper,
    "GOSDT": GOSDTWrapper,
    "LogisticRegression": LogisticRegressionWrapper,
    "GreedyTree": GreedyDecisionTreeWrapper,
    "RandomForest": RandomForestWrapper,
    "BMARandomForest": BMARandomForestWrapper
}

SELECTOR_REGISTRY = {
    "Passive": PassiveSelector,
    "Random": PassiveSelector,
    "QBC": QBCSelector, 
    "Uncertainty": UncertaintySelector,
    "HammingDiversity": HammingDiversitySelector,
    "RashomonExpectedModelChange": ModelChangeSelector
}

### ARGUMENT PARSING HELPERS ###
def parse_additional_args(args: list) -> Dict[str, Any]:
    config = {}
    for arg in args:
        if "=" not in arg:
            raise ValueError(f"Invalid argument format: {arg}. Use key=value.")
        key, value = arg.split("=", 1)
        try:
            config[key] = float(value) if '.' in value else int(value)
        except ValueError:
            if value.lower() == 'true': config[key] = True
            elif value.lower() == 'false': config[key] = False
            else: config[key] = value
    return config

### MAIN FUNCTION ###
def main():
    
    ## 0. Arguments ##
    parser = argparse.ArgumentParser(description="Run an active learning experiment.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--selector_model", type=str, required=True, choices=SELECTOR_MODEL_REGISTRY.keys())
    parser.add_argument("--predictor_model", type=str, required=True, choices=PREDICTOR_MODEL_REGISTRY.keys())
    parser.add_argument("--selector", type=str, required=True, choices=SELECTOR_REGISTRY.keys())
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method_number", type=int, required=True)
    parser.add_argument("--rashomon_threshold", type=float, default=0.05)
    parser.add_argument("--study_dir", type=str, default="study1_active_learning")
    args, unknown_args = parser.parse_known_args()    
    additional_config = parse_additional_args(unknown_args)

    predictor_params = {"random_state": args.seed}
    selector_params = {
        "random_state": args.seed, 
        "rashomon_threshold": args.rashomon_threshold,
        **additional_config 
    }
    
    if "PySORTD" in args.selector_model:
        selector_params["rashomon_multiplier"] = args.rashomon_threshold

    ## 1. Data Setup ##
    df_full = load_data(args.dataset, base_path=Path("src/data/"))
    df_test, df_working_pool = split_test_pool(df_full, test_proportion=0.2, random_state=args.seed)

    ## 2. Initial Observations ##
    initial_train_size = 20 
    if initial_train_size >= len(df_working_pool): 
        initial_train_size = len(df_working_pool) // 2
        
    initial_indices = get_random_initial_indices(
        y_train=df_working_pool["Y"].values,
        n_initial=initial_train_size,
        random_state=args.seed
    )
    df_train = df_working_pool.iloc[initial_indices]
    df_candidate = df_working_pool.drop(df_train.index)

    ## 2.5 Neutral Ground Calibration ##
    predictor_model_class = PREDICTOR_MODEL_REGISTRY[args.predictor_model]

    calib_base_params = {"random_state": args.seed}
    calib_base_params.update(additional_config)     
    calib_base_params["rashomon_threshold"] = args.rashomon_threshold

    print(f"--- Calibration: Finding Neutral Ground for {args.dataset} ---")
    calibration_results = calibrate_hyperparameters(
        df_pilot=df_train,
        model_class=predictor_model_class,
        base_params=calib_base_params
    )

    ## 3. Instantiate Models with Calibrated Params ##
    selector_model_class = SELECTOR_MODEL_REGISTRY[args.selector_model]    
    selector_class = SELECTOR_REGISTRY[args.selector]    
    
    current_selector_params = selector_params.copy()
    strategy_params = selector_params.copy()

    # 3a. Convert Calibrated Adder to Effective Multiplier
    min_loss = calibration_results["pilot_loss_proxy"]
    eps_adder = calibration_results["rashomon_epsilon_adder"]
    effective_multiplier = (min_loss + eps_adder) / max(min_loss, 1e-6)

    ## 3b. Update Shared Params
    shared_updates = {
        "beta": calibration_results.get("beta", 10.0),
        "max_depth": calibration_results["max_depth"],
        "regularization": calibration_results["regularization"],
        "rashomon_multiplier": effective_multiplier
    }    

    if args.selector_model == "RandomForest":
        shared_updates["beta"] = 0.0
    current_selector_params.update(shared_updates)
    strategy_params.update(shared_updates)

    # 3c. Specific Logic for UNREAL / PySORTD
    if "PySORTD" in args.selector_model:
        current_selector_params["rashomon_multiplier"] = effective_multiplier
        strategy_params["beta"] = calibration_results.get("beta", 0.0)

    # 3d. Update Predictor Params 
    predictor_params.update({
        "regularization": calibration_results["regularization"],
        "max_depth": 5 
    })

    # 3e. Instantiate everything
    selector_model = selector_model_class(**current_selector_params)
    predictor_model = predictor_model_class(**predictor_params)
    selector = selector_class(**strategy_params)
    
    ## 4. Train Oracle Model ##
    oracle_model = predictor_model_class(**predictor_params)
    oracle_model.fit(df_working_pool.drop(columns="Y"), df_working_pool["Y"])

    ## 5. Run Simulation ##
    print(f"Starting Experiment: {args.dataset} | Sel:{args.selector_model} | Pred:{args.predictor_model} | Strat:{args.selector}")
    sim_config = SimulationConfig(
        selector_model=selector_model,
        predictor_model=predictor_model,
        oracle_model=oracle_model,
        selector=selector, 
        df_train=df_train,
        df_candidate=df_candidate,
        df_test=df_test
    )    
    results = run_learning_procedure(sim_config, calibrated_params=calibration_results)

    ## 6. Save Results ##
    output_dir = Path("results") / args.study_dir / args.dataset / f"M{args.method_number}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"M{args.method_number}_S{args.seed}.pkl"
    output_path = output_dir / filename
    
    with open(output_path, "wb") as f:
        print(f"Saving results to {output_path}...")
        pickle.dump(results, f)
        print("Save successful.")

### RUN MAIN ####
if __name__ == "__main__":
    main()