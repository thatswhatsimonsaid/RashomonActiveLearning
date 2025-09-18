"""
Main entry point for running active learning simulations.

This script parses command-line arguments to configure and execute a full
active learning experiment, saving the results to a file.
"""

### LIBRARIES ###
import argparse
import pickle
from pathlib import Path
from typing import Dict, Any

### IMPORTS ###
from src.utils.data_handler import load_data, split_data
from src.utils.models import ( ModelWrapper, TreeFarmsWrapper, RandomForestWrapper, GaussianProcessWrapper, BNNWrapper)
from src.utils.selectors import Selector, PassiveSelector, QBCSelector, BALDSelector
from src.utils.learning_procedure import SimulationConfig, run_learning_procedure

### REGISTRIES ###
MODEL_REGISTRY = {
    "TreeFarms": TreeFarmsWrapper,
    "RandomForest": RandomForestWrapper,
    "GPC": GaussianProcessWrapper,
    "BNN": BNNWrapper,
}
SELECTOR_REGISTRY = {
    "Passive": PassiveSelector,
    "QBC": QBCSelector,
    "BALD": BALDSelector,
}

### ARGUMENT PARSING HELPERS ###
def parse_additional_args(args: list) -> Dict[str, Any]:
    """Helper function to parse key=value pairs for model/selector configs."""
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
    parser = argparse.ArgumentParser(description="Run an active learning experiment.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=MODEL_REGISTRY.keys())
    parser.add_argument("--selector", type=str, required=True, choices=SELECTOR_REGISTRY.keys())
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method_number", type=int, required=True)
    
    args, unknown_args = parser.parse_known_args()
    additional_config = parse_additional_args(unknown_args)

    # print("--- Starting Experiment ---")
    # print(f"Dataset: {args.dataset}, Model: {args.model}, Selector: {args.selector}, Seed: {args.seed}")
    # print(f"Method Number: {args.method_number}, Additional Config: {additional_config}")

    ## 1. Load and split data ##
    df = load_data(args.dataset, base_path=Path("src/data/processed"))
    df_train, df_test, df_candidate = split_data(
        df, test_proportion=0.2, candidate_proportion_of_remainder=0.8
    )

    ## 2. Instantiate model and selector ##
    model_class = MODEL_REGISTRY[args.model]
    selector_class = SELECTOR_REGISTRY[args.selector]
    config_params = {"random_state": args.seed, **additional_config}
    model = model_class(**config_params)
    selector = selector_class(**config_params)

    ## 3. Create simulation config ##
    sim_config = SimulationConfig(
        model=model, selector=selector, df_train=df_train,
        df_test=df_test, df_candidate=df_candidate
    )

    ## 4. Run the learning procedure ##
    results = run_learning_procedure(sim_config)

    ## 5. Save the results ##
    output_dir = Path(f"results/{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)    
    filename = f"M{args.method_number}_S{args.seed}.pkl"
    output_path = output_dir / filename
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    # print(f"\n--- Experiment Complete ---")
    # print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()