### SUMMARY ###
"""
A simulation study to compare how accurately different committee types
(Random Forest vs. DUREAL/UNREAL Proxies) estimate a "ground truth" vote entropy.

This script operates in two modes for cluster execution:
1. generate: Trains a massive RF, calculates ground truth data, and saves it.
2. analyze: Loads the ground truth data and runs the committee comparisons.
"""

### LIBRARIES ###
import sys
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Add the project's root directory to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

### IMPORTS ###
from sklearn.metrics import accuracy_score
from src.utils.data_handler import load_data
from src.utils.models import RandomForestWrapper


### HELPER FUNCTION ###
def _calculate_vote_entropy(committee_preds: pd.DataFrame) -> np.ndarray:
    """Calculates vote entropy for each sample in a set of predictions."""
    proportions = committee_preds.apply(
        lambda row: row.value_counts(normalize=True), axis=1
    ).fillna(0)
    entropy = -np.sum(proportions * np.log2(proportions + 1e-9), axis=1)
    return entropy.values


### MODE 1: UNIVERSE GENERATION ###
def generate_universe(args):
    """Trains a massive RF and saves the ground truth data."""
    print(f"--- [Mode: GENERATE] for Dataset: {args.dataset} ---")
    print(f"Universe size: {args.n_universe}, Seed: {args.seed}")

    # 1. Load Data
    print("[1/3] Loading data...")
    df = load_data(args.dataset, base_path=PROJECT_ROOT / "src/data/processed")
    X = df.drop(columns="Y")
    y = df["Y"]

    # 2. Generate Universe of Trees
    print(f"[2/3] Generating 'universe' of {args.n_universe} trees...")
    universe_model_wrapper = RandomForestWrapper(
        n_estimators=args.n_universe, random_state=args.seed, n_jobs=-1
    )
    universe_model_wrapper.fit(X, y)
    universe_model = universe_model_wrapper.model

    # 3. Calculate and Save Ground Truth Data
    print("[3/3] Calculating and saving ground truth data...")
    universe_preds_df = universe_model_wrapper.get_raw_ensemble_predictions(X)

    ### Uncomment this section to have the true universe only contain unique patterns ###
    # Stats cluster: Universal trees have duplicates
    # HYAK cluster: Universal trees are unique
    universe_preds_df = universe_preds_df.T.drop_duplicates().T
    print(f"  -> Found {len(universe_preds_df.columns)} unique patterns.")

    # 4. Calculate ground truth entropy
    true_entropy = _calculate_vote_entropy(universe_preds_df)

    # Evaluate each individual tree from the universe
    tree_accuracies = [
        (accuracy_score(y, tree.predict(X.values)), i)
        for i, tree in enumerate(universe_model.estimators_)
    ]

    # Save the necessary data to disk
    output_dir = PROJECT_ROOT / "results/entropy_study"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset}_universe_data.pkl"

    universe_data = {
        "true_entropy": true_entropy,
        "tree_accuracies": tree_accuracies,
        "universe_model": universe_model, # Save the model to sample from it
        "X": X, # Save X for predictions
        "params": vars(args)
    }
    with open(output_path, "wb") as f:
        pickle.dump(universe_data, f)

    print(f"Universe data saved to {output_path}")


### MODE 2: ANALYSIS AND PLOTTING ###
def analyze_universe(args):
    """Loads saved universe data and runs the committee comparisons."""
    print(f"--- [Mode: ANALYZE] for Dataset: {args.dataset} ---")
    
    # 1. Load Universe Data
    print("[1/4] Loading pre-generated universe data...")
    universe_file = PROJECT_ROOT / "results/entropy_study" / f"{args.dataset}_universe_data.pkl"
    if not universe_file.exists():
        raise FileNotFoundError(f"Universe data file not found: {universe_file}. Please run the 'generate' mode first.")
    
    with open(universe_file, "rb") as f:
        universe_data = pickle.load(f)

    true_entropy = universe_data["true_entropy"]
    tree_accuracies = universe_data["tree_accuracies"]
    universe_model = universe_data["universe_model"]
    X = universe_data["X"]
    n_universe = universe_data["params"]["n_universe"]

    # 2. Create Committees
    print("[2/4] Creating filtered and sampled committees...")
    
    # DUREAL Proxy
    tree_accuracies.sort(key=lambda x: x[0], reverse=True)
    best_tree_indices = [idx for acc, idx in tree_accuracies[:args.n_estimate]]
    best_trees_committee = [universe_model.estimators_[i] for i in best_tree_indices]

    # UNREAL Proxy
    best_tree_preds = np.array([tree.predict(X.values) for tree in best_trees_committee]).T
    best_tree_preds_df = pd.DataFrame(best_tree_preds)
    unique_indices_local = best_tree_preds_df.T.drop_duplicates().index
    unique_best_trees_committee = [best_trees_committee[i] for i in unique_indices_local]

    # Randomly Sampled RF
    random.seed(args.seed)
    randomly_sampled_committee = random.sample(universe_model.estimators_, args.n_estimate)

    # 3. Calculate Entropy Estimates
    print("[3/4] Calculating entropy estimates...")
    # Random
    random_preds_df = pd.DataFrame(np.array([tree.predict(X.values) for tree in randomly_sampled_committee]).T)
    random_entropy = _calculate_vote_entropy(random_preds_df)
    # DUREAL
    dureal_preds_df = pd.DataFrame(np.array([tree.predict(X.values) for tree in best_trees_committee]).T)
    dureal_entropy = _calculate_vote_entropy(dureal_preds_df)
    # UNREAL
    unreal_preds_df = pd.DataFrame(np.array([tree.predict(X.values) for tree in unique_best_trees_committee]).T)
    unreal_entropy = _calculate_vote_entropy(unreal_preds_df)

    mse_random = np.mean((random_entropy - true_entropy) ** 2)
    mse_dureal = np.mean((dureal_entropy - true_entropy) ** 2)
    mse_unreal = np.mean((unreal_entropy - true_entropy) ** 2)

    print("\n--- Comparison Results ---")
    print(f"MSE (Randomly Sampled vs. True): {mse_random:.6f}")
    print(f"MSE (DUREAL Proxy vs. True):     {mse_dureal:.6f}")
    print(f"MSE (UNREAL Proxy vs. True):     {mse_unreal:.6f}")

    # 4. Plot Results
    print("[4/4] Generating plot...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6), sharex=True, sharey=True)
    fig.suptitle(f'Entropy Estimation Comparison on {args.dataset} Dataset', fontsize=16)
    plt.style.use('seaborn-v0_8-whitegrid')

    ax1.scatter(true_entropy, random_entropy, alpha=0.4, label=f'MSE: {mse_random:.4f}')
    ax1.set_title(f"Randomly Sampled RF ({args.n_estimate} Trees)")
    
    ax2.scatter(true_entropy, dureal_entropy, alpha=0.4, color='g', label=f'MSE: {mse_dureal:.4f}')
    ax2.set_title(f"DUREAL Proxy (Top {args.n_estimate} Trees)")

    ax3.scatter(true_entropy, unreal_entropy, alpha=0.4, color='m', label=f'MSE: {mse_unreal:.4f}')
    ax3.set_title(f"UNREAL Proxy ({len(unique_best_trees_committee)} Unique Trees)")

    for ax in [ax1, ax2, ax3]:
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect Match (y=x)')
        ax.set_xlabel(f"True Entropy (from {n_universe} Tree Universe)")
        ax.set_ylabel("Estimated Entropy")
        ax.grid(True)
        ax.legend()
    
    output_path = PROJECT_ROOT / "results/images" / f"{args.dataset}_entropy_estimation_cluster.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


### MAIN SCRIPT LOGIC ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the entropy estimation simulation study.")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Operation mode')

    # --- Parent parser for shared arguments ---
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset.")
    parent_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    # --- Parser for "generate" mode ---
    parser_gen = subparsers.add_parser('generate', parents=[parent_parser], help='Generate and save the universe data.')
    parser_gen.add_argument("--n_universe", type=int, default=1_000_000, help="Number of trees in the universe.")
    parser_gen.set_defaults(func=generate_universe)

    # --- Parser for "analyze" mode ---
    parser_an = subparsers.add_parser('analyze', parents=[parent_parser], help='Analyze the pre-generated universe data.')
    parser_an.add_argument("--n_estimate", type=int, default=1000, help="Number of trees in the smaller committees.")
    parser_an.set_defaults(func=analyze_universe)

    args = parser.parse_args()
    args.func(args)