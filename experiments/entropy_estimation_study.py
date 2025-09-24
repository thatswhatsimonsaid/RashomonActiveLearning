### SUMMARY ###
"""
A simulation study to compare how accurately different committee types
(Random Forest vs. Rashomon) estimate a "ground truth" vote entropy.

The study follows these steps:
1.  Generate a "universe" of trees using TreeFarms with a large threshold.
2.  Calculate the vote entropy from this universe as the "ground truth".
3.  Train a standard Random Forest and a TreeFarms model with a small threshold.
4.  Calculate the vote entropy from these two models as the "estimates".
5.  Compare the estimates to the ground truth using Mean Squared Error and a plot.
"""

### LIBRARIES ###
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### IMPORTS ###
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.utils.data_handler import load_data
from src.utils.models import TreeFarmsWrapper, RandomForestWrapper

### HELPER FUNCTION ###
def _calculate_vote_entropy(committee_preds: pd.DataFrame) -> np.ndarray:
    """Calculates vote entropy for each sample in a set of predictions."""
    # Proportions of votes for each class, for each data point (row)
    proportions = committee_preds.apply(
        lambda row: row.value_counts(normalize=True), axis=1
    ).fillna(0)
    # Shannon entropy formula: - sum(p * log(p))
    entropy = -np.sum(proportions * np.log2(proportions + 1e-9), axis=1)
    return entropy.values

### MAIN SIMULATION FUNCTION ###
def run_entropy_simulation(dataset_name: str, seed: int = 42):
    """
    Runs the full simulation study for a given dataset.
    """
    print(f"--- Starting Entropy Estimation Study for Dataset: {dataset_name} ---")

    # 1. Take a dataset D
    print("[1/5] Loading data...")
    df = load_data(dataset_name, base_path=PROJECT_ROOT / "src/data/processed")
    X = df.drop(columns="Y")
    y = df["Y"]

    # 2. Run TreeFarms with epsilon=0.5 to get the "universe" of all good trees
    print("[2/5] Generating 'universe' of trees (T_ALL)...")
    tf_all_model = TreeFarmsWrapper(rashomon_threshold=0.5, regularization=0.01)
    tf_all_model.fit(X, y)
    t_all_preds = tf_all_model.get_raw_ensemble_predictions(X)

    # 3. Calculate the "true" vote entropy from this universe
    print("[3/5] Calculating 'ground truth' entropy...")
    true_entropy = _calculate_vote_entropy(t_all_preds)
    
    # 4. Calculate the Random Forest entropy estimate
    print("[4/5] Calculating Random Forest entropy estimate...")
    rf_model = RandomForestWrapper(n_estimators=len(tf_all_model.all_trees_), random_state=seed)
    rf_model.fit(X, y)
    rf_preds = rf_model.get_raw_ensemble_predictions(X)
    rf_entropy_estimate = _calculate_vote_entropy(rf_preds)

    # 5. Calculate the Rashomon entropy estimate
    print("[5/5] Calculating Rashomon set entropy estimate...")
    # Per your plan, we re-run with a smaller epsilon
    tf_rashomon_model = TreeFarmsWrapper(rashomon_threshold=0.02, regularization=0.01)
    tf_rashomon_model.fit(X, y)
    tf_rashomon_preds = tf_rashomon_model.get_raw_ensemble_predictions(X)
    rashomon_entropy_estimate = _calculate_vote_entropy(tf_rashomon_preds)
    
    print("\n--- Simulation Complete ---")

    return {
        "true_entropy": true_entropy,
        "rf_estimate": rf_entropy_estimate,
        "rashomon_estimate": rashomon_entropy_estimate,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the entropy estimation simulation study.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use.")
    args = parser.parse_args()
    
    # Run the simulation
    results = run_entropy_simulation(dataset_name=args.dataset)

    # Compare the estimates to the true entropy
    mse_rf = np.mean((results["rf_estimate"] - results["true_entropy"])**2)
    mse_rashomon = np.mean((results["rashomon_estimate"] - results["true_entropy"])**2)

    print("\n--- Comparison Results ---")
    print(f"Mean Squared Error (Random Forest vs. True): {mse_rf:.6f}")
    print(f"Mean Squared Error (Rashomon vs. True):   {mse_rashomon:.6f}")
    
    if mse_rashomon < mse_rf:
        print("\nConclusion: The Rashomon set provided a closer estimate to the true entropy. ✅")
    else:
        print("\nConclusion: The Random Forest provided a closer estimate to the true entropy.")

    # Generate a visual comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    fig.suptitle(f'Entropy Estimation Comparison on {args.dataset} Dataset', fontsize=16)

    # Plot for Random Forest
    ax1.scatter(results["true_entropy"], results["rf_estimate"], alpha=0.5, label=f'MSE: {mse_rf:.4f}')
    ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Match (y=x)')
    ax1.set_title("Random Forest Estimate")
    ax1.set_xlabel("True Entropy")
    ax1.set_ylabel("Estimated Entropy")
    ax1.grid(True)
    ax1.legend()

    # Plot for Rashomon Estimate
    ax2.scatter(results["true_entropy"], results["rashomon_estimate"], alpha=0.5, color='g', label=f'MSE: {mse_rashomon:.4f}')
    ax2.plot([0, 1], [0, 1], 'r--', label='Perfect Match (y=x)')
    ax2.set_title("Rashomon Estimate")
    ax2.set_xlabel("True Entropy")
    ax2.grid(True)
    ax2.legend()
    
    # Save the plot
    output_dir = PROJECT_ROOT / "results/images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset}_entropy_estimation.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nScatter plot saved to: {output_path}")