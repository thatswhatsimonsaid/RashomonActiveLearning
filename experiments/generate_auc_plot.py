### SUMMARY ###
"""
A standalone script to generate and save a ROC AUC comparison plot for
Random Forest and TreeFarms models on a specified dataset.

This script is designed to be run from the command line, making it suitable
for execution on an HPC cluster.
"""

### LIBRARIES ###
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### IMPORTS ###
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from src.utils.data_handler import load_data
from src.utils.models import RandomForestWrapper, TreeFarmsWrapper

### FUNCTIONS ###
def run_auc_comparison(
    dataset_name: str,
    output_path: Path,
    rf_n_estimators: int = 100,
    treefarms_thresholds: list[float] = None,
    random_state: int = 42,
    test_size: float = 0.3,
):
    """
    Trains RF and TreeFarms models, and generates/saves an ROC AUC comparison plot.

    Args:
        dataset_name (str): The name of the dataset to load.
        output_path (Path): Path to save the output plot file.
        rf_n_estimators (int): Number of estimators for the RandomForest model.
        treefarms_thresholds (list[float]): A list of rashomon_threshold values for TreeFarms.
        random_state (int): Seed for reproducibility.
        test_size (float): Proportion of the dataset to hold out for testing.
    """
    if treefarms_thresholds is None:
        treefarms_thresholds = [0.001, 0.01, 0.05, 0.1]

    print("--- Starting AUC Comparison Simulation ---")
    print(f"Dataset: {dataset_name}")
    print(f"Random Forest Estimators: {rf_n_estimators}")
    print(f"TreeFarms Thresholds: {treefarms_thresholds}")

    # 1. Load and Split Data
    print("\n[1/4] Loading and splitting data...")
    df = load_data(dataset_name, base_path=Path("src/data/processed"))
    X = df.drop(columns="Y")
    y = df["Y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Data split: {len(X_train)} train, {len(X_test)} test samples.")

    # 2. Run Models and Collect Results
    print("\n[2/4] Training models and collecting results...")
    roc_results = {}

    # Train and Evaluate Random Forest
    rf_model = RandomForestWrapper(n_estimators=rf_n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)
    y_pred_proba_rf = rf_model.model.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
    auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
    roc_results['Random Forest'] = {'fpr': fpr_rf, 'tpr': tpr_rf, 'auc': auc_rf}
    print(f"  -> Random Forest AUC: {auc_rf:.4f}")

    # Train and Evaluate TreeFarms for each threshold
    for threshold in treefarms_thresholds:
        tf_model = TreeFarmsWrapper(rashomon_threshold=threshold, regularization=0.01)
        tf_model.fit(X_train, y_train)
        raw_preds = tf_model.get_raw_ensemble_predictions(X_test)
        y_pred_proba_tf = (raw_preds == 1).mean(axis=1)
        fpr_tf, tpr_tf, _ = roc_curve(y_test, y_pred_proba_tf)
        auc_tf = roc_auc_score(y_test, y_pred_proba_tf)
        model_name = f'TreeFarms (thresh={threshold})'
        roc_results[model_name] = {'fpr': fpr_tf, 'tpr': tpr_tf, 'auc': auc_tf}
        print(f"  -> {model_name} AUC: {auc_tf:.4f}")

    # 3. Plot the ROC Curves
    print("\n[3/4] Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))

    for name, results in roc_results.items():
        plt.plot(results['fpr'], results['tpr'], label=f"{name}, AUC = {results['auc']:.3f}")

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance (AUC = 0.5)')
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve Comparison on {dataset_name} Dataset", fontsize=14)
    plt.legend(loc='lower right', title="Model, AUC")
    plt.grid(True)
    
    # 4. Save the plot
    print(f"\n[4/4] Saving plot to {output_path}...")
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() # Close the figure to free memory

    print("\n--- Simulation Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and save an ROC AUC comparison plot for RF and TreeFarms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to use (e.g., MONK1, COMPAS)."
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to save the output PNG image file (e.g., results/images/auc_plot.png)."
    )
    parser.add_argument(
        "--rf_estimators",
        type=int,
        default=100,
        help="Number of estimators for the Random Forest model."
    )
    parser.add_argument(
        "--tf_thresholds",
        type=float,
        nargs='+',
        default=[0.001, 0.01, 0.05, 0.1],
        help="A list of rashomon_threshold values to test for TreeFarms."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    run_auc_comparison(
        dataset_name=args.dataset,
        output_path=args.output_file,
        rf_n_estimators=args.rf_estimators,
        treefarms_thresholds=args.tf_thresholds,
        random_state=args.seed,
    )