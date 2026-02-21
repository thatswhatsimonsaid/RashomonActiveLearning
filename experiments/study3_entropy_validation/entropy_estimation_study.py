### SUMMARY ###
"""
A simulation study comparing how DUREAL, UNREAL, and Random Forest models
estimate the true, population-level Shannon entropy from a known DGP.

This script supports a multi-threshold hyperparameter sweep with four modes:
1. run: Executes a single simulation for one seed and threshold.
2. aggregate: Aggregates raw metrics for a specific threshold.
3. plot: Generates boxplot/histogram comparisons for a specific threshold.
4. final_plot: Gathers all aggregated results and creates a summary plot
   showing performance across all tested thresholds.
"""

### LIBRARIES ###
import argparse
import pickle
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### IMPORTS ###
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.utils.models import RandomForestWrapper, TreeFarmsWrapper

### HELPER FUNCTIONS ###
def _calculate_vote_entropy(committee_preds: pd.DataFrame) -> np.ndarray:
    if committee_preds.empty or committee_preds.shape[1] == 0: return np.zeros(len(committee_preds))
    proportions = committee_preds.apply(lambda row: row.value_counts(normalize=True), axis=1).fillna(0)
    return -np.sum(proportions * np.log2(proportions + 1e-9), axis=1).values

def calculate_likelihood(X: pd.DataFrame, probs: list) -> np.ndarray:
    """Calculate P(X|class) for binary features assuming independence."""
    likelihood = np.ones(len(X))
    for i, col in enumerate(X.columns):
        likelihood *= np.where(X[col] == 1, probs[i], 1 - probs[i])
    return likelihood

def get_threshold_str(threshold: float) -> str:
    """Creates a filesystem-safe string from a float."""
    return str(threshold).replace('.', '_')

# ==============================================================================
# ### WORKFLOW MODES ###
# ==============================================================================

### A. RUN SINGLE SIMULATION ###
def run_dgp_simulation(args):
    """Runs one full simulation for a single seed and saves the metrics."""
    print(f"--- [Mode: RUN] Threshold: {args.tf_rashomon_threshold}, Seed: {args.seed} ---")

    # 1. Define a more complex Binary DGP and add label noise
    np.random.seed(args.seed)
    n_features = 6
    
    # Define the probabilities for the clean DGP
    probs_class0 = [0.8, 0.7, 0.3, 0.2, 0.5, 0.5]
    probs_class1 = [0.2, 0.3, 0.7, 0.8, 0.5, 0.5]

    # Generate the training data from the clean DGP
    X_train = pd.DataFrame(np.random.binomial(1, probs_class0, size=(args.n_train // 2, n_features)), columns=[f'f{i+1}' for i in range(n_features)])
    X_train = pd.concat([X_train, pd.DataFrame(np.random.binomial(1, probs_class1, size=(args.n_train // 2, n_features)), columns=[f'f{i+1}' for i in range(n_features)])])
    y_train = pd.Series([0] * (args.n_train // 2) + [1] * (args.n_train // 2))

    # Add noise by random flips 
    n_flips = int(args.noise_level * args.n_train)
    flip_indices = np.random.choice(y_train.index, size=n_flips, replace=False)
    y_train.loc[flip_indices] = 1 - y_train.loc[flip_indices]
    
    # Generate the candidate set (used for entropy calculation) from the clean DGP
    X_candidate = pd.DataFrame(np.random.binomial(1, probs_class0, size=(args.n_candidate // 2, n_features)), columns=[f'f{i+1}' for i in range(n_features)])
    X_candidate = pd.concat([X_candidate, pd.DataFrame(np.random.binomial(1, probs_class1, size=(args.n_candidate // 2, n_features)), columns=[f'f{i+1}' for i in range(n_features)])])
    
    # 2. Calculate "True" Population-Level Entropy
    likelihood0 = calculate_likelihood(X_candidate, probs_class0)
    likelihood1 = calculate_likelihood(X_candidate, probs_class1)
    p1 = likelihood1 / (likelihood0 + likelihood1 + 1e-9)
    p0 = 1 - p1
    true_entropy = - (p0 * np.log2(p0 + 1e-9) + p1 * np.log2(p1 + 1e-9))

    # 3. Train Models
    # Train the standard RF with bagging
    rf_bagging_model = RandomForestWrapper(
        n_estimators=args.rf_estimators, random_state=args.seed, bootstrap=True
    )
    rf_bagging_model.fit(X_train, y_train)

    # Train the modified RF without bagging
    rf_no_bagging_model = RandomForestWrapper(
        n_estimators=args.rf_estimators, random_state=args.seed, bootstrap=False
    )
    rf_no_bagging_model.fit(X_train, y_train)
    
    # Train TreeFarms
    tf_model = TreeFarmsWrapper(
        regularization=args.tf_regularization, rashomon_threshold=args.tf_rashomon_threshold
    )
    tf_model.fit(X_train, y_train)

    # 4. Calculate Entropy Estimates
    rf_bagging_preds = rf_bagging_model.get_raw_ensemble_predictions(X_candidate)
    rf_bagging_estimate = _calculate_vote_entropy(rf_bagging_preds)
    
    rf_no_bagging_preds = rf_no_bagging_model.get_raw_ensemble_predictions(X_candidate)
    rf_no_bagging_estimate = _calculate_vote_entropy(rf_no_bagging_preds)
    
    dureal_preds = tf_model.get_raw_ensemble_predictions(X_candidate)
    dureal_estimate = _calculate_vote_entropy(dureal_preds)
    
    unreal_preds = dureal_preds.T.drop_duplicates().T
    unreal_estimate = _calculate_vote_entropy(unreal_preds)

    # --- (Your diagnostic print statement is still fine here) ---
    dureal_committee_size = dureal_preds.shape[1]
    unreal_committee_size = unreal_preds.shape[1]
    print(
        f"[DIAGNOSTIC] Seed: {args.seed}, "
        f"Threshold: {args.tf_rashomon_threshold}, "
        f"DUREAL models: {dureal_committee_size}, "
        f"UNREAL models: {unreal_committee_size}"
    )

    # 5. Calculate Metrics
    results = {
        'seed': args.seed,
        
        # Metrics for RF with Bagging
        'rmse_rf_bagging': np.sqrt(np.mean((rf_bagging_estimate - true_entropy)**2)),
        'mae_rf_bagging': np.mean(np.abs(rf_bagging_estimate - true_entropy)),
        'n_models_rf_bagging': rf_bagging_preds.shape[1],

        # Metrics for RF without Bagging
        'rmse_rf_no_bagging': np.sqrt(np.mean((rf_no_bagging_estimate - true_entropy)**2)),
        'mae_rf_no_bagging': np.mean(np.abs(rf_no_bagging_estimate - true_entropy)),
        'n_models_rf_no_bagging': rf_no_bagging_preds.shape[1],

        # Metrics for DUREAL and UNREAL
        'rmse_dureal': np.sqrt(np.mean((dureal_estimate - true_entropy)**2)),
        'mae_dureal': np.mean(np.abs(dureal_estimate - true_entropy)),
        'n_models_dureal': dureal_committee_size,

        'rmse_unreal': np.sqrt(np.mean((unreal_estimate - true_entropy)**2)),
        'mae_unreal': np.mean(np.abs(unreal_estimate - true_entropy)),
        'n_models_unreal': unreal_committee_size,

        # Arrays 
        'arr_true_entropy': true_entropy,
        'arr_rf_bagging': rf_bagging_estimate,
        'arr_rf_no_bagging': rf_no_bagging_estimate,
        'arr_dureal': dureal_estimate,
        'arr_unreal': unreal_estimate,
    }
    
    # 6. Save Raw Results
    threshold_str = get_threshold_str(args.tf_rashomon_threshold)
    output_dir = PROJECT_ROOT / f"results/study3_entropy_validation/raw/thresh_{threshold_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"metrics_seed_{args.seed}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Metrics for seed {args.seed} saved to {output_path}")


    # 7. Generate Entropy Scatter Plot if requested
    if args.save_scatter_plot:
        estimates = {
            'RF (Bagging)': rf_bagging_estimate,
            'RF (No Bagging)': rf_no_bagging_estimate,
            'DUREAL': dureal_estimate,
            'UNREAL': unreal_estimate
        }
        mses = {
            'RF (Bagging)': np.mean((rf_bagging_estimate - true_entropy)**2),
            'RF (No Bagging)': np.mean((rf_no_bagging_estimate - true_entropy)**2),
            'DUREAL': np.mean((dureal_estimate - true_entropy)**2),
            'UNREAL': np.mean((unreal_estimate - true_entropy)**2),
        }
        
        output_dir_images = PROJECT_ROOT / f"results/study3_entropy_validation/images/scatter_plots/thresh_{threshold_str}"
        output_path_png = output_dir_images / f"scatter_seed_{args.seed}.png"
        
        plot_entropy_scatter(true_entropy, estimates, mses, output_path_png, args.seed, args.tf_rashomon_threshold)

### B. AGGREGATE RESULTS ###
def aggregate_results(args):
    """Aggregates raw metrics for a specific threshold into CSV files."""
    print(f"--- [Mode: AGGREGATE] Threshold: {args.tf_rashomon_threshold} ---")
    
    threshold_str = get_threshold_str(args.tf_rashomon_threshold)
    raw_results_dir = PROJECT_ROOT / f"results/study3_entropy_validation/raw/thresh_{threshold_str}"
    pkl_files = list(raw_results_dir.glob("metrics_seed_*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No raw result .pkl files found in {raw_results_dir}")

    all_metrics = [pickle.load(open(f, 'rb')) for f in pkl_files]
    df = pd.DataFrame(all_metrics)

    output_dir = PROJECT_ROOT / f"results/study3_entropy_validation/aggregated/thresh_{threshold_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / "aggregated_metrics.csv", index=False)
    print(f"Aggregated results from {len(pkl_files)} runs saved to {output_dir / 'aggregated_metrics.csv'}")

    summary_data = []
    models = ['rf_bagging', 'rf_no_bagging', 'dureal', 'unreal']
    model_names = ['RF (Bagging)', 'RF (No Bagging)', 'DUREAL', 'UNREAL']
    for key, name in zip(models, model_names):
        summary_data.append({
            'Model': name,
            'Mean Num Models': df[f'n_models_{key}'].mean(),
            'Mean RMSE': df[f'rmse_{key}'].mean(), 'Variance RMSE': df[f'rmse_{key}'].var(),
            'Mean MAE': df[f'mae_{key}'].mean(), 'Variance MAE': df[f'mae_{key}'].var()
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False, float_format='%.6f')
    print(f"Summary statistics table saved to {output_dir / 'summary_metrics.csv'}")

### C. PLOT AGGREGATED RESULTS ###
def plot_results(args):
    """Loads aggregated metrics for a specific threshold and creates comparison charts."""
    print(f"--- [Mode: PLOT] Threshold: {args.tf_rashomon_threshold} ---")
    
    threshold_str = get_threshold_str(args.tf_rashomon_threshold)
    data_path = PROJECT_ROOT / f"results/study3_entropy_validation/aggregated/thresh_{threshold_str}/aggregated_metrics.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Aggregated metrics not found at {data_path}. Please run 'aggregate' mode first.")
    df = pd.read_csv(data_path)

    model_labels = ['RF (Bagging)', 'RF (No Bagging)', 'DUREAL', 'UNREAL']
    model_keys = ['rf_bagging', 'rf_no_bagging', 'dureal', 'unreal']
    colors = ['purple', 'orange', 'blue', 'green']
    metrics = ['rmse', 'mae']
    
    # Boxplot
    fig_box, axes_box = plt.subplots(1, 2, figsize=(16, 7))
    for i, metric in enumerate(metrics):
        ax = axes_box[i]
        data_to_plot = [df[f'{metric}_{model}'] for model in model_keys]
        bplot = ax.boxplot(data_to_plot, labels=model_labels, patch_artist=True, vert=True)
        for patch, color in zip(bplot['boxes'], colors): patch.set_facecolor(color); patch.set_alpha(0.7)
        for median in bplot['medians']: median.set_color('black'); median.set_linewidth(1.5)
        ax.set_title(f"Distribution of {metric.upper()}", fontsize=14); ax.set_ylabel(metric.upper())
        ax.grid(True, axis='y', linestyle='--', alpha=0.6); ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    fig_box.suptitle(f"Aggregated Performance (Thresh={args.tf_rashomon_threshold}, {len(df)} Runs) - Boxplot", fontsize=16)
    plt.tight_layout()
    
    output_dir = PROJECT_ROOT / f"results/study3_entropy_validation/images/thresh_{threshold_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "comparison_boxplot.png", dpi=300)
    plt.close(fig_box)
    print(f"Boxplot saved to {output_dir / 'comparison_boxplot.png'}")

### D. PLOT FINAL SUMMARY ###
def plot_final_summary(args):
    """Gathers all summary metrics and plots performance across thresholds."""
    metric_to_plot = args.metric.upper()
    print(f"--- [Mode: FINAL_PLOT] Plotting Mean {metric_to_plot} ---")
    
    aggregated_dir = PROJECT_ROOT / "results/study3_entropy_validation/aggregated"
    summary_files = list(aggregated_dir.glob("thresh_*/summary_metrics.csv"))
    
    if not summary_files:
        raise FileNotFoundError(f"No summary metric files found in {aggregated_dir}.")
        
    all_summaries = []
    for f in summary_files:
        try:
            threshold = float(f.parent.name.replace('thresh_', '').replace('_', '.'))
            summary_df = pd.read_csv(f)
            summary_df['threshold'] = threshold
            all_summaries.append(summary_df)
        except (ValueError, IndexError):
            print(f"Warning: Could not parse threshold from {f.parent.name}. Skipping.")
            continue
    
    if not all_summaries:
        raise ValueError("Could not find or parse any valid summary files.")

    final_df = pd.concat(all_summaries).sort_values('threshold')

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'RF (Bagging)': 'purple', 
        'RF (No Bagging)': 'orange', 
        'DUREAL': 'blue', 
        'UNREAL': 'green'
    }

    column_name = f'Mean {metric_to_plot}'
    for model_name in final_df['Model'].unique():
        model_df = final_df[final_df['Model'] == model_name]
        ax.plot(model_df['threshold'], model_df[column_name], marker='o', linestyle='-', label=model_name, color=colors.get(model_name))
        
    ax.set_xlabel("Rashomon Threshold (epsilon)", fontsize=12)
    ax.set_ylabel(f"{column_name} (Lower is Better)", fontsize=12)
    ax.set_title(f"Model Performance ({metric_to_plot}) vs. Rashomon Threshold", fontsize=14, fontweight='bold')
    ax.legend(title="Model")
    ax.grid(True, which='both', linestyle='--', alpha=0.7)

    output_path = PROJECT_ROOT / f"results/study3_entropy_validation/images/final_summary_by_threshold_{args.metric}.png"    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Final summary plot saved to {output_path}")

### E. ###

def plot_entropy_scatter(true_entropy, estimates, mses, output_path, seed, threshold):
    """Generates a scatter plot of True vs. Estimated entropy for multiple models."""
    
    n_models = len(estimates)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5), sharex=True, sharey=True)
    fig.suptitle(f"Entropy Estimation Comparison (Thresh={threshold}, Seed={seed})", fontsize=16)

    model_colors = {
        'RF (Bagging)': 'purple', 
        'RF (No Bagging)': 'orange', 
        'DUREAL': 'blue', 
        'UNREAL': 'green'
    }

    for i, (model_name, est_entropy) in enumerate(estimates.items()):
        ax = axes[i]
        mse = mses[model_name]
        color = model_colors.get(model_name, 'gray')
        
        ax.scatter(true_entropy, est_entropy, alpha=0.4, color=color, label=f"MSE: {mse:.4f}")
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect Match (y=x)')
        
        ax.set_title(f"{model_name}")
        ax.set_xlabel("True Entropy")
        ax.set_ylabel("Estimated Entropy")
        ax.grid(True)
        ax.legend(loc='upper left')
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Entropy scatter plot saved to {output_path}")

### F. ###
def create_scatter_from_pkl(args):
    """Loads a single, rich .pkl file and generates the entropy scatter plot."""
    threshold_str = get_threshold_str(args.tf_rashomon_threshold)
    print(f"--- [Mode: SCATTER_PLOT] for Thresh={args.tf_rashomon_threshold}, Seed={args.seed} ---")

    # 1. Load the rich pkl file
    pkl_path = PROJECT_ROOT / f"results/study3_entropy_validation/raw/thresh_{threshold_str}/metrics_seed_{args.seed}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Result file not found: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    # 2. Extract the necessary arrays
    true_entropy = results['arr_true_entropy']
    estimates = {
        'RF (Bagging)': results['arr_rf_bagging'],
        'RF (No Bagging)': results['arr_rf_no_bagging'],
        'DUREAL': results['arr_dureal'],
        'UNREAL': results['arr_unreal']
    }
    mses = {
        'RF (Bagging)': results['rmse_rf_bagging']**2,
        'RF (No Bagging)': results['rmse_rf_no_bagging']**2,
        'DUREAL': results['rmse_dureal']**2,
        'UNREAL': results['rmse_unreal']**2,
    }

    # 3. Define output path and generate plot
    output_dir_images = PROJECT_ROOT / f"results/study3_entropy_validation/images/scatter_plots/thresh_{threshold_str}"
    output_path_png = output_dir_images / f"scatter_seed_{args.seed}.png"    
    plot_entropy_scatter(true_entropy, estimates, mses, output_path_png, args.seed, args.tf_rashomon_threshold)

# ==============================================================================
# ### MAIN EXECUTION ###
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGP Entropy Estimation Study")
    
    parser.add_argument('mode', type=str, choices=['run', 'aggregate', 'plot', 'final_plot', 'scatter_plot'],
                        help="Execution mode for the script")
                        
    parser.add_argument('--n_train', type=int, default=200)
    parser.add_argument('--n_candidate', type=int, default=1000)
    parser.add_argument('--rf_estimators', type=int, default=100)
    parser.add_argument('--tf_regularization', type=float, default=0.01)
    parser.add_argument('--tf_rashomon_threshold', type=float, required=False)
    parser.add_argument('--noise_level', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_scatter_plot', action='store_true',
                        help="If set, saves an entropy scatter plot for the 'run' mode.")
    parser.add_argument('--metric', type=str, choices=['rmse', 'mae'], default='rmse',
                        help="The metric to plot in 'final_plot' mode.")
    args = parser.parse_args()    
    if args.mode in ['run', 'aggregate', 'plot', 'scatter_plot'] and args.tf_rashomon_threshold is None:
        parser.error(f"--tf_rashomon_threshold is required for mode '{args.mode}'")
    
    if args.mode == 'run':
        run_dgp_simulation(args)
    elif args.mode == 'aggregate':
        aggregate_results(args)
    elif args.mode == 'plot':
        plot_results(args)
    elif args.mode == 'final_plot':
        plot_final_summary(args)
    elif args.mode == 'scatter_plot':
        create_scatter_from_pkl(args)