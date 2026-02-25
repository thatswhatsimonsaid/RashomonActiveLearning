### LIBRARIES ###
import argparse
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### PATH SETUP ###
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

### IMPORTS ###
from src.utils.models import BMARandomForestWrapper, PySORTDWrapper, RandomForestWrapper
from src.utils.query_strategies import QBCSelector

# Shared helper for Shannon Entropy
def shannon_entropy(p_class1):
    p1 = np.clip(p_class1, 1e-9, 1 - 1e-9)
    p0 = 1 - p1
    return -(p0 * np.log2(p0) + p1 * np.log2(p1))


def calculate_theoretical_entropy(X, p_c0, p_c1):
    """Calculates the ground-truth Shannon entropy of the DGP."""
    def get_lik(X_df, p_vec):
        lik = np.ones(len(X_df))
        for i, col in enumerate(X_df.columns):
            lik *= np.where(X_df[col] == 1, p_vec[i], 1 - p_vec[i])
        return lik

    l0 = get_lik(X, p_c0)
    l1 = get_lik(X, p_c1)
    prob_c1 = l1 / (l0 + l1 + 1e-9)
    return shannon_entropy(prob_c1)


def run_validation_sim(args):
    print(f"--- [RUN] Seed: {args.seed} | Beta: {args.beta} ---")
    np.random.seed(args.seed)
    
    # 1. Generate Binary DGP (6 Features)
    p_c0 = [0.8, 0.7, 0.3, 0.2, 0.5, 0.5]
    p_c1 = [0.2, 0.3, 0.7, 0.8, 0.5, 0.5]
    cols = [f'f{i}' for i in range(6)]

    def gen_data(n):
        X = pd.concat([
            pd.DataFrame(np.random.binomial(1, p_c0, size=(n//2, 6)), columns=cols),
            pd.DataFrame(np.random.binomial(1, p_c1, size=(n//2, 6)), columns=cols)
        ]).reset_index(drop=True)
        y = pd.Series([0]*(n//2) + [1]*(n//2))
        return X, y

    X_train, y_train = gen_data(args.n_train)
    X_cand, _ = gen_data(args.n_candidate)
    
    # Add Label Noise
    flips = np.random.choice(y_train.index, int(args.noise * args.n_train), replace=False)
    y_train.loc[flips] = 1 - y_train.loc[flips]

    # 2. Ground Truth Entropy
    true_entropy = calculate_theoretical_entropy(X_cand, p_c0, p_c1)

    # 3. Fit Production Wrappers
    models = {
        "QBC-RF": RandomForestWrapper(n_estimators=args.n_committee, max_depth=args.max_depth),
        "BMA-RF": BMARandomForestWrapper(n_estimators=args.n_committee, max_depth=args.max_depth),
        "UNREAL": PySORTDWrapper(max_depth=args.max_depth, regularization=1e-9, 
                                 rashomon_multiplier=args.multiplier)
    }

    estimates = {}
    selector = QBCSelector(beta=args.beta)

    for name, m in models.items():
        m.fit(X_train, y_train)
        b = 0.0 if name == "QBC-RF" else args.beta
        res = selector.select(m, pd.concat([X_train, y_train.rename('Y')], axis=1), 
                              pd.concat([X_cand, pd.Series(0, index=X_cand.index, name='Y')], axis=1))
        estimates[name] = res["AllEntropies"].values

    # 4. Metrics & Export
    results = {'seed': args.seed, 'true_h': true_entropy}
    for name, est in estimates.items():
        results[f'rmse_{name}'] = np.sqrt(np.mean((est - true_entropy)**2))
        results[f'mae_{name}'] = np.mean(np.abs(est - true_entropy))
        results[f'arr_{name}'] = est

    out_dir = PROJECT_ROOT / "results/study3_entropy_validation/raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"seed_{args.seed}.pkl", 'wb') as f:
        pickle.dump(results, f)


def plot_validation_results():
    """Aggregates seeds and plots the True vs Estimated Scatter."""
    raw_dir = PROJECT_ROOT / "results/study3_entropy_validation/raw"
    files = list(raw_dir.glob("seed_*.pkl"))
    if not files: return

    # Average over seeds
    all_res = [pickle.load(open(f, 'rb')) for f in files]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    methods = ["QBC-RF", "BMA-RF", "UNREAL"]
    colors = ["darkred", "forestgreen", "darkblue"]

    true_h = all_res[0]['true_h']
    
    for i, (m, col) in enumerate(zip(methods, colors)):
        # Mean estimate across seeds
        avg_est = np.mean([r[f'arr_{m}'] for r in all_res], axis=0)
        rmse = np.mean([r[f'rmse_{m}'] for r in all_res])
        
        axes[i].scatter(true_h, avg_est, alpha=0.3, color=col, s=10)
        axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[i].set_title(f"{m}\nAvg RMSE: {rmse:.4f}", fontweight='bold')
        axes[i].set_xlabel("Theoretical Entropy")
        axes[i].grid(True, alpha=0.3)

    axes[0].set_ylabel("Estimated Vote Entropy")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "results/study3_entropy_validation/validation_scatter.png", dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['run', 'plot'], default='run')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--beta', type=float, default=200.0)
    parser.add_argument('--multiplier', type=float, default=500.0)
    parser.add_argument('--n_train', type=int, default=20) 
    parser.add_argument('--n_candidate', type=int, default=500)
    parser.add_argument('--n_committee', type=int, default=10)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--noise', type=float, default=0.1)
    args = parser.parse_args()

    if args.mode == 'run': run_validation_sim(args)
    else: plot_validation_results()