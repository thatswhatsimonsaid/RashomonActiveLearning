import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Helper to predict on the 2D PCA grid
def get_decision_boundary_grid(pca_model, clf_model, xx, yy):
    """
    Projects 2D grid points back to high-dim space and predicts class.
    """
    # Flatten grid to (N, 2)
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
    
    # Inverse transform to High-Dim space
    grid_points_high_dim = pca_model.inverse_transform(grid_points_2d)
    
    # Predict
    if hasattr(clf_model, "predict"):
        Z = clf_model.predict(grid_points_high_dim)
    elif hasattr(clf_model, "classify"): 
        Z = np.array([clf_model.classify(row)[0] for row in grid_points_high_dim])
    else:
        Z = np.zeros(grid_points_high_dim.shape[0]) # Fallback
        
    return Z.reshape(xx.shape)

def plot_pca_evolution(result_file_path: Path, data_file_path: Path):
    # 1. Load Results
    try:
        with open(result_file_path, 'rb') as f:
            results = pickle.load(f)
    except Exception as e:
        print(f"Error loading result file: {e}")
        return

    if not hasattr(results, 'entropy_history') or not results.entropy_history:
        print("Error: No entropy history found.")
        return

    # 2. Load Data
    try:
        with open(data_file_path, 'rb') as f:
            df_pool = pickle.load(f)
    except Exception as e:
        print(f"Error loading data file: {e}")
        return

    # 3. Setup PCA (Compute ONCE)
    print("Computing PCA projection...")
    feature_cols = df_pool.select_dtypes(include=np.number).columns.tolist()
    if 'Y' in feature_cols: feature_cols.remove('Y')
    
    X_high_dim = df_pool[feature_cols].values
    
    # Use fixed seed for consistent projection across runs
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_high_dim)
    
    df_pool['PC1'] = X_pca[:, 0]
    df_pool['PC2'] = X_pca[:, 1]
    
    # 4. Create Grid for Decision Boundary Background
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    # 5. Output Dir
    dataset_dir = result_file_path.parent.parent
    output_dir = dataset_dir / "entropy_plots_pca" / result_file_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating PCA plots in: {output_dir}")

    first_candidates = list(results.entropy_history[0].keys())
    currently_labeled = df_pool.index.difference(first_candidates).tolist()

    # 6. Iterate
    for iter_idx, entropy_dict in enumerate(tqdm(results.entropy_history, desc="Plotting Frames")):
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # --- A. Plot Decision Boundary (Background Contour) ---
        if hasattr(results, 'best_tree_history') and iter_idx < len(results.best_tree_history):
            current_tree = results.best_tree_history[iter_idx]
            if current_tree is not None:
                try:
                    Z = get_decision_boundary_grid(pca, current_tree, xx, yy)
                    # Use alpha=0.15 for very subtle background regions
                    ax.contourf(xx, yy, Z, alpha=0.15, cmap='coolwarm', levels=[-0.5, 0.5, 1.5], zorder=0)
                except Exception:
                    pass

        # --- B. Plot UNLABELED/Candidate Background (Light Grey dots) ---
        ax.scatter(
            df_pool['PC1'], df_pool['PC2'],
            c='lightgrey', s=20, alpha=0.4, zorder=1, label='Unlabeled'
        )
        
        # --- C. Plot CANDIDATES (Colored by Entropy Heatmap) ---
        if entropy_dict:
            cand_indices = list(entropy_dict.keys())
            cand_entropies = list(entropy_dict.values())
            cand_df = df_pool.loc[cand_indices]
            
            sc = ax.scatter(
                cand_df['PC1'], cand_df['PC2'],
                c=cand_entropies, cmap='coolwarm', vmin=0, vmax=1.0,
                s=60, edgecolors='grey', linewidth=0.5, alpha=1.0, zorder=2
            )
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Vote Entropy", rotation=270, labelpad=15)
            
        # --- D. Plot LABELED (Dark Green with transparency) ---
        if currently_labeled:
            labeled_df = df_pool.loc[currently_labeled]
            ax.scatter(
                labeled_df['PC1'], labeled_df['PC2'],
                c='#006400',       # Dark Green
                marker='X',        # Big X marker
                s=50,              # Reasonable size
                linewidth=1.5,     # Thicker lines
                alpha=0.3,         # <--- CHANGE: Added transparency to reduce dominance
                zorder=3,          # On top of everything
                label='Labeled'
            )

        # --- E. Format ---
        ax.set_title(f"Iteration {iter_idx}: Entropy & Boundary (PCA)", fontsize=14)
        var1 = pca.explained_variance_ratio_[0]
        var2 = pca.explained_variance_ratio_[1]
        ax.set_xlabel(f"PC1 ({var1:.1%} Var)")
        ax.set_ylabel(f"PC2 ({var2:.1%} Var)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Save
        output_path = output_dir / f"iter_{iter_idx:04d}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Update Labeled for next frame
        if hasattr(results, 'selection_history') and iter_idx < len(results.selection_history):
            selected_id = results.selection_history[iter_idx]
            if selected_id is not None:
                currently_labeled.append(selected_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", type=Path)
    parser.add_argument("data_file", type=Path)
    args = parser.parse_args()

    if args.result_file.exists() and args.data_file.exists():
        plot_pca_evolution(args.result_file, args.data_file)
    else:
        print("Files not found.")