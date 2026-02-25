### Libraries ###
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import entropy
import warnings
import re
import matplotlib.patheffects as pe
from treefarms import TREEFARMS
import statsmodels.api as sm
from src.utils.models import BMARandomForestWrapper, PySORTDWrapper, RandomForestWrapper
from src.utils.query_strategies import QBCSelector

### Set up ###
N_NOISE_FOR_EXPERIMENT= 5
warnings.filterwarnings("ignore", category=UserWarning)
plt.style.use('seaborn-v0_8-whitegrid')
BINARIZED_FEATURES = ['X1_bin_0.33', 'X1_bin_0.66', 'X2_bin_0.33', 'X2_bin_0.66']+ [f'N{i}_bin_0.5' for i in range(N_NOISE_FOR_EXPERIMENT)]

# ==============================================================================
# --- DATA GENERATION ---
# ==============================================================================

def true_dgp(x_cont):
    """L-Shaped Decision Boundary as defined in Manuscript Section 3."""
    if x_cont[0] <= 0.5: return 0 
    elif x_cont[0] > 0.5 and x_cont[1] <= 0.5: return 0 
    else: return 1 

def create_data_pool(n_points: int, pool_seed: int) -> pd.DataFrame:
    """Generates the synthetic data pool with signal and 'Haystack' noise."""
    n_noise_vars = 18
    np.random.seed(pool_seed)
    
    X_signal = np.random.rand(n_points, 2)
    X_noise = np.random.rand(n_points, n_noise_vars)
    
    X_total = np.hstack([X_signal, X_noise])
    cols = ['X1_cont', 'X2_cont'] + [f'N{i}_cont' for i in range(n_noise_vars)]
    
    df_pool = pd.DataFrame(X_total, columns=cols)
    df_pool['label'] = [true_dgp(x) for x in X_signal]
    
    # Binarize signal features
    for val in [0.33, 0.66]:
        df_pool[f'X1_bin_{val}'] = (df_pool['X1_cont'] > val).astype(int)
        df_pool[f'X2_bin_{val}'] = (df_pool['X2_cont'] > val).astype(int)
        
    # Binarize a subset of noise features
    for i in range(min(n_noise_vars, N_NOISE_FOR_EXPERIMENT)):
        df_pool[f'N{i}_bin_0.5'] = (df_pool[f'N{i}_cont'] > 0.5).astype(int)

    return df_pool

def distance_to_boundary(x_cont, metric='axis'):
    """Calculates signed distance to L-shaped boundary. Negative = Interior."""
    x1, x2 = x_cont[0], x_cont[1]
    abs_dist_x, abs_dist_y = abs(x1 - 0.5), abs(x2 - 0.5)
    magnitude = np.sqrt(abs_dist_x**2 + abs_dist_y**2) if metric == 'euclidean' else min(abs_dist_x, abs_dist_y)
    sign = -1 if (x1 - 0.5 >= 0 and x2 - 0.5 >= 0) else 1
    return sign * magnitude

# ==============================================================================
# --- PRODUCTION-ALIGNED QUERY LOGIC ---
# ==============================================================================

def get_weighted_entropies(model_wrapper, df_train, df_candidate, beta=50.0):
    selector = QBCSelector(beta=beta)
    
    # 1. Subset Candidate Grid (Already handled in task, but good safety)
    cand_copy = df_candidate[BINARIZED_FEATURES].copy()
    cand_copy['Y'] = 0

    # 2. Subset Training Data to match the 4 features models expect
    train_copy = df_train[BINARIZED_FEATURES].copy()
    train_copy['Y'] = df_train['label']

    selection_output = selector.select(
        model=model_wrapper,
        df_train=train_copy, 
        df_candidate=cand_copy
    )
    
    return selection_output["AllEntropies"]

# ==============================================================================
# --- MODEL UTILS (LEGACY SUPPORT) ---
# ==============================================================================

def predict_tree(tree, X_binarized: pd.DataFrame):
    """Universal predict interface for SKLearn and TreeFarms objects."""
    if isinstance(tree, DecisionTreeClassifier):
        return tree.predict(X_binarized)
    return np.array([tree.classify(row)[0] for row in X_binarized.values])

# ==============================================================================
# --- STATS & TOPOLOGICAL ANALYSIS ---
# ==============================================================================

def calculate_n_eff(diff_array: np.ndarray) -> int:
    """Calculates effective sample size for correlated results."""
    diff = diff_array.flatten()
    if np.std(diff) == 0 or len(diff) <= 1: return 1
    corr = np.corrcoef(diff[:-1], diff[1:])[0, 1]
    rho = np.clip(corr, -0.99, 0.99) if not np.isnan(corr) else 0
    return max(1, int(len(diff) * (1 - rho) / (1 + rho)))

def run_localized_analysis(df_grid_results, near_thresh=0.1, far_thresh=0.2):
    """
    Formally tests for topological 'sharpness' using BCI and Interaction Modeling.
    """
    # Identify Zones
    near_mask = df_grid_results['dist_to_boundary'].abs() <= near_thresh
    far_mask = df_grid_results['dist_to_boundary'].abs() >= far_thresh
    
    # 1. Calculate BCI (Boundary Concentration Index)
    rf_snr = df_grid_results.loc[near_mask, 've_rf'].mean() / df_grid_results.loc[far_mask, 've_rf'].mean()
    unreal_snr = df_grid_results.loc[near_mask, 've_unreal'].mean() / df_grid_results.loc[far_mask, 've_unreal'].mean()
    
    # 2. Formal Interaction Test (The 'Pattern' Test)
    df_long = pd.melt(df_grid_results, id_vars=['dist_to_boundary'], 
                      value_vars=['ve_rf', 've_unreal'], 
                      var_name='method', value_name='entropy')
    
    df_long['is_unreal'] = (df_long['method'] == 've_unreal').astype(int)
    df_long['abs_dist'] = df_long['dist_to_boundary'].abs()
    df_long['interaction'] = df_long['is_unreal'] * df_long['abs_dist']
    
    # Regression: Entropy = B0 + B1(Method) + B2(Dist) + B3(Interaction)
    X = sm.add_constant(df_long[['is_unreal', 'abs_dist', 'interaction']])
    model = sm.OLS(df_long['entropy'], X).fit()
    
    beta_3 = model.params['interaction']
    p_beta_3 = model.pvalues['interaction']

    return {
        "BCI_RF": rf_snr,
        "BCI_UNREAL": unreal_snr,
        "BCI_Improvement": (unreal_snr / rf_snr) - 1,
        "Interaction_Beta": beta_3,
        "Interaction_Pval": p_beta_3
    }