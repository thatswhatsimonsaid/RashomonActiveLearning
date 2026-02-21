### Libraries ###
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import entropy, mode
from scipy import stats
import warnings
import re
import matplotlib.patheffects as pe
from treefarms import TREEFARMS

import statsmodels.api as sm


### Set up ###
warnings.filterwarnings("ignore", category=UserWarning)
plt.style.use('seaborn-v0_8-whitegrid')
BINARIZED_FEATURES = ['X1_bin_0.33', 'X1_bin_0.66', 'X2_bin_0.33', 'X2_bin_0.66']

# ==============================================================================
# --- DATA GENERATION ---
# ==============================================================================

def true_dgp(x_cont):
    """L-Shaped Decision Boundary as defined in Manuscript Section 3."""
    if x_cont[0] <= 0.5: return 0 
    elif x_cont[0] > 0.5 and x_cont[1] <= 0.5: return 0 
    else: return 1 

def create_data_pool(n_points: int, pool_seed: int) -> pd.DataFrame:
    n_noise_vars = 18
    np.random.seed(pool_seed)
    # Signal features
    X_signal = np.random.rand(n_points, 2)
    # Noise features (The "Haystack")
    X_noise = np.random.rand(n_points, n_noise_vars)
    
    X_total = np.hstack([X_signal, X_noise])
    cols = ['X1_cont', 'X2_cont'] + [f'N{i}_cont' for i in range(n_noise_vars)]
    
    df_pool = pd.DataFrame(X_total, columns=cols)
    
    # Target based ONLY on signal
    df_pool['label'] = [true_dgp(x) for x in X_signal]
    
    # Binarize signal (as before)
    for val in [0.33, 0.66]:
        df_pool[f'X1_bin_{val}'] = (df_pool['X1_cont'] > val).astype(int)
        df_pool[f'X2_bin_{val}'] = (df_pool['X2_cont'] > val).astype(int)
        
    # Binarize noise (to give RF more ways to get "lost")
    for i in range(min(n_noise_vars, 5)): # Just binarize a few noise vars to keep it complex
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
# --- MODEL UTILS ---
# ==============================================================================

def predict_tree(tree, X_binarized: pd.DataFrame):
    """Universal predict interface for SKLearn and TreeFarms objects."""
    if isinstance(tree, DecisionTreeClassifier):
        return tree.predict(X_binarized)
    return np.array([tree.classify(row)[0] for row in X_binarized.values])

class MockModelWrapper:
    """Standardizes committee predictions into a DataFrame."""
    def __init__(self, committee_models):
        self.committee = committee_models

    def get_raw_ensemble_predictions(self, X_data_binarized: pd.DataFrame) -> pd.DataFrame:
        preds = {f"Tree_{i}": predict_tree(t, X_data_binarized) for i, t in enumerate(self.committee)}
        return pd.DataFrame(preds, index=X_data_binarized.index)

# ==============================================================================
# --- COMMITTEE FACTORIES ---
# ==============================================================================

def get_rf_committee(X_train_bin, y_train, n_committee, model_seed) -> list:
    rf = RandomForestClassifier(n_estimators=n_committee, max_depth=3, random_state=model_seed)
    rf.fit(X_train_bin, y_train)
    return list(rf.estimators_)

def get_rashomon_committee(X_train_bin, y_train, n_committee, model_seed, reg, thresh) -> list:
    """Trains TreeFarms committee with Accuracy Ordering."""
    config = {
        "depth_budget": 3,
        "rashomon_ignore_trivial_extensions": True,
        "regularization": reg,
        "rashomon_bound_adder": thresh,
        "verbose": False
    }
    tf = TREEFARMS(config)
    tf.fit(X_train_bin, y_train)
    all_trees = [tf[i] for i in range(tf.get_tree_count())]
    if not all_trees: return []

    accuracies = [np.mean(predict_tree(t, X_train_bin) == y_train.values) for t in all_trees]
    sorted_indices = np.argsort(accuracies)[::-1]
    best_acc = accuracies[sorted_indices[0]]    
    scoped_trees = [all_trees[i] for i in sorted_indices if (best_acc - accuracies[i]) <= thresh]
    return scoped_trees[:n_committee]

# ==============================================================================
# --- ACTIVE LEARNING LOGIC ---
# ==============================================================================

def get_qbc_selection(committee_models, df_train, df_candidate, use_unique_trees, model_seed=0):
    """Calculates Vote Entropy using Full-Pool Uniqueness Filter."""
    if not committee_models:
        return df_candidate.sample(1, random_state=model_seed).index[0], pd.Series(0.0, index=df_candidate.index)

    mock = MockModelWrapper(committee_models)
    X_train_bin = df_train[BINARIZED_FEATURES]
    X_cand_bin = df_candidate[BINARIZED_FEATURES]
    cand_votes = mock.get_raw_ensemble_predictions(X_cand_bin)
    
    if use_unique_trees:
        train_votes = mock.get_raw_ensemble_predictions(X_train_bin)
        all_votes = pd.concat([train_votes, cand_votes])
        unique_cols = all_votes.T.drop_duplicates().index
        cand_votes = cand_votes.loc[:, unique_cols]
        
    if cand_votes.empty:
        return df_candidate.sample(1, random_state=model_seed).index[0], pd.Series(0.0, index=df_candidate.index)

    entropies = cand_votes.apply(lambda row: entropy(row.value_counts(normalize=True), base=2), axis=1)
    return entropies.idxmax(), entropies

def get_entropies_for_grid(committee, df_train, use_unique_trees, grid_points_bin):
    """Grid evaluator for Continuous Heatmap (Study 3)."""
    _, entropies = get_qbc_selection(committee, df_train, grid_points_bin, use_unique_trees)
    return entropies

# ==============================================================================
# --- STATS ---
# ==============================================================================

def calculate_n_eff(diff_array: np.ndarray) -> int:
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
    
    # Encode Method: 0 for RF, 1 for UNREAL
    df_long['is_unreal'] = (df_long['method'] == 've_unreal').astype(int)
    df_long['abs_dist'] = df_long['dist_to_boundary'].abs()
    
    # Interaction term: is_unreal * abs_dist
    df_long['interaction'] = df_long['is_unreal'] * df_long['abs_dist']
    
    # Regression: Entropy = B0 + B1(Method) + B2(Dist) + B3(Interaction)
    X = sm.add_constant(df_long[['is_unreal', 'abs_dist', 'interaction']])
    model = sm.OLS(df_long['entropy'], X).fit()
    
    # B3 (interaction) is the test for "Pattern"
    beta_3 = model.params['interaction']
    p_beta_3 = model.pvalues['interaction']

    return {
        "BCI_RF": rf_snr,
        "BCI_UNREAL": unreal_snr,
        "BCI_Improvement": (unreal_snr / rf_snr) - 1,
        "Interaction_Beta": beta_3,
        "Interaction_Pval": p_beta_3
    }