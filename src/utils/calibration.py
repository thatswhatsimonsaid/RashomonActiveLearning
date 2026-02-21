import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import average_precision_score
from typing import Dict, Any, Type, List
from sklearn.ensemble import RandomForestClassifier

def calibrate_hyperparameters(
    df_pilot: pd.DataFrame,
    model_class: Type,
    base_params: Dict[str, Any],
    depth_grid: List[int] = [3, 5],
    lambda_grid: List[float] = [0.01, 0.001],
    beta_grid: List[float] = [1.0, 5.0, 10.0, 25.0, 50.0], # Search for temperature
    min_rashomon_size: int = 2
) -> Dict[str, Any]:
    
    X = df_pilot.drop(columns="Y")
    y = df_pilot["Y"]
    loo = LeaveOneOut()

    # --- STAGE 1: NEUTRAL PREDICTOR TUNING (Same as before) ---
    best_acc, best_d, best_lambda = -1, 5, 0.001
    for depth in depth_grid:
        for reg in lambda_grid:
            accuracies = []
            for train_idx, val_idx in loo.split(X):
                temp_model = model_class(**{**base_params, "max_depth": depth, "regularization": reg})
                temp_model.fit(X.iloc[train_idx], y.iloc[train_idx])
                pred = temp_model.predict(X.iloc[val_idx])
                accuracies.append(1 if pred[0] == y.iloc[val_idx].values[0] else 0)
            avg_acc = np.mean(accuracies)
            if avg_acc > best_acc:
                best_acc, best_d, best_lambda = avg_acc, depth, reg

    # --- STAGE 2: RASHOMON EXPANSION (Additive Slack) ---
    # We still use the Adder to ensure we have a Version Space to work with
    current_epsilon_adder = 0.05 
    min_loss_proxy = 1.0 - best_acc
    
    while True:
        eff_multiplier = (min_loss_proxy + current_epsilon_adder) / max(min_loss_proxy, 1e-6)
        final_pilot_model = model_class(**{
            **base_params, "max_depth": best_d, 
            "regularization": best_lambda, "rashomon_multiplier": eff_multiplier
        })
        final_pilot_model.fit(X, y)
        
        # We only need at least 2 models to do weighting!
        if final_pilot_model.get_rashomon_size() >= min_rashomon_size or current_epsilon_adder >= 0.25:
            break
        current_epsilon_adder += 0.05

    # --- STAGE 3: BETA CALIBRATION (The "Sharpness" of Uncertainty) ---
    raw_preds_df = final_pilot_model.get_raw_ensemble_predictions(X)
    if hasattr(final_pilot_model, "model") and isinstance(final_pilot_model.model, RandomForestClassifier):
        losses = final_pilot_model.get_ensemble_losses(X, y)
    else:
        losses = final_pilot_model.get_ensemble_losses()
    errors = (final_pilot_model.predict(X) != y.values).astype(int)
    
    best_beta, best_auprc = beta_grid[0], -1

    if np.sum(errors) == 0:
        # If perfect accuracy, we want a 'democratic' committee (Low Beta)
        # to maximize discovery of alternative structures.
        best_beta = 5.0 
    else:
        for beta in beta_grid:
            # Weighted Entropy Calculation for this Beta
            adj_losses = losses - np.min(losses)
            w = np.exp(-beta * adj_losses)
            w /= np.sum(w)
            
            p = np.dot(raw_preds_df.values, w)
            p = np.clip(p, 1e-9, 1-1e-9)
            entropy = -(p * np.log(p) + (1-p) * np.log(1-p))
            
            # Score: How well does this entropy predict actual errors?
            score = average_precision_score(errors, entropy)
            if score > best_auprc:
                best_auprc, best_beta = score, beta

    return {
        "max_depth": best_d,
        "regularization": best_lambda,
        "rashomon_epsilon_adder": current_epsilon_adder,
        "beta": best_beta, # <--- The new key
        "pilot_accuracy": best_acc,
        "pilot_loss_proxy": min_loss_proxy
    }