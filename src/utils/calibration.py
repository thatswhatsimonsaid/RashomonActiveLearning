### Libraries ###
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import average_precision_score
from typing import Dict, Any, Type, List
from sklearn.ensemble import RandomForestClassifier

### Calibrate Function ###
def calibrate_hyperparameters(
    df_pilot: pd.DataFrame,
    model_class: Type,
    base_params: Dict[str, Any],
    depth_grid: List[int] = [3, 5],
    lambda_grid: List[float] = [0.01, 0.001],
    beta_grid: List[float] = [1.0, 10.0, 50.0, 100.0, 250.0, 500.0],
    min_rashomon_size: int = 2
) -> Dict[str, Any]:
    
    X = df_pilot.drop(columns="Y")
    y = df_pilot["Y"]
    loo = LeaveOneOut()

    ### STAGE 1: NEUTRAL PREDICTOR TUNING ###
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

    ### STAGE 2: RASHOMON EXPANSION ###
    current_epsilon_adder = 0.05 
    min_loss_proxy = 1.0 - best_acc
    
    while True:
        eff_multiplier = (min_loss_proxy + current_epsilon_adder) / max(min_loss_proxy, 1e-6)
        final_pilot_model = model_class(**{
            **base_params, "max_depth": best_d, 
            "regularization": best_lambda, "rashomon_multiplier": eff_multiplier
        })
        final_pilot_model.fit(X, y)        
        if final_pilot_model.get_rashomon_size() >= min_rashomon_size or current_epsilon_adder >= 0.25:
            break
        current_epsilon_adder += 0.05

    ### STAGE 3: BETA CALIBRATION ###
    raw_preds_df = final_pilot_model.get_raw_ensemble_predictions(X)
    
    if hasattr(final_pilot_model, "get_ensemble_losses"):
        losses = final_pilot_model.get_ensemble_losses(X, y)
    else:
        losses = np.zeros(final_pilot_model.get_rashomon_size())

    errors = (final_pilot_model.predict(X) != y.values).astype(int)
    best_beta, best_auprc = beta_grid[0], -1

    if np.sum(errors) == 0:
        best_beta = 25.0 
    else:
        for beta in beta_grid:
            adj_losses = losses - np.min(losses)
            w = np.exp(-beta * adj_losses)
            w /= np.sum(w)            
            p = np.dot(raw_preds_df.values, w)
            p = np.clip(p, 1e-9, 1-1e-9)            
            entropy = -(p * np.log(p) + (1-p) * np.log(1-p))             
            score = average_precision_score(errors, entropy)
            if score > best_auprc:
                best_auprc, best_beta = score, beta

    ### Return ###
    return {
        "max_depth": best_d,
        "regularization": best_lambda,
        "rashomon_epsilon_adder": current_epsilon_adder,
        "beta": best_beta, 
        "pilot_accuracy": best_acc,
        "pilot_loss_proxy": min_loss_proxy
    }