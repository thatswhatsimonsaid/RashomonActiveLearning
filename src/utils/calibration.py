### Libraries ###
import inspect
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import average_precision_score
from typing import Dict, Any, Type, List
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier

### Calibrate Function ###
def calibrate_hyperparameters(
    df_pilot: pd.DataFrame,
    model_class: Type,
    base_params: Dict[str, Any],
    depth_grid: List[int] = [3, 5],
    lambda_grid: List[float] = [0.01, 0.001],
    beta_grid: List[float] = [1.0, 10.0, 50.0, 100.0, 250.0, 500.0],
    min_rashomon_size: int = 2,
    task: str = "classification",
) -> Dict[str, Any]:
    if task == "regression":
        return _calibrate_regression(
            df_pilot=df_pilot,
            model_class=model_class,
            base_params=base_params,
            depth_grid=depth_grid,
            lambda_grid=lambda_grid,
            beta_grid=beta_grid,
            min_rashomon_size=min_rashomon_size,
        )

    X = df_pilot.drop(columns="Y")
    y = df_pilot["Y"]
    loo = LeaveOneOut()

    ### STAGE 1: NEUTRAL PREDICTOR TUNING ###
    sig = inspect.signature(model_class.__init__)
    has_reg = "regularization" in sig.parameters
    best_acc, best_d, best_lambda = -1, 5, 0.001    
    effective_lambda_grid = lambda_grid if has_reg else [None]

    for depth in depth_grid:
        for reg in effective_lambda_grid:
            accuracies = []
            for train_idx, val_idx in loo.split(X):
                run_params = {**base_params, "max_depth": depth}
                if has_reg:
                    run_params["regularization"] = reg
                
                temp_model = model_class(**run_params)
                temp_model.fit(X.iloc[train_idx], y.iloc[train_idx])
                pred = temp_model.predict(X.iloc[val_idx])
                accuracies.append(1 if pred[0] == y.iloc[val_idx].values[0] else 0)
            
            avg_acc = np.mean(accuracies)
            if avg_acc > best_acc:
                best_acc, best_d, best_lambda = avg_acc, depth, reg

    ### STAGE 2: RASHOMON EXPANSION ###
    has_rashomon = "rashomon_multiplier" in sig.parameters
    fixed_epsilon = base_params.get("rashomon_threshold") 
    
    current_epsilon_adder = 0.0
    min_loss_proxy = 1.0 - best_acc

    if not has_rashomon:
        final_pilot_model = model_class(**{**base_params, "max_depth": best_d})
        final_pilot_model.fit(X, y)
    elif fixed_epsilon is not None:
        # 1. USE FIXED EPSILON MODE
        current_epsilon_adder = float(fixed_epsilon)
        eff_multiplier = (min_loss_proxy + current_epsilon_adder) / max(min_loss_proxy, 1e-6)
        final_pilot_model = model_class(**{
            **base_params, "max_depth": best_d, 
            "regularization": best_lambda, "rashomon_multiplier": eff_multiplier
        })
        final_pilot_model.fit(X, y)
        print(f"--- Calibration: Using FIXED epsilon {current_epsilon_adder} ---")
    else:
        # 2. AUTO-CALIBRATION MODE (Original logic)
        current_epsilon_adder = 0.05 
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
    requested_beta = base_params.get("beta", 0.0)
    if requested_beta != "calibrated":
        # FAST TRACK: UNREAL without BMA
        best_beta = float(requested_beta)
        print(f"--- Fast-Tracking: Using fixed beta={best_beta} ---")
    else:
        # FULL CALIBRATION: UNREAL with BMA
        print(f"--- CALIBRATING ---")
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


def _fit_pilot(model_class, base_params, depth, reg, rashomon_multiplier, X, y):
    run_params = {**base_params, "max_depth": depth}
    if reg is not None:
        run_params["regularization"] = reg
    if rashomon_multiplier is None:
        run_params.pop("rashomon_multiplier", None)
    else:
        run_params["rashomon_multiplier"] = rashomon_multiplier
    model = model_class(**run_params)
    model.fit(X, y)
    return model


def _calibrate_regression(
    df_pilot: pd.DataFrame,
    model_class: Type,
    base_params: Dict[str, Any],
    depth_grid: List[int],
    lambda_grid: List[float],
    beta_grid: List[float],
    min_rashomon_size: int,
) -> Dict[str, Any]:
    """
    Same three stages as classification, on squared error.

    Stage 1 minimizes leave-one-out MSE.
    Stage 2 sets the Rashomon excess ε. SORTD keeps trees whose objective is
    at most (1 + ε) times the optimal objective, so ε is passed through directly.
    Stage 3 picks β so that committee variance ranks in-sample squared error.
    """
    X = df_pilot.drop(columns="Y")
    y = df_pilot["Y"]
    loo = LeaveOneOut()
    sig = inspect.signature(model_class.__init__)
    has_reg = "regularization" in sig.parameters
    has_rashomon = "rashomon_multiplier" in sig.parameters
    effective_lambda_grid = lambda_grid if has_reg else [None]

    best_mse, best_d, best_lambda = np.inf, depth_grid[0], effective_lambda_grid[0]
    # Stage 1 scores the optimal tree. ε = 0 keeps the solver from enumerating a set.
    stage1_multiplier = 0.0 if has_rashomon else None
    for depth in depth_grid:
        for reg in effective_lambda_grid:
            squared_errors = []
            for train_idx, val_idx in loo.split(X):
                temp_model = _fit_pilot(
                    model_class, base_params, depth, reg, stage1_multiplier,
                    X.iloc[train_idx], y.iloc[train_idx],
                )
                pred = np.asarray(temp_model.predict(X.iloc[val_idx]), dtype=float).reshape(-1)
                squared_errors.append((pred[0] - float(y.iloc[val_idx].values[0])) ** 2)
            avg_mse = float(np.mean(squared_errors))
            if avg_mse < best_mse:
                best_mse, best_d, best_lambda = avg_mse, depth, reg

    fixed_epsilon = base_params.get("rashomon_threshold")
    current_epsilon = 0.0
    if not has_rashomon:
        final_pilot_model = _fit_pilot(model_class, base_params, best_d, best_lambda, None, X, y)
    elif fixed_epsilon is not None:
        current_epsilon = float(fixed_epsilon)
        final_pilot_model = _fit_pilot(model_class, base_params, best_d, best_lambda, current_epsilon, X, y)
        print(f"--- Calibration: Using FIXED regression epsilon {current_epsilon} ---")
    else:
        current_epsilon = 0.05
        while True:
            final_pilot_model = _fit_pilot(model_class, base_params, best_d, best_lambda, current_epsilon, X, y)
            if final_pilot_model.get_rashomon_size() >= min_rashomon_size or current_epsilon >= 0.25:
                break
            current_epsilon += 0.05

    in_sample = np.asarray(final_pilot_model.predict(X), dtype=float).reshape(-1)
    pilot_mse = float(np.mean((in_sample - y.to_numpy(dtype=float)) ** 2))

    requested_beta = base_params.get("beta", 0.0)
    if requested_beta != "calibrated":
        best_beta = float(requested_beta)
        print(f"--- Fast-Tracking: Using fixed beta={best_beta} ---")
    else:
        print("--- CALIBRATING regression beta ---")
        # Squared-error gaps can be much smaller than classification error rates.
        search_betas = []
        for beta in (1e-4, 1e-3, 1e-2, 0.1, *beta_grid):
            if beta not in search_betas:
                search_betas.append(beta)
        raw_preds_df = final_pilot_model.get_raw_ensemble_predictions(X)
        if raw_preds_df is None:
            best_beta = 0.0
            raw_preds_df = pd.DataFrame(index=X.index)
        if hasattr(final_pilot_model, "get_ensemble_losses") and raw_preds_df.shape[1] > 0:
            losses = final_pilot_model.get_ensemble_losses(X, y)
        else:
            losses = np.zeros(raw_preds_df.shape[1])
        point_error = (in_sample - y.to_numpy(dtype=float)) ** 2
        best_beta, best_rank = search_betas[0], -np.inf
        if raw_preds_df.shape[1] < 2 or np.std(point_error) < 1e-12:
            best_beta = 0.0
        else:
            for beta in search_betas:
                adj_losses = losses - np.min(losses)
                weights = np.exp(-beta * adj_losses)
                weights = weights / np.sum(weights)
                variance = _prediction_variance(raw_preds_df.values.astype(float), weights)
                rank = spearmanr(point_error, variance).correlation
                if rank is not None and np.isfinite(rank) and rank > best_rank:
                    best_rank, best_beta = rank, beta

    return {
        "max_depth": best_d,
        "regularization": best_lambda,
        "rashomon_epsilon_adder": current_epsilon,
        "beta": best_beta,
        "pilot_accuracy": float("nan"),
        "pilot_loss_proxy": pilot_mse,
        "task": "regression",
    }


def _prediction_variance(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = predictions @ weights
    centered = predictions - mean.reshape(-1, 1)
    return (centered ** 2) @ weights