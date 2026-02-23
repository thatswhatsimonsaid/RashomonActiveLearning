### Summary ###
"""
Defines a standard interface for predictive models and evaluation.
"""

### Libraries ###
from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
import pandas as pd
import copyreg
import inspect
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score

import os
from concurrent.futures import ThreadPoolExecutor

### GLOBAL PICKLE FIX (For PySORTD C++ Objects) ###
try:
    from pysortd.csortd import SolverResult
    def pickle_solver_result(obj):
        return str, ("<Unpicklable SolverResult Dropped>",)
    copyreg.pickle(SolverResult, pickle_solver_result)
except ImportError:
    pass

### PySORTD Import & Compatibility Patch ###
try:
    from pysortd import SORTDClassifier

    # Scikit-Learn 1.6+ removed _validate_data, breaking PySORTD.
    if hasattr(SORTDClassifier, "fit") and not hasattr(SORTDClassifier, "_validate_data"):
        from sklearn.utils.validation import check_array, check_X_y
        
        def _patch_validate_data(self, X, y="no_validation", **check_params):
            # 1. Remove 'reset' argument #
            if "reset" in check_params:
                del check_params["reset"]
            
            # 2. Redirect to correct validation function #
            if y == "no_validation":
                return check_array(X, **check_params)
            return check_X_y(X, y, **check_params)
            
        setattr(SORTDClassifier, "_validate_data", _patch_validate_data)

except ImportError:
    SORTDClassifier = None

### MODEL WRAPPER INTERFACE ###
class ModelWrapper(ABC):
    """Abstract base class for all model wrappers."""

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Trains the model on the provided data."""
        pass

    @abstractmethod
    def predict(self, X_data: pd.DataFrame) -> np.ndarray:
        """Generates predictions for the given data."""
        pass

    def get_raw_ensemble_predictions(self, X_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Returns predictions from all members of the ensemble/Rashomon set."""
        return None
    
    def get_rashomon_size(self) -> int:
        return 1


### RANDOM FOREST WRAPPER (BASELINE) ### 
class RandomForestWrapper(ModelWrapper):
    def __init__(self, n_estimators: int = 100, random_state: int = 42, **kwargs):
        rf_sig = inspect.signature(RandomForestClassifier.__init__)
        valid_params = rf_sig.parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state,
            bootstrap=True,
            **filtered_kwargs 
        )
        self.is_fitted_ = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)
        self.is_fitted_ = True
        return self

    def predict(self, X_data: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_: raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(X_data)
    
    def get_raw_ensemble_predictions(self, X_data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_: raise RuntimeError("Model has not been fitted yet.")
        X_np = X_data.values
        all_preds = np.stack([tree.predict(X_np) for tree in self.model.estimators_], axis=1)
        df = pd.DataFrame(all_preds, index=X_data.index)
        df.columns = [f"tree_{i}" for i in range(df.shape[1])]
        return df
    
    @property
    def estimators_(self):
        return self.model.estimators_
    
    def get_ensemble_losses(self, X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        """Returns the training error for each tree in the forest."""
        if not self.is_fitted_: raise RuntimeError("Not fitted")
        
        losses = []
        X_np = X_train.values
        y_np = y_train.values
        for tree in self.model.estimators_:
            preds = tree.predict(X_np)
            loss = 1.0 - accuracy_score(y_np, preds)
            losses.append(loss)
            
        return np.array(losses)


### LOGISTIC REGRESSION WRAPPER ###
class LogisticRegressionWrapper(ModelWrapper):
    def __init__(self, random_state: int = 42, **kwargs):
        lr_sig = inspect.signature(LogisticRegression.__init__)
        valid_params = lr_sig.parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        
        self.model = LogisticRegression(random_state=random_state, max_iter=1000, **filtered_kwargs)
        self.is_fitted_ = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)
        self.is_fitted_ = True
        return self

    def predict(self, X_data: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_: raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(X_data)

### GREEDY DECISION TREE WRAPPER (BASELINE) ###
class GreedyDecisionTreeWrapper(ModelWrapper):
    def __init__(self, random_state: int = 42, max_depth: Optional[int] = None, **kwargs):
        dt_sig = inspect.signature(DecisionTreeClassifier.__init__)
        valid_params = dt_sig.parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        self.model = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, **filtered_kwargs)
        self.is_fitted_ = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)
        self.is_fitted_ = True
        return self

    def predict(self, X_data: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_: raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(X_data)


### GOSDT WRAPPER ###
class GOSDTWrapper(ModelWrapper):
    """Placeholder wrapper if GOSDT is needed, otherwise PySORTD covers it."""
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y): pass
    def predict(self, X): return np.zeros(len(X))

### PySORTD WRAPPER ###
class PySORTDWrapper(ModelWrapper):
    """
    Wrapper for PySORTD. 
    Acts as the 'Best Tree' Predictor AND the 'Rashomon Set' Selector.
    """
    def __init__(self, regularization: float = 0.01, rashomon_multiplier: float = 0.1, max_num_trees: int = 100, max_depth: int = 3, time_limit: int = 60, **kwargs):
        if SORTDClassifier is None: raise ImportError("pysortd is not installed.")
        
        self.config = {
            "optimization_task": "cost-complex-accuracy",
            "cost_complexity": regularization,
            "use_rashomon_multiplier": True, 
            "rashomon_multiplier": rashomon_multiplier,
            "max_num_trees": max_num_trees,
            "max_depth": max_depth,
            "time_limit": time_limit,
            "verbose": False
        }
        self.model = SORTDClassifier(**self.config)
        self.rashomon_size_ = 0
        self.committee_size_ = 0
        self.is_fitted_ = False
        self.epsilon = 0.0 

    def __getstate__(self):
        """Only pickles safe Python primitives. Discards C++ objects."""
        return {
            "config": self.config,
            "rashomon_size_": self.rashomon_size_,
            "committee_size_": self.committee_size_,
            "is_fitted_": self.is_fitted_,
            "epsilon": self.epsilon
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = None 

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        X_np = np.ascontiguousarray(X_train.values, dtype=np.intc)
        y_np = np.ascontiguousarray(y_train.values, dtype=np.intc)
        
        self.model.fit(X_np, y_np)
        
        self.rashomon_size_ = self.model.rashomon_set_size
        self.is_fitted_ = True
        return self

    def predict(self, X_data: pd.DataFrame) -> np.ndarray:
        X_np = np.ascontiguousarray(X_data.values, dtype=np.intc)
        return self.model.predict(X_np)

    def _predict_single_tree(self, tree_obj, X_np):
        """Vectorized tree traversal using Boolean masks."""
        n_samples = X_np.shape[0]
        preds = np.zeros(n_samples, dtype=int)
        
        def traverse(node, current_mask):
            if not np.any(current_mask):
                return
            
            try: is_leaf = node.is_leaf_node()
            except TypeError: is_leaf = node.is_leaf_node
            
            if is_leaf:
                preds[current_mask] = node.label
            else:
                left_mask = current_mask & (X_np[:, node.feature] == 0)
                right_mask = current_mask & (X_np[:, node.feature] == 1)
                
                traverse(node.left_child, left_mask)
                traverse(node.right_child, right_mask)

        initial_mask = np.ones(n_samples, dtype=bool)
        traverse(tree_obj, initial_mask)
        return preds

    def get_raw_ensemble_predictions(self, X_data: pd.DataFrame, cache: bool = True) -> pd.DataFrame:
        """
        Fastest implementation: Uses vectorized Boolean masks and optional caching.
        """
        if not self.is_fitted_: raise RuntimeError("Not fitted")
        
        # 1. Check Cache (Avoid redundant C++ object traversal)
        if cache and hasattr(self, '_last_X') and self._last_X.equals(X_data):
            return self._last_preds_cache

        X_np = np.ascontiguousarray(X_data.values, dtype=np.intc)
        n_trees = self.rashomon_size_
        n_samples = X_np.shape[0]
        all_preds = np.empty((n_samples, n_trees), dtype=np.int8)
        
        # 2. Vectorized Traversal (This is the heavy lifting)
        for i in range(n_trees):
            tree_obj = self.model.get_tree_n(i)
            all_preds[:, i] = self._predict_single_tree(tree_obj, X_np)
            
        df = pd.DataFrame(all_preds, index=X_data.index)
        df.columns = [f"tree_{i}" for i in range(df.shape[1])]
        
        # 3. Store in Cache
        if cache:
            self._last_X = X_data
            self._last_preds_cache = df
            
        return df

    def get_ensemble_losses(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        High-speed loss calculation using cached predictions and vectorized NumPy.
        """
        # Uses cache=True to avoid re-calculating if X is the same as the query pool
        all_preds_df = self.get_raw_ensemble_predictions(X, cache=True)
        
        y_true = y.values.flatten().reshape(-1, 1)
        # Fast NumPy comparison
        errors = (all_preds_df.values != y_true).mean(axis=0)
        
        reg = self.config.get("cost_complexity", 0.01)
        
        # Complexity (Batch-process C++ calls)
        # Consistent with Eq. 1 (using leaf nodes)
        n_leaves = np.array([
            self.model.get_tree_n(i).get_num_leaf_nodes() 
            for i in range(self.rashomon_size_)
        ])
        
        return errors + (reg * n_leaves)

### METRIC UTILS ###
def calculate_oracle_agreement(current_model: ModelWrapper, oracle_model: ModelWrapper, df_test: pd.DataFrame) -> float:
    X_test = df_test.drop(columns="Y")
    preds_current = current_model.predict(X_test)
    preds_oracle = oracle_model.predict(X_test)
    return float(np.mean(preds_current == preds_oracle))

def evaluate_model(model: ModelWrapper, df_test: pd.DataFrame) -> Dict[str, float]:
    X_test = df_test.drop(columns="Y")
    y_test_true = df_test["Y"]
    predictions = model.predict(X_test)
    f1 = f1_score(y_test_true, predictions, average='micro')
    acc = accuracy_score(y_test_true, predictions)
    return {"f1_micro": float(f1), "accuracy": float(acc)}

def evaluate_models(predictor_model, oracle_model, df_test) -> dict:
    from src.utils.tree_utils import (
        calculate_ted_score, 
        calculate_oracle_agreement
    )

    metrics = evaluate_model(predictor_model, df_test)
    metrics["oracle_agreement"] = calculate_oracle_agreement(
        current_model=predictor_model,
        oracle_model=oracle_model,
        df_test=df_test
    )

    metrics["tree_edit_distance"] = calculate_ted_score(
        model=predictor_model,
        oracle=oracle_model
    )
    
    return metrics