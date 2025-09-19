### SUMMARY ###
"""
Defines a standard interface for predictive models and evaluation.

This module provides a base class, `ModelWrapper`, to ensure that all models
share a common API for fitting, predicting, and accessing predictions.
"""

### LIBRARIES ###
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import f1_score, accuracy_score
from treefarms import TREEFARMS

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
        """
        Returns a DataFrame of predictions from each member of the model ensemble.
        Returns None if the model is not an ensemble type.
        """
        return None

    def predict_log_proba_ensemble(self, X_data: pd.DataFrame, n_samples: int) -> Optional[torch.Tensor]:
        """
        Returns a tensor of log probabilities from an ensemble or Bayesian model.
        Returns None if the model does not support this.
        Shape: (n_points, n_ensemble_members, n_classes)
        """
        return None

### TREEFARMS WRAPPER ###
class TreeFarmsWrapper(ModelWrapper):
    def __init__(self, regularization: float, rashomon_threshold: float, **kwargs):
        self.config = {
            "regularization": regularization,
            "rashomon_bound_adder": rashomon_threshold
        }
        self.model = TREEFARMS(self.config)
        self.all_trees_ = []
        self.is_fitted_ = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)
        self.all_trees_ = [self.model[i] for i in range(self.model.get_tree_count())]
        self.is_fitted_ = True
        return self

    def _predict_single_tree(self, tree_model, X_data: pd.DataFrame) -> pd.Series:
        predictions = [tree_model.classify(row)[0] for row in X_data.values]
        return pd.Series(predictions, index=X_data.index)

    def predict(self, X_data: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted yet.")
        ensemble_preds = self.get_raw_ensemble_predictions(X_data)
        mode_predictions, _ = stats.mode(ensemble_preds.values, axis=1)
        return mode_predictions.flatten().astype(int)

    def get_raw_ensemble_predictions(self, X_data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted yet.")
        pred_list = [self._predict_single_tree(tree, X_data) for tree in self.all_trees_]
        df = pd.concat(pred_list, axis=1)
        df.columns = [f"tree_{i}" for i in range(df.shape[1])]
        return df

### RANDOM FOREST WRAPPER ###
class RandomForestWrapper(ModelWrapper):
    def __init__(self, n_estimators: int = 100, random_state: int = 42, **kwargs):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.is_fitted_ = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train, y_train)
        self.is_fitted_ = True
        return self

    def predict(self, X_data: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(X_data)

    def get_raw_ensemble_predictions(self, X_data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted yet.")
        preds = np.array([tree.predict(X_data.values) for tree in self.model.estimators_]).T
        df = pd.DataFrame(preds, index=X_data.index)
        df.columns = [f"tree_{i}" for i in range(df.shape[1])]
        return df

### GAUSSIAN PROCESS CLASSIFIER WRAPPER ###
class GaussianProcessWrapper(ModelWrapper):
    """A wrapper for scikit-learn's GaussianProcessClassifier."""

    def __init__(self, random_state: int = 42, **kwargs):
        self.model = GaussianProcessClassifier(kernel=RBF(1.0), random_state=random_state)
        self.is_fitted_ = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.model.fit(X_train.values, y_train.values)
        self.is_fitted_ = True
        return self

    def predict(self, X_data: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted yet.")
        return self.model.predict(X_data.values)

    def predict_log_proba_ensemble(self, X_data: pd.DataFrame, n_samples: int) -> torch.Tensor:
        """
        Generates log-probability samples by duplicating the GPC's deterministic output.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted yet.")
        
        # GPC is deterministic, so we get one probability distribution
        probs = self.model.predict_proba(X_data.values)
        log_probs = np.log(probs + 1e-9) # Add epsilon for stability
        
        # Duplicate n_samples times to create a fake "ensemble" dimension
        log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32)
        return log_probs_tensor.unsqueeze(1).repeat(1, n_samples, 1)

### BAYESIAN NEURAL NETWORK ###
class _BNN(nn.Module):
    """Internal PyTorch module for the Bayesian Neural Network."""
    def __init__(self, input_size, num_classes, hidden_size=50, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))
    
### BAYESIAN NEURAL NETWORK WRAPPER ###
class BNNWrapper(ModelWrapper):
    """A wrapper for a Bayesian Neural Network using MC Dropout."""

    def __init__(self, epochs=50, lr=0.001, random_state=42, **kwargs):
        self.epochs = epochs
        self.lr = lr
        self.random_state = random_state
        self.model: Optional[_BNN] = None
        self.is_fitted_ = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        torch.manual_seed(self.random_state)
        X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_train.values, dtype=torch.long)

        self.model = _BNN(X_train.shape[1], len(y_train.unique()))
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train() # Enable dropout
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        self.is_fitted_ = True
        return self

    def predict(self, X_data: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted yet.")
            
        log_probs_tensor = self.predict_log_proba_ensemble(X_data, n_samples=100)
        mean_probs = torch.exp(log_probs_tensor).mean(dim=1)
        return torch.argmax(mean_probs, dim=1).cpu().numpy()

    def predict_log_proba_ensemble(self, X_data: pd.DataFrame, n_samples: int) -> torch.Tensor:
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted yet.")
            
        self.model.train() # Enable dropout for Monte Carlo samples
        X_tensor = torch.tensor(X_data.values, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = [torch.log_softmax(self.model(X_tensor), dim=1) for _ in range(n_samples)]
        
        return torch.stack(outputs, dim=1)

### MODEL EVALUATION FUNCTION ###
def evaluate_model(model: ModelWrapper, df_test: pd.DataFrame) -> Dict[str, float]:
    X_test = df_test.drop(columns="Y")
    y_test = df_test["Y"]
    predictions = model.predict(X_test)
    
    f1 = f1_score(y_test, predictions, average='micro')
    acc = accuracy_score(y_test, predictions)

    return {"f1_micro": float(f1), "accuracy": float(acc)}