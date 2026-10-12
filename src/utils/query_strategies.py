### Summary ###
"""
Defines selection strategies for the active learning process.
"""

### Libraries ###
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
import pandas as pd
from src.utils.models import ModelWrapper
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier


### SELECTOR WRAPPER INTERFACE ###
class Selector(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def select(
        self,
        model: ModelWrapper,
        df_train: pd.DataFrame,
        df_candidate: pd.DataFrame) -> Dict[str, Any]:
        """
        Selects a single sample from the candidate set to be labeled.
        Returns a dict containing:
          - "IndexRecommendation": The index of the selected point.
          - "AllEntropies": (Optional) A pd.Series of scores for all candidates (for visualization).
        """
        pass

### PASSIVE SELECTOR ###
class PassiveSelector(Selector):
    def __init__(self, random_state: int = 42, **kwargs):

        super().__init__(**kwargs)
        self.random_state = random_state

    def select(
        self,
        model: ModelWrapper,
        df_train: pd.DataFrame,
        df_candidate: pd.DataFrame) -> Dict[str, Any]:

        if len(df_candidate) == 0:
            return {"IndexRecommendation": None}
        recommended_index = df_candidate.sample(n=1, random_state=self.random_state).index[0]
        all_entropies = pd.Series(0.0, index=df_candidate.index)
        
        return {
            "IndexRecommendation": int(recommended_index),
            "AllEntropies": all_entropies
        }

### QUERY-BY-COMMITTEE SELECTOR ###
class QBCSelector(Selector):
    def __init__(self, beta: float = 10.0, **kwargs):
        """
        Args:
            beta: Inverse temperature for Gibbs weighting.
                  High beta = Trust the leaders.
                  Low beta = Listen to the whole Rashomon set.
        """
        super().__init__(**kwargs)
        self.beta = beta

    def select(
        self,
        model: Any, 
        df_train: pd.DataFrame,
        df_candidate: pd.DataFrame) -> Dict[str, Any]:

        if len(df_candidate) == 0:
            return {"IndexRecommendation": None}

        X_candidate = df_candidate.drop(columns="Y")

        # 1. Get raw predictions (N_samples, N_trees)
        raw_preds_df = model.get_raw_ensemble_predictions(X_candidate)
        
        # 2. Get losses for Gibbs weighting
        if df_train is not None and hasattr(model, "get_ensemble_losses"):
            losses = model.get_ensemble_losses(df_train.drop(columns="Y"), df_train["Y"])
        else:
            # Fallback for models that don't support ensembles
            losses = np.zeros(raw_preds_df.shape[1])

        # If the set collapsed to 1 tree (should be rare!!!), fallback to random
        if raw_preds_df.shape[1] < 2:
            self.effective_committee_size_ = 1.0 
            recommended_index = df_candidate.sample(n=1).index[0]
            return {
                "IndexRecommendation": int(recommended_index),
                "AllEntropies": pd.Series(0.0, index=df_candidate.index)
            }

        # 3. Calculate Gibbs Weights
        adj_losses = losses - np.min(losses)
        weights = np.exp(-self.beta * adj_losses)
        weights /= np.sum(weights)
        
        # Calculate Effective Committee Size using Shannon Entropy
        shannon_entropy = -np.sum(weights * np.log(weights + 1e-12))
        self.effective_committee_size_ = np.exp(shannon_entropy) 

        # 4. Weighted Entropy 
        p = np.dot(raw_preds_df.values, weights)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        uncertainty_scores = -(p * np.log(p) + (1 - p) * np.log(1 - p))

        # 5. Result Extraction
        top_local_index = np.argmax(uncertainty_scores)
        recommended_index = df_candidate.index[top_local_index]
        
        return {
            "IndexRecommendation": int(recommended_index),
            "AllEntropies": pd.Series(uncertainty_scores, index=df_candidate.index)
        }

def _gibbs_weights(losses: np.ndarray, beta: float) -> np.ndarray:
    adj_losses = np.asarray(losses, dtype=float) - np.min(losses)
    weights = np.exp(-beta * adj_losses)
    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0:
        return np.full(len(adj_losses), 1.0 / len(adj_losses))
    return weights / total


def _weighted_prediction_variance(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Variance of the Gibbs-weighted committee prediction at each row."""
    mean = predictions @ weights
    centered = predictions - mean.reshape(-1, 1)
    return (centered ** 2) @ weights


### REGRESSION QUERY-BY-COMMITTEE ###
class RegressionQBCSelector(Selector):
    """
    Regression analogue of QBCSelector.

    Committee members are weighted by a Gibbs posterior on their squared-error
    loss. beta = 0 is a uniform Rashomon committee. The query is the unlabeled
    point with the largest weighted predictive variance.
    """

    def __init__(self, beta: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = float(beta) if beta != "calibrated" else 0.0

    def select(
        self,
        model: Any,
        df_train: pd.DataFrame,
        df_candidate: pd.DataFrame) -> Dict[str, Any]:

        if len(df_candidate) == 0:
            return {"IndexRecommendation": None}

        X_candidate = df_candidate.drop(columns="Y")
        raw_preds_df = model.get_raw_ensemble_predictions(X_candidate)

        if df_train is not None and hasattr(model, "get_ensemble_losses"):
            losses = model.get_ensemble_losses(df_train.drop(columns="Y"), df_train["Y"])
        else:
            losses = np.zeros(raw_preds_df.shape[1])

        if raw_preds_df.shape[1] < 2:
            self.effective_committee_size_ = 1.0
            recommended_index = df_candidate.sample(n=1).index[0]
            return {
                "IndexRecommendation": int(recommended_index),
                "AllEntropies": pd.Series(0.0, index=df_candidate.index)
            }

        weights = _gibbs_weights(losses, self.beta)
        shannon_entropy = -np.sum(weights * np.log(weights + 1e-12))
        self.effective_committee_size_ = float(np.exp(shannon_entropy))

        scores = _weighted_prediction_variance(raw_preds_df.values.astype(float), weights)
        top_local_index = int(np.argmax(scores))
        recommended_index = df_candidate.index[top_local_index]
        return {
            "IndexRecommendation": int(recommended_index),
            "AllEntropies": pd.Series(scores, index=df_candidate.index)
        }


### REGRESSION UNCERTAINTY ###
class RegressionUncertaintySelector(Selector):
    """Queries the pool point whose leaf has the largest training variance."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def select(
        self,
        model: ModelWrapper,
        df_train: pd.DataFrame,
        df_candidate: pd.DataFrame) -> Dict[str, Any]:

        if len(df_candidate) == 0:
            return {"IndexRecommendation": None}
        if not hasattr(model, "leaf_variance"):
            raise AttributeError("The selected model does not expose leaf_variance.")

        X_candidate = df_candidate.drop(columns="Y")
        scores = np.asarray(model.leaf_variance(X_candidate), dtype=float).reshape(-1)
        top_local_index = int(np.argmax(scores))
        recommended_index = df_candidate.index[top_local_index]
        return {
            "IndexRecommendation": int(recommended_index),
            "AllEntropies": pd.Series(scores, index=df_candidate.index)
        }

### UNCERTAINTY SELECTOR ###
class UncertaintySelector(Selector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def select(
        self,
        model: ModelWrapper,
        df_train: pd.DataFrame,
        df_candidate: pd.DataFrame) -> Dict[str, Any]:

        if len(df_candidate) == 0:
            return {"IndexRecommendation": None}

        X_candidate = df_candidate.drop(columns="Y")
        
        # 1. Get class probabilities (n_samples, n_classes)
        try:
            inner_model = model.model if hasattr(model, 'model') else model
            probs = inner_model.predict_proba(X_candidate)
        except AttributeError:
            raise AttributeError("The selected model does not support predict_proba for Uncertainty Sampling.")

        # 2. Calculate Uncertainty Score
        epsilon = 1e-9
        entropy = -np.sum(probs * np.log(probs + epsilon), axis=1)

        # 3. Find the maximum entropy (highest uncertainty)
        top_local_index = np.argmax(entropy)
        recommended_index = df_candidate.index[top_local_index]
        
        all_entropies_series = pd.Series(entropy, index=df_candidate.index)

        return {
            "IndexRecommendation": int(recommended_index),
            "AllEntropies": all_entropies_series
        }

def _ridge_predictions(df_train: pd.DataFrame, X_candidate: pd.DataFrame) -> np.ndarray:
    """Ridge regression used by GSy and iGS (Wu, Lin, and Huang, 2019), with λ = 0.01."""
    X_train = df_train.drop(columns="Y")
    y_train = df_train["Y"].to_numpy(dtype=float)
    ridge = Ridge(alpha=0.01)
    ridge.fit(X_train, y_train)
    return np.asarray(ridge.predict(X_candidate), dtype=float).reshape(-1)


def _output_gaps(predictions: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """For each candidate, distance from its prediction to the nearest labeled output."""
    gaps = np.abs(predictions.reshape(-1, 1) - y_train.reshape(1, -1))
    return gaps.min(axis=1)


### GREEDY SAMPLING ON THE INPUTS (GSx) ###
class GSxSelector(Selector):
    """
    Furthest-first traversal in the input space.

    The candidate that maximizes the Euclidean distance to the nearest labeled
    row is queried. This is GSx from Wu, Lin, and Huang (2019). On binary
    features it ranks rows the same way Hamming coreset does.
    """

    def select(self, model, df_train, df_candidate) -> Dict[str, Any]:
        if len(df_candidate) == 0:
            return {"IndexRecommendation": None}

        X_train = df_train.drop(columns="Y").to_numpy(dtype=float)
        X_cand = df_candidate.drop(columns="Y").to_numpy(dtype=float)
        distances = cdist(X_cand, X_train, metric="euclidean")
        scores = distances.min(axis=1)
        chosen = int(np.argmax(scores))
        return {
            "IndexRecommendation": int(df_candidate.index[chosen]),
            "AllEntropies": pd.Series(scores, index=df_candidate.index),
        }


### GREEDY SAMPLING ON THE OUTPUT (GSy) ###
class GSySelector(Selector):
    """
    Query the candidate whose ridge prediction is farthest from every labeled output.

    d_n = min_m |f(x_n) - y_m|, and the maximum d_n is labeled. f is ridge
    regression with λ = 0.01, as in Wu, Lin, and Huang (2019).
    """

    def select(self, model, df_train, df_candidate) -> Dict[str, Any]:
        if len(df_candidate) == 0:
            return {"IndexRecommendation": None}

        X_cand = df_candidate.drop(columns="Y")
        predictions = _ridge_predictions(df_train, X_cand)
        y_train = df_train["Y"].to_numpy(dtype=float)
        scores = _output_gaps(predictions, y_train)
        chosen = int(np.argmax(scores))
        return {
            "IndexRecommendation": int(df_candidate.index[chosen]),
            "AllEntropies": pd.Series(scores, index=df_candidate.index),
        }


### IMPROVED GREEDY SAMPLING (iGS) ###
class ImprovedGreedySelector(Selector):
    """
    Query the candidate farthest from the labeled set in input and output together.

    d_n = min_m ||x_n - x_m|| |f(x_n) - y_m|. f is the same ridge regression as GSy.
    """

    def select(self, model, df_train, df_candidate) -> Dict[str, Any]:
        if len(df_candidate) == 0:
            return {"IndexRecommendation": None}

        X_train = df_train.drop(columns="Y").to_numpy(dtype=float)
        X_cand = df_candidate.drop(columns="Y")
        input_distances = cdist(X_cand.to_numpy(dtype=float), X_train, metric="euclidean")
        predictions = _ridge_predictions(df_train, X_cand)
        y_train = df_train["Y"].to_numpy(dtype=float)
        output_distances = np.abs(predictions.reshape(-1, 1) - y_train.reshape(1, -1))
        scores = (input_distances * output_distances).min(axis=1)
        chosen = int(np.argmax(scores))
        return {
            "IndexRecommendation": int(df_candidate.index[chosen]),
            "AllEntropies": pd.Series(scores, index=df_candidate.index),
        }


### Coreset/Hamming Diversity ###
class HammingDiversitySelector(Selector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def select(self, model, df_train, df_candidate) -> Dict[str, Any]:
        if len(df_candidate) == 0:
            return {"IndexRecommendation": None}

        X_train = df_train.drop(columns="Y").values
        X_cand = df_candidate.drop(columns="Y").values
        distances = cdist(X_cand, X_train, metric='hamming')

        # This is the Max-Min/Coreset approach
        min_distances = np.min(distances, axis=1)
        recommended_idx = df_candidate.index[np.argmax(min_distances)]

        return {
            "IndexRecommendation": int(recommended_idx),
            "AllEntropies": pd.Series(min_distances, index=df_candidate.index)
        }

# ### Rashomon Expected Model Change (NOT USED) ###
class ModelChangeSelector(Selector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def select(self, model, df_train, df_candidate) -> Dict[str, Any]:
        if len(df_candidate) == 0:
            return {"IndexRecommendation": None}

        X_cand = df_candidate.drop(columns="Y")
        
        # 1. Get predictions from all models in Rashomon Set
        all_preds = model.get_raw_ensemble_predictions(X_cand)
        
        # 2. Get predictions from the single best tree (usually the first one)
        best_tree_preds = all_preds.iloc[:, 0]
        
        # 3. Calculate how many trees in the Rashomon set disagree with the best tree
        # Basically 'If I label this, I am likely to flip the best tree'
        disagreements = all_preds.apply(lambda col: col != best_tree_preds).sum(axis=1)
        
        recommended_idx = df_candidate.index[np.argmax(disagreements)]

        return {
            "IndexRecommendation": int(recommended_idx),
            "AllEntropies": pd.Series(disagreements, index=df_candidate.index)
        }