### Summary ###
"""
Defines selection strategies for the active learning process.
"""

### Libraries ###
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from src.utils.models import ModelWrapper
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier


### CLASSIFICATION PATTERN HELPERS ###
def extract_classification_patterns(
    pred_matrix: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse committee members into distinct classification patterns.

    A pattern is the full prediction vector of one tree across samples
    (one column of a samples×trees prediction matrix).

    Args:
        pred_matrix: Array of shape (n_samples, n_trees) with hard labels.
        weights: Optional per-tree weights of shape (n_trees,). Defaults to uniform.

    Returns:
        unique_patterns: (n_unique, n_samples) distinct prediction vectors.
        inverse: (n_trees,) maps each tree index to its pattern id.
        pattern_weights: (n_unique,) total weight mass on each pattern.
    """
    if pred_matrix.ndim != 2:
        raise ValueError(f"pred_matrix must be 2D, got shape {pred_matrix.shape}")

    n_samples, n_trees = pred_matrix.shape
    if weights is None:
        weights = np.ones(n_trees, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.shape[0] != n_trees:
            raise ValueError(
                f"weights length {weights.shape[0]} != n_trees {n_trees}"
            )

    # np.unique on rows of trees×samples
    tree_vectors = np.ascontiguousarray(pred_matrix.T)
    unique_patterns_t, inverse, counts = np.unique(
        tree_vectors, axis=0, return_inverse=True, return_counts=True
    )
    n_unique = unique_patterns_t.shape[0]
    pattern_weights = np.zeros(n_unique, dtype=float)
    np.add.at(pattern_weights, inverse, weights)

    return unique_patterns_t, inverse, pattern_weights


def _gibbs_weights(losses: np.ndarray, beta: float) -> np.ndarray:
    adj_losses = losses - np.min(losses)
    weights = np.exp(-beta * adj_losses)
    weights /= np.sum(weights)
    return weights


def _vote_entropy_scores(pred_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Binary vote entropy for each sample under committee weights."""
    p = np.dot(pred_matrix, weights)
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def _effective_committee_size(weights: np.ndarray) -> float:
    shannon_entropy = -np.sum(weights * np.log(weights + 1e-12))
    return float(np.exp(shannon_entropy))


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
    def __init__(self, beta: float = 10.0, use_unique_trees: bool = False, **kwargs):
        """
        Args:
            beta: Inverse temperature for Gibbs weighting.
                  High beta = Trust the leaders.
                  Low beta = Listen to the whole Rashomon set.
            use_unique_trees: If True, collapse functionally identical trees
                  (same prediction vector on the candidate pool) into one
                  pattern before computing vote entropy. Pattern mass is the
                  sum of member Gibbs weights.
        """
        super().__init__(**kwargs)
        self.beta = beta
        self.use_unique_trees = use_unique_trees

    def select(
        self,
        model: Any, 
        df_train: pd.DataFrame,
        df_candidate: pd.DataFrame) -> Dict[str, Any]:

        if len(df_candidate) == 0:
            return {"IndexRecommendation": None}

        X_candidate = df_candidate.drop(columns="Y")
        X_train = df_train.drop(columns="Y")
        y_train = df_train["Y"]
        
        # 1. Get raw predictions (N_samples, N_trees)
        raw_preds_df = model.get_raw_ensemble_predictions(X_candidate)
        raw_preds = raw_preds_df.values
        
        # 2. Get losses for Gibbs weighting
        if hasattr(model, "get_ensemble_losses"):
            losses = model.get_ensemble_losses(X_train, y_train)
        else:
            # Fallback for models that don't support ensembles
            losses = np.zeros(raw_preds.shape[1])

        # If the set collapsed to 1 tree (should be rare!!!), fallback to random
        if raw_preds.shape[1] < 2:
            self.effective_committee_size_ = 1.0 
            recommended_index = df_candidate.sample(n=1).index[0]
            return {
                "IndexRecommendation": int(recommended_index),
                "AllEntropies": pd.Series(0.0, index=df_candidate.index)
            }

        # 3. Calculate Gibbs Weights
        weights = _gibbs_weights(losses, self.beta)

        # Optional: collapse to distinct classification patterns on the candidate pool
        if self.use_unique_trees:
            unique_patterns, _, pattern_weights = extract_classification_patterns(
                raw_preds, weights=weights
            )
            # unique_patterns is (n_unique, n_samples); vote matrix needs samples×patterns
            vote_matrix = unique_patterns.T.astype(float)
            vote_weights = pattern_weights / np.sum(pattern_weights)
        else:
            vote_matrix = raw_preds.astype(float)
            vote_weights = weights

        self.effective_committee_size_ = _effective_committee_size(vote_weights)

        # 4. Weighted Vote Entropy
        uncertainty_scores = _vote_entropy_scores(vote_matrix, vote_weights)

        # 5. Result Extraction
        top_local_index = np.argmax(uncertainty_scores)
        recommended_index = df_candidate.index[top_local_index]
        
        return {
            "IndexRecommendation": int(recommended_index),
            "AllEntropies": pd.Series(uncertainty_scores, index=df_candidate.index)
        }


### PATTERN DISAMBIGUATION SELECTOR ###
class PatternDisambiguationSelector(Selector):
    """
    Select the unlabeled point that most increases distinct classification
    patterns among Rashomon / committee members.

    Motivation
    ----------
    Vote entropy asks: "how split is the committee at x?"
    This criterion asks: "if we append each tree's prediction at x to its
    prediction vector on the labeled set, how many distinct vectors remain?"

    A point that merely re-expresses an existing disagreement among already
    distinct trees scores lower than a point that splits trees that were
    previously functionally identical on the labeled data — i.e., it
    *disambiguates* committee members from each other.

    Notes
    -----
    - Does not require the true label of x (no refit / lookahead).
    - `extract_classification_patterns` is the swap-point if you have a
      preferred pattern-extraction routine.
    - Optional Gibbs weighting via `beta` (same convention as QBCSelector).
    """

    def __init__(
        self,
        beta: float = 0.0,
        score_mode: str = "count",
        **kwargs,
    ):
        """
        Args:
            beta: Gibbs inverse temperature for pattern masses.
            score_mode:
                "count" — raw number of distinct (train-pattern, pred_at_x) pairs.
                "effective" — exp(H) of Gibbs mass over those extended patterns.
        """
        super().__init__(**kwargs)
        if score_mode not in {"count", "effective"}:
            raise ValueError(
                f"score_mode must be 'count' or 'effective', got {score_mode!r}"
            )
        self.beta = beta
        self.score_mode = score_mode

    def select(
        self,
        model: Any,
        df_train: pd.DataFrame,
        df_candidate: pd.DataFrame,
    ) -> Dict[str, Any]:

        if len(df_candidate) == 0:
            return {"IndexRecommendation": None}

        X_candidate = df_candidate.drop(columns="Y")
        X_train = df_train.drop(columns="Y")
        y_train = df_train["Y"]

        preds_cand_df = model.get_raw_ensemble_predictions(X_candidate)
        preds_cand = preds_cand_df.values.astype(np.int8)
        n_cand, n_trees = preds_cand.shape

        if n_trees < 2:
            self.effective_committee_size_ = 1.0
            recommended_index = df_candidate.sample(n=1).index[0]
            return {
                "IndexRecommendation": int(recommended_index),
                "AllEntropies": pd.Series(0.0, index=df_candidate.index),
            }

        preds_train = model.get_raw_ensemble_predictions(X_train).values.astype(np.int8)

        if hasattr(model, "get_ensemble_losses"):
            losses = model.get_ensemble_losses(X_train, y_train)
        else:
            losses = np.zeros(n_trees)

        weights = _gibbs_weights(losses, self.beta)
        self.effective_committee_size_ = _effective_committee_size(weights)

        # Base patterns on the labeled set; inverse[t] = pattern id of tree t
        _, train_pattern_ids, _ = extract_classification_patterns(
            preds_train, weights=weights
        )

        # Pack (train_pattern_id, pred_at_x) into a single integer code per tree.
        # Binary labels assumed (as elsewhere in this codebase).
        extended_codes = (
            train_pattern_ids[None, :].astype(np.int64) * 2
            + preds_cand.astype(np.int64)
        )  # (n_cand, n_trees)

        if self.score_mode == "count":
            scores = np.empty(n_cand, dtype=float)
            for i in range(n_cand):
                scores[i] = np.unique(extended_codes[i]).size
        else:
            scores = np.empty(n_cand, dtype=float)
            for i in range(n_cand):
                codes = extended_codes[i]
                # Aggregate Gibbs mass per extended pattern
                uniq, inv = np.unique(codes, return_inverse=True)
                mass = np.zeros(uniq.size, dtype=float)
                np.add.at(mass, inv, weights)
                mass /= mass.sum()
                scores[i] = _effective_committee_size(mass)

        # Tie-break with ordinary vote entropy so pure disagreement still matters
        # when pattern counts are flat (e.g. trees already fully distinct on train).
        vote_entropy = _vote_entropy_scores(preds_cand.astype(float), weights)
        # Lexicographic argmax via a tiny entropy nudge in [0, log2]
        combined = scores + (vote_entropy / (vote_entropy.max() + 1e-12)) * 1e-6

        top_local_index = int(np.argmax(combined))
        recommended_index = df_candidate.index[top_local_index]

        return {
            "IndexRecommendation": int(recommended_index),
            "AllEntropies": pd.Series(scores, index=df_candidate.index),
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