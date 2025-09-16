"""
Defines selection strategies for the active learning process.

This module provides a base class, `Selector`, and specific implementations
for different query strategies like Passive sampling, Query-by-Committee (QBC),
and Bayesian Active Learning by Disagreement (BALD).
"""

### LIBRARIES ###
from abc import ABC, abstractmethod
from typing import Dict, List
import math
import numpy as np
import pandas as pd
import torch
from scipy import stats
from src.utils.models import ModelWrapper

### SELECTOR WRAPPER INTERFACE ###
class Selector(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def select(
        self,
        model: ModelWrapper,
        df_train: pd.DataFrame,
        df_candidate: pd.DataFrame
    ) -> Dict[str, int]:
        """
        Selects a single sample from the candidate set to be labeled.
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
        df_candidate: pd.DataFrame
    ) -> Dict[str, int]:
        if len(df_candidate) == 0:
            return {"IndexRecommendation": None}
        recommended_index = df_candidate.sample(n=1, random_state=self.random_state).index[0]
        return {"IndexRecommendation": int(recommended_index)}

### QUERY-BY-COMMITTEE SELECTOR ###
class QBCSelector(Selector):
    def __init__(self, use_unique_trees: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_unique_trees = use_unique_trees

    def _calculate_vote_entropy(self, committee_preds: pd.DataFrame) -> np.ndarray:
        proportions = committee_preds.apply(
            lambda row: row.value_counts(normalize=True), axis=1
        ).fillna(0)
        entropy = -np.sum(proportions * np.log(proportions + 1e-9), axis=1)
        return entropy.values

    def select(
        self,
        model: ModelWrapper,
        df_train: pd.DataFrame,
        df_candidate: pd.DataFrame
    ) -> Dict[str, int]:
        if len(df_candidate) == 0:
            return {"IndexRecommendation": None}
        X_candidate = df_candidate.drop(columns="Y")
        candidate_preds_raw = model.get_raw_ensemble_predictions(X_candidate)
        if candidate_preds_raw is None:
            raise TypeError("QBCSelector requires a model with an ensemble (e.g., RandomForest, TreeFarms).")
        
        committee_preds = candidate_preds_raw
        if self.use_unique_trees:
            X_train = df_train.drop(columns="Y")
            train_preds_raw = model.get_raw_ensemble_predictions(X_train)
            all_preds_combined = pd.concat([candidate_preds_raw, train_preds_raw], axis=0)
            unique_patterns = all_preds_combined.T.drop_duplicates().T
            committee_preds = candidate_preds_raw[unique_patterns.columns]
        
        if committee_preds.empty:
            return {"IndexRecommendation": None}
            
        uncertainty_scores = self._calculate_vote_entropy(committee_preds)
        top_local_index = np.argmax(uncertainty_scores)
        recommended_index = df_candidate.index[top_local_index]
        return {"IndexRecommendation": int(recommended_index)}

### BALD SELECTOR ###
class BALDSelector(Selector):
    """
    Selects a single sample using the Bayesian Active Learning by Disagreement (BALD) strategy.
    """
    def __init__(self, n_ensemble_samples: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.n_ensemble_samples = n_ensemble_samples

    def _calculate_bald_scores(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Calculates BALD scores from a tensor of log probabilities."""
        # E_theta[H(y|x, theta)]
        conditional_entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=(1, 2)) / self.n_ensemble_samples
        
        # H(E_theta[p(y|x, theta)])
        mean_log_probs = torch.logsumexp(log_probs, dim=1) - math.log(self.n_ensemble_samples)
        entropy_of_mean = -torch.sum(torch.exp(mean_log_probs) * mean_log_probs, dim=1)
        
        return entropy_of_mean - conditional_entropy

    def select(
        self,
        model: ModelWrapper,
        df_train: pd.DataFrame,
        df_candidate: pd.DataFrame
    ) -> Dict[str, int]:

        if len(df_candidate) == 0:
            return {"IndexRecommendation": None}

        X_candidate = df_candidate.drop(columns="Y")
        
        # Get log probabilities from the model
        log_probs_tensor = model.predict_log_proba_ensemble(X_candidate, self.n_ensemble_samples)
        
        if log_probs_tensor is None:
            raise TypeError("BALDSelector requires a model that supports predict_log_proba_ensemble (e.g., BNN, GPC).")
        
        # Calculate BALD scores
        bald_scores = self._calculate_bald_scores(log_probs_tensor)
        
        # Select the single top candidate
        top_local_index = torch.argmax(bald_scores).item()
        recommended_index = df_candidate.index[top_local_index]

        return {"IndexRecommendation": int(recommended_index)}