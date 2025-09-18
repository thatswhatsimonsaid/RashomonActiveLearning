### LIBRARIES ###
import pandas as pd
import numpy as np
import torch
import pytest

### TEST THIS CLASS ###
from src.utils.selectors import BALDSelector

### MOCK MODEL CLASS ###
class MockBaldModel:
    def __init__(self, log_proba_tensor: torch.Tensor):
        self._log_proba = log_proba_tensor

    def predict_log_proba_ensemble(self, X_data: pd.DataFrame, n_samples: int) -> torch.Tensor:
        return self._log_proba

### SET UP FOR MOCK MODEL ###
@pytest.fixture
def mock_bald_model():
    """A pytest fixture that creates a mock model with robust log probabilities."""
    
    ## Configuration ##
    NUM_CANDIDATES = 100
    NUM_ENSEMBLE_MEMBERS = 10
    NUM_CLASSES = 2
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    ## Generate Data ##
    candidate_indices = range(1000, 1000 + NUM_CANDIDATES)
    random_logits = torch.randn(NUM_CANDIDATES, NUM_ENSEMBLE_MEMBERS, NUM_CLASSES)
    log_probs = torch.log_softmax(random_logits, dim=-1)

    ## Plant a perfectly uncertain candidate ##
    bald_winner_loc = 50 
    log_probs[bald_winner_loc, :5, :] = torch.tensor([0.0, -float('inf')])
    log_probs[bald_winner_loc, 5:, :] = torch.tensor([-float('inf'), 0.0])
    
    ## Return ##
    expected_winner_index = candidate_indices[bald_winner_loc]
    return MockBaldModel(log_probs), expected_winner_index, candidate_indices


### TEST FUNCTION ###
def test_bald_selector_picks_highest_disagreement(mock_bald_model):
    """
    Tests that BALDSelector picks the candidate with the highest confident disagreement.
    """
    # 1. ARRANGE
    model, expected_index, candidate_indices = mock_bald_model    
    selector = BALDSelector(n_ensemble_samples=10)
    df_candidate = pd.DataFrame({'Y': np.zeros(len(candidate_indices))}, index=candidate_indices)

    # 2. ACT
    result = selector.select(
        model=model,
        df_train=pd.DataFrame(),
        df_candidate=df_candidate
    )
    queried_index = result["IndexRecommendation"]

    # 3. ASSERT
    assert queried_index == expected_index