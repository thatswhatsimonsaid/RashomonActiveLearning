### LIBRARIES ###
import pandas as pd
import numpy as np
import pytest

### TEST THIS CLASS ###
from src.utils.selectors import QBCSelector

### MOCK MODEL CLASS ###
class MockModel:
    def __init__(self, vote_dataframe: pd.DataFrame):
        self._vote_df = vote_dataframe
    
    def get_raw_ensemble_predictions(self, X_data: pd.DataFrame) -> pd.DataFrame:
        return self._vote_df.loc[X_data.index]

### SET UP FOR MOCK MODEL ###
@pytest.fixture
def mock_qbc_model() -> MockModel:
    """
    A pytest fixture that creates a mock model with a robust, programmatically
    generated set of ensemble votes.
    """

    ## Configuration ##
    NUM_CANDIDATES = 100
    NUM_TREES = 11 
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    ## Generate data ##
    candidate_indices = range(0, NUM_CANDIDATES)    
    votes = np.random.randint(0, 2, size=(NUM_CANDIDATES, NUM_TREES))
    
    ## Plant a perfectly certain candidate and uncertain candidate ##

    # 1. Create a "perfectly uncertain" candidate with a 50/50 split in votes.
    most_uncertain_idx_loc = 1
    half_trees = NUM_TREES // 2
    votes[most_uncertain_idx_loc, :half_trees] = 0
    votes[most_uncertain_idx_loc, half_trees:] = 1
    
    # 2. Create a "perfectly certain" candidate (zero entropy)
    most_certain_idx_loc = 2 
    votes[most_certain_idx_loc, :] = 0

    ## Return ##
    expected_winner_index = candidate_indices[most_uncertain_idx_loc]
    vote_df = pd.DataFrame(votes, index=candidate_indices)    
    return MockModel(vote_df), expected_winner_index, candidate_indices

### TEST FUNCTION ###
def test_qbc_selector_picks_most_uncertain(mock_qbc_model):
    """
    Tests that the QBCSelector picks the candidate with the highest vote entropy
    from a large, randomized set of votes.
    """
    # 1. ARRANGE
    model, expected_index, candidate_indices = mock_qbc_model
    selector = QBCSelector(use_unique_trees=False)
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