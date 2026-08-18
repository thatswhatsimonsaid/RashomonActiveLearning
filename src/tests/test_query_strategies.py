### Libraries ###
import pandas as pd
import numpy as np
import pytest
from src.utils.query_strategies import PassiveSelector, QBCSelector

### Mock model ###
class MockQBCModel:
    def __init__(self, vote_matrix, index):
        self.vote_df = pd.DataFrame(vote_matrix, index=index)
    
    def get_raw_ensemble_predictions(self, X_data):
        return self.vote_df.loc[X_data.index]

@pytest.fixture
def dummy_candidates():
    indices = range(10)
    df = pd.DataFrame({'Y': [0]*10}, index=indices)
    return df, indices

def test_passive_selector(dummy_candidates):
    df_cand, _ = dummy_candidates
    selector = PassiveSelector(random_state=42)
    
    res = selector.select(None, None, df_cand)
    assert res["IndexRecommendation"] in df_cand.index

def test_qbc_selector_uncertainty(dummy_candidates):
    """Tests that QBC picks the split vote (50/50) over the consensus vote (100/0)."""
    df_cand, indices = dummy_candidates
    n_trees = 10
    votes = np.zeros((10, n_trees))
    votes[1, :5] = 1 # Candidate 1 is split
    
    model = MockQBCModel(votes, indices)
    selector = QBCSelector(use_unique_trees=False)
    
    res = selector.select(model, None, df_cand)
    assert res["IndexRecommendation"] == 1