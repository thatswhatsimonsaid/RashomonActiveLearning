### Libraries ###
import pandas as pd
import numpy as np
import pytest
from src.utils.query_strategies import (
    PassiveSelector,
    QBCSelector,
    PatternDisambiguationSelector,
    extract_classification_patterns,
)

### Mock model ###
class MockQBCModel:
    def __init__(self, vote_matrix, index, train_matrix=None, train_index=None, losses=None):
        self.vote_df = pd.DataFrame(vote_matrix, index=index)
        if train_matrix is None:
            train_matrix = np.zeros((1, vote_matrix.shape[1]))
            train_index = [0]
        self.train_df = pd.DataFrame(train_matrix, index=train_index)
        self.losses = (
            np.zeros(vote_matrix.shape[1]) if losses is None else np.asarray(losses, dtype=float)
        )

    def get_raw_ensemble_predictions(self, X_data):
        # Prefer candidate matrix; fall back to train matrix by index membership
        if set(X_data.index).issubset(set(self.vote_df.index)):
            return self.vote_df.loc[X_data.index]
        return self.train_df.loc[X_data.index]

    def get_ensemble_losses(self, X_train, y_train):
        return self.losses


@pytest.fixture
def dummy_candidates():
    indices = range(10)
    df = pd.DataFrame({"Y": [0] * 10}, index=indices)
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
    votes[1, :5] = 1  # Candidate 1 is split

    model = MockQBCModel(votes, indices)
    selector = QBCSelector(use_unique_trees=False)

    res = selector.select(model, pd.DataFrame({"Y": [0]}, index=[0]), df_cand)
    assert res["IndexRecommendation"] == 1


def test_extract_classification_patterns_collapses_duplicates():
    # 3 trees, 2 samples: trees 0 and 2 identical
    preds = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
        ]
    )
    unique, inverse, masses = extract_classification_patterns(
        preds, weights=np.array([0.2, 0.5, 0.3])
    )
    assert unique.shape[0] == 2
    assert inverse[0] == inverse[2]
    assert inverse[0] != inverse[1]
    assert pytest.approx(masses[inverse[0]]) == 0.5
    assert pytest.approx(masses[inverse[1]]) == 0.5


def test_qbc_use_unique_trees_collapses_before_entropy():
    """
    With duplicates, unique-tree QBC should behave like a 2-member committee
    even if raw tree count is large.
    """
    cand_idx = [10, 11]
    df_cand = pd.DataFrame({"Y": [0, 0]}, index=cand_idx)
    df_train = pd.DataFrame({"Y": [0]}, index=[0])

    # 4 trees but only 2 patterns on the candidate pool:
    # pattern A: [0,1], pattern B: [1,0]; duplicated twice each.
    votes = np.array(
        [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ]
    )
    model = MockQBCModel(votes, cand_idx)
    selector = QBCSelector(beta=0.0, use_unique_trees=True)
    res = selector.select(model, df_train, df_cand)

    # Both candidates are perfectly split across the two patterns → equal entropy;
    # just check ECS reflects 2 unique patterns, not 4 trees.
    assert pytest.approx(selector.effective_committee_size_, rel=1e-6) == 2.0
    assert res["IndexRecommendation"] in cand_idx


def test_pattern_disambiguation_prefers_splitting_duplicates():
    """
    Candidate 0: 50/50 vote, but only re-states an existing train disagreement
    between already-distinct trees → does not raise pattern count.

    Candidate 1: milder 25/75 vote, but splits a previously identical pair →
    raises the number of distinct extended patterns.

    Vote entropy would prefer candidate 0; pattern disambiguation prefers 1.
    """
    cand_idx = [10, 11]
    train_idx = [0, 1]
    df_cand = pd.DataFrame({"Y": [0, 0]}, index=cand_idx)
    df_train = pd.DataFrame({"Y": [0, 1]}, index=train_idx)

    # 4 trees. On train (2 points):
    #   T0,T1 share pattern [0,0]
    #   T2 pattern [0,1]
    #   T3 pattern [1,0]
    train_preds = np.array(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
    )

    # On candidates:
    # cand0: preds [0,1,0,1] — 50/50, but T0/T1 already get different labels? 
    #   Wait we need cand0 to NOT split T0/T1 and have high entropy from T2/T3.
    # cand0: T0=0, T1=0, T2=1, T3=1 → 50/50, does not split the duplicate pair
    # cand1: T0=0, T1=1, T2=0, T3=0 → 25/75, splits the duplicate pair
    cand_preds = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 0],
        ]
    )

    model = MockQBCModel(
        vote_matrix=cand_preds,
        index=cand_idx,
        train_matrix=train_preds,
        train_index=train_idx,
    )

    qbc = QBCSelector(beta=0.0, use_unique_trees=False)
    qbc_res = qbc.select(model, df_train, df_cand)
    assert qbc_res["IndexRecommendation"] == 10  # cand0, higher vote entropy

    pattern_sel = PatternDisambiguationSelector(beta=0.0, score_mode="count")
    pat_res = pattern_sel.select(model, df_train, df_cand)
    assert pat_res["IndexRecommendation"] == 11  # cand1, splits duplicates

    # Baseline distinct patterns on train: 3. Extending by cand1 yields 4;
    # extending by cand0 stays at 3.
    assert pat_res["AllEntropies"].loc[10] == 3
    assert pat_res["AllEntropies"].loc[11] == 4
