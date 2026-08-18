### Libraries ###
import pandas as pd
import numpy as np
import pytest
from src.utils.models import (
    RandomForestWrapper, 
    LogisticRegressionWrapper, 
    GreedyDecisionTreeWrapper,
    calculate_oracle_agreement
)

@pytest.fixture
def dummy_data():
    X = pd.DataFrame({'f1': np.random.rand(20), 'f2': np.random.rand(20)})
    y = pd.Series(np.random.randint(0, 2, size=20))
    return X, y

def test_rf_kwargs_filtering(dummy_data):
    """Tests that passing 'rashomon_threshold' doesn't crash RF."""
    X, y = dummy_data
    model = RandomForestWrapper(n_estimators=5, rashomon_threshold=0.05) 
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 20

def test_greedy_tree_kwargs_filtering(dummy_data):
    """Tests that passing 'rashomon_threshold' doesn't crash GreedyTree."""
    X, y = dummy_data
    model = GreedyDecisionTreeWrapper(max_depth=3, rashomon_threshold=0.05)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 20

def test_oracle_agreement(dummy_data):
    """Tests the oracle agreement logic."""
    X, y = dummy_data
    model_a = GreedyDecisionTreeWrapper(random_state=42)
    model_b = GreedyDecisionTreeWrapper(random_state=42) # Identical
    
    model_a.fit(X, y)
    model_b.fit(X, y)
    
    df_eval = pd.concat([y.rename("Y"), X], axis=1)
    agreement = calculate_oracle_agreement(model_a, model_b, df_eval)
    assert agreement == 1.0