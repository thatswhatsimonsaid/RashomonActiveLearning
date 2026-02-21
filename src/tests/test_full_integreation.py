### Libraries ###
import pytest
import pandas as pd
import numpy as np
from src.utils.models import RandomForestWrapper, PySORTDWrapper
from src.utils.query_strategies import PassiveSelector, QBCSelector
from src.utils.learning_procedure import SimulationConfig, run_learning_procedure

### Tiny dataset ###
@pytest.fixture
def tiny_binary_dataset():
    """
    Creates a tiny, fast dataset (N=50, D=5) for integration testing.
    Returns: df_train, df_candidate, df_test, df_full
    """
    np.random.seed(42)
    N = 50
    D = 5
    X = np.random.randint(0, 2, size=(N, D))
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
    
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(D)])
    df["Y"] = y
    
    df_train = df.iloc[:5].copy()
    df_test = df.iloc[5:20].copy()
    df_candidate = df.iloc[20:].copy()
    
    return df_train, df_candidate, df_test, df

### Run simulation ###
def run_short_simulation(selector_model, predictor_model, selector, dataset_tuple):
    """Helper to run a short 5-step active learning loop."""

    ## Set up ##
    df_train, df_candidate, df_test, df_full = dataset_tuple    
    df_candidate_small = df_candidate.iloc[:5].copy()

    ## Oracle ##
    oracle = predictor_model.__class__(**predictor_model.__dict__.get("config", {}))
    if isinstance(predictor_model, RandomForestWrapper):
        oracle = RandomForestWrapper(n_estimators=10, random_state=42)
    elif isinstance(predictor_model, PySORTDWrapper):
        oracle = PySORTDWrapper(**predictor_model.config)
    oracle.fit(df_full.drop(columns="Y"), df_full["Y"])

    ## Configuration ##
    config = SimulationConfig(
        selector_model=selector_model,
        predictor_model=predictor_model,
        oracle_model=oracle,
        selector=selector,
        df_train=df_train,
        df_candidate=df_candidate_small,
        df_test=df_test
    )
    
    ## Run ##
    results = run_learning_procedure(config)
    return results

### Test passive learning ###
def test_method_1_passive_rf(tiny_binary_dataset):
    """
    METHOD 1: Random Sampling + Random Forest
    """
    print("\n[TEST] Method 1: Passive + RF")
    
    # Setup
    sel_model = RandomForestWrapper(n_estimators=10, random_state=42)
    pred_model = RandomForestWrapper(n_estimators=10, random_state=42)
    selector = PassiveSelector(random_state=42)
    
    # Execute
    results = run_short_simulation(sel_model, pred_model, selector, tiny_binary_dataset)
    
    # Verify
    assert len(results.accuracy_history) == 6  # 5 steps + 1 initial
    assert results.selection_history[0] is not None
    print("  -> Method 1 Passed!")

### Test QBC-RF ###
def test_method_2_qbc_rf(tiny_binary_dataset):
    """
    METHOD 2: QBC + Random Forest (Uncertainty Sampling)
    """
    print("\n[TEST] Method 2: QBC + RF")
    
    # Setup
    sel_model = RandomForestWrapper(n_estimators=10, random_state=42)
    pred_model = RandomForestWrapper(n_estimators=10, random_state=42)
    selector = QBCSelector(use_unique_trees=False) # Standard QBC
    
    # Execute
    results = run_short_simulation(sel_model, pred_model, selector, tiny_binary_dataset)
    
    # Verify
    assert len(results.accuracy_history) == 6
    assert any(len(e) > 0 for e in results.entropy_history)
    print("  -> Method 2 Passed!")

### Test UNREAL ###
def test_method_3_unreal_pysortd(tiny_binary_dataset):
    """
    METHOD 3: UNREAL (QBC + PySORTD)
    """
    print("\n[TEST] Method 3: UNREAL (PySORTD)")
    
    # Setup
    params = {
        "regularization": 0.001, 
        "rashomon_multiplier": 0.1, 
        "max_depth": 3
    }
    sel_model = PySORTDWrapper(**params)
    pred_model = PySORTDWrapper(**params)
    selector = QBCSelector(use_unique_trees=True) # Unique Tree QBC
    
    # Execute
    results = run_short_simulation(sel_model, pred_model, selector, tiny_binary_dataset)
    
    # Verify
    assert len(results.accuracy_history) == 6
    assert len(results.rashomon_size_history) == 6
    assert all(r >= 1 for r in results.rashomon_size_history)
    print("  -> Method 3 Passed!")