### Libraries ###
import pandas as pd
import numpy as np
import pytest
from src.utils.data_handler import split_test_pool, get_random_initial_indices

def test_split_test_pool_correct_sizes():
    """Tests that split_test_pool partitions the DataFrame correctly."""
    # 1. ARRANGE
    total_rows = 100
    df = pd.DataFrame({
        'Y': range(total_rows),
        'feature1': range(total_rows)
    })
    test_prop = 0.20
    
    # 2. ACT
    df_test, df_pool = split_test_pool(df, test_proportion=test_prop, random_state=42)

    # 3. ASSERT
    assert len(df_test) == 20
    assert len(df_pool) == 80

def test_split_test_pool_no_overlap():
    """Tests that there is no data leakage between sets."""
    df = pd.DataFrame({'Y': range(100), 'feature1': range(100)})
    df_test, df_pool = split_test_pool(df, 0.20, random_state=42)
    
    test_indices = set(df_test.index)
    pool_indices = set(df_pool.index)
    assert test_indices.isdisjoint(pool_indices)

def test_get_random_initial_indices_stratified():
    """Tests that we get at least one sample per class."""
    y = np.array([0]*50 + [1]*50)
    indices = get_random_initial_indices(y, n_initial=10, random_state=42)
    
    assert len(indices) == 10
    selected_y = y[indices]
    assert 0 in selected_y
    assert 1 in selected_y