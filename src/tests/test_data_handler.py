### LIBRARIES ###
import pandas as pd
import pytest

### TEST THIS FUNCTION ###
from src.utils.data_handler import split_data

### TEST FUNCTIONS ###
def test_split_data_correct_sizes():
    """
    Tests that split_data partitions the DataFrame into the correct proportions.
    """
    # 1. ARRANGE
    total_rows = 100
    df = pd.DataFrame({
        'Y': range(total_rows),
        'feature1': range(total_rows)
    })
    
    test_prop = 0.2
    cand_prop_of_remainder = 0.5
    
    # Calculate expected sizes
    expected_test_size = int(total_rows * test_prop) # 100 * 0.2 = 20
    remaining_size = total_rows - expected_test_size # 100 - 20 = 80
    expected_cand_size = int(remaining_size * cand_prop_of_remainder) # 80 * 0.5 = 40
    expected_train_size = remaining_size - expected_cand_size # 80 - 40 = 40

    # 2. ACT
    df_train, df_test, df_candidate = split_data(
        df,
        test_proportion=test_prop,
        candidate_proportion_of_remainder=cand_prop_of_remainder
    )

    # 3. ASSERT
    assert len(df_train) == expected_train_size
    assert len(df_test) == expected_test_size
    assert len(df_candidate) == expected_cand_size

def test_split_data_no_overlap():
    """
    Tests that there is no data leakage (no overlapping indices) between the
    train, test, and candidate sets.
    """
    # 1. ARRANGE
    df = pd.DataFrame({'Y': range(100), 'feature1': range(100)})

    # 2. ACT
    df_train, df_test, df_candidate = split_data(df, 0.2, 0.5)
    train_indices = set(df_train.index)
    test_indices = set(df_test.index)
    candidate_indices = set(df_candidate.index)

    # # 3. ASSERT: 
    assert train_indices.isdisjoint(test_indices)
    assert train_indices.isdisjoint(candidate_indices)
    assert test_indices.isdisjoint(candidate_indices)