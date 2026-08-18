### Summary ###
"""
Handles all data loading and preparation for the active learning simulations.
"""

### Libraries ###
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

### Load Data ###
def load_data(dataset_name: str, base_path: Path = Path("src/data")) -> pd.DataFrame:
    """Loads a pre-processed pickled DataFrame."""
    filepath = base_path / f"{dataset_name}.pkl"
    with open(filepath, 'rb') as file:
        data = pickle.load(file).dropna()
    return data

### Split data ###
def split_test_pool(
    df: pd.DataFrame,
    test_proportion: float,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the full dataset into a hold-out Test Set and a Working Pool.
    
    Args:
        df: The full dataset.
        test_proportion: Fraction of data to hold out for testing.
        random_state: Seed for reproducibility.
        
    Returns:
        df_test, df_pool_remaining
    """

    X = df.drop(columns="Y")
    y = df["Y"]
    X_pool, X_test, y_pool, y_test = train_test_split(X, y, test_size=test_proportion, random_state=random_state)
    df_test = pd.concat([y_test, X_test], axis=1)
    df_pool_remaining = pd.concat([y_pool, X_pool], axis=1)
    return df_test, df_pool_remaining

def get_random_initial_indices(y_train: np.ndarray, n_initial: int, random_state: int = 42) -> np.ndarray:
    """
    Selects n_initial indices randomly from the training set.
    Ensures that at least one example of each class is included if possible.
    """

    # Initialize #
    np.random.seed(random_state)
    classes = np.unique(y_train)
    indices = []

    # Stratified Pick #
    for c in classes:
        c_indices = np.where(y_train == c)[0]
        if len(c_indices) > 0:
            indices.append(np.random.choice(c_indices))

    # Fill the rest randomly
    remaining_needed = n_initial - len(indices)
    if remaining_needed > 0:
        all_indices = np.arange(len(y_train))
        available = np.setdiff1d(all_indices, indices)        
        if len(available) < remaining_needed:
            raise ValueError(f"Not enough data to pick {n_initial} samples.") 
        extra_indices = np.random.choice(available, size=remaining_needed, replace=False)
        indices.extend(extra_indices)

    # Return #
    return np.array(indices)