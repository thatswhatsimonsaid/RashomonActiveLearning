"""
Handles all data loading and preparation for the active learning simulations.
"""

### LIBRARIES ###
import os
import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

### LOAD DATA ###
def load_data(dataset_name: str, base_path: Path = Path("data/processed")) -> pd.DataFrame:
    """Loads a pre-processed pickled DataFrame.

    This function assumes it is run from the project's root directory.

    Args:
        dataset_name: The name of the dataset file (without the .pkl extension).
        base_path: The directory containing the processed data.

    Returns:
        The loaded pandas DataFrame with any rows containing NA values dropped.
        
    Raises:
        FileNotFoundError: If the specified dataset file cannot be found.
    """
    filepath = base_path / f"{dataset_name}.pkl"
    
    if not filepath.exists():
        raise FileNotFoundError(f"File '{filepath}' not found. Make sure you are in the project root.")
        
    with open(filepath, 'rb') as file:
        data = pickle.load(file).dropna()
        
    return data

### TRAIN TEST CANDIDATE SPLIT ###
def split_data(
    df: pd.DataFrame,
    test_proportion: float,
    candidate_proportion_of_remainder: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a DataFrame into training, test, and candidate sets.

    The split is performed in two stages:
    1. The original DataFrame is split into a test set and a combined train/candidate set.
    2. The train/candidate set is then split into the initial training set and the candidate set.

    Args:
        df: The original DataFrame, assuming the target column is named 'Y'.
        test_proportion: The proportion of the original data to reserve for the test set (e.g., 0.2).
        candidate_proportion_of_remainder: The proportion of the *remaining* data (after the test
                                           split) to allocate to the candidate set. For example, if
                                           test_proportion is 0.2, the remainder is 80% of the data.
                                           A value of 0.8 here would make the candidate set 0.8 * 0.8 = 64%
                                           of the original data.
    Returns:
        A tuple containing the initial training, test, and candidate DataFrames.
    """
    ## 1. Separate features (X) and target (y) ##
    X = df.drop(columns="Y")
    y = df["Y"]

    ## 2. First split: separate the test set ##
    X_train_cand, X_test, y_train_cand, y_test = train_test_split(
        X, y, test_size=test_proportion, random_state=42 # Add random_state for reproducibility
    )

    ## 3. Second split: separate initial training and candidate sets ##
    X_train, X_cand, y_train, y_cand = train_test_split(
        X_train_cand, y_train_cand, test_size=candidate_proportion_of_remainder, random_state=42
    )

    ## 4. Reconstruct DataFrames ##
    df_train = pd.concat([y_train, X_train], axis=1)
    df_test = pd.concat([y_test, X_test], axis=1)
    df_candidate = pd.concat([y_cand, X_cand], axis=1)

    return df_train, df_test, df_candidate