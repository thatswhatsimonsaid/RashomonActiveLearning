### LIBRARIES ###
import pandas as pd
import numpy as np
import pytest

### TEST THIS CLASS ###
from src.utils.models import TreeFarmsWrapper

### FIXTURES ###
@pytest.fixture
def binary_classification_data():
    """Provides a simple, fast dataset with binary features."""
    X = pd.DataFrame({
        'feature1': np.random.randint(0, 2, size=10),
        'feature2': np.random.randint(0, 2, size=10)
    })
    y = pd.Series(np.random.randint(0, 2, size=10))
    return X, y

### TEST FUNCTIONS ###
def test_treefarms_fit_predict(binary_classification_data):
    """
    Tests that the TreeFarmsWrapper can fit and predict using binary data.
    """
    # 1. ARRANGE
    X_train, y_train = binary_classification_data
    model = TreeFarmsWrapper(regularization=0.1, rashomon_threshold=0.1)

    # 2. ACT
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)

    # 3. ASSERT
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_train)

def test_treefarms_get_raw_predictions(binary_classification_data):
    """
    Tests the get_raw_ensemble_predictions method for TreeFarms using binary data.
    """
    # 1. ARRANGE
    X_train, y_train = binary_classification_data
    model = TreeFarmsWrapper(regularization=0.1, rashomon_threshold=0.1)

    # 2. ACT
    model.fit(X_train, y_train)
    raw_preds = model.get_raw_ensemble_predictions(X_train)

    # 3. ASSERT
    assert isinstance(raw_preds, pd.DataFrame)
    assert raw_preds.shape[0] == len(X_train)