### LIBRARIES ###
import pandas as pd
import numpy as np
import pytest

### TEST THIS CLASS ###
from src.utils.models import RandomForestWrapper

### FIXTURES ###
@pytest.fixture
def dummy_classification_data():
    """Provides a simple dataset for training and prediction."""
    X = pd.DataFrame({'feature1': np.random.rand(20), 'feature2': np.random.rand(20)})
    y = pd.Series(np.random.randint(0, 2, size=20))
    return X, y

### TEST FUNCTIONS ###
def test_randomforest_fit_predict(dummy_classification_data):
    """Tests that the RandomForestWrapper can fit and predict."""
    X_train, y_train = dummy_classification_data
    model = RandomForestWrapper(n_estimators=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_train)

def test_randomforest_get_raw_predictions(dummy_classification_data):
    """Tests the get_raw_ensemble_predictions method for RandomForest."""
    X_train, y_train = dummy_classification_data
    model = RandomForestWrapper(n_estimators=5)
    model.fit(X_train, y_train)
    raw_preds = model.get_raw_ensemble_predictions(X_train)
    assert isinstance(raw_preds, pd.DataFrame)
    assert raw_preds.shape[0] == len(X_train)