### LIBRARIES ###
import pandas as pd
import numpy as np
import torch
import pytest
from src.utils.models import BNNWrapper

### FIXTURES ###
@pytest.fixture
def dummy_classification_data():
    """Provides a simple dataset for training and prediction."""
    X = pd.DataFrame({'feature1': np.random.rand(20), 'feature2': np.random.rand(20)})
    y = pd.Series(np.random.randint(0, 2, size=20))
    return X, y

### TEST FUNCTIONS ###
def test_bnn_fit_predict(dummy_classification_data):
    """Tests that the BNNWrapper can fit and predict."""
    X_train, y_train = dummy_classification_data
    model = BNNWrapper(epochs=2)
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_train)

def test_bnn_get_log_probas(dummy_classification_data):
    """Tests the predict_log_proba_ensemble method for BNN."""
    X_train, y_train = dummy_classification_data
    model = BNNWrapper(epochs=2)
    n_samples = 10
    model.fit(X_train, y_train)
    log_probas = model.predict_log_proba_ensemble(X_train, n_samples=n_samples)
    assert isinstance(log_probas, torch.Tensor)
    assert log_probas.shape == (len(X_train), n_samples, len(y_train.unique()))