### LIBRARIES ###
import pandas as pd
import numpy as np
import torch
import pytest

### TEST THESE CLASSES ###
from src.utils.models import (
    TreeFarmsWrapper,
    RandomForestWrapper,
    GaussianProcessWrapper,
    BNNWrapper
)

### FIXTURES ###
@pytest.fixture
def dummy_classification_data():
    """Provides a simple dataset for training and prediction."""
    X = pd.DataFrame({
        'feature1': np.random.rand(20),
        'feature2': np.random.rand(20)
    })
    y = pd.Series(np.random.randint(0, 2, size=20))
    return X, y

### TEST PARAMETERS ###
@pytest.mark.parametrize("model_class, model_params", [
    (TreeFarmsWrapper, {"regularization": 0.01, "rashomon_threshold": 0.05}),
    (RandomForestWrapper, {"n_estimators": 5}),
    (GaussianProcessWrapper, {}),
    (BNNWrapper, {"epochs": 2})
])

### TEST FUNCTIONS ###
def test_model_fit_predict(model_class, model_params, dummy_classification_data):
    """
    A generic test that runs for ALL models.
    It verifies that each model can be initialized, fit, and can predict.
    """
    # 1. ARRANGE
    X_train, y_train = dummy_classification_data
    model = model_class(**model_params)

    # 2. ACT
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)

    # 3. ASSERT
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_train)

### TESTS FOR SPECIFIC METHODS ###
@pytest.mark.parametrize("ensemble_model_class, model_params", [
    (TreeFarmsWrapper, {"regularization": 0.01, "rashomon_threshold": 0.05}),
    (RandomForestWrapper, {"n_estimators": 5})
])
def test_ensemble_models_get_raw_predictions(ensemble_model_class, model_params, dummy_classification_data):
    """
    Tests the get_raw_ensemble_predictions method for ONLY the ensemble models.
    """
    # 1. ARRANGE
    X_train, y_train = dummy_classification_data
    model = ensemble_model_class(**model_params)
    
    # 2. ACT
    model.fit(X_train, y_train)
    raw_preds = model.get_raw_ensemble_predictions(X_train)

    # 3. ASSERT
    assert isinstance(raw_preds, pd.DataFrame)
    assert raw_preds.shape[0] == len(X_train)
    assert raw_preds.shape[1] > 0 # Should have at least one "tree"

### TESTS FOR PROBABILISTIC MODELS ###
@pytest.mark.parametrize("prob_model_class, model_params", [
    (GaussianProcessWrapper, {}),
    (BNNWrapper, {"epochs": 2})
])
def test_probabilistic_models_get_log_probas(prob_model_class, model_params, dummy_classification_data):
    """
    Tests the predict_log_proba_ensemble method for ONLY the probabilistic models.
    """
    # 1. ARRANGE
    X_train, y_train = dummy_classification_data
    model = prob_model_class(**model_params)
    n_samples = 10

    # 2. ACT
    model.fit(X_train, y_train)
    log_probas = model.predict_log_proba_ensemble(X_train, n_samples=n_samples)

    # 3. ASSERT
    assert isinstance(log_probas, torch.Tensor)
    assert log_probas.shape == (len(X_train), n_samples, len(y_train.unique()))