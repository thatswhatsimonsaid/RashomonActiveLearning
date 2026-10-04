### Libraries ###
import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.utils.models import PySORTDWrapper

@pytest.fixture
def dummy_data():
    X = pd.DataFrame({'f1': [0, 1, 0, 1], 'f2': [0, 0, 1, 1]})
    y = pd.Series([0, 1, 0, 1])
    return X, y

@pytest.fixture
def mock_sortd_classifier():
    """Patches SORTDClassifier in models.py (scoped to the test) with a mocked C++ model."""
    mock_cpp = MagicMock()
    mock_cpp.rashomon_set_size = 5
    mock_cpp.predict.return_value = np.array([0, 1, 0, 1])
    with patch("src.utils.models.SORTDClassifier", return_value=mock_cpp) as mock_cls:
        yield mock_cls, mock_cpp

def test_pysortd_config_mapping(mock_sortd_classifier):
    """Tests that arguments map to the internal config dict correctly."""
    model = PySORTDWrapper(
        regularization=0.02, 
        rashomon_multiplier=0.05, 
        max_depth=5
    )
    
    assert model.config["cost_complexity"] == 0.02
    assert model.config["rashomon_multiplier"] == 0.05
    assert model.config["max_depth"] == 5

def test_pysortd_fit_predict(dummy_data, mock_sortd_classifier):
    """Tests fit and predict logic with mocked backend."""
    X, y = dummy_data
    mock_cls, mock_cpp = mock_sortd_classifier

    # Construct inside the patch so the wrapper holds the mocked model
    model = PySORTDWrapper()
    assert model.model is mock_cpp

    model.fit(X, y)
    assert model.is_fitted_
    assert model.get_rashomon_size() == 5
    mock_cpp.fit.assert_called_once()

    preds = model.predict(X)
    assert len(preds) == 4
