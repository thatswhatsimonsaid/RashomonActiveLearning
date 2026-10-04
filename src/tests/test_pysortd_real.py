### Summary ###
"""
Light tests against the real (compiled) pysortd backend.
Checks the sklearn _validate_data compatibility patch in src.utils.models
and basic Rashomon set behaviour. Skipped if pysortd is not installed.
"""

### Libraries ###
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pysortd")
import src.utils.models  # noqa: F401  (installs the compatibility patch)
from pysortd import SORTDClassifier, SORTDRegressor
from pysortd.base import BaseSORTDSolver
from src.utils.models import PySORTDWrapper

### Data ###
@pytest.fixture
def binary_X():
    rng = np.random.default_rng(0)
    return rng.integers(0, 2, size=(120, 6)).astype(np.intc), rng

def test_validate_data_available_on_base(binary_X):
    """The patch (or sklearn itself) must provide _validate_data to every pysortd estimator."""
    assert hasattr(BaseSORTDSolver, "_validate_data")
    assert hasattr(SORTDClassifier, "_validate_data")
    assert hasattr(SORTDRegressor, "_validate_data")

def test_sortd_classifier_fit_predict(binary_X):
    X, rng = binary_X
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(np.intc)
    model = SORTDClassifier("cost-complex-accuracy", max_depth=2, cost_complexity=0.01,
                            rashomon_multiplier=0.1, max_num_trees=100, time_limit=30)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),)
    assert np.mean(preds == y) == 1.0
    assert model.rashomon_set_size >= 1

def test_sortd_regressor_fit_predict(binary_X):
    X, rng = binary_X
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + rng.normal(0, 0.1, len(X))
    model = SORTDRegressor(max_depth=2, cost_complexity=0.01, rashomon_multiplier=0.1,
                           max_num_trees=1000, time_limit=30)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),)
    assert np.issubdtype(preds.dtype, np.floating)
    assert np.mean((preds - y) ** 2) < 0.1
    # Any tree in the Rashomon set can be used for prediction
    tree = model.get_tree_n(model.rashomon_set_size - 1)
    assert model.predict(X, tree=tree).shape == (len(y),)

def test_rashomon_set_grows_with_multiplier(binary_X):
    X, rng = binary_X
    y = X[:, 0] + 0.5 * X[:, 2] + rng.normal(0, 0.5, len(X))
    sizes = []
    for mult in [0.0, 0.1, 0.5]:
        model = SORTDRegressor(max_depth=2, cost_complexity=0.01, rashomon_multiplier=mult,
                               max_num_trees=10**6, time_limit=30)
        model.fit(X, y)
        sizes.append(model.rashomon_set_size)
    assert sizes[0] >= 1
    assert sizes[0] <= sizes[1] <= sizes[2]
    assert sizes[2] > sizes[0]

def test_pysortd_wrapper_real_backend(binary_X):
    """PySORTDWrapper end-to-end: ensemble predictions and losses line up with the set size."""
    X, rng = binary_X
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
    X_df = pd.DataFrame(X.astype(int), columns=[f"V{i}" for i in range(X.shape[1])])
    y_s = pd.Series(y)
    model = PySORTDWrapper(regularization=0.01, rashomon_multiplier=0.5, max_num_trees=200, max_depth=2)
    model.fit(X_df, y_s)
    n = model.get_rashomon_size()
    assert n >= 1
    assert model.get_raw_ensemble_predictions(X_df).shape == (len(y), n)
    assert model.get_ensemble_losses(X_df, y_s).shape == (n,)
