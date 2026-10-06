### Libraries ###
import numpy as np
import pandas as pd
import pytest

from sklearn.tree import DecisionTreeRegressor

from src.utils.calibration import calibrate_hyperparameters
from src.utils.data_handler import get_random_initial_indices
from src.utils.models import GreedyRegressionTreeWrapper, RandomForestRegressorWrapper
from src.utils.preprocess_data import generate_synthetic_regression
from src.utils.query_strategies import RegressionQBCSelector, RegressionUncertaintySelector
from src.utils.learning_procedure import SimulationConfig, run_learning_procedure


class FakeRegressionCommittee:
    def __init__(self, predictions, losses, index):
        self.predictions = pd.DataFrame(predictions, index=index)
        self.losses = np.asarray(losses, dtype=float)

    def get_raw_ensemble_predictions(self, X_data):
        return self.predictions.loc[X_data.index]

    def get_ensemble_losses(self, X_train, y_train):
        return self.losses


def test_regression_qbc_queries_the_disagreement():
    index = [0, 1, 2]
    predictions = np.array([
        [0.0, 4.0],
        [1.0, 1.0],
        [2.0, 2.2],
    ])
    model = FakeRegressionCommittee(predictions, losses=[0.0, 0.0], index=index)
    pool = pd.DataFrame({"Y": [0.0, 0.0, 0.0]}, index=index)
    labeled = pool.iloc[:1].copy()

    result = RegressionQBCSelector(beta=0.0).select(model, labeled, pool)
    assert result["IndexRecommendation"] == 0
    assert result["AllEntropies"].loc[0] > result["AllEntropies"].loc[1]


def test_regression_uncertainty_uses_leaf_variance():
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 40)
    noise = np.where(x > 0, rng.normal(0, 1.5, size=40), rng.normal(0, 0.05, size=40))
    frame = pd.DataFrame({"X0": x, "Y": x + noise})
    model = GreedyRegressionTreeWrapper(max_depth=1, random_state=0)
    model.fit(frame[["X0"]], frame["Y"])

    result = RegressionUncertaintySelector().select(model, frame.iloc[:10], frame)
    chosen = frame.loc[result["IndexRecommendation"], "X0"]
    assert chosen > 0


def test_initial_regression_sample_is_not_stratified_by_value():
    y = np.array([0.1, 0.2, 0.3, 0.4, 10.0, 11.0])
    chosen = get_random_initial_indices(y, n_initial=3, random_state=1, stratify=False)
    assert len(chosen) == 3
    assert len(set(chosen)) == 3


def test_short_regression_loop_records_squared_error():
    frame = generate_synthetic_regression(n_samples=40, n_features=3, alpha=0.0, phi=0.1, random_state=0)
    train = frame.iloc[:8].copy()
    candidate = frame.iloc[8:13].copy()
    test = frame.iloc[13:].copy()
    selector_model = RandomForestRegressorWrapper(n_estimators=10, random_state=0, max_depth=3)
    predictor = GreedyRegressionTreeWrapper(max_depth=2, random_state=0)
    oracle = GreedyRegressionTreeWrapper(max_depth=2, random_state=0)
    oracle.fit(frame.drop(columns="Y"), frame["Y"])

    results = run_learning_procedure(SimulationConfig(
        selector_model=selector_model,
        predictor_model=predictor,
        oracle_model=oracle,
        selector=RegressionQBCSelector(beta=0.0),
        df_train=train,
        df_candidate=candidate,
        df_test=test,
        task="regression",
    ))

    assert len(results.mse_history) == 6
    assert len(results.rmse_history) == 6
    assert len(results.r2_history) == 6
    assert results.accuracy_history == []
    assert all(np.isfinite(value) for value in results.mse_history)
    assert results.selection_history[0] is not None


def test_pysortd_regressor_enumerates_and_learns():
    pytest.importorskip("pysortd")
    from src.utils.models import PySORTDRegressorWrapper
    frame = generate_synthetic_regression(n_samples=60, n_features=4, alpha=0.0, phi=0.2, random_state=1)
    params = {
        "regularization": 0.01,
        "rashomon_multiplier": 0.10,
        "max_depth": 2,
        "max_num_trees": 20,
        "time_limit": 10,
        "n_thresholds": 3,
    }
    selector_model = PySORTDRegressorWrapper(**params)
    predictor = PySORTDRegressorWrapper(**params)
    selector_model.fit(frame.iloc[:25].drop(columns="Y"), frame.iloc[:25]["Y"])
    assert selector_model.get_rashomon_size() >= 1
    committee = selector_model.get_raw_ensemble_predictions(frame.iloc[:5].drop(columns="Y"))
    best = selector_model.predict(frame.iloc[:5].drop(columns="Y"))
    assert committee.shape[0] == 5
    assert committee.shape[1] == selector_model.get_rashomon_size()
    matches_best = [
        np.allclose(committee.iloc[:, i].to_numpy(), best, rtol=1e-5, atol=1e-5)
        for i in range(committee.shape[1])
    ]
    assert any(matches_best)

    oracle = PySORTDRegressorWrapper(**params)
    oracle.fit(frame.drop(columns="Y"), frame["Y"])
    results = run_learning_procedure(SimulationConfig(
        selector_model=PySORTDRegressorWrapper(**params),
        predictor_model=predictor,
        oracle_model=oracle,
        selector=RegressionQBCSelector(beta=0.0),
        df_train=frame.iloc[:12].copy(),
        df_candidate=frame.iloc[12:16].copy(),
        df_test=frame.iloc[16:].copy(),
        task="regression",
    ))
    assert len(results.mse_history) == 5
    assert len(results.rashomon_size_history) == 5
    assert all(size >= 1 for size in results.rashomon_size_history)

    train_X = frame.iloc[:25].drop(columns="Y")
    train_y = frame.iloc[:25]["Y"]
    committee_train = selector_model.get_raw_ensemble_predictions(train_X)
    best_train = selector_model.predict(train_X)
    match_idx = [
        i for i in range(committee_train.shape[1])
        if np.allclose(committee_train.iloc[:, i].to_numpy(), best_train, rtol=1e-5, atol=1e-5)
    ]
    assert match_idx
    losses = selector_model.get_ensemble_losses(train_X, train_y)
    assert np.min(losses[match_idx]) <= np.min(losses) + 1e-6

    optimal_only = PySORTDRegressorWrapper(**{**params, "rashomon_multiplier": 0.0})
    optimal_only.fit(train_X, train_y)
    assert optimal_only.get_rashomon_size() >= 1
    optimal_loss = optimal_only.get_ensemble_losses(train_X, train_y)
    assert np.min(optimal_loss) <= np.min(losses) + 1e-5


class _RashomonStub:
    """One CART tree presented as a two-member committee, for calibration only."""

    def __init__(self, max_depth=1, regularization=0.01, rashomon_multiplier=0.1, random_state=0, **kwargs):
        self.rashomon_multiplier = rashomon_multiplier
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_rashomon_size(self):
        return 2

    def get_raw_ensemble_predictions(self, X):
        pred = np.asarray(self.predict(X), dtype=float)
        disagreement = np.linspace(0.0, 1.0, len(pred))
        return pd.DataFrame({"tree_0": pred, "tree_1": pred + disagreement}, index=X.index)

    def get_ensemble_losses(self, X, y):
        return np.array([0.1, 0.4])


def test_regression_calibration_keeps_epsilon_and_beta():
    frame = generate_synthetic_regression(n_samples=24, n_features=3, alpha=0.0, phi=0.2, random_state=0)
    pilot = frame.iloc[:10]
    fixed = calibrate_hyperparameters(
        pilot,
        GreedyRegressionTreeWrapper,
        base_params={"random_state": 0, "rashomon_threshold": 0.10, "beta": 0.0},
        depth_grid=[1, 2],
        lambda_grid=[0.01],
        task="regression",
    )
    assert fixed["max_depth"] in (1, 2)
    assert fixed["rashomon_epsilon_adder"] == 0.0
    assert fixed["beta"] == 0.0
    assert np.isfinite(fixed["pilot_loss_proxy"])
    assert fixed["task"] == "regression"

    tuned = calibrate_hyperparameters(
        pilot,
        _RashomonStub,
        base_params={"random_state": 0, "rashomon_threshold": 0.10, "beta": "calibrated"},
        depth_grid=[1],
        lambda_grid=[0.01],
        beta_grid=[1.0, 10.0],
        task="regression",
    )
    assert tuned["rashomon_epsilon_adder"] == 0.10
    assert tuned["beta"] in (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)
