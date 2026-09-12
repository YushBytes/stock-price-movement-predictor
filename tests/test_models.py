"""Phase 5 tests for Logistic Regression training and training-only scaling.

Uses small, deterministic synthetic data purely for unit testing -- never
as the actual project dataset (the real SPY data is only ever produced by
src.data_loader.fetch_ohlcv).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import train_logistic_regression  # noqa: E402


def _linearly_separable_data(n=40, seed=0):
    rng = np.random.RandomState(seed)
    x = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series((x["f1"] + x["f2"] > 0).astype(int))
    return x, y


# --- Model tests ---------------------------------------------------------------

def test_logistic_regression_trains_successfully():
    x_train, y_train = _linearly_separable_data()
    model = train_logistic_regression(x_train, y_train, random_seed=42)
    assert hasattr(model, "coef_")


def test_logistic_regression_output_is_binary():
    x_train, y_train = _linearly_separable_data()
    model = train_logistic_regression(x_train, y_train, random_seed=42)
    predictions = model.predict(x_train)
    assert set(predictions.tolist()) <= {0, 1}


def test_prediction_length_matches_input_length():
    x_train, y_train = _linearly_separable_data(n=40)
    model = train_logistic_regression(x_train, y_train, random_seed=42)
    x_new = x_train.iloc[:15]
    predictions = model.predict(x_new)
    assert len(predictions) == 15


def test_model_is_deterministic_given_same_seed():
    x_train, y_train = _linearly_separable_data()
    model_a = train_logistic_regression(x_train, y_train, random_seed=7)
    model_b = train_logistic_regression(x_train, y_train, random_seed=7)
    np.testing.assert_allclose(model_a.coef_, model_b.coef_)


def test_train_logistic_regression_rejects_empty_input():
    with pytest.raises(ValueError):
        train_logistic_regression(pd.DataFrame(), pd.Series([], dtype=int), random_seed=42)


def test_train_logistic_regression_rejects_mismatched_lengths():
    x_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y_train = pd.Series([0, 1])
    with pytest.raises(ValueError):
        train_logistic_regression(x_train, y_train, random_seed=42)


def test_model_does_not_train_with_target_as_a_feature():
    # Guard against accidentally including the target column in X: a
    # feature-column list construction step (as used in the notebook)
    # must never select "Target" into the training matrix.
    combined = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "Target": [0, 1, 0, 1]})
    feature_columns = [c for c in combined.columns if c != "Target"]
    assert "Target" not in feature_columns
    x_train = combined[feature_columns]
    y_train = combined["Target"]
    model = train_logistic_regression(x_train, y_train, random_seed=42)
    assert "Target" not in x_train.columns


# --- Scaler tests (training-only fitting) --------------------------------------

def test_scaler_fits_on_training_data():
    x_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = StandardScaler()
    scaler.fit(x_train)
    assert scaler.mean_[0] == pytest.approx(3.0)


def test_validation_and_test_can_be_transformed_without_refitting():
    x_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    x_val = pd.DataFrame({"f1": [10.0, 20.0]})
    x_test = pd.DataFrame({"f1": [100.0, 200.0]})

    scaler = StandardScaler()
    scaler.fit(x_train)
    mean_after_fit = scaler.mean_[0]

    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    assert scaler.mean_[0] == mean_after_fit  # transform() must not change fitted parameters
    assert x_val_scaled.shape == (2, 1)
    assert x_test_scaled.shape == (2, 1)


def test_test_data_does_not_change_scaler_parameters():
    x_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    x_test = pd.DataFrame({"f1": [1000.0, 2000.0, 3000.0]})  # wildly different distribution

    scaler = StandardScaler()
    scaler.fit(x_train)
    mean_before, var_before = scaler.mean_.copy(), scaler.var_.copy()

    scaler.transform(x_test)  # must NOT refit

    np.testing.assert_array_equal(scaler.mean_, mean_before)
    np.testing.assert_array_equal(scaler.var_, var_before)

    # Scaling test data with train's parameters should NOT land near mean 0 --
    # concrete proof the scaler used train's (not test's) statistics.
    x_test_scaled = scaler.transform(x_test)
    assert abs(x_test_scaled.mean()) > 1.0
