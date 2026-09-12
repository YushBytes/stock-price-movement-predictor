"""Phase 7 tests for the combined engineered feature set (raw lagged OHLCV
+ technical indicators).

All tests use small, deterministic, hand-built DataFrames -- never the
real market dataset -- so they run without a network connection.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (  # noqa: E402
    ENGINEERED_FEATURE_COLUMNS,
    RAW_FEATURE_COLUMNS,
    TECHNICAL_INDICATOR_COLUMNS,
    build_engineered_features,
)
from src.models import train_logistic_regression  # noqa: E402
from src.evaluation import compute_classification_metrics  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def _ohlcv_df(n=120, seed=0):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    steps = rng.normal(loc=0.05, scale=1.0, size=n)
    close = 100 + np.cumsum(steps)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close + rng.normal(0, 0.1, n),
            "High": close + np.abs(rng.normal(0, 0.2, n)),
            "Low": close - np.abs(rng.normal(0, 0.2, n)),
            "Close": close,
            "Volume": rng.randint(1000, 2000, n),
        }
    )


# --- Feature counts and names -------------------------------------------------

def test_exactly_28_engineered_features():
    assert len(ENGINEERED_FEATURE_COLUMNS) == 28


def test_exactly_20_raw_features():
    assert len(RAW_FEATURE_COLUMNS) == 20


def test_exactly_8_technical_features():
    assert len(TECHNICAL_INDICATOR_COLUMNS) == 8


def test_engineered_feature_names_match_expected_columns():
    df = _ohlcv_df()
    result = build_engineered_features(df)
    feature_columns = [c for c in result.columns if c != "Date"]
    assert feature_columns == ENGINEERED_FEATURE_COLUMNS


def test_engineered_columns_are_raw_then_indicator_in_order():
    assert ENGINEERED_FEATURE_COLUMNS[:20] == RAW_FEATURE_COLUMNS
    assert ENGINEERED_FEATURE_COLUMNS[20:] == TECHNICAL_INDICATOR_COLUMNS


# --- Structural correctness ----------------------------------------------------

def test_no_target_in_engineered_features():
    df = _ohlcv_df()
    df["Target"] = 0
    result = build_engineered_features(df)
    assert "Target" not in result.columns


def test_no_nan_values_in_engineered_features():
    df = _ohlcv_df()
    result = build_engineered_features(df)
    feature_columns = [c for c in result.columns if c != "Date"]
    assert not result[feature_columns].isna().any().any()


def test_chronological_order_preserved():
    df = _ohlcv_df()
    result = build_engineered_features(df)
    assert result["Date"].is_monotonic_increasing


def test_deterministic_output():
    df = _ohlcv_df()
    result_a = build_engineered_features(df)
    result_b = build_engineered_features(df)
    pd.testing.assert_frame_equal(result_a, result_b)


def test_input_dataframe_is_not_mutated():
    df = _ohlcv_df()
    original = df.copy()
    build_engineered_features(df)
    pd.testing.assert_frame_equal(df, original)


def test_no_future_suggesting_feature_names():
    forbidden_substrings = ["future", "next_close", "close_plus", "close_tomorrow", "target_shifted", "tomorrow"]
    suspicious = [c for c in ENGINEERED_FEATURE_COLUMNS if any(s in c.lower() for s in forbidden_substrings)]
    assert suspicious == []


def test_engineered_alignment_is_the_intersection_of_both_warmups():
    df = _ohlcv_df(n=100)
    result = build_engineered_features(df)
    # The engineered dataset must start no earlier than either individual
    # builder's own first valid date (i.e. it reflects the LATER cutoff).
    from src.features import build_raw_features, build_technical_indicators
    raw_only = build_raw_features(df)
    indicators_only = build_technical_indicators(df)
    expected_start = max(raw_only["Date"].min(), indicators_only["Date"].min())
    assert result["Date"].min() == expected_start


def test_missing_close_column_is_rejected():
    df = _ohlcv_df().drop(columns=["Close"])
    with pytest.raises(KeyError):
        build_engineered_features(df)


def test_empty_input_is_rejected():
    df = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    with pytest.raises(ValueError):
        build_engineered_features(df)


# --- Causality: perturbation test -----------------------------------------------

def test_future_perturbation_does_not_alter_earlier_engineered_rows():
    df = _ohlcv_df(n=120)
    before = build_engineered_features(df)

    perturbed = df.copy()
    perturbed.loc[perturbed.index[-1], "Close"] = 999999.0
    perturbed.loc[perturbed.index[-1], "Volume"] = 999999999
    after = build_engineered_features(perturbed)

    cutoff = 40  # well clear of the longest window (MACD's 26+9)
    pd.testing.assert_frame_equal(
        before.iloc[:-cutoff].reset_index(drop=True),
        after.iloc[:-cutoff].reset_index(drop=True),
    )


# --- Model training on the engineered feature set --------------------------------

def test_model_trains_successfully_on_engineered_features():
    df = _ohlcv_df(n=150)
    engineered = build_engineered_features(df)
    y = (engineered["Date"].dt.dayofyear % 2).astype(int)  # arbitrary deterministic binary target
    feature_columns = [c for c in engineered.columns if c != "Date"]
    x_scaled = StandardScaler().fit_transform(engineered[feature_columns])
    model = train_logistic_regression(x_scaled, y, random_seed=42)
    assert hasattr(model, "coef_")


def test_prediction_length_matches_test_labels():
    df = _ohlcv_df(n=150)
    engineered = build_engineered_features(df)
    feature_columns = [c for c in engineered.columns if c != "Date"]
    y = (engineered["Date"].dt.dayofyear % 2).astype(int)
    scaler = StandardScaler().fit(engineered[feature_columns])
    x_scaled = scaler.transform(engineered[feature_columns])
    model = train_logistic_regression(x_scaled, y, random_seed=42)
    predictions = model.predict(x_scaled[-20:])
    assert len(predictions) == 20


def test_metrics_are_valid_for_engineered_predictions():
    y_true = pd.Series([1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = pd.Series([1, 0, 1, 1, 0, 1, 0, 0])
    metrics = compute_classification_metrics(y_true, y_pred)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
