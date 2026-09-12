"""Phase 5 tests for causal raw OHLCV feature construction.

All tests use small, deterministic, hand-built DataFrames -- never the
real market dataset -- so they run without a network connection.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import RAW_PRICE_VOLUME_COLUMNS, build_raw_features  # noqa: E402


def _ohlcv_df(n=10):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [100.0 + i for i in range(n)],
            "High": [101.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [100.5 + i for i in range(n)],
            "Volume": [1000 + 10 * i for i in range(n)],
        }
    )


def test_required_raw_columns_are_present_in_output():
    df = _ohlcv_df()
    result = build_raw_features(df, lags=[1, 2])
    for column in RAW_PRICE_VOLUME_COLUMNS:
        for lag in (1, 2):
            assert f"{column}_lag{lag}" in result.columns


def test_lag_values_are_correct():
    df = _ohlcv_df(n=6)
    result = build_raw_features(df, lags=[1, 2])
    # After dropping the first max(lag)=2 rows, row 0 of the result
    # corresponds to row 2 of the input.
    assert result["Close_lag1"].iloc[0] == df["Close"].iloc[1]
    assert result["Close_lag2"].iloc[0] == df["Close"].iloc[0]


def test_lag1_uses_immediately_previous_observation():
    df = _ohlcv_df(n=8)
    result = build_raw_features(df, lags=[1])
    for i in range(len(result)):
        original_row = i + 1  # row 0 of result = row 1 of df (max_lag=1 dropped)
        assert result["Open_lag1"].iloc[i] == df["Open"].iloc[original_row - 1]


def test_lag2_uses_two_observations_back():
    df = _ohlcv_df(n=8)
    result = build_raw_features(df, lags=[1, 2])
    for i in range(len(result)):
        original_row = i + 2  # max_lag=2 dropped
        assert result["Open_lag2"].iloc[i] == df["Open"].iloc[original_row - 2]


def test_no_future_values_are_used():
    df = _ohlcv_df(n=10)
    result_before = build_raw_features(df, lags=[1, 2, 3])

    df_modified = df.copy()
    df_modified.loc[df_modified.index[-1], "Close"] = 999999.0  # change only the LAST row
    result_after = build_raw_features(df_modified, lags=[1, 2, 3])

    # Every row's lag features except the very last one must be unaffected
    # by a change to the last row's Close (a lag feature never looks ahead).
    pd.testing.assert_frame_equal(
        result_before.iloc[:-1].reset_index(drop=True),
        result_after.iloc[:-1].reset_index(drop=True),
    )


def test_initial_lag_rows_are_dropped_not_filled():
    df = _ohlcv_df(n=10)
    result = build_raw_features(df, lags=[1, 2, 3, 5])
    assert len(result) == len(df) - 5
    assert not result.isna().any().any()  # no NaN anywhere -- nothing was left unfilled either


def test_no_target_column_enters_raw_features():
    df = _ohlcv_df(n=10)
    df["Target"] = [0, 1] * 5  # deliberately present in the input
    result = build_raw_features(df, lags=[1, 2])
    assert "Target" not in result.columns


def test_input_dataframe_is_not_mutated():
    df = _ohlcv_df(n=10)
    original = df.copy()
    build_raw_features(df, lags=[1, 2, 3])
    pd.testing.assert_frame_equal(df, original)


def test_chronological_ordering_is_preserved():
    df = _ohlcv_df(n=10)
    result = build_raw_features(df, lags=[1, 2])
    assert result["Date"].is_monotonic_increasing
    assert result["Date"].tolist() == df["Date"].iloc[2:].tolist()


def test_missing_required_column_raises_clear_error():
    df = _ohlcv_df(n=5).drop(columns=["Volume"])
    with pytest.raises(KeyError):
        build_raw_features(df)


def test_empty_dataframe_raises_clear_error():
    df = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    with pytest.raises(ValueError):
        build_raw_features(df)


def test_negative_lag_is_rejected():
    df = _ohlcv_df(n=10)
    with pytest.raises(ValueError):
        build_raw_features(df, lags=[1, -1])


def test_unsorted_dates_are_rejected():
    df = _ohlcv_df(n=5)
    df.loc[0, "Date"] = df["Date"].iloc[-1] + pd.Timedelta(days=100)  # break monotonic order
    with pytest.raises(ValueError):
        build_raw_features(df)
