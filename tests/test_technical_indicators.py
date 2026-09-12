"""Phase 6 tests for causal technical-indicator feature construction.

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

from src.features import build_technical_indicators  # noqa: E402

INDICATOR_COLUMNS = ["SMA_10", "SMA_20", "RSI_14", "MACD", "MACD_SIGNAL", "MACD_HIST", "RETURN_1D", "VOLATILITY_20"]


def _price_series_df(n=80, seed=0):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    # A random walk with a small positive drift -- realistic enough that
    # SMA/RSI/MACD/volatility all produce non-degenerate values.
    steps = rng.normal(loc=0.05, scale=1.0, size=n)
    close = 100 + np.cumsum(steps)
    return pd.DataFrame({"Date": dates, "Close": close})


# --- Required indicators / structure -----------------------------------------

def test_required_indicators_are_created():
    df = _price_series_df()
    result = build_technical_indicators(df)
    for column in INDICATOR_COLUMNS:
        assert column in result.columns


def test_indicator_columns_are_numeric():
    df = _price_series_df()
    result = build_technical_indicators(df)
    for column in INDICATOR_COLUMNS:
        assert pd.api.types.is_numeric_dtype(result[column])


def test_rsi_range_is_valid_after_warmup():
    df = _price_series_df(n=120)
    result = build_technical_indicators(df)
    assert (result["RSI_14"] >= 0).all()
    assert (result["RSI_14"] <= 100).all()


def test_volatility_is_non_negative():
    df = _price_series_df(n=120)
    result = build_technical_indicators(df)
    assert (result["VOLATILITY_20"] >= 0).all()


def test_warmup_rows_are_dropped_and_no_nan_remains():
    df = _price_series_df(n=80)
    result = build_technical_indicators(df)
    assert len(result) < len(df)  # some warm-up rows must have been dropped
    assert not result[INDICATOR_COLUMNS].isna().any().any()


def test_no_target_column_in_technical_indicators():
    df = _price_series_df(n=80)
    df["Target"] = 0
    result = build_technical_indicators(df)
    assert "Target" not in result.columns


def test_no_future_columns_appear():
    df = _price_series_df(n=80)
    result = build_technical_indicators(df)
    forbidden_substrings = ["future", "next_close", "close_plus", "tomorrow", "target_shifted"]
    suspicious = [c for c in result.columns if any(s in c.lower() for s in forbidden_substrings)]
    assert suspicious == []


def test_chronological_ordering_is_preserved():
    df = _price_series_df(n=80)
    result = build_technical_indicators(df)
    assert result["Date"].is_monotonic_increasing
    assert result["Date"].tolist() == df["Date"].iloc[len(df) - len(result):].tolist()


def test_input_dataframe_is_not_mutated():
    df = _price_series_df(n=80)
    original = df.copy()
    build_technical_indicators(df)
    pd.testing.assert_frame_equal(df, original)


def test_missing_close_column_is_rejected():
    df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=10), "Open": range(10)})
    with pytest.raises(KeyError):
        build_technical_indicators(df)


def test_empty_input_is_rejected():
    df = pd.DataFrame(columns=["Date", "Close"])
    with pytest.raises(ValueError):
        build_technical_indicators(df)


def test_unsorted_dates_are_rejected():
    df = _price_series_df(n=10)
    df.loc[0, "Date"] = df["Date"].iloc[-1] + pd.Timedelta(days=100)
    with pytest.raises(ValueError):
        build_technical_indicators(df)


# --- Causality: the required perturbation test ---------------------------------

def test_future_perturbation_does_not_alter_earlier_indicator_values():
    df = _price_series_df(n=100)
    before = build_technical_indicators(df)

    perturbed = df.copy()
    perturbed.loc[perturbed.index[-1], "Close"] = 999999.0  # change only the LAST observation
    after = build_technical_indicators(perturbed)

    # Rows well clear of the perturbed timestamp (beyond any window length
    # used: max(20, 26+9)) must be byte-for-byte identical.
    cutoff = 40
    pd.testing.assert_frame_equal(
        before.iloc[:-cutoff].reset_index(drop=True),
        after.iloc[:-cutoff].reset_index(drop=True),
    )


def test_technical_feature_output_is_deterministic():
    df = _price_series_df(n=80)
    result_a = build_technical_indicators(df)
    result_b = build_technical_indicators(df)
    pd.testing.assert_frame_equal(result_a, result_b)
