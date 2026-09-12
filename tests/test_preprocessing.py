"""Phase 3 tests for leak-free target construction, class balance, and
chronological splitting.

All tests use small, deterministic, hand-built DataFrames -- never the real
market dataset -- so they run without a network connection.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402
    TARGET_COLUMN,
    assert_no_target_leakage,
    build_target,
    chronological_split,
    compute_class_balance,
    print_class_balance,
)


def _df(closes, with_dates=True):
    data = {"Close": closes}
    if with_dates:
        data = {"Date": pd.date_range("2024-01-01", periods=len(closes), freq="D"), **data}
    return pd.DataFrame(data)


# --- Target construction: core label correctness ----------------------------

def test_up_label_when_next_close_higher():
    df = _df([100.0, 105.0])
    result = build_target(df)
    assert result[TARGET_COLUMN].iloc[0] == 1


def test_down_label_when_next_close_lower():
    df = _df([100.0, 95.0])
    result = build_target(df)
    assert result[TARGET_COLUMN].iloc[0] == 0


def test_equal_close_produces_down_or_flat_label():
    df = _df([100.0, 100.0])
    result = build_target(df)
    assert result[TARGET_COLUMN].iloc[0] == 0


def test_final_row_is_removed():
    df = _df([100.0, 105.0, 95.0])
    result = build_target(df)
    assert len(result) == len(df) - 1
    # The dropped row's date must not appear anywhere in the result.
    assert df["Date"].iloc[-1] not in result["Date"].values


def test_multi_row_labels_are_all_correct():
    closes = [100.0, 105.0, 95.0, 95.0, 110.0]
    df = _df(closes)
    result = build_target(df)
    expected = [1, 0, 0, 1]  # 100->105 up, 105->95 down, 95->95 flat(=0), 95->110 up
    assert result[TARGET_COLUMN].tolist() == expected


# --- Chronological integrity -------------------------------------------------

def test_chronological_ordering_is_preserved():
    df = _df([100.0, 101.0, 102.0, 103.0])
    result = build_target(df)
    assert result["Date"].is_monotonic_increasing
    assert result["Close"].tolist() == [100.0, 101.0, 102.0]


def test_rows_are_not_shuffled():
    df = _df([100.0, 101.0, 102.0, 103.0, 104.0])
    result = build_target(df)
    # Row order in the output must match the row order of the input exactly
    # (minus the dropped final row) -- never reordered/shuffled.
    assert result["Close"].tolist() == df["Close"].iloc[:-1].tolist()


def test_build_target_rejects_unsorted_dates():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-03"]),
            "Close": [101.0, 100.0, 102.0],
        }
    )
    with pytest.raises(ValueError):
        build_target(df)


def test_build_target_rejects_duplicate_dates():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "Close": [100.0, 100.0, 101.0],
        }
    )
    with pytest.raises(ValueError):
        build_target(df)


def test_build_target_does_not_mutate_input():
    df = _df([100.0, 105.0, 95.0])
    original = df.copy()
    build_target(df)
    pd.testing.assert_frame_equal(df, original)


def test_build_target_preserves_original_index():
    df = _df([100.0, 105.0, 95.0])
    df.index = [10, 20, 30]
    result = build_target(df)
    assert list(result.index) == [10, 20]


def test_no_helper_columns_remain_in_output():
    df = _df([100.0, 105.0, 95.0])
    result = build_target(df)
    assert list(result.columns) == list(df.columns) + [TARGET_COLUMN]


# --- Target value sanity -----------------------------------------------------

def test_target_contains_only_binary_values():
    df = _df([100.0, 105.0, 95.0, 95.0, 110.0, 90.0])
    result = build_target(df)
    assert set(result[TARGET_COLUMN].unique()) <= {0, 1}


def test_target_not_accidentally_in_feature_columns():
    feature_columns = ["Open", "High", "Low", "Close", "Volume"]
    assert_no_target_leakage(feature_columns)  # should not raise

    with pytest.raises(AssertionError):
        assert_no_target_leakage(feature_columns + [TARGET_COLUMN])

    with pytest.raises(AssertionError):
        assert_no_target_leakage(feature_columns + ["Close_next"])


# --- Edge cases (Step 13) ----------------------------------------------------

def test_missing_close_column_raises_clear_error():
    df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=3), "Open": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError):
        build_target(df)


def test_empty_dataframe_raises_clear_error():
    df = pd.DataFrame(columns=["Date", "Close"])
    with pytest.raises(ValueError):
        build_target(df)


def test_one_row_dataframe_raises_clear_error():
    df = _df([100.0])
    with pytest.raises(ValueError):
        build_target(df)


def test_two_row_dataframe_produces_one_labeled_row():
    df = _df([100.0, 105.0])
    result = build_target(df)
    assert len(result) == 1
    assert result[TARGET_COLUMN].iloc[0] == 1


def test_non_numeric_close_raises_clear_error():
    df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=3), "Close": ["a", "b", "c"]})
    with pytest.raises(TypeError):
        build_target(df)


def test_nan_close_raises_clear_error():
    df = _df([100.0, np.nan, 105.0])
    with pytest.raises(ValueError):
        build_target(df)


def test_works_without_date_column():
    # A DataFrame with no Date column is allowed; row order alone defines time.
    df = pd.DataFrame({"Close": [100.0, 105.0, 95.0]})
    result = build_target(df)
    assert len(result) == 2
    assert result[TARGET_COLUMN].tolist() == [1, 0]


# --- Class balance ------------------------------------------------------------

def test_compute_class_balance_counts_and_percentages():
    target = pd.Series([1, 1, 1, 0, 0])
    balance = compute_class_balance(target)
    assert balance["n_total"] == 5
    assert balance["n_up"] == 3
    assert balance["n_down_or_flat"] == 2
    assert balance["pct_up"] == pytest.approx(60.0)
    assert balance["pct_down_or_flat"] == pytest.approx(40.0)


def test_compute_class_balance_rejects_non_binary_values():
    with pytest.raises(ValueError):
        compute_class_balance(pd.Series([0, 1, 2]))


def test_compute_class_balance_rejects_empty_series():
    with pytest.raises(ValueError):
        compute_class_balance(pd.Series([], dtype=int))


def test_print_class_balance_runs_without_error(capsys):
    balance = compute_class_balance(pd.Series([1, 0, 1]))
    print_class_balance(balance, label="test")
    captured = capsys.readouterr()
    assert "test" in captured.out
    assert "Total observations: 3" in captured.out


# --- Chronological split ------------------------------------------------------

def test_split_proportions_are_approximately_70_15_15():
    df = pd.DataFrame({"Close": range(1000)})
    train, val, test = chronological_split(df, 0.70, 0.15)
    assert len(train) == 700
    assert len(val) == 150
    assert len(test) == 150
    assert len(train) + len(val) + len(test) == len(df)


def test_split_never_shuffles_rows():
    df = pd.DataFrame({"Close": range(100)})
    train, val, test = chronological_split(df, 0.70, 0.15)
    assert train["Close"].tolist() == list(range(70))
    assert val["Close"].tolist() == list(range(70, 85))
    assert test["Close"].tolist() == list(range(85, 100))


def test_train_dates_precede_validation_dates():
    df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=100), "Close": range(100)})
    train, val, _ = chronological_split(df, 0.70, 0.15)
    assert train["Date"].max() < val["Date"].min()


def test_validation_dates_precede_test_dates():
    df = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=100), "Close": range(100)})
    _, val, test = chronological_split(df, 0.70, 0.15)
    assert val["Date"].max() < test["Date"].min()


def test_chronological_split_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        chronological_split(pd.DataFrame(columns=["Close"]), 0.70, 0.15)


def test_chronological_split_rejects_invalid_ratios():
    df = pd.DataFrame({"Close": range(10)})
    with pytest.raises(ValueError):
        chronological_split(df, 0.8, 0.3)  # sums to > 1
