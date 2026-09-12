"""Phase 2 tests for data loading and validation.

These tests use small, deterministic, hand-built DataFrames -- never the
real market dataset -- so they run without a network connection. The real
SPY dataset is only ever produced by ``src.data_loader.fetch_ohlcv``, which
is intentionally NOT exercised by these unit tests.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    REQUIRED_COLUMNS,
    _cache_matches_config,
    load_cached_data,
    save_raw_data,
    validate_ohlcv,
)


def _make_clean_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1000, 1100, 1200],
        }
    )


def test_validate_ohlcv_reports_zero_problems_on_clean_data():
    report = validate_ohlcv(_make_clean_df(), "2024-01-01", "2024-01-03")
    assert report["n_rows"] == 3
    assert report["has_all_required_columns"] is True
    assert report["is_chronological"] is True
    assert report["duplicate_dates"] == 0
    assert report["missing_values_total"] == 0
    assert report["invalid_ohlc_relationship_rows"] == 0
    assert report["negative_volume_rows"] == 0
    assert report["covers_requested_start"] is True
    assert report["covers_requested_end"] is True


def test_validate_ohlcv_detects_out_of_order_dates():
    shuffled = _make_clean_df().iloc[[2, 0, 1]].reset_index(drop=True)
    report = validate_ohlcv(shuffled)
    assert report["is_chronological"] is False


def test_validate_ohlcv_detects_duplicate_dates():
    df = _make_clean_df()
    with_dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    report = validate_ohlcv(with_dupe)
    assert report["duplicate_dates"] == 1


def test_validate_ohlcv_detects_invalid_high_low():
    df = _make_clean_df()
    df.loc[0, "High"] = 50.0  # High below Low -- invalid
    report = validate_ohlcv(df)
    assert report["invalid_high_low_rows"] == 1
    assert report["invalid_ohlc_relationship_rows"] >= 1


def test_validate_ohlcv_detects_close_outside_high_low():
    df = _make_clean_df()
    df.loc[0, "Close"] = 500.0  # far above High -- invalid
    report = validate_ohlcv(df)
    assert report["invalid_close_range_rows"] == 1


def test_validate_ohlcv_detects_negative_volume():
    df = _make_clean_df()
    df.loc[0, "Volume"] = -10
    report = validate_ohlcv(df)
    assert report["negative_volume_rows"] == 1


def test_validate_ohlcv_detects_missing_values():
    df = _make_clean_df()
    df.loc[0, "Close"] = None
    report = validate_ohlcv(df)
    assert report["missing_values_total"] == 1


def test_validate_ohlcv_handles_empty_dataframe():
    report = validate_ohlcv(pd.DataFrame(columns=REQUIRED_COLUMNS))
    assert report["is_empty"] is True
    assert report["n_rows"] == 0


def test_save_and_load_cached_data_roundtrip(tmp_path):
    df = _make_clean_df()
    csv_path = tmp_path / "TEST.csv"
    save_raw_data(df, csv_path)
    assert csv_path.exists()

    loaded = load_cached_data(csv_path)
    assert list(loaded.columns) == REQUIRED_COLUMNS
    assert len(loaded) == len(df)
    assert pd.api.types.is_datetime64_any_dtype(loaded["Date"])


def test_cache_matches_config_true_within_tolerance():
    assert _cache_matches_config(_make_clean_df(), "2024-01-01", "2024-01-03") is True


def test_cache_matches_config_false_when_start_far_off():
    assert _cache_matches_config(_make_clean_df(), "2010-01-01", None) is False


def test_cache_matches_config_false_when_end_not_covered():
    assert _cache_matches_config(_make_clean_df(), "2024-01-01", "2030-01-01") is False


def test_cache_matches_config_false_for_empty_dataframe():
    assert _cache_matches_config(pd.DataFrame(columns=REQUIRED_COLUMNS), "2024-01-01", None) is False
