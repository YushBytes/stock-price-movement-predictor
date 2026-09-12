"""Centralized, cached data-loading layer for the Streamlit dashboard.

All file I/O for the dashboard goes through this module so that:
- Paths are always relative to the repository root (never machine-specific).
- Every deterministic CSV load is cached with ``@st.cache_data``.
- Missing files surface a clear ``st.warning`` rather than crashing the app.

This module is read-only with respect to the ML pipeline; it never writes
to ``results/``, ``figures/``, ``models/``, or ``data/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Repository root — resolved relative to this file so all paths are portable.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _REPO_ROOT / "results"
_FIGURES_DIR = _REPO_ROOT / "figures"


# ---------------------------------------------------------------------------
# Verified project metadata
# Hard-coded constants from the verified experiment (not loaded from a file
# that might change). These never deviate from the confirmed notebook output.
# ---------------------------------------------------------------------------
VERIFIED_METADATA: dict = {
    "symbol": "SPY",
    "asset_name": "SPDR S&P 500 ETF Trust",
    "data_source": "Yahoo Finance (yfinance)",
    "raw_rows": 5_457,
    "date_start": "2005-01-03",
    "date_end": "2026-09-11",
    "ohlcv_columns": ["Open", "High", "Low", "Close", "Volume"],
    "n_raw_features": 20,
    "n_technical_features": 8,
    "n_engineered_features": 28,
    "feature_lags": [1, 2, 3, 5],
    "technical_indicators": [
        "SMA_10", "SMA_20", "RSI_14",
        "MACD", "MACD_SIGNAL", "MACD_HIST",
        "RETURN_1D", "VOLATILITY_20",
    ],
    # Chronological split dates
    "train_start": "2005-01-03",
    "train_end": "2020-03-05",
    "val_start": "2020-03-06",
    "val_end": "2023-06-05",
    "test_start": "2023-06-06",
    "test_end": "2026-09-10",
    # Test partition class balance
    "test_total": 819,
    "test_up": 463,
    "test_down_or_flat": 356,
    "test_pct_up": 56.53,
    "test_pct_down_or_flat": 43.47,
    # Full dataset labeled rows
    "labeled_rows": 5_456,
    # Partition row counts
    "train_rows": 3819,
    "val_rows": 818,
    "test_rows": 819,
    # Partition class balance
    "train_pct_up": 54.52,
    "val_pct_up": 53.18,
    "test_pct_up_labeled": 56.53,
}

INDICATOR_DESCRIPTIONS: dict[str, dict] = {
    "SMA_10": {
        "full_name": "Simple Moving Average (10-day)",
        "formula": "SMA_10[t] = mean(Close[t-9 … t])",
        "window": "10 trading days",
        "description": (
            "Trailing average of the last 10 closing prices. "
            "Acts as a short-term trend reference — price above SMA_10 "
            "suggests near-term momentum."
        ),
        "causal_note": "Uses only Close[t] and earlier. No centering. No future look-ahead.",
    },
    "SMA_20": {
        "full_name": "Simple Moving Average (20-day)",
        "formula": "SMA_20[t] = mean(Close[t-19 … t])",
        "window": "20 trading days",
        "description": (
            "Medium-term smoothed trend reference (~one calendar month). "
            "Crossovers between price and SMA_20 are widely used as "
            "momentum / mean-reversion signals."
        ),
        "causal_note": "Uses only Close[t] and earlier. No centering. No future look-ahead.",
    },
    "RSI_14": {
        "full_name": "Relative Strength Index (14-day)",
        "formula": "RSI = 100 − 100 / (1 + avg_gain / avg_loss)",
        "window": "14 trading days",
        "description": (
            "Ratio of average gains to average losses over 14 days, "
            "scaled 0–100. Readings above 70 are traditionally considered "
            "overbought; below 30, oversold."
        ),
        "causal_note": "Computed from Close changes at lags ≥1. No future data.",
    },
    "MACD": {
        "full_name": "MACD Line (12/26 EMA difference)",
        "formula": "MACD[t] = EMA_12(Close)[t] − EMA_26(Close)[t]",
        "window": "12 / 26 days",
        "description": (
            "Difference between a fast (12-day) and slow (26-day) "
            "exponential moving average of Close. Positive values indicate "
            "upward momentum; negative values indicate downward momentum."
        ),
        "causal_note": "Both EMAs are trailing. Warm-up rows (≤26) are dropped, not filled.",
    },
    "MACD_SIGNAL": {
        "full_name": "MACD Signal Line (9-day EMA of MACD)",
        "formula": "MACD_SIGNAL[t] = EMA_9(MACD)[t]",
        "window": "9 days (applied to MACD)",
        "description": (
            "A smoothed version of the MACD line. Crossovers between MACD "
            "and MACD_SIGNAL are classic trend-change indicators."
        ),
        "causal_note": "Derived only from past MACD values. No future look-ahead.",
    },
    "MACD_HIST": {
        "full_name": "MACD Histogram",
        "formula": "MACD_HIST[t] = MACD[t] − MACD_SIGNAL[t]",
        "window": "—",
        "description": (
            "The distance between MACD and its signal line. "
            "Visually shows momentum acceleration or deceleration."
        ),
        "causal_note": "Derived only from past values of MACD and MACD_SIGNAL.",
    },
    "RETURN_1D": {
        "full_name": "1-Day Return",
        "formula": "RETURN_1D[t] = (Close[t] − Close[t-1]) / Close[t-1]",
        "window": "1 day",
        "description": (
            "The simplest momentum signal: yesterday's percentage return. "
            "Captures very short-term continuation or reversal."
        ),
        "causal_note": "Uses Close[t] and Close[t-1] only. Strictly causal.",
    },
    "VOLATILITY_20": {
        "full_name": "20-Day Rolling Volatility",
        "formula": "VOLATILITY_20[t] = std(RETURN_1D[t-19 … t])",
        "window": "20 trading days",
        "description": (
            "Rolling standard deviation of daily returns. "
            "Higher values indicate a more uncertain market regime, "
            "which can affect how reliable other signals are."
        ),
        "causal_note": "Trailing window. No centering, no future data.",
    },
}


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_final_comparison() -> Optional[pd.DataFrame]:
    """Load ``results/final_comparison.csv``.

    Returns the DataFrame on success, or ``None`` if the file is missing.
    A ``st.warning`` is shown for the user when the file is absent.
    """
    path = _RESULTS_DIR / "final_comparison.csv"
    if not path.exists():
        st.warning(
            f"⚠️ Results file not found: `results/final_comparison.csv`  \n"
            "Run the Jupyter notebook to regenerate it."
        )
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as exc:
        st.error(f"Error reading `results/final_comparison.csv`: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_baseline_comparison() -> Optional[pd.DataFrame]:
    """Load ``results/baseline_comparison.csv``."""
    path = _RESULTS_DIR / "baseline_comparison.csv"
    if not path.exists():
        st.warning(f"⚠️ Results file not found: `results/baseline_comparison.csv`")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Error reading `results/baseline_comparison.csv`: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_raw_model_comparison() -> Optional[pd.DataFrame]:
    """Load ``results/raw_model_comparison.csv``."""
    path = _RESULTS_DIR / "raw_model_comparison.csv"
    if not path.exists():
        st.warning(f"⚠️ Results file not found: `results/raw_model_comparison.csv`")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Error reading `results/raw_model_comparison.csv`: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_engineered_model_comparison() -> Optional[pd.DataFrame]:
    """Load ``results/engineered_model_comparison.csv``."""
    path = _RESULTS_DIR / "engineered_model_comparison.csv"
    if not path.exists():
        st.warning(f"⚠️ Results file not found: `results/engineered_model_comparison.csv`")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Error reading `results/engineered_model_comparison.csv`: {exc}")
        return None


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def get_figure_path(filename: str) -> Path:
    """Return the absolute path to a figure file.

    Does NOT raise if missing — callers should check ``.exists()`` and use
    ``st.warning`` rather than crashing.
    """
    return _FIGURES_DIR / filename


def figure_exists(filename: str) -> bool:
    """Return True if the figure file exists on disk."""
    return get_figure_path(filename).exists()
