"""Data acquisition and caching.

Phase 2 will implement fetching daily OHLCV data for the symbol/date range
defined in config.py (via yfinance) and caching it to data/raw/ so the
notebook never depends on an undocumented local dataset.
"""

from pathlib import Path

import pandas as pd


def fetch_ohlcv(symbol: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    """Fetch daily OHLCV data for ``symbol`` between ``start_date`` and ``end_date``.

    Implemented in Phase 2.
    """
    raise NotImplementedError("Data fetching will be implemented in Phase 2.")


def load_cached_data(csv_path: Path) -> pd.DataFrame:
    """Load previously cached OHLCV data from ``csv_path``.

    Implemented in Phase 2.
    """
    raise NotImplementedError("Cached data loading will be implemented in Phase 2.")
