"""Data acquisition, caching, and validation for daily OHLCV data.

Fetches real daily OHLCV data via yfinance, caches it to data/raw/, and
validates it before it ever reaches feature engineering or modeling. No
function in this module invents, samples, or synthesizes prices -- every
value originates from the yfinance download.

Raw (not dividend/split-adjusted) prices are used throughout: yfinance's
default ``auto_adjust=True`` retroactively rewrites historical closes when a
new dividend/split is recorded, which is a form of look-ahead most people
don't think about. Fetching with ``auto_adjust=False`` keeps Close exactly
as it was actually quoted on the day, consistent with the leak-free
target/feature design in preprocessing.py and features.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def _display_path(path: Path) -> str:
    """Format ``path`` relative to the current working directory when possible.

    Used only for print statements, never for actual file I/O -- avoids
    baking a machine-specific absolute path (e.g. a local username) into
    notebook output committed to a public repository.
    """
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def fetch_ohlcv(symbol: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    """Download real daily OHLCV data for ``symbol`` from Yahoo Finance.

    Returns a DataFrame with exactly the columns in ``REQUIRED_COLUMNS``,
    sorted chronologically (oldest first) with duplicate dates removed.
    Raises ``ValueError`` if yfinance returns no data.
    """
    raw = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False, progress=False)

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for symbol={symbol!r}, "
            f"start_date={start_date!r}, end_date={end_date!r}. "
            "Check the symbol, date range, and network connectivity."
        )

    # A single-symbol download still comes back with MultiIndex columns
    # (Price, Ticker) in this yfinance version -- flatten to the price level.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()  # the Date index becomes an explicit column

    extra_columns = [c for c in raw.columns if c not in REQUIRED_COLUMNS]
    if extra_columns:
        print(f"Dropping non-core column(s) returned by yfinance: {extra_columns}")

    df = raw[REQUIRED_COLUMNS].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    duplicate_count = int(df.duplicated(subset="Date").sum())
    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate date row(s); keeping the last occurrence of each.")
        df = df.drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
    else:
        print("Found 0 duplicate date rows.")

    return df


def save_raw_data(df: pd.DataFrame, csv_path: Path) -> None:
    """Save ``df`` to ``csv_path``, creating parent directories if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def load_cached_data(csv_path: Path) -> pd.DataFrame:
    """Load previously cached OHLCV data from ``csv_path``."""
    return pd.read_csv(csv_path, parse_dates=["Date"])


def _cache_matches_config(df: pd.DataFrame, start_date: str, end_date: str | None) -> bool:
    """Check whether cached data's coverage still matches the requested config.

    Allows a small (<=7 calendar day) gap between the requested start date and
    the cached data's actual first date, since a requested date can land on a
    weekend/holiday with no trading data -- an exact-equality check would
    never match and would force a re-download on every single run.
    """
    if df.empty:
        return False
    actual_start = df["Date"].min().normalize()
    requested_start = pd.Timestamp(start_date).normalize()
    if abs((actual_start - requested_start).days) > 7:
        return False
    if end_date is not None:
        actual_end = df["Date"].max().normalize()
        requested_end = pd.Timestamp(end_date).normalize()
        if actual_end < requested_end - pd.Timedelta(days=7):
            return False
    return True


def load_or_fetch_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str | None,
    csv_path: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load cached OHLCV data if it matches the requested config, else fetch fresh.

    Never silently reuses stale data: if the cached file's date coverage
    doesn't match ``start_date``/``end_date``, it is re-downloaded.
    """
    if not force_refresh and csv_path.exists():
        cached = load_cached_data(csv_path)
        if _cache_matches_config(cached, start_date, end_date):
            print(
                f"Using cached data at {_display_path(csv_path)} "
                f"({len(cached)} rows, {cached['Date'].min().date()} to {cached['Date'].max().date()})."
            )
            return cached
        print(f"Cached data at {_display_path(csv_path)} does not match the requested configuration; re-downloading.")

    df = fetch_ohlcv(symbol, start_date, end_date)
    save_raw_data(df, csv_path)
    print(f"Downloaded and cached {len(df)} rows to {_display_path(csv_path)}.")
    return df


def validate_ohlcv(df: pd.DataFrame, requested_start: str | None = None, requested_end: str | None = None) -> dict:
    """Run structural and sanity checks on OHLCV data and return a report dict.

    Every value in the returned report is computed from ``df`` itself --
    nothing is assumed or invented.
    """
    report: dict = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "is_empty": bool(df.empty),
    }
    if df.empty:
        return report

    report["missing_required_columns"] = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report["has_all_required_columns"] = len(report["missing_required_columns"]) == 0

    report["first_date"] = df["Date"].min()
    report["last_date"] = df["Date"].max()
    report["dates_valid"] = bool(df["Date"].notna().all())
    report["is_chronological"] = bool(df["Date"].is_monotonic_increasing)
    report["duplicate_dates"] = int(df.duplicated(subset="Date").sum())

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    report["numeric_dtypes_ok"] = all(pd.api.types.is_numeric_dtype(df[c]) for c in numeric_columns)

    missing_by_column = df[REQUIRED_COLUMNS].isna().sum()
    report["missing_values_total"] = int(missing_by_column.sum())
    report["missing_values_by_column"] = missing_by_column.to_dict()

    numeric_values = df[numeric_columns].to_numpy(dtype=float)
    report["infinite_values_total"] = int(np.isinf(numeric_values).sum())

    invalid_high_low = int((df["High"] < df["Low"]).sum())
    invalid_open_range = int(((df["Open"] > df["High"]) | (df["Open"] < df["Low"])).sum())
    invalid_close_range = int(((df["Close"] > df["High"]) | (df["Close"] < df["Low"])).sum())
    report["invalid_high_low_rows"] = invalid_high_low
    report["invalid_open_range_rows"] = invalid_open_range
    report["invalid_close_range_rows"] = invalid_close_range
    report["invalid_ohlc_relationship_rows"] = invalid_high_low + invalid_open_range + invalid_close_range

    report["negative_volume_rows"] = int((df["Volume"] < 0).sum())

    # A requested start/end date may fall on a weekend/holiday with no trading
    # data, so "covers" allows a small (<=7 calendar day) gap to the nearest
    # actual trading day rather than requiring an exact or one-sided match.
    if requested_start is not None:
        gap_days = abs((report["first_date"] - pd.Timestamp(requested_start)).days)
        report["covers_requested_start"] = bool(gap_days <= 7)
    if requested_end is not None:
        gap_days = abs((pd.Timestamp(requested_end) - report["last_date"]).days)
        report["covers_requested_end"] = bool(gap_days <= 7)

    return report


def print_validation_report(report: dict) -> None:
    """Print a human-readable validation report, always showing real values."""
    print("=== Data Validation Report ===")
    print(f"Dataset shape: ({report['n_rows']}, {report['n_columns']})")
    print(f"Number of rows: {report['n_rows']}")
    print(f"Number of columns: {report['n_columns']}")
    if report["is_empty"]:
        print("Dataset is EMPTY.")
        return
    print(f"First date: {report['first_date']}")
    print(f"Last date: {report['last_date']}")
    print(f"Required columns present: {report['has_all_required_columns']}")
    if report["missing_required_columns"]:
        print(f"Missing columns: {report['missing_required_columns']}")
    print(f"Dates valid (no nulls): {report['dates_valid']}")
    print(f"Chronologically sorted: {report['is_chronological']}")
    print(f"Duplicate dates: {report['duplicate_dates']}")
    print(f"Numeric dtypes OK: {report['numeric_dtypes_ok']}")
    print(f"Missing values (total): {report['missing_values_total']}")
    print(f"Missing values by column: {report['missing_values_by_column']}")
    print(f"Infinite values (total): {report['infinite_values_total']}")
    print(f"Invalid High < Low rows: {report['invalid_high_low_rows']}")
    print(f"Invalid Open-out-of-[Low,High] rows: {report['invalid_open_range_rows']}")
    print(f"Invalid Close-out-of-[Low,High] rows: {report['invalid_close_range_rows']}")
    print(f"Negative volume rows: {report['negative_volume_rows']}")
    if "covers_requested_start" in report:
        print(f"Covers requested start date: {report['covers_requested_start']}")
    if "covers_requested_end" in report:
        print(f"Covers requested end date: {report['covers_requested_end']}")
