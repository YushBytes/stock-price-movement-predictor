"""Raw and technical-indicator feature construction.

Raw feature philosophy
-----------------------
At prediction time ``t``, a model may only use information already
observed at or before ``t``. ``build_raw_features`` builds *lagged*
OHLCV columns -- ``Open_lag1``, ``Close_lag2``, ``Volume_lag5``, etc. --
using only ``pandas.Series.shift(k)`` with a **positive** ``k``. A lag of
1 means "yesterday's value" (``Open_lag1[t] = Open[t-1]``), a lag of 2
means "the day before that" (``Open_lag2[t] = Open[t-2]``), and so on:
every one of these values was already observed strictly before row
``t``'s own Target is decided, so using them as features cannot leak
tomorrow's outcome. This is the same reasoning `preprocessing.py`
documents for the target: a negative shift (looking at ``t+1``) is
reserved exclusively for `build_target`, and this module never uses one.

`build_technical_indicators` (a later phase) will implement moving
averages, RSI, MACD, rolling volatility, and returns using the ``ta``
library or pandas. Every feature must be computed using only data
available at or before its own row's timestamp -- no centered rolling
windows and no negative ``shift`` values there either.
"""

import pandas as pd

RAW_PRICE_VOLUME_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
RAW_FEATURE_LAGS = [1, 2, 3, 5]


def build_raw_features(df: pd.DataFrame, lags: list[int] | None = None) -> pd.DataFrame:
    """Build causal lagged raw OHLCV features.

    For each column in ``RAW_PRICE_VOLUME_COLUMNS`` and each value in
    ``lags`` (default ``RAW_FEATURE_LAGS`` = [1, 2, 3, 5]), adds a column
    named ``f"{column}_lag{lag}"`` equal to ``df[column].shift(lag)`` --
    the value observed ``lag`` trading days before row ``t``. Only
    positive lags are accepted; a negative/future shift here would be a
    feature-leakage bug, so it is rejected with a ``ValueError`` rather
    than silently produced.

    The first ``max(lags)`` rows of the input cannot have every lag
    populated (there isn't enough preceding history) and are dropped --
    never forward-filled, backward-filled, or invented.

    Returns a **new** DataFrame containing ``Date`` (if present in ``df``)
    plus exactly the lagged feature columns. ``df`` itself is never
    mutated, and no same-day (lag 0) OHLCV column or ``Target`` is ever
    included in the output -- this function only knows about
    ``RAW_PRICE_VOLUME_COLUMNS``, so a ``Target`` column in the input, if
    present, is structurally impossible to leak into the output.

    Raises
    ------
    KeyError
        If any of ``RAW_PRICE_VOLUME_COLUMNS`` is missing from ``df``.
    ValueError
        If ``df`` is empty, ``lags`` contains a non-positive value, or
        (when a ``Date`` column is present) dates are not sorted
        chronologically.
    """
    if lags is None:
        lags = RAW_FEATURE_LAGS

    missing_columns = [c for c in RAW_PRICE_VOLUME_COLUMNS if c not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required column(s) for raw feature construction: {missing_columns}")

    if len(df) == 0:
        raise ValueError("Cannot build raw features from an empty DataFrame.")

    if not lags or any(lag <= 0 for lag in lags):
        raise ValueError(
            "All lags must be positive integers -- feature construction must never use a "
            f"negative/future shift; got {lags}."
        )

    if "Date" in df.columns and not df["Date"].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted chronologically (oldest to newest) before building lagged features.")

    result = pd.DataFrame(index=df.index)
    if "Date" in df.columns:
        result["Date"] = df["Date"]

    for column in RAW_PRICE_VOLUME_COLUMNS:
        for lag in lags:
            result[f"{column}_lag{lag}"] = df[column].shift(lag)

    max_lag = max(lags)
    # Rows without enough preceding history to fill every lag column are
    # dropped rather than filled with any invented/carried-forward value.
    result = result.iloc[max_lag:].reset_index(drop=True)

    return result


def build_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Build technical-indicator features. Implemented in a later phase."""
    raise NotImplementedError("Technical indicators will be implemented in a later phase.")
