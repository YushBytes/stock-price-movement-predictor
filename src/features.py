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

Technical indicator philosophy
--------------------------------
``build_technical_indicators`` computes SMA, RSI, MACD, rolling
volatility, and daily return, all via ``ta`` (never ``pandas-ta``, which
is unmaintained) or plain pandas ``rolling``/``pct_change``. Every one of
these is a trailing/backward-looking transformation -- a rolling window
ending at row ``t`` (never ``center=True``) or an EMA, both of which only
ever consult rows at or before ``t``. None of them use a negative
``shift``; that remains reserved exclusively for ``build_target`` in
``preprocessing.py``. Because ``Target[t]`` is defined from
``Close[t+1]`` vs. ``Close[t]``, any indicator computed using information
through day ``t`` (e.g. ``RSI_14[t]``) is valid for predicting
``Target[t]`` -- it does not need to "see" day ``t+1`` to be useful, and
it never does.
"""

import pandas as pd
import ta.momentum
import ta.trend

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


TECHNICAL_INDICATOR_WINDOWS = {
    "SMA_10": 10,
    "SMA_20": 20,
    "RSI_14": 14,
    "MACD_WINDOW_FAST": 12,
    "MACD_WINDOW_SLOW": 26,
    "MACD_WINDOW_SIGNAL": 9,
    "VOLATILITY_20": 20,
    "RETURN_1D": 1,
}


def build_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Build causal technical-indicator features from ``Close`` prices.

    Indicators (see README "Technical Indicators" for the rationale of
    each):

    - ``SMA_10``, ``SMA_20`` -- simple moving average of Close over
      10/20 days (``ta.trend.sma_indicator``, a plain trailing
      ``rolling(window).mean()`` -- verified not to use a centered window).
    - ``RSI_14`` -- 14-day Relative Strength Index
      (``ta.momentum.RSIIndicator``).
    - ``MACD``, ``MACD_SIGNAL``, ``MACD_HIST`` -- 12/26/9-day MACD
      (``ta.trend.MACD``).
    - ``RETURN_1D`` -- 1-day percentage return (``Close.pct_change(1)``).
    - ``VOLATILITY_20`` -- 20-day rolling standard deviation of
      ``RETURN_1D``.

    Every one of the above is a deterministic, backward-looking function
    of historical ``Close`` prices only; none of them uses ``shift`` with
    a negative argument or a centered rolling window.

    Rows within any indicator's warm-up period (the columns that need the
    most preceding history to produce their first real value) are
    dropped -- never forward-filled, backward-filled, or invented. After
    trimming, an internal check confirms no NaN remains anywhere in the
    output (a stray NaN afterward would indicate a genuine bug, not an
    expected warm-up gap).

    Returns a **new** DataFrame containing ``Date`` (if present in ``df``)
    plus exactly the indicator columns above. ``df`` itself is never
    mutated, and ``Target`` is never included in the output -- this
    function only reads ``Close``.

    Raises
    ------
    KeyError
        If ``Close`` is missing from ``df``.
    ValueError
        If ``df`` is empty, or (when a ``Date`` column is present) dates
        are not sorted chronologically.
    """
    if "Close" not in df.columns:
        raise KeyError("'Close' column not found in DataFrame; required for all technical indicators.")

    if len(df) == 0:
        raise ValueError("Cannot build technical indicators from an empty DataFrame.")

    if "Date" in df.columns and not df["Date"].is_monotonic_increasing:
        raise ValueError(
            "DataFrame must be sorted chronologically (oldest to newest) before building technical indicators."
        )

    close = df["Close"]

    result = pd.DataFrame(index=df.index)
    if "Date" in df.columns:
        result["Date"] = df["Date"]

    result["SMA_10"] = ta.trend.sma_indicator(close, window=TECHNICAL_INDICATOR_WINDOWS["SMA_10"])
    result["SMA_20"] = ta.trend.sma_indicator(close, window=TECHNICAL_INDICATOR_WINDOWS["SMA_20"])
    result["RSI_14"] = ta.momentum.RSIIndicator(close, window=TECHNICAL_INDICATOR_WINDOWS["RSI_14"]).rsi()

    macd_indicator = ta.trend.MACD(
        close,
        window_slow=TECHNICAL_INDICATOR_WINDOWS["MACD_WINDOW_SLOW"],
        window_fast=TECHNICAL_INDICATOR_WINDOWS["MACD_WINDOW_FAST"],
        window_sign=TECHNICAL_INDICATOR_WINDOWS["MACD_WINDOW_SIGNAL"],
    )
    result["MACD"] = macd_indicator.macd()
    result["MACD_SIGNAL"] = macd_indicator.macd_signal()
    result["MACD_HIST"] = macd_indicator.macd_diff()

    daily_return = close.pct_change(1)
    result["RETURN_1D"] = daily_return
    result["VOLATILITY_20"] = daily_return.rolling(window=TECHNICAL_INDICATOR_WINDOWS["VOLATILITY_20"]).std()

    indicator_columns = [c for c in result.columns if c != "Date"]

    # The warm-up cutoff is determined by whichever indicator column takes
    # longest to produce its first real value -- not an assumed constant.
    first_valid_positions = [result[c].reset_index(drop=True).first_valid_index() for c in indicator_columns]
    if any(pos is None for pos in first_valid_positions):
        raise ValueError("At least one indicator produced no valid values at all -- insufficient input data.")
    warmup_rows = max(first_valid_positions)

    result = result.iloc[warmup_rows:].reset_index(drop=True)

    if result[indicator_columns].isna().any().any():
        raise AssertionError("Unexpected NaN values remain in technical indicators after warm-up trimming.")

    return result
