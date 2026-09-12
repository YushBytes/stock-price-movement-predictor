"""Leak-free target construction, class-balance analysis, and chronological splitting.

Target definition
------------------
For trading day ``t``::

    Target[t] = 1  if Close[t+1] >  Close[t]   (UP)
    Target[t] = 0  if Close[t+1] <= Close[t]   (DOWN_OR_FLAT)

``Close[t+1]`` (tomorrow's close) is used ONLY inside ``build_target`` to
compute the label for row ``t``. It is never added to the returned
DataFrame as its own column, and it must never appear in a model's feature
matrix later in the project -- a negative ``shift`` like the one used here
is reserved for target construction only, never for feature construction
(see features.py).

The final row of any chronologically-sorted DataFrame has no following
observation, so no real Target can be computed for it; that row is dropped
rather than assigned a fabricated label.

Equal closes (``Close[t+1] == Close[t]``) are classified as DOWN_OR_FLAT
(0), not discarded -- an unchanged price is not a directional "up" move,
and dropping those rows would silently shrink the dataset and bias the
class balance based on an arbitrary tie-breaking choice.
"""

import pandas as pd

TARGET_COLUMN = "Target"


def build_target(df: pd.DataFrame, price_column: str = "Close") -> pd.DataFrame:
    """Return a copy of ``df`` with a leak-free next-day-direction target column.

    ``df`` is expected to already be sorted chronologically (oldest to
    newest) with unique dates if it has a ``Date`` column; this is validated
    below rather than assumed silently. The caller's DataFrame is never
    mutated -- a copy is returned.

    Adds a single new column, ``TARGET_COLUMN`` ("Target"), containing only
    0/1 values. No other helper/future-derived column is added or retained:
    ``Close[t+1]`` is used internally to compute the label and then
    discarded, never persisted as a feature.

    Raises
    ------
    KeyError
        If ``price_column`` is not a column of ``df``.
    ValueError
        If ``df`` has fewer than 2 rows, contains NaN values in
        ``price_column``, or (when a ``Date`` column is present) has
        duplicate or non-chronological dates.
    TypeError
        If ``price_column`` is not numeric.
    """
    if price_column not in df.columns:
        raise KeyError(f"'{price_column}' column not found in DataFrame. Available columns: {list(df.columns)}")

    if len(df) < 2:
        raise ValueError(
            "Need at least 2 rows to construct a next-day target (each row must have a "
            f"following observation to compare against); got {len(df)} row(s)."
        )

    if not pd.api.types.is_numeric_dtype(df[price_column]):
        raise TypeError(f"'{price_column}' column must be numeric; got dtype {df[price_column].dtype}.")

    if df[price_column].isna().any():
        raise ValueError(f"'{price_column}' column contains NaN values; cannot construct a reliable target.")

    if "Date" in df.columns:
        if df["Date"].duplicated().any():
            raise ValueError("Duplicate dates found; cannot determine unambiguous next-day ordering.")
        if not df["Date"].is_monotonic_increasing:
            raise ValueError("DataFrame must be sorted chronologically (oldest to newest) before target construction.")

    result = df.copy()

    # Look one row ahead ONLY to build the label. This value is never
    # attached to the returned DataFrame as a column.
    next_price = result[price_column].shift(-1)
    has_next_observation = next_price.notna()

    result[TARGET_COLUMN] = (next_price > result[price_column]).astype(int)

    # The row(s) with no following observation (the last row, in a properly
    # sorted DataFrame) get no real label -- drop them rather than fabricate one.
    result = result.loc[has_next_observation]

    return result


def compute_class_balance(target: pd.Series) -> dict:
    """Compute UP/DOWN_OR_FLAT counts and percentages for a binary target series.

    Every number is computed directly from ``target`` -- nothing is assumed.
    """
    if len(target) == 0:
        raise ValueError("Cannot compute class balance for an empty target series.")

    unique_values = set(pd.unique(target))
    if not unique_values <= {0, 1}:
        raise ValueError(f"Target must contain only 0 and 1; found {sorted(unique_values)}.")

    n_total = len(target)
    n_up = int((target == 1).sum())
    n_down_or_flat = int((target == 0).sum())

    return {
        "n_total": n_total,
        "n_up": n_up,
        "n_down_or_flat": n_down_or_flat,
        "pct_up": 100 * n_up / n_total,
        "pct_down_or_flat": 100 * n_down_or_flat / n_total,
    }


def print_class_balance(balance: dict, label: str = "") -> None:
    """Print a class-balance report using only values already in ``balance``."""
    heading = f"Class balance ({label}):" if label else "Class balance:"
    print(heading)
    print(f"  Total observations: {balance['n_total']}")
    print(f"  UP (1):           {balance['n_up']} ({balance['pct_up']:.2f}%)")
    print(f"  DOWN_OR_FLAT (0): {balance['n_down_or_flat']} ({balance['pct_down_or_flat']:.2f}%)")


def chronological_split(df: pd.DataFrame, train_ratio: float, validation_ratio: float):
    """Split ``df`` into (train, validation, test) partitions by row order only.

    The oldest ``train_ratio`` fraction of rows becomes the training set,
    the next ``validation_ratio`` fraction becomes the validation set, and
    the remainder becomes the test set. Never shuffles, samples randomly,
    or uses ``sklearn.model_selection.train_test_split``.

    Raises ``ValueError`` if ``df`` is empty or the ratios are invalid.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")
    if not (0 < train_ratio < 1) or not (0 < validation_ratio < 1):
        raise ValueError("train_ratio and validation_ratio must each be between 0 and 1.")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1 (test partition would be empty).")

    n = len(df)
    train_end = int(n * train_ratio)
    validation_end = train_end + int(n * validation_ratio)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    validation_df = df.iloc[train_end:validation_end].reset_index(drop=True)
    test_df = df.iloc[validation_end:].reset_index(drop=True)

    return train_df, validation_df, test_df


def assert_no_target_leakage(feature_columns, target_column: str = TARGET_COLUMN) -> None:
    """Assert that a feature-column list excludes the target and known future-derived helpers.

    Not exercised by a real feature matrix until Phase 6+, but established
    now so the safeguard exists from the moment target construction lands.
    """
    feature_columns = list(feature_columns)
    if target_column in feature_columns:
        raise AssertionError(f"Target column '{target_column}' must not appear in the feature column list.")
    forbidden_helper_columns = {"NextClose", "Close_next", "FutureClose", "Close_t_plus_1"}
    leaked = forbidden_helper_columns & set(feature_columns)
    if leaked:
        raise AssertionError(f"Future-derived helper column(s) found in feature columns: {leaked}")
