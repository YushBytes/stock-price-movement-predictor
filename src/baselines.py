"""Naive baseline predictors: persistence and majority-class.

Both are rule-based and require no model fitting -- they exist to give the
project a rigorous, honest reference point that any real model must beat.

Persistence baseline
---------------------
Predicts that the direction observed most recently is about to repeat.
Concretely, for evaluation row ``t`` (a specific trading day), the
prediction is the previously-completed one-day price direction::

    prediction[t] = direction[t-1]
    direction[k]  = 1 if Close[k] > Close[k-1] else 0

This uses ``Close[t-1]`` and ``Close[t-2]`` only -- never ``Close[t]``,
never ``Close[t+1]``, and never the row's own ``Target`` value (using
``Target[t]`` to predict ``Target[t]`` would be circular, since ``Target[t]``
is defined using ``Close[t+1]``, the very thing being predicted). Predicting
for the first row of a partition (e.g. the first day of the test set) is
handled correctly by supplying the full chronological history up to and
including that partition -- the immediately preceding trading day's
direction (from the validation period, if predicting for the start of the
test period) is real, already-observed information and is used rather than
discarded.

Majority-class baseline
-------------------------
Predicts the single class that is most frequent in the TRAINING partition
only, for every evaluation row. Validation and test data are never
consulted when choosing which class is "the majority" -- doing so would
leak information about the evaluation period into the baseline itself.
Ties (exactly 50/50 in training) are broken deterministically by
predicting UP (1); this is a documented, arbitrary-but-fixed choice, not a
data-dependent one.
"""

import pandas as pd


def persistence_baseline(full_df: pd.DataFrame, evaluation_df: pd.DataFrame, price_column: str = "Close") -> pd.Series:
    """Predict each row of ``evaluation_df`` using the previously observed direction.

    ``full_df`` must be the complete chronologically-sorted history
    (typically spanning train+validation+test), used only so that the
    first row(s) of ``evaluation_df`` can look back into whatever period
    precedes them without being discarded. ``evaluation_df`` determines
    the output length and row alignment (via its ``Date`` column) -- the
    returned Series always has exactly ``len(evaluation_df)`` predictions,
    indexed identically to ``evaluation_df``.

    Raises
    ------
    KeyError
        If ``Date`` is missing from either DataFrame, or ``price_column``
        is missing from ``full_df``.
    ValueError
        If ``evaluation_df`` is empty, or ``full_df`` lacks enough history
        immediately before some row of ``evaluation_df`` to compute a
        prediction (i.e. fewer than 2 preceding trading days exist).
    """
    if "Date" not in full_df.columns:
        raise KeyError("full_df must contain a 'Date' column.")
    if "Date" not in evaluation_df.columns:
        raise KeyError("evaluation_df must contain a 'Date' column.")
    if price_column not in full_df.columns:
        raise KeyError(f"'{price_column}' column not found in full_df.")
    if len(evaluation_df) == 0:
        raise ValueError("evaluation_df must not be empty.")

    full_sorted = full_df.sort_values("Date").reset_index(drop=True)
    price = full_sorted[price_column]

    prior_price = price.shift(1)
    direction = pd.Series(pd.NA, index=full_sorted.index, dtype="Int64")
    has_prior_day = prior_price.notna()
    direction.loc[has_prior_day] = (price[has_prior_day] > prior_price[has_prior_day]).astype(int)

    # prediction[k] = direction[k-1]: the direction observed as of the
    # previous trading day, never the current or a future one.
    prediction_by_date = pd.Series(direction.shift(1).to_numpy(), index=full_sorted["Date"].to_numpy())

    aligned = evaluation_df["Date"].map(prediction_by_date)
    if aligned.isna().any():
        missing_dates = evaluation_df.loc[aligned.isna(), "Date"].tolist()
        raise ValueError(
            f"Could not compute a persistence prediction for {len(missing_dates)} evaluation date(s) -- "
            f"full_df does not contain at least 2 trading days immediately before them: {missing_dates[:5]}"
        )

    result = aligned.astype(int)
    result.index = evaluation_df.index
    return result


def majority_class_baseline(train_target: pd.Series, n_predictions: int) -> pd.Series:
    """Predict the training partition's majority class for every evaluation row.

    The majority class is computed from ``train_target`` only -- validation
    and test data must never be passed here. On an exact 50/50 tie, UP (1)
    is predicted; this tie-break is fixed and documented, not data-driven.
    """
    if len(train_target) == 0:
        raise ValueError("train_target must not be empty.")
    unique_values = set(pd.unique(train_target))
    if not unique_values <= {0, 1}:
        raise ValueError(f"train_target must contain only 0 and 1; found {sorted(unique_values)}.")
    if n_predictions < 0:
        raise ValueError("n_predictions must be non-negative.")

    n_up = int((train_target == 1).sum())
    n_down_or_flat = int((train_target == 0).sum())
    majority_class = 1 if n_up >= n_down_or_flat else 0  # tie -> UP (1), see module docstring

    return pd.Series([majority_class] * n_predictions, dtype=int)
