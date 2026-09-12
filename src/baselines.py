"""Naive baseline predictors.

Phase 5 will implement the two required naive baselines:
  - Persistence: predict that tomorrow's direction repeats today's direction.
  - Majority-class: always predict the class most frequent in the training set.

Both are rule-based and require no model fitting.
"""

import pandas as pd


def persistence_baseline(df: pd.DataFrame, direction_column: str) -> pd.Series:
    """Predict tomorrow's direction as today's direction. Implemented in Phase 5."""
    raise NotImplementedError("Persistence baseline will be implemented in Phase 5.")


def majority_class_baseline(train_target: pd.Series, n_predictions: int) -> pd.Series:
    """Predict the training set's majority class for every row. Implemented in Phase 5."""
    raise NotImplementedError("Majority-class baseline will be implemented in Phase 5.")
