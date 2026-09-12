"""Shared evaluation metrics and plotting helpers.

Phase 9/10/11 will implement accuracy/precision/recall/F1 computation, the
four-way comparison table (Persistence vs Majority vs Raw vs Engineered),
and the predicted-vs-actual visualization over the test window. All
evaluation must be computed on the chronological test partition only.
"""

import pandas as pd


def compute_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute accuracy/precision/recall/F1 for one model's predictions.

    Implemented in Phase 9.
    """
    raise NotImplementedError("Evaluation metrics will be implemented in Phase 9.")


def build_comparison_table(results: dict) -> pd.DataFrame:
    """Build the four-way baseline/model comparison table. Implemented in Phase 10."""
    raise NotImplementedError("Comparison table will be implemented in Phase 10.")
