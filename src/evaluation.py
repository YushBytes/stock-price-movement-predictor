"""Shared evaluation metrics and comparison-table helpers.

Positive class convention: UP = 1, DOWN_OR_FLAT = 0 (matching
src/preprocessing.py's target definition) -- precision/recall/F1 below are
always computed with respect to the UP (1) class.

A later phase will add the predicted-vs-actual visualization over the test
window. All evaluation in this project is computed on the chronological
test partition only -- never on validation or training data.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute accuracy/precision/recall/F1 (positive class = UP = 1).

    If a predictor produces only one class (e.g. the majority-class
    baseline), precision/recall/F1 for the class it never predicts would
    otherwise be mathematically undefined (0/0). This is handled
    explicitly via scikit-learn's ``zero_division=0``: it is defined as 0
    in that case, a documented choice rather than a silently suppressed
    warning.

    Raises ``ValueError`` if the two inputs have different lengths or are
    empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must be the same length; got {len(y_true)} and {len(y_pred)}.")
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on empty input.")

    y_true = pd.Series(y_true).reset_index(drop=True).astype(int)
    y_pred = pd.Series(y_pred).reset_index(drop=True).astype(int)

    return {
        "n_predictions": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def build_comparison_table(results: dict) -> pd.DataFrame:
    """Build a comparison table from a ``{model_name: metrics_dict}`` mapping.

    Each ``metrics_dict`` must be a dict as returned by
    ``compute_classification_metrics`` (containing at least accuracy,
    precision, recall, f1). Row order matches insertion order of ``results``.
    """
    if not results:
        raise ValueError("results must not be empty.")

    rows = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
            }
        )
    return pd.DataFrame(rows)
