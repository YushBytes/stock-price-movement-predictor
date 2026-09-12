"""Phase 4 tests for classification-metric computation and the comparison table."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import build_comparison_table, compute_classification_metrics  # noqa: E402


def test_valid_predictions_produce_correct_metrics():
    y_true = pd.Series([1, 1, 0, 0, 1])
    y_pred = pd.Series([1, 0, 0, 0, 1])  # 4/5 correct
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["n_predictions"] == 5
    assert metrics["accuracy"] == pytest.approx(0.8)
    # TP=2, FP=0, FN=1 for class 1 -> precision=1.0, recall=2/3
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(2 / 3)


def test_binary_predictions_handled_correctly():
    y_true = pd.Series([0, 0, 0, 0])
    y_pred = pd.Series([0, 0, 0, 0])
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0


def test_zero_division_is_handled_explicitly_not_raised():
    # Predictor never predicts the positive class -> precision for class 1
    # is mathematically 0/0. Must resolve to 0.0, not raise or warn-crash.
    y_true = pd.Series([1, 1, 0, 0])
    y_pred = pd.Series([0, 0, 0, 0])
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_classification_metrics(pd.Series([1, 0]), pd.Series([1, 0, 1]))


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError):
        compute_classification_metrics(pd.Series([], dtype=int), pd.Series([], dtype=int))


def test_build_comparison_table_shape_and_columns():
    results = {
        "Persistence": {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5},
        "Majority Class": {"accuracy": 0.6, "precision": 0.6, "recall": 1.0, "f1": 0.75},
    }
    table = build_comparison_table(results)
    assert list(table.columns) == ["Model", "Accuracy", "Precision", "Recall", "F1"]
    assert table["Model"].tolist() == ["Persistence", "Majority Class"]
    assert table.loc[1, "Recall"] == 1.0


def test_build_comparison_table_rejects_empty_results():
    with pytest.raises(ValueError):
        build_comparison_table({})
