"""Phase 4 tests for the persistence and majority-class naive baselines.

All tests use small, deterministic, hand-built DataFrames -- never the real
market dataset -- so they run without a network connection.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import majority_class_baseline, persistence_baseline  # noqa: E402


def _dates(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="D")


# --- Persistence baseline -----------------------------------------------------

def test_persistence_predicts_up_after_previous_up_direction():
    # direction[d1] = 1 if Close[d1] > Close[d0]: 105 > 100 -> UP
    full_df = pd.DataFrame({"Date": _dates(3), "Close": [100.0, 105.0, 103.0]})
    evaluation_df = full_df.iloc[[2]]  # predict for d2 using direction observed at d1
    prediction = persistence_baseline(full_df, evaluation_df)
    assert prediction.iloc[0] == 1


def test_persistence_predicts_down_after_previous_down_direction():
    # direction[d1] = 1 if Close[d1] > Close[d0]: 95 > 100 -> False -> DOWN
    full_df = pd.DataFrame({"Date": _dates(3), "Close": [100.0, 95.0, 999.0]})
    evaluation_df = full_df.iloc[[2]]
    prediction = persistence_baseline(full_df, evaluation_df)
    assert prediction.iloc[0] == 0


def test_persistence_never_uses_current_or_future_target():
    # A 'Target' column with values that would give the OPPOSITE answer if
    # mistakenly consulted -- persistence must ignore it entirely and use
    # only historical Close prices.
    full_df = pd.DataFrame(
        {
            "Date": _dates(3),
            "Close": [100.0, 105.0, 103.0],  # direction[d1] = UP (1)
            "Target": [0, 0, 0],  # deliberately wrong / irrelevant values
        }
    )
    evaluation_df = full_df.iloc[[2]]
    prediction = persistence_baseline(full_df, evaluation_df)
    assert prediction.iloc[0] == 1  # matches Close-derived direction, not Target


def test_persistence_first_evaluation_row_uses_preceding_history():
    # full_df spans "train+validation+test"; evaluation_df is only the
    # "test" tail. The first test-window prediction must use the
    # immediately preceding ("validation-period") observation rather than
    # being dropped for lack of in-partition history.
    full_df = pd.DataFrame({"Date": _dates(5), "Close": [100.0, 90.0, 80.0, 200.0, 999.0]})
    evaluation_df = full_df.iloc[[3, 4]].reset_index(drop=True)  # the "test" partition
    prediction = persistence_baseline(full_df, evaluation_df)
    assert len(prediction) == 2
    # prediction for row index 3 (d3) = direction[d2] = 1 if Close[d2]>Close[d1] = 80>90 -> False -> 0
    assert prediction.iloc[0] == 0
    # prediction for row index 4 (d4) = direction[d3] = 1 if Close[d3]>Close[d2] = 200>80 -> True -> 1
    assert prediction.iloc[1] == 1


def test_persistence_output_length_matches_evaluation_length():
    full_df = pd.DataFrame({"Date": _dates(10), "Close": list(range(100, 110))})
    for k in (1, 3, 5):
        evaluation_df = full_df.iloc[-k:]
        prediction = persistence_baseline(full_df, evaluation_df)
        assert len(prediction) == k


def test_persistence_preserves_row_alignment_regardless_of_evaluation_order():
    full_df = pd.DataFrame({"Date": _dates(6), "Close": [100.0, 105.0, 95.0, 130.0, 120.0, 140.0]})
    evaluation_df = full_df.iloc[[4, 3]]  # deliberately out of chronological order
    prediction = persistence_baseline(full_df, evaluation_df)
    # Row for d4 (index 4): direction[d3] = 1 if Close[d3]>Close[d2] = 130>95 -> True -> 1
    # Row for d3 (index 3): direction[d2] = 1 if Close[d2]>Close[d1] = 95>105 -> False -> 0
    assert prediction.loc[4] == 1
    assert prediction.loc[3] == 0


def test_persistence_raises_when_insufficient_history():
    full_df = pd.DataFrame({"Date": _dates(2), "Close": [100.0, 105.0]})
    evaluation_df = full_df.iloc[[0]]  # no day before d0 exists
    with pytest.raises(ValueError):
        persistence_baseline(full_df, evaluation_df)


def test_persistence_raises_on_missing_date_column():
    full_df = pd.DataFrame({"Close": [100.0, 105.0, 103.0]})
    with pytest.raises(KeyError):
        persistence_baseline(full_df, full_df.iloc[[2]])


def test_persistence_raises_on_empty_evaluation_df():
    full_df = pd.DataFrame({"Date": _dates(3), "Close": [100.0, 105.0, 103.0]})
    with pytest.raises(ValueError):
        persistence_baseline(full_df, full_df.iloc[0:0])


# --- Majority-class baseline ---------------------------------------------------

def test_majority_up_training_data_produces_all_up_predictions():
    train_target = pd.Series([1, 1, 1, 0])
    prediction = majority_class_baseline(train_target, n_predictions=5)
    assert (prediction == 1).all()
    assert len(prediction) == 5


def test_majority_down_training_data_produces_all_down_predictions():
    train_target = pd.Series([0, 0, 0, 1])
    prediction = majority_class_baseline(train_target, n_predictions=4)
    assert (prediction == 0).all()
    assert len(prediction) == 4


def test_majority_class_ignores_evaluation_distribution_by_construction():
    # The function signature only accepts train_target -- there is no
    # parameter through which validation/test data could influence the
    # chosen majority class. Demonstrate the same train_target always
    # yields the same majority prediction regardless of n_predictions.
    train_target = pd.Series([1, 1, 0])
    small = majority_class_baseline(train_target, n_predictions=2)
    large = majority_class_baseline(train_target, n_predictions=200)
    assert small.iloc[0] == large.iloc[0] == 1


def test_majority_tie_breaks_deterministically_to_up():
    train_target = pd.Series([1, 1, 0, 0])  # exact 50/50 tie
    first_run = majority_class_baseline(train_target, n_predictions=3)
    second_run = majority_class_baseline(train_target, n_predictions=3)
    assert (first_run == 1).all()
    assert first_run.tolist() == second_run.tolist()  # deterministic, not random


def test_majority_output_length_is_exact():
    train_target = pd.Series([1, 0, 1])
    for n in (0, 1, 7, 100):
        prediction = majority_class_baseline(train_target, n_predictions=n)
        assert len(prediction) == n


def test_majority_rejects_non_binary_training_target():
    with pytest.raises(ValueError):
        majority_class_baseline(pd.Series([0, 1, 2]), n_predictions=3)


def test_majority_rejects_empty_training_target():
    with pytest.raises(ValueError):
        majority_class_baseline(pd.Series([], dtype=int), n_predictions=3)
