"""Model training for the raw-feature and engineered-feature classifiers.

Phase 5/7 will implement Logistic Regression (primary) and optionally a
Random Forest classifier, trained only on the training partition. Any
scaler used must be fit on the training partition only and reused
(never refit) to transform validation/test data.
"""

import pandas as pd


def train_logistic_regression(x_train: pd.DataFrame, y_train: pd.Series, random_seed: int):
    """Train a Logistic Regression classifier. Implemented in Phase 5."""
    raise NotImplementedError("Model training will be implemented in Phase 5.")
