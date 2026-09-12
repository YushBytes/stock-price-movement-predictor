"""Model training for the raw-feature and engineered-feature classifiers.

Uses scikit-learn's ``LogisticRegression`` with a deterministic
configuration (fixed ``random_state``, a generous ``max_iter`` so
convergence warnings don't mask a real problem). No hyperparameter search
is performed in this phase -- this establishes a clean, honest raw-feature
reference model, and repeatedly tuning against the test set is exactly the
kind of leakage this project avoids. Any scaler used upstream of this
function must be fit on the training partition only and reused (never
refit) to transform validation/test data -- see the notebook's "Raw
Price/Volume Model" section for that workflow.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(x_train: pd.DataFrame, y_train: pd.Series, random_seed: int) -> LogisticRegression:
    """Train a deterministic Logistic Regression classifier on the training partition only.

    Raises ``ValueError`` if ``x_train``/``y_train`` are empty or their
    lengths don't match.
    """
    if len(x_train) == 0:
        raise ValueError("x_train must not be empty.")
    if len(x_train) != len(y_train):
        raise ValueError(f"x_train and y_train must be the same length; got {len(x_train)} and {len(y_train)}.")

    model = LogisticRegression(random_state=random_seed, max_iter=1000)
    model.fit(x_train, y_train)
    return model
