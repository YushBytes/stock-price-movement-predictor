"""Leak-free target construction and chronological train/validation/test splitting.

Phase 3 will implement the next-day direction target (using only
Close[t] and Close[t+1]) and the chronological split logic. No function in
this module may shuffle rows or use information from beyond the row's own
time index.
"""

import pandas as pd


def build_target(df: pd.DataFrame, price_column: str = "Close") -> pd.DataFrame:
    """Add a leak-free next-day-direction target column.

    Implemented in Phase 3.
    """
    raise NotImplementedError("Target construction will be implemented in Phase 3.")


def chronological_split(df: pd.DataFrame, train_ratio: float, validation_ratio: float):
    """Split ``df`` into train/validation/test partitions by row order only.

    Implemented in Phase 3. Must never shuffle or randomly sample rows.
    """
    raise NotImplementedError("Chronological splitting will be implemented in Phase 3.")
