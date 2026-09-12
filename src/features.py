"""Raw and technical-indicator feature construction.

Phase 6/7 will implement raw OHLCV lag features and technical indicators
(moving averages, RSI, MACD, rolling volatility, returns) using the ``ta``
library or pandas. Every feature must be computed using only data available
at or before its own row's timestamp -- no centered rolling windows and no
negative ``shift`` values.
"""

import pandas as pd


def build_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build lagged raw OHLCV features. Implemented in Phase 6."""
    raise NotImplementedError("Raw feature construction will be implemented in Phase 6.")


def build_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Build technical-indicator features. Implemented in Phase 7."""
    raise NotImplementedError("Technical indicators will be implemented in Phase 7.")
