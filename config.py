"""Central configuration for the Stock Price Movement Predictor project.

All project-wide constants (data symbol, date range, random seed, output
paths) live here so that later modules import from this single source
instead of hard-coding values in multiple places.
"""

from pathlib import Path

# --- Project paths -----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# --- Data source ---------------------------------------------------------
# SPY (SPDR S&P 500 ETF Trust) chosen for high liquidity, long clean daily
# history, and absence of single-company idiosyncrasies (see README).
SYMBOL = "SPY"
DATA_SOURCE = "yfinance"
START_DATE = "2005-01-01"
END_DATE = None  # None = fetch up to the most recent available session

RAW_DATA_FILE = RAW_DATA_DIR / f"{SYMBOL}.csv"

# --- Reproducibility -------------------------------------------------------
RANDOM_SEED = 42

# --- Chronological split ratios (never randomly shuffled) ------------------
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15  # derived as the remainder; kept explicit for clarity


def ensure_output_directories() -> None:
    """Create all project output directories if they don't already exist."""
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
