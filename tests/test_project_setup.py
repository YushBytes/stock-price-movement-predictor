"""Phase 1 sanity tests.

These verify the project foundation only: configuration loads, the expected
directory structure exists, and every source module can be imported without
error. They intentionally do NOT test model correctness -- that arrives in
later phases once the pipeline itself exists.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402


def test_config_constants_loaded():
    assert config.SYMBOL == "SPY"
    assert config.DATA_SOURCE == "yfinance"
    assert isinstance(config.RANDOM_SEED, int)
    assert abs((config.TRAIN_RATIO + config.VALIDATION_RATIO + config.TEST_RATIO) - 1.0) < 1e-9


def test_expected_directories_exist():
    assert config.RAW_DATA_DIR.exists()
    assert config.PROCESSED_DATA_DIR.exists()
    assert config.MODELS_DIR.exists()
    assert config.RESULTS_DIR.exists()
    assert config.FIGURES_DIR.exists()
    assert (PROJECT_ROOT / "src").exists()
    assert (PROJECT_ROOT / "notebooks").exists()


def test_source_modules_import():
    from src import baselines, data_loader, evaluation, features, models, preprocessing  # noqa: F401


def test_ensure_output_directories_is_idempotent():
    config.ensure_output_directories()
    assert config.RAW_DATA_DIR.exists()
