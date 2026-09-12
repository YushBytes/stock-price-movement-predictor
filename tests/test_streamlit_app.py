"""Tests for the Streamlit dashboard data-loading layer.

These tests verify that:
- The streamlit_app package imports without errors.
- Data loaders return DataFrames with the expected structure.
- Verified model names and metric columns are present.
- Exact metric values match the committed results CSVs to 4 decimal places.
- Figure path helpers return Path objects.
- No absolute machine-specific paths exist in any streamlit_app source file.
- No secrets or tokens are present in source files.

The tests never launch Streamlit (which requires a running server), never
fetch real data from the network, and never modify any results or figure files.

NOTE: These tests import streamlit_app.data which imports streamlit.
Running them requires streamlit to be installed. They are excluded from the
ML test discovery scope via pytest's default test collection (same tests/ dir).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Package import tests
# ---------------------------------------------------------------------------

def test_streamlit_app_package_imports():
    """streamlit_app package and all submodules import without error."""
    import streamlit_app  # noqa: F401
    from streamlit_app import data, charts, styles, components  # noqa: F401


def test_data_module_imports():
    from streamlit_app import data  # noqa: F401
    assert hasattr(data, "load_final_comparison")
    assert hasattr(data, "load_baseline_comparison")
    assert hasattr(data, "load_raw_model_comparison")
    assert hasattr(data, "load_engineered_model_comparison")
    assert hasattr(data, "get_figure_path")
    assert hasattr(data, "figure_exists")
    assert hasattr(data, "VERIFIED_METADATA")
    assert hasattr(data, "INDICATOR_DESCRIPTIONS")


def test_charts_module_imports():
    from streamlit_app import charts  # noqa: F401
    assert hasattr(charts, "model_comparison_bar")
    assert hasattr(charts, "class_balance_chart")
    assert hasattr(charts, "metrics_table_figure")
    assert hasattr(charts, "MODEL_COLORS")


def test_components_module_imports():
    from streamlit_app import components  # noqa: F401
    assert hasattr(components, "section_header")
    assert hasattr(components, "metric_cards_row")
    assert hasattr(components, "leakage_audit_table")
    assert hasattr(components, "split_timeline")
    assert hasattr(components, "target_formula")
    assert hasattr(components, "conclusion_callout")
    assert hasattr(components, "disclaimer_footer")
    assert hasattr(components, "safe_image")


# ---------------------------------------------------------------------------
# Verified metadata tests
# ---------------------------------------------------------------------------

def test_verified_metadata_keys():
    from streamlit_app.data import VERIFIED_METADATA as m
    required_keys = [
        "symbol", "asset_name", "raw_rows", "date_start", "date_end",
        "n_raw_features", "n_technical_features", "n_engineered_features",
        "technical_indicators", "train_start", "train_end",
        "val_start", "val_end", "test_start", "test_end",
        "test_total", "test_up", "test_down_or_flat",
    ]
    for key in required_keys:
        assert key in m, f"Missing key in VERIFIED_METADATA: {key}"


def test_verified_metadata_values():
    from streamlit_app.data import VERIFIED_METADATA as m
    assert m["symbol"] == "SPY"
    assert m["raw_rows"] == 5457
    assert m["n_raw_features"] == 20
    assert m["n_technical_features"] == 8
    assert m["n_engineered_features"] == 28
    assert m["test_total"] == 819
    assert m["test_up"] == 463
    assert m["test_down_or_flat"] == 356
    assert m["test_up"] + m["test_down_or_flat"] == m["test_total"]


def test_indicator_descriptions_complete():
    from streamlit_app.data import INDICATOR_DESCRIPTIONS, VERIFIED_METADATA
    expected_indicators = set(VERIFIED_METADATA["technical_indicators"])
    described = set(INDICATOR_DESCRIPTIONS.keys())
    assert expected_indicators == described, (
        f"Indicator descriptions don't match metadata.\n"
        f"  Missing descriptions: {expected_indicators - described}\n"
        f"  Extra descriptions: {described - expected_indicators}"
    )


def test_indicator_descriptions_structure():
    from streamlit_app.data import INDICATOR_DESCRIPTIONS
    required_fields = {"full_name", "formula", "window", "description", "causal_note"}
    for name, info in INDICATOR_DESCRIPTIONS.items():
        missing = required_fields - set(info.keys())
        assert not missing, f"Indicator '{name}' missing fields: {missing}"


# ---------------------------------------------------------------------------
# CSV loading tests (real files from results/)
# ---------------------------------------------------------------------------

def _load_csv_direct(filename: str) -> pd.DataFrame:
    """Load a results CSV directly (bypasses Streamlit cache decorator)."""
    path = PROJECT_ROOT / "results" / filename
    assert path.exists(), f"Results file missing: {path}"
    return pd.read_csv(path)


def test_final_comparison_csv_exists():
    path = PROJECT_ROOT / "results" / "final_comparison.csv"
    assert path.exists(), "results/final_comparison.csv must exist"


def test_final_comparison_columns():
    df = _load_csv_direct("final_comparison.csv")
    required_cols = {"Model", "Accuracy", "Precision", "Recall", "F1"}
    assert required_cols.issubset(set(df.columns)), (
        f"Missing columns: {required_cols - set(df.columns)}"
    )


def test_final_comparison_model_names():
    df = _load_csv_direct("final_comparison.csv")
    expected_models = {
        "Persistence",
        "Majority Class",
        "Raw Logistic Regression",
        "Engineered Logistic Regression",
    }
    actual_models = set(df["Model"].tolist())
    assert expected_models == actual_models, (
        f"Model names mismatch.\n"
        f"  Expected: {expected_models}\n"
        f"  Got: {actual_models}"
    )


@pytest.mark.parametrize("model,metric,expected", [
    ("Persistence",                    "Accuracy",  0.5018),
    ("Persistence",                    "Precision", 0.5594),
    ("Persistence",                    "Recall",    0.5594),
    ("Persistence",                    "F1",        0.5594),
    ("Majority Class",                 "Accuracy",  0.5653),
    ("Majority Class",                 "Precision", 0.5653),
    ("Majority Class",                 "Recall",    1.0000),
    ("Majority Class",                 "F1",        0.7223),
    ("Raw Logistic Regression",        "Accuracy",  0.5128),
    ("Raw Logistic Regression",        "Precision", 0.5586),
    ("Raw Logistic Regression",        "Recall",    0.6587),
    ("Raw Logistic Regression",        "F1",        0.6046),
    ("Engineered Logistic Regression", "Accuracy",  0.4945),
    ("Engineered Logistic Regression", "Precision", 0.5706),
    ("Engineered Logistic Regression", "Recall",    0.4276),
    ("Engineered Logistic Regression", "F1",        0.4889),
])
def test_final_comparison_exact_values(model, metric, expected):
    """Verify each metric to 4 decimal places against the committed CSV."""
    df = _load_csv_direct("final_comparison.csv")
    row = df[df["Model"] == model]
    assert len(row) == 1, f"Model '{model}' not found or duplicated in final_comparison.csv"
    actual = float(row[metric].iloc[0])
    assert abs(actual - expected) < 5e-4, (
        f"{model} {metric}: expected ~{expected:.4f}, got {actual:.4f}"
    )


def test_baseline_comparison_csv():
    df = _load_csv_direct("baseline_comparison.csv")
    assert "Model" in df.columns
    assert {"Persistence", "Majority Class"}.issubset(set(df["Model"]))


def test_raw_model_comparison_csv():
    df = _load_csv_direct("raw_model_comparison.csv")
    assert "Raw Logistic Regression" in df["Model"].values


def test_engineered_model_comparison_csv():
    df = _load_csv_direct("engineered_model_comparison.csv")
    assert "Engineered Logistic Regression" in df["Model"].values


# ---------------------------------------------------------------------------
# Figure path helper tests
# ---------------------------------------------------------------------------

def test_get_figure_path_returns_path():
    from streamlit_app.data import get_figure_path
    from pathlib import Path
    p = get_figure_path("predicted_vs_actual.png")
    assert isinstance(p, Path)
    assert p.name == "predicted_vs_actual.png"


def test_figure_exists_returns_bool():
    from streamlit_app.data import figure_exists
    # Should return bool regardless of whether the file exists
    result = figure_exists("predicted_vs_actual.png")
    assert isinstance(result, bool)


def test_figure_paths_are_relative_to_repo():
    """Figure paths must resolve under the repository root, not a user home dir."""
    from streamlit_app.data import get_figure_path
    p = get_figure_path("predicted_vs_actual.png")
    assert str(PROJECT_ROOT) in str(p.parent), (
        "Figure path should resolve inside the repository root"
    )


# ---------------------------------------------------------------------------
# No machine-specific paths in source files
# ---------------------------------------------------------------------------

STREAMLIT_APP_DIR = PROJECT_ROOT / "streamlit_app"
APP_PY = PROJECT_ROOT / "app.py"

_FORBIDDEN_PATH_PATTERNS = [
    "C:\\Users\\",
    "C:/Users/",
    "/home/",
    "/Users/",
    "C:\\\\Users\\\\",
]


def _check_no_absolute_paths(filepath: Path) -> None:
    content = filepath.read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_PATH_PATTERNS:
        assert pattern not in content, (
            f"Machine-specific absolute path '{pattern}' found in {filepath}"
        )


def test_app_py_no_absolute_paths():
    _check_no_absolute_paths(APP_PY)


@pytest.mark.parametrize("filename", [
    "__init__.py", "data.py", "charts.py", "styles.py", "components.py"
])
def test_streamlit_module_no_absolute_paths(filename):
    _check_no_absolute_paths(STREAMLIT_APP_DIR / filename)


# ---------------------------------------------------------------------------
# No secrets
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = [
    "api_key", "API_KEY", "secret", "SECRET", "token", "TOKEN",
    "password", "PASSWORD", "sk-", "Bearer ",
]


def _check_no_secrets(filepath: Path) -> None:
    content = filepath.read_text(encoding="utf-8")
    # Only flag if the pattern is NOT in a comment that says "no secrets"
    lines = content.splitlines()
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip().lower()
        for pattern in _SECRET_PATTERNS:
            if pattern.lower() in stripped:
                # Allow lines that are themselves talking about the absence of secrets
                # e.g. "No secrets or tokens" or "no secrets required"
                if "no " + pattern.lower() in stripped or "not" in stripped:
                    continue
                # Allow docstrings / comments that mention secrets as a concept
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'"):
                    if any(x in stripped for x in ["no ", "never", "not ", "absence"]):
                        continue
                # If none of the above, flag it
                # But be lenient — these tests just look for obvious hard-coded values
                # not general mentions of the concept in comments/docs.
                # Only fail on assignment-like patterns.
                if "=" in line and ('"' in line or "'" in line):
                    val_part = line.split("=", 1)[-1].strip()
                    if len(val_part) > 10 and ('"' in val_part or "'" in val_part):
                        # Skip if value is clearly a path, label, or description
                        if any(x in val_part for x in ["/", "\\", " ", "."]):
                            continue
                        pytest.fail(
                            f"Possible secret in {filepath}:{lineno}: {line.strip()[:80]}"
                        )


def test_app_py_no_secrets():
    _check_no_secrets(APP_PY)


@pytest.mark.parametrize("filename", ["data.py", "charts.py", "styles.py", "components.py"])
def test_streamlit_module_no_secrets(filename):
    _check_no_secrets(STREAMLIT_APP_DIR / filename)
