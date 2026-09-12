# Stock Price Movement Predictor

## Overview

A reproducible machine-learning project that predicts whether a liquid
equity/index will close **higher or lower tomorrow than today** (a binary
Up/Down classification problem), built for the GitHub Community SRM (GCSRM)
Recruitment 2026 Technical Track — AI & Machine Learning, Option A.

This is an educational project demonstrating correct time-series ML
methodology (no leakage, no shuffling, honest baselines) — it is **not** a
trading system and makes no claim of profitability.

## Problem Statement

Given daily OHLCV (Open, High, Low, Close, Volume) history for a single
liquid instrument, predict the **direction** (Up or Down) of the next
trading day's closing price relative to today's close, using only
information available as of today's close.

## Objective

- Build a leak-free, chronologically-evaluated binary classifier.
- Compare it honestly against two naive baselines (persistence and
  majority-class) and against a simpler raw-feature model, to determine
  whether engineered technical indicators add real predictive value.
- Demonstrate, in a way a reviewer can verify line by line, that no future
  information ever reaches a feature or a fitted preprocessor.

## Current Status

**Phase 1 — Project foundation.** The repository structure, configuration,
dependency pinning, and module skeletons are in place. No data has been
fetched and no model has been trained yet. All performance-related sections
below are placeholders until the corresponding phase is implemented.

## Planned Approach

1. Fetch daily OHLCV data for the selected symbol via `yfinance` and cache
   it locally (Phase 2).
2. Construct a leak-free next-day-direction target using only
   `Close[t]` and `Close[t+1]` (Phase 3).
3. Report class balance and basic exploratory statistics (Phase 4).
4. Establish two naive baselines — persistence and majority-class
   (Phase 5).
5. Train a Logistic Regression model on raw OHLCV features (Phase 6).
6. Engineer technical indicators (moving averages, RSI, MACD, rolling
   volatility, returns) computed causally (Phase 7).
7. Train a second model on the engineered features (Phase 8).
8. Evaluate all four approaches on an untouched, chronologically final
   test window and produce a four-way comparison (Phases 9–10).
9. Visualize predicted vs. actual direction over the test window
   (Phase 11).

Full architecture, dataset rationale, and leakage-prevention plan were
established in the Phase 0 design document and are reflected in the module
docstrings under `src/`.

## Technology Stack

- Python 3.12
- pandas, numpy
- scikit-learn
- matplotlib
- `ta` (maintained technical-analysis library — **not** `pandas-ta`)
- Jupyter Notebook

## Project Structure

```
stock-price-movement-predictor/
├── data/
│   ├── raw/            # cached OHLCV pull (fetched by code, not committed)
│   └── processed/      # optional feature-engineered snapshots
├── notebooks/
│   └── stock_price_movement_predictor.ipynb   # main deliverable notebook
├── src/
│   ├── data_loader.py      # fetch/cache OHLCV data
│   ├── preprocessing.py    # leak-free target + chronological splitting
│   ├── features.py         # raw + technical-indicator features
│   ├── baselines.py        # persistence + majority-class baselines
│   ├── models.py           # Logistic Regression training
│   └── evaluation.py       # metrics + comparison table helpers
├── models/              # saved fitted model artifacts (generated, gitignored)
├── results/             # exported metrics tables (generated, gitignored)
├── figures/             # saved plots (generated, gitignored)
├── tests/               # basic project/setup tests
├── config.py            # all project constants (symbol, dates, seed, paths)
├── requirements.txt     # pinned dependency versions
├── .gitignore
└── README.md
```

## Reproducibility

- Every project constant (symbol, date range, random seed, output paths)
  lives in `config.py` — nothing is hard-coded across multiple files.
- Raw data is never committed; it is fetched reproducibly through
  `src/data_loader.py` from a pinned date range and cached to `data/raw/`.
- Dependencies are pinned in `requirements.txt` to exact versions verified
  to install together under **Python 3.12**.
- All train/validation/test splits are chronological (row-order based).
  Random shuffling and `sklearn.train_test_split(shuffle=True)` are never
  used on this time-series data.
- Any preprocessing step that "learns" from data (e.g. `StandardScaler`)
  is fit only on the training partition and reused, never refit, on
  validation/test data.

To set up locally:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
jupyter notebook notebooks/stock_price_movement_predictor.ipynb
```

## Future Implementation Phases

| Phase | Scope |
|---|---|
| 2 | Data acquisition and validation |
| 3 | Leak-free target construction |
| 4 | Class balance and exploratory analysis |
| 5 | Persistence and majority-class baselines |
| 6 | Raw OHLCV model |
| 7 | Technical indicators |
| 8 | Engineered-feature model |
| 9 | Time-series evaluation |
| 10 | Four-way comparison |
| 11 | Prediction visualization |
| 12–15 | Notebook polish, documentation, reproducibility audit, final review |

Results, metrics, and conclusions will be added to this README **only**
after they have been produced and verified in the notebook — nothing here
is fabricated or anticipated.
