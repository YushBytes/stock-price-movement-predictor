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

## Option A Requirements Checklist

Verified directly against the repository (not assumed from prior notes)
during the Phase 8 final audit.

| Requirement | Status | Evidence |
|---|---|---|
| Daily historical OHLCV data | PASS | Real SPY data via `yfinance` (`src/data_loader.py`), 5,457 rows, 2005-01-03 to 2026-09-11, cached at `data/raw/SPY.csv` |
| Leak-free next-day target | PASS | `build_target()` (`src/preprocessing.py`): `Target[t]=1` iff `Close[t+1]>Close[t]`; `Close[t+1]` used only inside this function, never persisted as a column |
| At least 3 technical indicators | PASS | 5 indicator families, 8 columns: `SMA_10, SMA_20, RSI_14, MACD, MACD_SIGNAL, MACD_HIST, RETURN_1D, VOLATILITY_20` (`src/features.py`) |
| Explicit feature-rows-vs-target verification | PASS | Notebook Section 7 ("Leak-Free Verification"): real `Date`/`Close[t]`/`Close[t+1]` (labeled "LABEL ONLY — NOT A FEATURE")/`Target` table, with an independent recomputation assertion |
| Persistence baseline | PASS | `persistence_baseline()` (`src/baselines.py`), uses only `Close[t-1]`/`Close[t-2]`; test accuracy 0.5018 |
| Majority-class baseline | PASS | `majority_class_baseline()`, computed from `train_df["Target"]` only; test accuracy 0.5653 |
| Raw price/volume model | PASS | `build_raw_features()`, 20 lagged OHLCV columns + `LogisticRegression`; test accuracy 0.5128 |
| Engineered technical-indicator model | PASS | `build_engineered_features()`, 28 columns (20 raw + 8 indicators) + `LogisticRegression`; test accuracy 0.4945 |
| Strictly time-based splitting | PASS | `chronological_split()`; train 2005-01-03→2020-03-05, validation 2020-03-06→2023-06-05, test 2023-06-06→2026-09-10; no `train_test_split`, no shuffling anywhere in `src/` (verified by direct source search) |
| Scalers fitted only on training data | PASS | `StandardScaler.fit(X_train)` only, in both the raw and engineered pipelines; proven (not just asserted) via the non-zero scaled-test-mean signature (~6.6) |
| Directional classification metrics | PASS | Accuracy/Precision/Recall/F1 for all four models (`compute_classification_metrics`), positive class = UP |
| Class balance report | PASS | Full/train/validation/test UP vs. DOWN_OR_FLAT counts and percentages reported (see "Target Construction and Leakage Prevention" below) |
| Four-way comparison table | PASS | `results/final_comparison.csv`; all four models, same 819-row test target |
| Predicted vs. actual plot | PASS | `figures/predicted_vs_actual.png` — actual vs. engineered-model predicted direction over the real test dates |
| Reproducible Jupyter Notebook | PASS | `notebooks/stock_price_movement_predictor.ipynb` executes top-to-bottom with a fresh kernel, exit code 0, no manual intervention |
| README.md | PASS | This file |
| Public GitHub-ready repository | PASS | Isolated git repo (not nested in an unrelated parent directory), `.gitignore` excludes venv/cache/generated data while keeping source/tests/docs/results tracked, no secrets or machine-specific absolute paths in any tracked file (verified by direct search this phase) |
| No temporal data leakage | PASS | See "Leakage Audit" in the Phase 8 report; only permitted future-look is `Close[t+1]` inside `build_target` |
| Honest generalization analysis | PASS | "Findings" below states plainly that no ML model beat the Majority Class baseline — not adjusted or hidden |

## Current Status

**Phase 7 — Engineered feature model and final four-way comparison
(project complete through Option A's required scope).** The engineered
Logistic Regression (28 features: 20 raw lagged OHLCV + 8 technical
indicators) has been trained and evaluated on the identical chronological
test partition used by every other model. The final four-way comparison
(Persistence, Majority Class, Raw Logistic Regression, Engineered
Logistic Regression) is complete — see "Final Results" and "Findings"
below. **The honest result: the engineered model did not improve on the
raw model, and no machine-learning model in this project beat the
Majority Class baseline.** This is reported as-is, not adjusted.

## Planned Approach

1. ~~Fetch daily OHLCV data for the selected symbol via `yfinance` and cache
   it locally~~ — **done** (Phase 2, see "Data Acquisition" above).
2. ~~Construct a leak-free next-day-direction target using only
   `Close[t]` and `Close[t+1]`~~ — **done** (Phase 3, see "Target
   Construction and Leakage Prevention" below).
3. ~~Report class balance and basic exploratory statistics~~ — **done**
   (Phase 3, folded into the same section since it needs the target/split).
4. ~~Establish two naive baselines — persistence and majority-class~~ —
   **done** (Phase 4, see "Naive Baselines" below).
5. ~~Train a Logistic Regression model on raw OHLCV features~~ — **done**
   (Phase 5, see "Raw Price/Volume Model" below).
6. ~~Engineer technical indicators (moving averages, RSI, MACD, rolling
   volatility, returns) computed causally~~ — **done** (Phase 6, see
   "Technical Indicators" below).
7. ~~Train a second model on the engineered features~~ — **done** (Phase 7,
   see "Engineered Feature Model" below).
8. ~~Evaluate all four approaches on an untouched, chronologically final
   test window and produce a four-way comparison~~ — **done** (Phase 7,
   see "Final Results" below).
9. ~~Visualize predicted vs. actual direction over the test window~~ —
   **done** (Phase 7, `figures/predicted_vs_actual.png`).

Full architecture, dataset rationale, and leakage-prevention plan were
established in the Phase 0 design document and are reflected in the module
docstrings under `src/`.

## Data Acquisition

- **Asset:** SPY (SPDR S&P 500 ETF Trust)
- **Source:** Yahoo Finance via the `yfinance` package (`fetch_ohlcv` in
  `src/data_loader.py`), downloaded with `auto_adjust=False` to keep raw,
  as-quoted OHLC prices rather than Yahoo's retroactively dividend/split
  -adjusted close (see "Reproducibility" below for why).
- **Frequency:** Daily
- **Raw data location:** `data/raw/SPY.csv` — generated by code, **not**
  committed to git (see `.gitignore`); a fresh clone must regenerate it.

**To acquire the data:**

```python
import config
from src.data_loader import load_or_fetch_ohlcv

df = load_or_fetch_ohlcv(config.SYMBOL, config.START_DATE, config.END_DATE, config.RAW_DATA_FILE)
```

Running the notebook's "Data Loading" cell does exactly this. Behavior:

- If `data/raw/SPY.csv` already exists **and** its date coverage matches
  `config.py`'s `START_DATE`/`END_DATE` (within a small tolerance for
  weekends/holidays), the cached file is loaded — no network call.
- Otherwise (missing, or the configuration changed), fresh data is
  downloaded from Yahoo Finance and the cache is overwritten.
- Duplicate dates, if any are returned by the source, are removed (keeping
  the last occurrence) and the number removed is printed — never silently
  discarded.

**Data validation performed** (`validate_ohlcv` / `print_validation_report`
in `src/data_loader.py`), run on the real downloaded dataset and asserted
in the notebook:

1. Dataset is non-empty and has all required columns (`Date, Open, High,
   Low, Close, Volume`).
2. Dates are non-null and sorted chronologically (oldest → newest).
3. No duplicate dates remain.
4. OHLCV columns are numeric; missing and infinite values are counted.
5. `High >= Low`, and `Open`/`Close` fall within `[Low, High]`.
6. `Volume` is non-negative.
7. Actual date coverage is compared against the requested `config.py` range.

As of the most recent run captured in the committed notebook, the dataset
contained **5,457 rows spanning 2005-01-03 to 2026-09-11**, with **zero**
missing values, duplicate dates, infinite values, invalid OHLC
relationships, or negative-volume rows. These numbers will grow on future
runs as new trading days occur (since `END_DATE` is `None`, meaning "up to
the most recent available session") — re-run the notebook for current
figures rather than treating this snapshot as fixed.

## Target Construction and Leakage Prevention

**Exact definition.** For trading day `t`:

```
Target[t] = 1   if Close[t+1] >  Close[t]     (UP)
Target[t] = 0   if Close[t+1] <= Close[t]     (DOWN_OR_FLAT)
```

- **Encoding:** `1 = UP`, `0 = DOWN_OR_FLAT`. This single convention
  (`Target`, capitalized) is used consistently everywhere in the project —
  no alternate spelling or casing is introduced elsewhere.
- **Equal closes are DOWN_OR_FLAT, not discarded.** An unchanged price is
  not a directional "up" move. Silently dropping tie rows would shrink the
  dataset and bias the class balance based on an arbitrary tie-breaking
  choice, so ties are kept and labeled 0.
- **The final row is removed.** The last row in a chronologically-sorted
  dataset has no following trading day, so `Close[t+1]` doesn't exist for
  it — no label is fabricated; that row is dropped instead.
- **How future information is kept out of features:** `Close[t+1]` is used
  *only* inside `build_target` (`src/preprocessing.py`) to compute the
  label for row `t`. It is never attached to the returned DataFrame as a
  column, and it must never appear in any model's feature matrix built in
  later phases — `assert_no_target_leakage()` exists specifically to check
  a feature-column list against the target column and known future-derived
  helper names once a real feature matrix is built (Phase 5+). A negative
  `shift` (looking at `t+1`) is used only for target construction; feature
  construction (Phase 5/6) will never use a negative shift.
- **Manual verification:** the notebook's "Leak-Free Verification" section
  prints `Date`, `Close[t]`, `Close[t+1]` (explicitly labeled "USED ONLY
  FOR LABEL CONSTRUCTION — NOT A MODEL FEATURE"), and `Target` side by
  side for real rows of the dataset, and independently recomputes `Target`
  from that table to assert it matches — so an evaluator doesn't have to
  take the implementation's word for it.

**Chronological splitting.** `chronological_split()` divides the labeled
dataset into TRAIN (oldest `TRAIN_RATIO`, 70%), VALIDATION (next
`VALIDATION_RATIO`, 15%), and TEST (remaining ~15%) by row order alone —
never `sklearn.train_test_split`, never shuffled, never randomly sampled.
Random splitting is not used because it would let the model "see" data
from time periods after (or interleaved with) what it's tested on,
producing an artificially inflated and meaningless accuracy — the whole
point of a time-series evaluation is to simulate only ever having the past
available, so the split must respect real chronological order.

**Class-balance analysis** (`compute_class_balance()` / real numbers from
the current dataset, 5,456 labeled rows after the final row was dropped):

| Partition | Rows | UP (1) | DOWN_OR_FLAT (0) | % UP |
|---|---|---|---|---|
| Full dataset | 5,456 | 2,980 | 2,476 | 54.62% |
| Train | 3,819 | 2,082 | 1,737 | 54.52% |
| Validation | 818 | 435 | 383 | 53.18% |
| Test | 819 | 463 | 356 | 56.53% |

The classes are mildly imbalanced (more UP days than DOWN_OR_FLAT days,
consistent with SPY's long-term uptrend) but not severely so — this is
reported honestly here, not glossed over, because it matters for choosing
and interpreting classification metrics in later phases. These are real
numbers from the notebook's last executed run, not estimates; re-running
the notebook as new trading days accumulate will shift them slightly.

No model has been trained yet and no accuracy claim is made here — this
section is about the data pipeline, not predictive performance.

## Naive Baselines

Both baselines are evaluated on the same chronological test partition
(`test_df`, 819 rows, 2023-06-06 to 2026-09-10) established in Phase 3,
against the same `test_df["Target"]` values, using the same positive-class
convention (UP = 1). Neither baseline fits anything — they are rule-based
reference points a real model must beat.

### Persistence Baseline

- **Definition:** predicts that tomorrow's direction matches the most
  recently observed one-day price direction: `prediction[t] =
  direction[t-1]`, where `direction[k] = 1 if Close[k] > Close[k-1] else 0`.
- **Why it's a useful baseline:** it captures naive momentum/trend-following
  — "whatever just happened will keep happening" — without any model
  fitting, so it's the simplest possible non-constant predictor a real
  model should outperform.
- **How historical direction is used:** `persistence_baseline()`
  (`src/baselines.py`) is given the *entire* chronological history
  (train+validation+test) for context, so the first prediction in the test
  window can correctly use the last observed direction from the
  *validation* period rather than being dropped for lack of in-partition
  history. Only `test_df`'s dates and row count determine what gets
  predicted and how many predictions come back.
- **How leakage is avoided:** the prediction for row `t` only ever reads
  `Close[t-1]` and `Close[t-2]` — never `Close[t]`, never `Close[t+1]`, and
  never the row's own `Target` value (using `Target[t]` to predict
  `Target[t]` would be circular, since `Target[t]` is itself defined using
  `Close[t+1]`, the very thing being predicted).

### Majority-Class Baseline

- **Definition:** always predicts the single class most frequent in the
  **training partition**, for every evaluation row.
- **Training-only class determination:** `majority_class_baseline()` takes
  only `train_df["Target"]` as input — there is no parameter through which
  validation or test data could influence which class is chosen. On the
  real data, train is 54.52% UP / 45.48% DOWN_OR_FLAT, so the majority
  class is **UP (1)**. An exact 50/50 tie (not the case here) is broken
  deterministically toward UP (1) — a fixed, documented rule, not a
  data-dependent one.
- **Why it's necessary with imbalanced classes:** accuracy alone can be
  misleading when classes aren't 50/50 — a model could look "accurate"
  just by leaning toward the majority class. This baseline makes that
  effect explicit and quantifiable, so later models are judged against
  what "doing nothing clever" already achieves, not against a naive 50%.

### Actual Baseline Results

Computed by `src/evaluation.py::compute_classification_metrics` on the
real test partition (819 predictions each), independently cross-checked
by manually recomputing accuracy in the notebook:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Persistence | 0.5018 | 0.5594 | 0.5594 | 0.5594 |
| Majority Class | 0.5653 | 0.5653 | 1.0000 | 0.7223 |

Saved to `results/baseline_comparison.csv` (committed to the repo as
deliverable evidence — see "Project Structure" below). The
majority baseline's recall is trivially 1.0 because it always predicts UP
— it "catches" every actual UP day by never predicting anything else, at
the cost of also predicting UP on every DOWN_OR_FLAT day. Neither number
here is being claimed as "good" — they exist purely as the reference point
Phase 5+'s real models must beat, and the fact that persistence lands
almost exactly at a coin flip (50.18%) is an expected, honest result for
a liquid index's daily direction, not a bug.

## Raw Price/Volume Model

The first real machine-learning model in this project — **raw price/volume
only, no technical indicators yet** (Option A requires training on raw
price/volume before retraining on engineered indicators; indicators are a
later phase).

### Raw Feature Philosophy and Lag Structure

At prediction time `t`, the model may only use information already
observed at or before `t`. `build_raw_features()` (`src/features.py`)
builds causally-lagged columns for each of `Open, High, Low, Close,
Volume` at lags **1, 2, 3, and 5** trading days (e.g. `Close_lag1[t] =
Close[t-1]`, `Volume_lag5[t] = Volume[t-5]`) — 20 features in total, no
same-day (lag 0) column. Every one of these values was observed strictly
before `Target[t]` is decided, so using it as a feature cannot leak the
outcome being predicted.

### Why Lagging Prevents Look-Ahead

Only `pandas.Series.shift(k)` with a **positive** `k` is used — never
`shift(-1)`/`shift(-2)`. A negative shift (looking at `t+1`) is reserved
exclusively for `build_target()`; `build_raw_features()` rejects a
non-positive lag with a `ValueError` rather than silently accepting one,
so this isn't just a convention, it's enforced.

### Handling Lag-Induced Missing Rows

The first `max(lags) = 5` rows of the dataset can't have every lag
populated (fewer than 5 days of preceding history exist) — they are
**dropped**, never forward-filled, backward-filled, or invented. Applying
this to the full chronological dataset before partitioning means only the
very start of the **training** partition loses rows: **5 rows lost**,
train shrinks from 3,819 to 3,814 usable feature rows. The validation
(818 rows) and test (819 rows) partitions are **fully intact**, because
features for their first rows can correctly look back into whatever
partition immediately precedes them (the same reasoning the persistence
baseline already relies on) — the existing `train_df`/`validation_df`/
`test_df` partitions from Phase 3 are reused by `Date` membership rather
than re-splitting from scratch, so the test window stays identical to the
one the baselines were evaluated on: **2023-06-06 to 2026-09-10**.

### Logistic Regression and Training-Only Scaling

`sklearn.linear_model.LogisticRegression` (`src/models.py`,
`random_state` fixed to `config.RANDOM_SEED`, `max_iter=1000`, no
hyperparameter search — repeatedly tuning against the test set is exactly
the kind of leakage this project avoids) fits a linear decision boundary
in scaled feature space and outputs a probability of UP; thresholding at
0.5 gives the binary prediction.

`sklearn.preprocessing.StandardScaler` is fit **only** on `X_train`:

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

`scaler.fit_transform(X_test)` and `scaler.fit_transform(X_val)` are never
called. Concrete evidence this is real, not just asserted: after scaling,
`X_train`'s per-feature mean/std land at ~0/~1 (by definition of fitting
on it) — but `X_test`, scaled with `X_train`'s parameters, does **not**
land near 0/1 (its scaled means come out around **6.6**, stds around
**1.58**, for the real dataset). That gap is exactly what you'd expect
when a scaler's parameters come from a *different, earlier* period than
the data being transformed — direct proof the scaler never saw test data
during fitting.

### Role of Validation vs. Test

The validation partition exists for development/diagnostics; the test
partition is the final, untouched evaluation window. This phase performs
no hyperparameter tuning, so validation isn't used to pick anything here
— but the split discipline (fit only on train, never peek at test) is
already in place for when tuning is introduced.

### Actual Test Metrics and Comparison with Baselines

Computed by `compute_classification_metrics` on the same 819-row test
target used by both baselines, independently cross-checked by manually
recomputing accuracy in the notebook:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Persistence | 0.5018 | 0.5594 | 0.5594 | 0.5594 |
| Majority Class | 0.5653 | 0.5653 | 1.0000 | 0.7223 |
| Raw Logistic Regression | 0.5128 | 0.5586 | 0.6587 | 0.6046 |

Saved to `results/raw_model_comparison.csv` (committed alongside
`results/baseline_comparison.csv`, which is left untouched).

**Honest discussion of the result:** the raw model beats Persistence
(51.28% vs. 50.18% accuracy) but does **not** beat Majority Class (56.53%)
on accuracy. This is not being reported as a success. With a mildly
imbalanced test set (56.53% UP), a model that leans toward predicting UP
more often (its recall of 0.6587 vs. Persistence's 0.5594 shows it does)
can still trail a baseline that predicts UP unconditionally on raw
accuracy, precisely because that baseline's simplicity is well-suited to
an imbalanced test period. Twenty lagged raw price/volume numbers appear
to carry only weak, if any, genuine directional signal beyond what the
class imbalance itself already provides — consistent with market
efficiency and exactly the kind of honest, unglamorous result this
project is designed to surface rather than hide. Whether engineered
technical indicators do any better is answered in "Findings" below (the
short answer: no, they did not).

## Technical Indicators

Feature engineering only in this section — model training on these
indicators is covered in "Engineered Feature Model" below.

### Indicators Selected and Exact Windows

| Indicator | Column(s) | Window | Purpose |
|---|---|---|---|
| Simple Moving Average | `SMA_10`, `SMA_20` | 10, 20 days | Smoothed recent-trend reference; price relative to it is a classic momentum/mean-reversion signal. |
| Relative Strength Index | `RSI_14` | 14 days | Ratio of average gains to losses, scaled 0–100; extreme readings often precede a reversal or continuation. |
| MACD | `MACD`, `MACD_SIGNAL`, `MACD_HIST` | 12/26/9 days | Fast-EMA-minus-slow-EMA trend/momentum signal; crossovers are widely used as trend-change signals. |
| Daily Return | `RETURN_1D` | 1 day | Simplest possible momentum/reversal signal (`Close.pct_change(1)`). |
| Rolling Volatility | `VOLATILITY_20` | 20 days | 20-day rolling standard deviation of `RETURN_1D`; volatility regime affects how reliable directional signals are. |

Implemented via `build_technical_indicators()` (`src/features.py`), using
the `ta` library (never `pandas-ta`, which is unmaintained) for SMA/RSI/
MACD and plain pandas (`pct_change`, `rolling().std()`) for return/
volatility.

### Causal Calculation

Every indicator above is a **trailing** (backward-looking) transformation
of historical `Close` prices: `ta`'s SMA/RSI/MACD implementations were
inspected directly (their source contains no `center=True` and no
negative `shift`) before being adopted, and the return/volatility
formulas use only `pct_change(1)`/`rolling(window)`, which by construction
never look ahead. `Target[t] = 1 if Close[t+1] > Close[t]`, so an
indicator computed using information available *through* day `t` (e.g.
`RSI_14[t]`) is valid for predicting `Target[t]` without needing to see
day `t+1` — this timing relationship is spelled out explicitly in the
notebook's "Technical Indicators" section.

A **perturbation test** makes this concrete rather than just asserted:
the notebook changes a single real observation (the very last day's
`Close`) and rebuilds every indicator, then asserts that every row more
than 40 rows before that change is byte-for-byte identical to the
un-perturbed version. It passes on the real dataset.

### Warm-Up Handling

Rows within any indicator's warm-up period are **dropped**, never
forward-filled, backward-filled, or invented. On the real dataset, this
removes **33 rows** (2005-01-10 to 2005-02-17) — `MACD`'s 26+9-day
requirement is the longest warm-up among the five indicators, and
therefore determines the cutoff for all of them together.

### Raw vs. Engineered Features

The next phase's engineered feature set will be `build_raw_features()`'s
20 lagged raw OHLCV columns (Phase 5) **plus** these 8 technical-indicator
columns — indicators are added *on top of* raw features, not used as a
replacement, so the eventual comparison actually tests whether indicators
add signal beyond raw price/volume history rather than testing two
unrelated feature sets against each other.

This section covers feature engineering only — see "Engineered Feature
Model" and "Findings" below for the actual answer to whether these
indicators improve prediction accuracy (they did not, under this setup).

## Engineered Feature Model

**28 total features**: the 20 raw lagged OHLCV columns from Phase 5
(`build_raw_features`) plus the 8 technical indicators from Phase 6
(`build_technical_indicators`), combined by `build_engineered_features()`
(`src/features.py`) — which reuses both existing builders directly rather
than duplicating any formula, and aligns them by `Date` (an inner join),
so the combined dataset naturally begins wherever the **later** of the
two warm-up cutoffs falls (33 rows, driven by the technical indicators —
identical to Phase 6's cutoff, since 33 > 5).

- **Model:** `sklearn.linear_model.LogisticRegression`, the exact same
  deterministic configuration as the raw model (`random_state =
  config.RANDOM_SEED`, `max_iter=1000`, no hyperparameter search).
- **Scaler:** `StandardScaler` fit **only** on `X_train_eng`
  (3,786 rows × 28 features — train lost 33 rows to indicator warm-up);
  `X_val_eng`/`X_test_eng` transformed with those fitted parameters only.
  Concrete proof, not just assertion: scaled `X_test_eng` means land
  around **6.63** (not ~0), the same signature seen in the raw model,
  confirming the scaler never saw test data.
- **Chronological evaluation:** the existing `train_df`/`validation_df`/
  `test_df` partitions from Phase 3 are reused by `Date` membership —
  never re-split — so the **test period is unchanged**:
  **2023-06-06 to 2026-09-10, 819 observations**, identical to every
  other model in this project.
- **Metrics** (`compute_classification_metrics`, same function used for
  every other model): Accuracy 0.4945, Precision 0.5706, Recall 0.4276,
  F1 0.4889 — independently cross-checked by manually recomputing
  accuracy in the notebook (matches to 9 decimal places).

## Final Results

The complete four-way comparison, all four models evaluated on the
identical 819-row test target:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Persistence | 0.5018 | 0.5594 | 0.5594 | 0.5594 |
| Majority Class | 0.5653 | 0.5653 | 1.0000 | 0.7223 |
| Raw Logistic Regression | 0.5128 | 0.5586 | 0.6587 | 0.6046 |
| Engineered Logistic Regression | 0.4945 | 0.5706 | 0.4276 | 0.4889 |

Saved to `results/final_comparison.csv` (committed to the repo, along with
`results/baseline_comparison.csv`, `results/raw_model_comparison.csv`,
and `results/engineered_model_comparison.csv` — all four are preserved
side by side; nothing is overwritten).

**Class imbalance and why accuracy alone is insufficient.** The test
partition is mildly imbalanced toward UP (463 UP / 56.53% vs. 356
DOWN_OR_FLAT / 43.47%, from Phase 3's class-balance analysis, unchanged).
This is exactly why Majority Class's 0.5653 accuracy is not impressive on
its own merits, and exactly why a model's accuracy must be read alongside
precision/recall/F1: Majority Class's F1 of 0.7223 looks strong only
because it trivially achieves recall = 1.0 (it predicts UP unconditionally,
so it never misses a real UP day) at the cost of precision no better than
its own accuracy. **A model with accuracy near 0.51 is not demonstrating
useful predictive superiority over a rule that does no learning at all.**

## Findings

- **Highest accuracy:** Majority Class (0.5653).
- **Highest F1:** Majority Class (0.7223).
- **Did the engineered model beat the raw model?** No — worse on
  Accuracy (0.4945 vs. 0.5128), Recall (0.4276 vs. 0.6587), and F1 (0.4889
  vs. 0.6046); marginally higher Precision (0.5706 vs. 0.5586), which does
  not offset the rest.
- **Did the engineered model beat Persistence?** No (0.4945 vs. 0.5018).
- **Did the engineered model beat Majority Class?** No (0.4945 vs. 0.5653).
- **Did technical indicators improve the raw model?** **No.** Adding the
  8 technical indicators to the 20 raw lagged features made this
  particular linear model's out-of-sample accuracy *worse*, not better.
  A plausible (untested) explanation: the indicators are largely
  redundant derivations of the same `Close` series already present in the
  raw lags, so for a linear model they may add correlated noise/dimensions
  rather than independent signal.
- **Did any model beat the naive baselines?** The raw model beat
  Persistence but not Majority Class. The engineered model beat neither.
  **No machine-learning model in this project beat the Majority Class
  baseline.**

**Limitations:** a single linear model (no hyperparameter search or
alternative architectures tried); one instrument (SPY) and one fixed
chronological split; five indicator families at conventional fixed
windows, not selected or tuned; no transaction costs, slippage, or
position sizing modeled; **no statistical significance testing was
performed** on any metric difference above, so none of these differences
should be read as proven non-random. This project makes **no claim** that
any model here can reliably predict market direction, and **no claim of
real-world trading profitability** — the goal throughout has been a
rigorous, honest time-series classification experiment, and an honest
experiment is allowed to produce a negative result.

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
├── results/             # exported metrics tables (generated, committed as deliverable evidence)
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
- Raw OHLC prices (`auto_adjust=False`) are used instead of Yahoo's default
  adjusted close, because that adjustment retroactively rewrites historical
  prices whenever a new dividend/split is recorded — a subtle form of
  look-ahead bias this project avoids from the data layer up.
- The downloaded dataset is validated (structure, ordering, duplicates,
  numeric sanity, OHLC relationships, volume) every time it's loaded, and
  the notebook asserts on that validation report rather than assuming the
  data is clean.
- The next-day target is built with a strict, tested definition
  (`build_target`) that never fabricates a label for the unlabeled final
  row and never lets `Close[t+1]` leak into the feature set — see "Target
  Construction and Leakage Prevention" above.
- Raw feature lags use only positive `shift(k)` values (`build_raw_features`
  rejects a non-positive lag outright); `LogisticRegression` is trained
  with a fixed `random_state` and no hyperparameter search, so re-running
  the notebook reproduces the same model and metrics.

To set up locally:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
jupyter notebook notebooks/stock_price_movement_predictor.ipynb
```

To run the test suite (126 deterministic tests, no network access required
— they use small hand-built DataFrames, never the real dataset):

```bash
pytest tests/ -v
```

To regenerate everything from scratch (fresh data pull, all figures, all
results CSVs): open the notebook and Restart Kernel + Run All. The first
run downloads real data from Yahoo Finance (`data/raw/SPY.csv`, not
committed); every run after that reuses the cached file unless
`config.py`'s date range changes.

## Future Implementation Phases

| Phase | Scope |
|---|---|
| 2 | ~~Data acquisition and validation~~ — **done** |
| 3 | ~~Leak-free target construction, chronological split, class balance~~ — **done** |
| 4 | ~~Persistence and majority-class baselines~~ — **done** |
| 5 | ~~Raw OHLCV model~~ — **done** |
| 6 | ~~Technical indicators~~ — **done** |
| 7 | ~~Engineered-feature model, time-series evaluation, four-way
comparison, prediction visualization~~ — **done** |
| 8 | Final project audit / polish (see "Recommendation for Phase 8" in the Phase 7 report) |

All Option A required deliverables are implemented and evaluated as of
Phase 7. Results, metrics, and conclusions in this README were added
**only** after they were produced and verified in the notebook — nothing
here is fabricated or anticipated.
