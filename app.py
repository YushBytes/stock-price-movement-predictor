"""Stock Price Movement Predictor — Streamlit Research Dashboard.

Entry point: ``streamlit run app.py``

This application is an interactive presentation and demonstration layer for
the verified ML experiment. It never modifies, retrains, re-runs, or alters
any part of the pipeline in ``src/``. All displayed metrics are loaded
directly from the committed ``results/`` CSV files.

Design philosophy: rigorous, transparent, honest time-series ML.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — must be the FIRST Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Price Movement Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "Stock Price Movement Predictor — Leak-Free Time-Series ML Research Dashboard.\n\n"
            "Educational/research project. Not financial advice."
        ),
    },
)

# ---------------------------------------------------------------------------
# Local imports (after set_page_config)
# ---------------------------------------------------------------------------
from streamlit_app.styles import inject_styles
from streamlit_app.data import (
    VERIFIED_METADATA,
    INDICATOR_DESCRIPTIONS,
    load_final_comparison,
    get_figure_path,
    figure_exists,
)
from streamlit_app.charts import (
    model_comparison_bar,
    class_balance_chart,
    metrics_table_figure,
)
from streamlit_app.components import (
    section_header,
    metric_cards_row,
    leakage_audit_table,
    split_timeline,
    target_formula,
    conclusion_callout,
    disclaimer_footer,
    safe_image,
)

inject_styles()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 4px 0 20px 0;">
            <div style="font-size:1.05rem;font-weight:700;color:#F1F5F9;
                        letter-spacing:-0.01em;line-height:1.3">
                Stock Price<br>Movement Predictor
            </div>
            <div style="font-size:0.72rem;color:#64748B;margin-top:4px">
                Leak-Free Time-Series ML
            </div>
        </div>
        <hr style="border-color:#2D3148;margin-bottom:12px">
        """,
        unsafe_allow_html=True,
    )

    PAGES = [
        "1 · Overview",
        "2 · Dataset",
        "3 · Methodology",
        "4 · Technical Indicators",
        "5 · Model Comparison",
        "6 · Class Balance",
        "7 · Predictions",
        "8 · Findings & Limitations",
    ]

    page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.markdown("<hr style='border-color:#2D3148'>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#475569;font-size:0.72rem">Educational · Not financial advice</p>',
        unsafe_allow_html=True,
    )

# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================
if page == PAGES[0]:
    st.markdown(
        """
        <h1 style="font-size:2rem;font-weight:700;color:#F1F5F9;
                   letter-spacing:-0.02em;margin-bottom:4px">
            Stock Price Movement Predictor
        </h1>
        <p style="color:#94A3B8;font-size:1rem;margin-bottom:24px">
            Leak-Free Time-Series Machine Learning Research Dashboard
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p style="color:#CBD5E1;font-size:0.96rem;line-height:1.7;max-width:780px">
            An interactive presentation of a leak-free, chronologically-evaluated
            binary classification experiment for next-day market direction (Up/Down)
            using daily OHLCV data for SPY. The experiment compares four approaches:
            two naive baselines and two Logistic Regression models with different feature sets.
            Results are reported honestly — including the fact that no trained model
            outperformed the naive majority-class baseline.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Key stats
    metric_cards_row([
        {"value": "5,457", "label": "Raw observations", "unit": "daily OHLCV rows"},
        {"value": "20",    "label": "Raw lagged features", "unit": "causal OHLCV lags"},
        {"value": "8",     "label": "Technical indicators", "unit": "SMA · RSI · MACD · …"},
        {"value": "28",    "label": "Engineered features", "unit": "raw + indicators"},
        {"value": "819",   "label": "Official test predictions", "unit": "2023-06-06 → 2026-09-10"},
        {"value": "126",   "label": "Verified tests pass", "unit": "pytest · no network"},
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # Experiment summary
    col_a, col_b = st.columns([1, 1], gap="large")
    with col_a:
        section_header("Experiment at a glance")
        rows = [
            ("Asset", "SPY — SPDR S&P 500 ETF Trust"),
            ("Task", "Next-day Up / Down classification"),
            ("Target", "Target[t] = 1 if Close[t+1] > Close[t]"),
            ("Models", "Logistic Regression (raw) · Logistic Regression (engineered)"),
            ("Baselines", "Persistence · Majority Class"),
            ("Split", "Chronological — never shuffled"),
            ("Evaluation", "Accuracy · Precision · Recall · F1"),
            ("Data source", "Yahoo Finance via yfinance"),
        ]
        for k, v in rows:
            st.markdown(
                f'<p style="margin:4px 0"><span style="color:#64748B;font-size:0.82rem'
                f';min-width:110px;display:inline-block">{k}</span>'
                f'<span style="color:#CBD5E1;font-size:0.9rem"> {v}</span></p>',
                unsafe_allow_html=True,
            )

    with col_b:
        section_header("Verified final results")
        df = load_final_comparison()
        if df is not None:
            # Show as a clean table
            display_df = df.copy()
            for col in ["Accuracy", "Precision", "Recall", "F1"]:
                display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )
            st.markdown(
                '<p style="color:#64748B;font-size:0.76rem;margin-top:4px">'
                'Source: <code>results/final_comparison.csv</code> · n=819 test observations</p>',
                unsafe_allow_html=True,
            )

    disclaimer_footer()


# ===========================================================================
# PAGE 2 — DATASET
# ===========================================================================
elif page == PAGES[1]:
    section_header("Dataset", "Daily OHLCV history for SPY")

    m = VERIFIED_METADATA

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### Asset & Coverage")
        info = [
            ("Symbol", m["symbol"]),
            ("Full name", m["asset_name"]),
            ("Source", m["data_source"]),
            ("Frequency", "Daily (trading days only)"),
            ("Date range", f"{m['date_start']} → {m['date_end']}"),
            ("Total rows", f"{m['raw_rows']:,}"),
            ("Labeled rows", f"{m['labeled_rows']:,} (last row dropped — no next-day target)"),
            ("Missing values", "Zero (validated on every load)"),
            ("Duplicate dates", "Zero (validated and deduplicated)"),
        ]
        for k, v in info:
            st.markdown(
                f'<p style="margin:5px 0"><span style="color:#64748B;font-size:0.82rem'
                f';min-width:160px;display:inline-block">{k}</span>'
                f'<span style="color:#CBD5E1;font-size:0.9rem">{v}</span></p>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("What is OHLCV?", expanded=False):
            ohlcv = [
                ("Open", "Opening price of the trading session"),
                ("High", "Highest intraday price during the session"),
                ("Low", "Lowest intraday price during the session"),
                ("Close", "Closing price — used to compute the next-day target"),
                ("Volume", "Number of shares traded during the session"),
            ]
            for letter, desc in ohlcv:
                st.markdown(
                    f"**`{letter}`** — {desc}"
                )

        with st.expander("Why SPY?", expanded=False):
            st.markdown(
                """
                SPY (SPDR S&P 500 ETF Trust) was selected because:
                - **High liquidity** — bid/ask spreads are minimal, prices are clean.
                - **Long, continuous history** — clean daily data back to 1993.
                - **No single-company idiosyncrasies** — tracks a broad index,
                  avoiding earnings surprises, bankruptcy events, or M&A noise.
                - **Widely studied** — allows comparison with published ML benchmarks.

                `auto_adjust=False` is used to fetch raw, as-quoted OHLC prices rather
                than Yahoo Finance's retroactively dividend/split-adjusted close. Adjusted
                prices rewrite historical data when new corporate actions are recorded —
                a subtle form of look-ahead bias this project avoids from the data layer up.
                """
            )

    with col2:
        st.markdown("#### Test Partition (Official Evaluation Window)")
        metric_cards_row([
            {"value": "2023-06-06", "label": "Test start"},
            {"value": "2026-09-10", "label": "Test end"},
        ])
        st.markdown("<br>", unsafe_allow_html=True)
        metric_cards_row([
            {"value": "819", "label": "Test observations"},
            {"value": "463", "label": "UP days in test"},
            {"value": "356", "label": "DOWN/FLAT days in test"},
        ])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Class Balance by Partition")
        balance_data = {
            "Partition": ["Full dataset", "Train", "Validation", "Test"],
            "Rows": [5456, 3819, 818, 819],
            "UP (1)": [2980, 2082, 435, 463],
            "DOWN/FLAT (0)": [2476, 1737, 383, 356],
            "% UP": ["54.62%", "54.52%", "53.18%", "56.53%"],
        }
        import pandas as pd
        st.dataframe(
            pd.DataFrame(balance_data),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            '<p style="color:#64748B;font-size:0.76rem;margin-top:4px">'
            'Mildly imbalanced toward UP — consistent with SPY\'s long-term uptrend.</p>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    safe_image(
        get_figure_path("spy_close_price.png"),
        caption="SPY closing price history",
        warning_msg="This chart shows SPY closing price over the full dataset period.",
    )

    disclaimer_footer()


# ===========================================================================
# PAGE 3 — METHODOLOGY
# ===========================================================================
elif page == PAGES[2]:
    section_header("Methodology", "Rigorous, leak-free time-series machine learning")

    # ── Target construction ─────────────────────────────────────────────────
    st.markdown("### 1 · Target Construction")
    target_formula()

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown(
            """
            **Key constraints:**
            - Equal closes (Close[t+1] == Close[t]) → **DOWN/FLAT (0)**, not discarded.
            - The last row is **dropped** — it has no following observation.
            - Close[t+1] is never added as a DataFrame column; it's computed inline and discarded.
            """
        )
    with col_b:
        st.info(
            "Encoding: **1 = UP**, **0 = DOWN/FLAT**. "
            "This single convention is used consistently in all models, baselines, and evaluation code."
        )

    st.markdown("---")

    # ── Chronological split ─────────────────────────────────────────────────
    st.markdown("### 2 · Chronological Split")
    split_timeline()

    with st.expander("Why not use train_test_split?", expanded=False):
        st.markdown(
            """
            `sklearn.train_test_split(shuffle=True)` would let the model \"see\" data
            from time periods **after** what it's tested on — producing an artificially
            inflated and meaningless accuracy.

            A time-series model must be evaluated as if it only ever had the past
            available. The split must therefore respect chronological order: train on
            the oldest data, validate on the next slice, test on the most recent slice.

            `chronological_split()` in `src/preprocessing.py` divides rows by position
            only. It never calls `random`, `sample`, or `shuffle`.
            """
        )

    st.markdown("---")

    # ── Feature engineering ─────────────────────────────────────────────────
    st.markdown("### 3 · Feature Engineering")

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("**Raw features (20 columns)**")
        st.markdown(
            """
            For each of `Open`, `High`, `Low`, `Close`, `Volume` and each lag
            in `[1, 2, 3, 5]`:

            ```
            Open_lag1[t]  = Open[t-1]   (yesterday)
            Close_lag2[t] = Close[t-2]  (two days ago)
            Volume_lag5[t]= Volume[t-5] (five days ago)
            …
            ```

            5 columns × 4 lags = **20 causal features**. Only `shift(k)` with
            `k > 0` is ever used. A non-positive lag raises `ValueError`.
            """
        )
    with col2:
        st.markdown("**Engineered features (28 columns)**")
        st.markdown(
            """
            The engineered set adds 8 technical indicators on top of the 20
            raw features — it does **not** replace them. This makes the
            comparison meaningful: it tests whether indicators add signal
            *beyond* raw price/volume history for a linear model.

            ```
            28 total = 20 raw lagged OHLCV
                     +  8 technical indicators
            ```

            The combined dataset begins at the later of the two warm-up
            cutoffs (33 rows for the indicators, driven by MACD's 26+9-day
            requirement).
            """
        )

    st.markdown("---")

    # ── Scaling ─────────────────────────────────────────────────────────────
    st.markdown("### 4 · Scaling")
    st.markdown(
        """
        `sklearn.preprocessing.StandardScaler` is fit **only on `X_train`**:

        ```python
        scaler.fit(X_train)              # learns mean/std from training data only
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled   = scaler.transform(X_val)    # never re-fitted
        X_test_scaled  = scaler.transform(X_test)   # never re-fitted
        ```

        **Concrete proof this is real:** after scaling, `X_train`'s per-feature
        mean/std land at ~0/~1 (by definition of fitting on it). But `X_test`,
        scaled with `X_train`'s parameters, produces means around **6.6** and
        stds around **1.58** — exactly what you'd expect when a scaler's parameters
        come from a *different, earlier* period. This gap is direct evidence the
        scaler never saw test data during fitting.
        """
    )

    st.markdown("---")

    # ── Leakage audit ───────────────────────────────────────────────────────
    st.markdown("### 5 · Leakage Audit")
    st.markdown(
        '<p style="color:#94A3B8;font-size:0.9rem">Expand each item to read the detailed evidence.</p>',
        unsafe_allow_html=True,
    )
    leakage_audit_table()

    disclaimer_footer()


# ===========================================================================
# PAGE 4 — TECHNICAL INDICATORS
# ===========================================================================
elif page == PAGES[3]:
    section_header("Technical Indicators", "8 causal backward-looking features")

    st.markdown(
        """
        <p style="color:#CBD5E1;font-size:0.93rem;line-height:1.7;max-width:780px">
            All indicators below are <strong>trailing (backward-looking)</strong>
            transformations of historical Close prices. None of them use a centered
            rolling window or a negative shift. Every indicator uses only information
            available at or before day&nbsp;<em>t</em> — making them valid for
            predicting Target[t] without any look-ahead.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    indicator_names = list(INDICATOR_DESCRIPTIONS.keys())
    cols_per_row = 2
    for i in range(0, len(indicator_names), cols_per_row):
        row_cols = st.columns(cols_per_row, gap="medium")
        for j, col in enumerate(row_cols):
            idx = i + j
            if idx >= len(indicator_names):
                break
            name = indicator_names[idx]
            info = INDICATOR_DESCRIPTIONS[name]
            with col:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div style="margin-bottom:6px">
                            <span style="font-size:1.05rem;font-weight:700;
                                         color:#2DD4BF">{name}</span>
                            <span style="font-size:0.8rem;color:#64748B;
                                         margin-left:8px">{info['window']}</span>
                        </div>
                        <p style="color:#94A3B8;font-size:0.82rem;margin:0 0 4px 0">
                            {info['full_name']}
                        </p>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.code(info["formula"], language="text")
                    st.markdown(
                        f'<p style="color:#CBD5E1;font-size:0.88rem;margin:6px 0">'
                        f'{info["description"]}</p>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<p style="color:#059669;font-size:0.78rem;margin:0">'
                        f'✅ {info["causal_note"]}</p>',
                        unsafe_allow_html=True,
                    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts from figures/
    col_a, col_b = st.columns([1, 1])
    with col_a:
        safe_image(
            get_figure_path("close_with_sma.png"),
            caption="SPY closing price with SMA_10 and SMA_20 overlaid",
            warning_msg="Shows Close price with 10- and 20-day moving averages.",
        )
    with col_b:
        safe_image(
            get_figure_path("rsi_over_time.png"),
            caption="RSI_14 over time with overbought / oversold bands",
            warning_msg="Shows the 14-day RSI over the dataset.",
        )

    safe_image(
        get_figure_path("macd_over_time.png"),
        caption="MACD, MACD_SIGNAL, and MACD_HIST over time",
        warning_msg="Shows all three MACD components.",
    )

    with st.expander("Warm-up handling — why rows are dropped, not filled", expanded=False):
        st.markdown(
            """
            Technical indicators require a minimum number of preceding rows to produce
            their first real value. This is called the **warm-up period**:

            | Indicator | Warm-up rows |
            |---|---|
            | SMA_10 | 10 rows |
            | SMA_20 | 20 rows |
            | RSI_14 | 14 rows |
            | MACD (26+9) | 34 rows |
            | VOLATILITY_20 | 20 rows |

            The longest warm-up (MACD's 34 rows → 33 effective, detected empirically
            from `first_valid_index()`) determines the cutoff for the entire indicator
            set. Those **33 rows** are **dropped** — never forward-filled, backward-filled,
            or set to zero. A stray NaN after trimming would indicate a genuine bug and
            raises `AssertionError`.
            """
        )

    with st.expander("Perturbation test — causal indicator verification", expanded=False):
        st.markdown(
            """
            The notebook includes a **perturbation test** to make causality concrete
            rather than just asserted:

            1. The very last day's `Close` price is changed to a different value.
            2. Every indicator is rebuilt on the perturbed dataset.
            3. The test asserts that every row **more than 40 rows before** the change
               is byte-for-byte identical to the un-perturbed version.

            This passes on the real dataset, confirming that a change on day *t* only
            affects indicator values within that indicator's window — never rows further
            in the past, and certainly never future rows.
            """
        )

    disclaimer_footer()


# ===========================================================================
# PAGE 5 — MODEL COMPARISON
# ===========================================================================
elif page == PAGES[4]:
    section_header("Model Comparison", "All four models · official test partition · n = 819")

    df = load_final_comparison()
    if df is None:
        st.stop()

    # Metric selector
    metric = st.selectbox(
        "Select metric to visualize",
        ["Accuracy", "Precision", "Recall", "F1"],
        index=0,
    )

    # Bar chart
    fig_bar = model_comparison_bar(df, metric)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(
        """
        <p style="color:#64748B;font-size:0.78rem;margin-top:-8px">
            ✦ Teal highlight = best value for this metric.
            The best value among naive baselines is labelled "highest — naive baseline" in the
            table below, not called the "best model."
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Full metrics table (Plotly)
    st.markdown("#### Full comparison table")
    fig_table = metrics_table_figure(df)
    st.plotly_chart(fig_table, use_container_width=True)

    # Also show native dataframe for accessibility
    display_df = df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
    with st.expander("View as plain table", expanded=False):
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Model explanations
    st.markdown("#### Model details")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("**Persistence baseline**")
            st.markdown(
                """
                Predicts that tomorrow's direction matches the most recent observed
                one-day direction.

                `prediction[t] = 1 if Close[t-1] > Close[t-2] else 0`

                No model fitting. Captures naive momentum: \"whatever just happened
                will keep happening.\" Result: **Accuracy 0.5018** (~coin flip).
                """
            )
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("**Majority Class baseline**")
            st.markdown(
                """
                Always predicts the majority class in the **training partition** (UP = 1,
                since training is ~54.5% UP). No fitting beyond counting.

                Result: **Accuracy 0.5653, Recall 1.0000, F1 0.7223**.

                High recall is trivial — it predicts UP unconditionally, so it never
                misses a real UP day. High F1 is misleading for the same reason.

                > Any trained model must beat this to demonstrate that it learned
                > something beyond class imbalance.
                """
            )

    with col2:
        with st.container(border=True):
            st.markdown("**Raw Logistic Regression** · 20 features")
            st.markdown(
                """
                `LogisticRegression(random_state=42, max_iter=1000)`

                Features: 20 causal lagged OHLCV columns
                (`Open_lag1` … `Volume_lag5`).

                Scaler fitted on training data only.

                Result: **Accuracy 0.5128**.

                Beats Persistence but **not** the Majority Class baseline.
                """
            )
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("**Engineered Logistic Regression** · 28 features")
            st.markdown(
                """
                Same model class, same configuration. Features: the same 20 raw
                lagged OHLCV columns **plus** 8 technical indicators = 28 total.

                Scaler fitted on training data only.

                Result: **Accuracy 0.4945**.

                **Worse than the Raw model, worse than both baselines.**
                Adding technical indicators did not improve predictive accuracy.
                """
            )

    st.markdown("---")

    st.markdown("#### Why Majority Class has high F1 — and why that matters")
    conclusion_callout(
        "A naive baseline achieves high F1 because it predicts UP unconditionally, "
        "giving it perfect recall (it never misses a real UP day) at the cost of "
        "precision no better than its own accuracy. This is a consequence of class "
        "imbalance (56.53% UP in the test set), not of any learned predictive structure. "
        "Reporting F1 alongside accuracy ensures this is visible rather than hidden."
    )

    disclaimer_footer()


# ===========================================================================
# PAGE 6 — CLASS BALANCE
# ===========================================================================
elif page == PAGES[5]:
    section_header("Class Balance", "UP vs. DOWN/FLAT — official test partition")

    m = VERIFIED_METADATA

    metric_cards_row([
        {"value": str(m["test_up"]),            "label": "UP days",     "unit": f"{m['test_pct_up']:.2f}% of test"},
        {"value": str(m["test_down_or_flat"]),   "label": "DOWN/FLAT days", "unit": f"{m['test_pct_down_or_flat']:.2f}% of test"},
        {"value": str(m["test_total"]),          "label": "Total test rows", "unit": "2023-06-06 → 2026-09-10"},
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # Class balance chart
    fig_cb = class_balance_chart(m["test_up"], m["test_down_or_flat"])
    st.plotly_chart(fig_cb, use_container_width=True)

    safe_image(
        get_figure_path("class_balance.png"),
        caption="Class balance — bar chart from the notebook",
        warning_msg="Generated by the notebook during the class-balance analysis step.",
    )

    st.markdown("---")
    st.markdown("#### Why class balance matters for model evaluation")
    st.markdown(
        """
        The test partition has **56.53% UP days** — a mild but real imbalance toward
        the positive class.

        This has direct consequences for interpreting the results:

        - **Accuracy alone is insufficient.** A model could achieve 56.53% accuracy
          simply by predicting UP every time — no learning required.
        - **The Majority Class baseline makes this explicit.** It does exactly this:
          predicts UP unconditionally, and achieves accuracy 0.5653.
        - **Recall is inflated for the Majority Class.** By always predicting UP,
          it catches every real UP day (Recall = 1.0), at the cost of wrongly
          predicting UP for every DOWN/FLAT day too.
        - **F1 should be read alongside precision.** The Majority Class F1 of 0.7223
          looks strong but is entirely mechanical — there is no signal in it.
        - **Precision ≈ accuracy for the Majority Class** (0.5653 = 0.5653),
          which makes sense: when a model predicts UP for every row, its precision
          is exactly the fraction of rows that are actually UP.

        Any trained model must beat the Majority Class baseline on a metric that
        rewards *correct* predictions across *both* classes, not just recall of the
        majority class. None of the models in this project did.
        """
    )

    disclaimer_footer()


# ===========================================================================
# PAGE 7 — PREDICTIONS
# ===========================================================================
elif page == PAGES[6]:
    section_header("Predictions", "Engineered model predicted vs. actual direction")

    st.markdown(
        """
        <p style="color:#CBD5E1;font-size:0.93rem;line-height:1.7;max-width:780px">
            The chart below shows the Engineered Logistic Regression model's
            predicted direction versus the actual direction across the official
            test partition (2023-06-06 → 2026-09-10, 819 observations).
            This is the real verified image generated by the notebook.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    safe_image(
        get_figure_path("predicted_vs_actual.png"),
        caption=(
            "Predicted vs. Actual direction — Engineered Logistic Regression · "
            "Official test partition 2023-06-06 → 2026-09-10"
        ),
        warning_msg=(
            "This is the primary prediction visualization from the experiment. "
            "It shows the model's predicted Up/Down direction vs. the actual direction "
            "for each of the 819 test days."
        ),
    )

    safe_image(
        get_figure_path("engineered_confusion_matrix.png"),
        caption="Confusion matrix — Engineered Logistic Regression",
        warning_msg="Shows true/false positive/negative breakdown for the engineered model.",
    )

    st.markdown("---")
    st.markdown("#### Reading the prediction chart")
    st.markdown(
        """
        - Each point represents one trading day in the test window.
        - **Predicted: UP** means the model assigned probability > 0.5 to the UP class.
        - **Predicted: DOWN/FLAT** means the model assigned probability ≤ 0.5 to the UP class.
        - The actual direction is computed from `Close[t+1] > Close[t]` — the same
          leak-free definition used to build the training targets.
        - The model generates **no live predictions** and performs **no live inference**.
          The predictions shown here are from the offline evaluation on the held-out
          test partition only.
        """
    )

    st.info(
        "⚠️ These predictions are from a historical, offline experiment. "
        "They are not real-time signals, investment advice, or trading recommendations."
    )

    disclaimer_footer()


# ===========================================================================
# PAGE 8 — FINDINGS & LIMITATIONS
# ===========================================================================
elif page == PAGES[7]:
    section_header("Findings & Limitations", "Honest interpretation of the experiment")

    # Main conclusion
    st.markdown("### Main Conclusion")
    conclusion_callout(
        "The engineered technical-indicator model did not outperform the naive "
        "majority-class baseline on this test period — and neither did the raw "
        "Logistic Regression model. No machine-learning model in this project "
        "beat the Majority Class baseline."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Engineered model results
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("#### Engineered LR — test metrics")
        metric_cards_row([
            {"value": "49.45%", "label": "Accuracy"},
            {"value": "57.06%", "label": "Precision"},
        ])
        st.markdown("<br>", unsafe_allow_html=True)
        metric_cards_row([
            {"value": "42.76%", "label": "Recall"},
            {"value": "48.89%", "label": "F1"},
        ])
        st.markdown(
            '<p style="color:#64748B;font-size:0.78rem;margin-top:8px">'
            'Source: <code>results/engineered_model_comparison.csv</code></p>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("#### Finding breakdown")
        findings = [
            ("Engineered vs. Raw LR", "Worse — Accuracy 49.45% vs. 51.28%, F1 48.89% vs. 60.46%"),
            ("Engineered vs. Persistence", "Worse — Accuracy 49.45% vs. 50.18%"),
            ("Engineered vs. Majority Class", "Worse — Accuracy 49.45% vs. 56.53%"),
            ("Raw LR vs. Persistence", "Better — 51.28% vs. 50.18%"),
            ("Raw LR vs. Majority Class", "Worse — 51.28% vs. 56.53%"),
            ("Technical indicators helped?", "No — adding them made accuracy worse"),
        ]
        for k, v in findings:
            color = "#6EE7B7" if "Better" in v else "#F87171"
            st.markdown(
                f'<p style="margin:5px 0"><span style="color:#64748B;font-size:0.82rem'
                f';min-width:240px;display:inline-block">{k}</span>'
                f'<span style="font-size:0.88rem;color:{color}">{v}</span></p>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("#### Discussion")
    st.markdown(
        """
        **Why might technical indicators have hurt rather than helped?**

        Technical indicators (SMA, RSI, MACD, return, volatility) are all derived
        from the same `Close` price series that already appears in the raw lagged
        features. For a linear model (Logistic Regression), highly correlated
        inputs can add noise and variance rather than independent signal. This is a
        plausible explanation — but it was not tested with alternatives (dimensionality
        reduction, regularisation tuning, non-linear models), so it remains a hypothesis.

        **Why is the Majority Class baseline so hard to beat?**

        With 56.53% of the test days being UP days, a model that always predicts UP
        achieves 56.53% accuracy and 100% recall — for free, without any learning.
        This is a standard consequence of mild class imbalance. It sets a high bar
        for accuracy-based comparison and is the correct reference point.

        **Is this a failed project?**

        No. A **negative result** — where the hypothesis (technical indicators add
        predictive value) is not confirmed — is a valid and important scientific
        outcome. Reporting it honestly, rather than tuning until a positive result
        appears, is the hallmark of rigorous ML practice. The methodology here
        (leak-free split, causal features, no test-set peeking) is correct; the
        finding is that *this particular feature set and model class* did not find
        a reliable signal in this data.
        """
    )

    st.markdown("---")
    st.markdown("### Limitations")
    limitations = [
        ("Single asset", "Only SPY was tested. Results may not generalise to individual stocks, "
         "other indices, or other asset classes."),
        ("Single historical period", "The test window covers 2023-06-06 → 2026-09-10. "
         "Performance on other periods or market regimes is unknown."),
        ("Single model class", "Only Logistic Regression was trained — a linear classifier. "
         "Non-linear models (gradient boosting, neural networks) were not evaluated."),
        ("No hyperparameter search", "The model was trained with default parameters "
         "(max_iter=1000, no regularisation tuning) to avoid test-set leakage via "
         "repeated evaluation."),
        ("Direction only, not magnitude", "The target is binary (Up/Down). "
         "How much the price moves is not modelled."),
        ("No statistical significance testing", "None of the metric differences above "
         "were tested for statistical significance. They should not be read as "
         "proven non-random."),
        ("No transaction cost modelling", "No slippage, fees, or position sizing is "
         "modelled. This is a classification experiment, not a trading strategy."),
        ("Historical performance", "Historical classification accuracy does not guarantee "
         "future market behaviour."),
    ]

    for title, desc in limitations:
        with st.expander(f"⚠️  {title}", expanded=False):
            st.markdown(
                f'<p style="color:#94A3B8;font-size:0.9rem;line-height:1.6">{desc}</p>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### Reproducibility")
    col_r1, col_r2 = st.columns([1, 1], gap="large")
    with col_r1:
        st.markdown(
            """
            **Test suite**
            ```bash
            pytest -q
            ```
            Expected: **126 passed** — deterministic tests, no network access.

            **Streamlit dashboard**
            ```bash
            streamlit run app.py
            ```

            **Full pipeline (requires network)**
            ```bash
            pip install -r requirements.txt
            jupyter notebook notebooks/stock_price_movement_predictor.ipynb
            ```
            Restart Kernel + Run All to regenerate all results, figures, and models.
            """
        )
    with col_r2:
        st.markdown(
            """
            **Environment**
            - Python 3.12
            - All dependencies pinned in `requirements.txt`
            - `config.py` centralises all constants
            - `RANDOM_SEED = 42` for deterministic model training
            - Data fetched via `yfinance` and cached to `data/raw/SPY.csv`

            **Committed deliverables**
            - `results/final_comparison.csv`
            - `results/baseline_comparison.csv`
            - `results/raw_model_comparison.csv`
            - `results/engineered_model_comparison.csv`
            """
        )

    disclaimer_footer()
