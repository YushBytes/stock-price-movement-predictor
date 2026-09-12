"""Reusable Streamlit UI components for the research dashboard.

Each function in this module produces one self-contained UI element.
Functions use ``st.markdown(unsafe_allow_html=True)`` only for styled
elements that Streamlit's native widgets cannot produce; native widgets
are preferred everywhere else.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Section header
# ---------------------------------------------------------------------------

def section_header(title: str, subtitle: str = "") -> None:
    """Render a left-bordered section heading with an optional subtitle."""
    sub_html = f'<p>{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header">
            <h2>{title}</h2>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------

def metric_cards_row(cards: list[dict]) -> None:
    """Render a row of metric cards.

    Parameters
    ----------
    cards:
        List of dicts with keys ``value``, ``label``, and optionally
        ``unit``.  One card per column.
    """
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            unit_html = f'<div class="metric-unit">{card.get("unit", "")}</div>' if card.get("unit") else ""
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{card["value"]}</div>
                    <div class="metric-label">{card["label"]}</div>
                    {unit_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Leakage audit checklist
# ---------------------------------------------------------------------------

_LEAKAGE_CHECKS = [
    (
        "Target label construction",
        "Target[t] = 1 if Close[t+1] > Close[t]. "
        "Close[t+1] is used only inside build_target() to compute the label and is never "
        "persisted as a column or passed to any model.",
    ),
    (
        "No future value in model features",
        "build_raw_features() and build_technical_indicators() use only positive shifts "
        "(lag ≥ 1). A non-positive lag raises ValueError — enforced by code, not just convention.",
    ),
    (
        "Chronological train / validation / test split",
        "chronological_split() divides rows by position only (oldest → newest). "
        "sklearn.train_test_split with shuffle is never called anywhere in src/.",
    ),
    (
        "Scaler fitted on training partition only",
        "StandardScaler.fit(X_train) is called once. X_val and X_test are transformed "
        "with those parameters — never refitted. Proof: scaled X_test means land ~6.6, "
        "not ~0, confirming the scaler never saw test data.",
    ),
    (
        "Technical indicators are causal",
        "Every indicator (SMA, RSI, MACD, return, volatility) uses only trailing windows "
        "ending at row t. The ta library was inspected for center=True and negative shift "
        "before adoption. A perturbation test confirms only rows within 40 of the change "
        "are affected.",
    ),
    (
        "Warm-up rows dropped, not filled",
        "The first max(lags)=5 rows (raw features) and first 33 rows (technical indicators) "
        "are dropped from the dataset. No forward-fill, backward-fill, or invented values.",
    ),
]


def leakage_audit_table() -> None:
    """Render the six-point leakage audit checklist."""
    for title, explanation in _LEAKAGE_CHECKS:
        with st.expander(f"✅  PASS — {title}", expanded=False):
            st.markdown(
                f'<p style="color:#94A3B8;font-size:0.9rem;line-height:1.6">{explanation}</p>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Chronological split timeline
# ---------------------------------------------------------------------------

def split_timeline() -> None:
    """Render the train / validation / test timeline bar."""
    st.markdown(
        """
        <div class="split-row">
            <div class="split-segment split-train">
                <strong>TRAIN</strong><br>
                2005-01-03 → 2020-03-05<br>
                <small>3,819 rows · 70%</small>
            </div>
            <div class="split-segment split-val">
                <strong>VALIDATION</strong><br>
                2020-03-06 → 2023-06-05<br>
                <small>818 rows · 15%</small>
            </div>
            <div class="split-segment split-test">
                <strong>TEST</strong><br>
                2023-06-06 → 2026-09-10<br>
                <small>819 rows · 15%</small>
            </div>
        </div>
        <p style="color:#64748B;font-size:0.78rem;margin-top:4px">
            Split is strictly chronological (row-order). No shuffling. No random sampling.
        </p>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Target formula
# ---------------------------------------------------------------------------

def target_formula() -> None:
    """Render the target construction formula."""
    st.markdown(
        """
        <div class="formula-block">
            Target[t] = 1 &nbsp;&nbsp;if&nbsp;&nbsp; Close[t+1] &gt; Close[t] &nbsp;&nbsp; (UP)<br>
            Target[t] = 0 &nbsp;&nbsp;if&nbsp;&nbsp; Close[t+1] ≤ Close[t] &nbsp;&nbsp; (DOWN / FLAT)
        </div>
        <p style="color:#94A3B8;font-size:0.86rem;margin-top:6px">
            <strong style="color:#F87171">Close[t+1]</strong> is used 
            <em>only</em> to compute the label for row&nbsp;t inside 
            <code>build_target()</code>. It is never added to the 
            DataFrame as a column and never passed to any model as a feature.
        </p>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Conclusion callout
# ---------------------------------------------------------------------------

def conclusion_callout(text: str) -> None:
    """Render an amber-bordered conclusion box."""
    st.markdown(
        f'<div class="conclusion-box">{text}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Disclaimer footer
# ---------------------------------------------------------------------------

def disclaimer_footer() -> None:
    """Render a subtle educational/research disclaimer."""
    st.markdown(
        """
        <div class="disclaimer">
            <strong>Disclaimer:</strong> This is an educational and research project demonstrating
            rigorous time-series machine-learning methodology. It is <strong>not financial advice</strong>
            and makes no claim of trading profitability. Historical classification results do not
            guarantee future market behaviour. The experiment is designed to be honest about its
            limitations, including the fact that no trained model beat the naive majority-class baseline.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Figure display with graceful fallback
# ---------------------------------------------------------------------------

def safe_image(path, caption: str, warning_msg: str) -> None:
    """Display an image if it exists, otherwise show a warning."""
    from pathlib import Path

    p = Path(path)
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.warning(
            f"⚠️ Figure not available: `{p.name}`  \n"
            f"{warning_msg}  \n"
            "Run `jupyter notebook` → Restart Kernel + Run All to regenerate figures."
        )
