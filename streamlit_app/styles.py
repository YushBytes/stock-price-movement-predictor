"""CSS injection for the dark professional research-dashboard theme.

Call ``inject_styles()`` once at the top of ``app.py`` after
``st.set_page_config``. All selectors target Streamlit's rendered HTML;
only stable class names / element selectors are used so the styles degrade
gracefully if Streamlit updates its internal DOM structure.
"""

import streamlit as st


_CSS = """
<style>
/* ── Base & typography ─────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0F1117;
    color: #E2E8F0;
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
}

/* ── Sidebar ───────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #1A1D2E;
    border-right: 1px solid #2D3148;
}
[data-testid="stSidebar"] .stRadio label {
    color: #CBD5E1;
    font-size: 0.92rem;
    padding: 4px 0;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #2DD4BF;
}

/* ── Section headers ───────────────────────────────────────────────── */
.section-header {
    border-left: 3px solid #2DD4BF;
    padding-left: 12px;
    margin-top: 1.6rem;
    margin-bottom: 0.6rem;
}
.section-header h2 {
    color: #F1F5F9;
    font-size: 1.4rem;
    font-weight: 600;
    margin: 0;
    letter-spacing: -0.01em;
}
.section-header p {
    color: #94A3B8;
    font-size: 0.88rem;
    margin: 2px 0 0 0;
}

/* ── Metric cards ──────────────────────────────────────────────────── */
.metric-card {
    background: #1A1D2E;
    border: 1px solid #2D3148;
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #2DD4BF;
    line-height: 1.1;
}
.metric-card .metric-label {
    font-size: 0.78rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 4px;
}
.metric-card .metric-unit {
    font-size: 0.82rem;
    color: #64748B;
    margin-top: 2px;
}

/* ── Info / status badges ──────────────────────────────────────────── */
.badge-pass {
    display: inline-block;
    background: #064E3B;
    color: #6EE7B7;
    border: 1px solid #059669;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* ── Formula block ─────────────────────────────────────────────────── */
.formula-block {
    background: #1A1D2E;
    border: 1px solid #2D3148;
    border-left: 3px solid #2DD4BF;
    border-radius: 6px;
    padding: 14px 20px;
    font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
    font-size: 1rem;
    color: #A5F3FC;
    margin: 12px 0;
}

/* ── Timeline / split rows ─────────────────────────────────────────── */
.split-row {
    display: flex;
    gap: 1px;
    border-radius: 6px;
    overflow: hidden;
    margin: 12px 0;
}
.split-segment {
    padding: 10px 14px;
    font-size: 0.82rem;
    font-weight: 500;
}
.split-train { background: #1E3A5F; color: #93C5FD; flex: 3.5; }
.split-val   { background: #3B2E58; color: #C4B5FD; flex: 0.75; }
.split-test  { background: #1F3A2F; color: #6EE7B7; flex: 0.75; }

/* ── Conclusion callout ────────────────────────────────────────────── */
.conclusion-box {
    background: #1A1D2E;
    border: 1px solid #2D3148;
    border-left: 4px solid #F59E0B;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 16px 0;
    color: #FDE68A;
    font-size: 0.97rem;
    line-height: 1.6;
}

/* ── Disclaimer footer ─────────────────────────────────────────────── */
.disclaimer {
    background: #12151F;
    border: 1px solid #1E2235;
    border-radius: 6px;
    padding: 10px 16px;
    color: #475569;
    font-size: 0.78rem;
    line-height: 1.5;
    margin-top: 32px;
}

/* ── Streamlit native overrides ────────────────────────────────────── */
[data-testid="stMetricValue"] { color: #2DD4BF !important; }
div.stTabs [data-baseweb="tab"] { color: #94A3B8; }
div.stTabs [aria-selected="true"] { color: #2DD4BF !important; }
hr { border-color: #2D3148; }

/* dataframe table */
[data-testid="stDataFrame"] {
    border-radius: 6px;
    overflow: hidden;
}
</style>
"""


def inject_styles() -> None:
    """Inject the application-wide CSS into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)
