"""Plotly chart helpers for the Streamlit research dashboard.

Every function here produces a Plotly figure configured with the dark
research-dashboard theme. Charts are for visualization and communication
only — no ML re-computation happens here.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd

# ---------------------------------------------------------------------------
# Shared theme
# ---------------------------------------------------------------------------
_BG_MAIN = "#0F1117"
_BG_CARD = "#1A1D2E"
_GRID = "#2D3148"
_TEXT = "#E2E8F0"
_MUTED = "#94A3B8"
_TEAL = "#2DD4BF"
_AMBER = "#F59E0B"
_RED = "#F87171"
_PURPLE = "#A78BFA"
_BLUE = "#60A5FA"

# Color palette per model (consistent across all charts)
MODEL_COLORS: dict[str, str] = {
    "Persistence": "#60A5FA",          # blue
    "Majority Class": "#A78BFA",       # purple
    "Raw Logistic Regression": _TEAL,  # teal
    "Engineered Logistic Regression": _AMBER,  # amber
}

_LAYOUT_BASE = dict(
    paper_bgcolor=_BG_MAIN,
    plot_bgcolor=_BG_CARD,
    font=dict(family="Inter, Segoe UI, system-ui, sans-serif", color=_TEXT, size=12),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(
        gridcolor=_GRID,
        linecolor=_GRID,
        tickfont=dict(color=_MUTED),
        title_font=dict(color=_MUTED),
    ),
    yaxis=dict(
        gridcolor=_GRID,
        linecolor=_GRID,
        tickfont=dict(color=_MUTED),
        title_font=dict(color=_MUTED),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor=_GRID,
        borderwidth=1,
        font=dict(color=_MUTED, size=11),
    ),
)


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def model_comparison_bar(df: pd.DataFrame, metric: str) -> go.Figure:
    """Horizontal bar chart comparing all models on a single metric.

    Parameters
    ----------
    df:
        DataFrame with columns ``Model`` and the selected ``metric``.
        Expected to come from ``data.load_final_comparison()``.
    metric:
        One of ``Accuracy``, ``Precision``, ``Recall``, ``F1``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    df_sorted = df.sort_values(metric, ascending=True).copy()

    colors = [MODEL_COLORS.get(m, _TEAL) for m in df_sorted["Model"]]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_sorted[metric],
            y=df_sorted["Model"],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color=_BG_MAIN, width=1),
            ),
            text=[f"{v:.4f}" for v in df_sorted[metric]],
            textposition="outside",
            textfont=dict(color=_TEXT, size=11),
            hovertemplate=(
                "<b>%{y}</b><br>"
                + f"{metric}: " + "%{x:.4f}<extra></extra>"
            ),
        )
    )

    # Reference line at 0.5 (coin-flip level) — useful for Accuracy
    if metric == "Accuracy":
        fig.add_vline(
            x=0.5,
            line=dict(color=_RED, width=1, dash="dot"),
            annotation_text="0.5 (coin flip)",
            annotation_font=dict(color=_RED, size=10),
            annotation_position="top right",
        )

    layout = dict(
        **_LAYOUT_BASE,
        title=dict(
            text=f"<b>{metric}</b> — all models, official test partition (n=819)",
            font=dict(color=_TEXT, size=14),
            x=0,
        ),
        xaxis=dict(
            **_LAYOUT_BASE["xaxis"],
            range=[0, min(df_sorted[metric].max() * 1.18, 1.05)],
            title=metric,
            tickformat=".2f",
        ),
        height=260,
        bargap=0.35,
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Class balance chart
# ---------------------------------------------------------------------------

def class_balance_chart(up_count: int, down_count: int) -> go.Figure:
    """Horizontal stacked bar showing class proportion in the test partition."""
    total = up_count + down_count
    pct_up = 100 * up_count / total
    pct_down = 100 * down_count / total

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="UP (1)",
        x=[up_count],
        y=["Test partition"],
        orientation="h",
        marker_color=_TEAL,
        text=[f"UP: {up_count} ({pct_up:.1f}%)"],
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate=f"UP: {up_count} ({pct_up:.2f}%)<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        name="DOWN / FLAT (0)",
        x=[down_count],
        y=["Test partition"],
        orientation="h",
        marker_color=_PURPLE,
        text=[f"DOWN: {down_count} ({pct_down:.1f}%)"],
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate=f"DOWN/FLAT: {down_count} ({pct_down:.2f}%)<extra></extra>",
    ))

    layout = dict(
        **_LAYOUT_BASE,
        barmode="stack",
        title=dict(
            text="<b>Class balance</b> — official test partition (n=819)",
            font=dict(color=_TEXT, size=14),
            x=0,
        ),
        xaxis=dict(
            **_LAYOUT_BASE["xaxis"],
            title="Observations",
        ),
        height=180,
        bargap=0.4,
        showlegend=True,
        legend=dict(
            **_LAYOUT_BASE["legend"],
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="left",
            x=0,
        ),
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Metrics radar / full table helper — returns a styled Plotly table
# ---------------------------------------------------------------------------

def metrics_table_figure(df: pd.DataFrame) -> go.Figure:
    """Render a Plotly table of all four model metrics."""
    metric_cols = ["Accuracy", "Precision", "Recall", "F1"]

    # Format to 4 decimal places
    cell_values = [df["Model"].tolist()]
    for col in metric_cols:
        cell_values.append([f"{v:.4f}" for v in df[col]])

    # Highlight best value per metric column in teal
    fill_colors: list[list[str]] = [
        ["#1A1D2E"] * len(df)
    ]
    for col in metric_cols:
        best_idx = df[col].idxmax()
        colors = []
        for i in df.index:
            if i == best_idx:
                colors.append("#0D3D38")  # dark teal highlight
            else:
                colors.append("#1A1D2E")
        fill_colors.append(colors)

    fig = go.Figure(
        go.Table(
            columnwidth=[180, 80, 80, 80, 80],
            header=dict(
                values=["<b>Model</b>"] + [f"<b>{c}</b>" for c in metric_cols],
                fill_color="#12151F",
                line_color=_GRID,
                align=["left"] + ["center"] * 4,
                font=dict(color=_TEAL, size=12),
                height=32,
            ),
            cells=dict(
                values=cell_values,
                fill_color=fill_colors,
                line_color=_GRID,
                align=["left"] + ["center"] * 4,
                font=dict(color=_TEXT, size=11),
                height=30,
            ),
        )
    )

    fig.update_layout(
        paper_bgcolor=_BG_MAIN,
        margin=dict(l=0, r=0, t=0, b=0),
        height=190,
    )
    return fig
