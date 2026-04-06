"""Plotting helpers for forecasting experiments."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
]


def plot_forecast_vs_actual(
    dates: Sequence[pd.Timestamp],
    actual: Sequence[float],
    forecasts: Dict[str, Sequence[float]],
    title: str = "Forecast vs Actual",
    train_dates: Optional[Sequence[pd.Timestamp]] = None,
    train_values: Optional[Sequence[float]] = None,
    quantile_lo: Optional[Sequence[float]] = None,
    quantile_hi: Optional[Sequence[float]] = None,
) -> go.Figure:
    """Build a Plotly chart comparing actual and forecasted values."""
    fig = go.Figure()

    if train_dates is not None and train_values is not None:
        context = min(60, len(train_dates))
        fig.add_trace(
            go.Scatter(
                x=list(train_dates)[-context:],
                y=list(train_values)[-context:],
                name="Train (context)",
                line={"color": "gray", "width": 1, "dash": "dot"},
            )
        )

    fig.add_trace(
        go.Scatter(
            x=list(dates),
            y=list(actual),
            name="Actual",
            line={"color": "black", "width": 2},
        )
    )

    for idx, (label, values) in enumerate(forecasts.items()):
        fig.add_trace(
            go.Scatter(
                x=list(dates),
                y=list(values),
                name=label,
                line={"color": PLOT_COLORS[idx % len(PLOT_COLORS)], "width": 1.5, "dash": "dash"},
            )
        )

    if quantile_lo is not None and quantile_hi is not None:
        fig.add_trace(
            go.Scatter(
                x=list(dates),
                y=list(quantile_hi),
                mode="lines",
                line={"width": 0},
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=list(dates),
                y=list(quantile_lo),
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(31,119,180,0.15)",
                name="10th-90th pctl",
            )
        )

    fig.update_layout(
        title={"text": title, "font": {"size": 16}},
        xaxis_title="Date",
        yaxis_title="Value",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        height=450,
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def plot_metric_bars(
    metrics_df: pd.DataFrame,
    metrics_to_plot: Sequence[str],
    group_col: str = "model",
    title: str = "Model Comparison",
) -> go.Figure:
    """Build metric-wise horizontal bars for quick model comparison."""
    n_metrics = len(metrics_to_plot)
    fig = make_subplots(rows=1, cols=n_metrics, subplot_titles=[m.upper() for m in metrics_to_plot])

    for col_idx, metric in enumerate(metrics_to_plot):
        if metric not in metrics_df.columns:
            continue
        sub = metrics_df[[group_col, metric]].dropna().sort_values(metric)
        fig.add_trace(
            go.Bar(
                y=sub[group_col],
                x=sub[metric],
                orientation="h",
                marker_color=PLOT_COLORS[: len(sub)],
                text=np.round(sub[metric], 4),
                textposition="outside",
                showlegend=False,
            ),
            row=1,
            col=col_idx + 1,
        )

    fig.update_layout(
        title={"text": title, "font": {"size": 16}},
        height=max(300, 40 * len(metrics_df)),
        template="plotly_white",
    )
    return fig
