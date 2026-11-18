"""Interactive Plotly visualization routines for the research dashboard."""

from typing import Any, Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS: Dict[str, str] = {
    "actual": "#0f3e72",
    "predicted": "#0ea5e9",
    "anomaly": "#ef4444",
    "threshold": "#f59e0b",
    "background": "#ffffff",
    "grid": "#e2e8f0",
}


def plot_actual_vs_predicted(df: pd.DataFrame, max_points: int = 750) -> go.Figure:
    """Plot observed consumption against model forecast."""
    plot_df = df.iloc[-max_points:].copy()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["actual"],
            mode="lines",
            name="Actual Consumption (kWh)",
            line=dict(color=COLORS["actual"], width=1.8),
        )
    )
    if "predicted" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df["timestamp"],
                y=plot_df["predicted"],
                mode="lines",
                name="XGBoost Forecast (kWh)",
                line=dict(color=COLORS["predicted"], width=1.5, dash="dash"),
            )
        )

    fig.update_layout(
        title="Observed Electricity Load vs. XGBoost Forecast",
        xaxis_title="Timestamp",
        yaxis_title="Energy (kWh)",
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=50, b=40),
        height=420,
    )
    return fig


def plot_anomaly_detection_dashboard(df: pd.DataFrame, max_points: int = 750) -> go.Figure:
    """Create a two-panel Plotly figure showing load + flagged anomalies and residual vs threshold."""
    plot_df = df.iloc[-max_points:].copy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Electricity Consumption (kWh) & Detected Anomalies",
            "Absolute Residual |y - y_hat| vs. Adaptive Rolling MAD Threshold",
        ),
        row_heights=[0.55, 0.45],
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["actual"],
            mode="lines",
            name="Actual (kWh)",
            line=dict(color=COLORS["actual"], width=1.6),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["predicted"],
            mode="lines",
            name="Forecast (kWh)",
            line=dict(color=COLORS["predicted"], width=1.4, dash="dash"),
        ),
        row=1,
        col=1,
    )

    anoms = plot_df[plot_df["is_anomaly"] == True]  # noqa: E712
    if not anoms.empty:
        fig.add_trace(
            go.Scatter(
                x=anoms["timestamp"],
                y=anoms["actual"],
                mode="markers",
                name="Detected Anomaly",
                marker=dict(color=COLORS["anomaly"], size=8, symbol="circle-open", line=dict(width=2)),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["residual"],
            mode="lines",
            name="Residual |e_t|",
            line=dict(color="#64748b", width=1.2),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["threshold"],
            mode="lines",
            name="Adaptive MAD Threshold",
            line=dict(color=COLORS["threshold"], width=1.8),
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Energy (kWh)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (kWh)", row=2, col=1)
    fig.update_xaxes(title_text="Timestamp", row=2, col=1)

    fig.update_layout(
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=45, r=20, t=60, b=40),
        height=620,
    )
    return fig


def plot_consumption_patterns(df_hourly: pd.DataFrame) -> go.Figure:
    """Plot average diurnal (24h) and day-of-week consumption profiles."""
    df_work = df_hourly.copy()
    df_work["Datetime"] = pd.to_datetime(df_work["Datetime"])
    df_work["hour"] = df_work["Datetime"].dt.hour
    df_work["dow"] = df_work["Datetime"].dt.day_name()

    hourly_avg = df_work.groupby("hour")["Global_active_power"].mean().reset_index()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_avg = (
        df_work.groupby("dow")["Global_active_power"]
        .mean()
        .reindex(dow_order)
        .reset_index()
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Average Diurnal Profile (24-Hour)", "Average Weekly Profile"),
    )
    fig.add_trace(
        go.Scatter(
            x=hourly_avg["hour"],
            y=hourly_avg["Global_active_power"],
            mode="lines+markers",
            name="Hourly Mean (kWh)",
            line=dict(color=COLORS["actual"], width=2.2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=dow_avg["dow"],
            y=dow_avg["Global_active_power"],
            name="Daily Mean (kWh)",
            marker_color=COLORS["predicted"],
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
    fig.update_xaxes(title_text="Day of Week", row=1, col=2)
    fig.update_yaxes(title_text="Mean Load (kWh)", row=1, col=1)
    fig.update_yaxes(title_text="Mean Load (kWh)", row=1, col=2)

    fig.update_layout(
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        showlegend=False,
        margin=dict(l=40, r=20, t=50, b=40),
        height=380,
    )
    return fig


def plot_comparison_metrics(metrics_comp: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Plot side-by-side comparison bars for Adaptive MAD Detector vs Fixed Baseline."""
    metric_keys = ["precision", "recall", "f1_score"]
    metric_labels = ["Precision", "Recall", "F1-Score"]

    fig = go.Figure()
    palette = [COLORS["actual"], COLORS["threshold"]]

    for idx, (model_name, m_dict) in enumerate(metrics_comp.items()):
        vals = [float(m_dict.get(k, 0.0)) * 100.0 for k in metric_keys]
        fig.add_trace(
            go.Bar(
                name=model_name,
                x=metric_labels,
                y=vals,
                marker_color=palette[idx % len(palette)],
                text=[f"{v:.1f}%" for v in vals],
                textposition="outside",
            )
        )

    fig.update_layout(
        barmode="group",
        title="Adaptive MAD Detector vs. Fixed-Threshold Baseline (%)",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 115]),
        plot_bgcolor=COLORS["background"],
        paper_bgcolor=COLORS["background"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=400,
    )
    return fig
