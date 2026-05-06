"""Plotly chart builders for dashboard and report figures."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


def write_html_compact(fig: go.Figure, path: str | Path) -> Path:
    """Write a Plotly figure as HTML using CDN-loaded plotly.js.

    The default ``fig.write_html`` inlines the entire ~3MB plotly.js bundle in
    every file. Routing through the CDN keeps each artifact under ~50KB.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs="cdn", full_html=True)
    return output


def create_counterfactual_figure(
    result: dict[str, object],
    target_variable: str,
) -> go.Figure:
    """Create factual vs counterfactual chart with confidence interval."""

    factual = result["factual"][target_variable]
    counterfactual = result["counterfactual"][target_variable]
    ci_lower = result["confidence_interval"]["lower"][target_variable]
    ci_upper = result["confidence_interval"]["upper"][target_variable]
    x_values = list(range(1, len(counterfactual) + 1))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=factual,
            name="Factual",
            line={"color": "#2563eb", "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=counterfactual,
            name="Counterfactual",
            line={"color": "#dc2626", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values + x_values[::-1],
            y=ci_upper + ci_lower[::-1],
            fill="toself",
            fillcolor="rgba(220,38,38,0.14)",
            line={"color": "rgba(255,255,255,0)"},
            name="95% interval",
        )
    )
    fig.update_layout(
        title=f"Counterfactual impact on {target_variable}",
        xaxis_title="Forecast step",
        yaxis_title=target_variable,
        hovermode="x unified",
        template="plotly_white",
        margin={"l": 40, "r": 24, "t": 54, "b": 40},
    )
    return fig


def create_model_comparison_figure(
    actual: list[float],
    predictions: dict[str, list[float]],
    *,
    shock_index: int | None = None,
    title: str = "Post-shock Forecasting Comparison",
) -> go.Figure:
    """Create actual-vs-model comparison chart."""

    x_values = list(range(len(actual)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=actual,
            name="Actual",
            line={"color": "#111827", "width": 3},
        )
    )
    colors = ["#6b7280", "#dc2626", "#059669", "#7c3aed"]
    for idx, (model_name, values) in enumerate(predictions.items()):
        fig.add_trace(
            go.Scatter(
                x=x_values[: len(values)],
                y=values,
                name=model_name,
                line={"color": colors[idx % len(colors)], "width": 2},
            )
        )

    if shock_index is not None:
        fig.add_vline(
            x=shock_index,
            line_width=2,
            line_dash="dot",
            line_color="#f59e0b",
            annotation_text="Shock",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Timestep",
        yaxis_title="Target value",
        hovermode="x unified",
        template="plotly_white",
        margin={"l": 40, "r": 24, "t": 54, "b": 40},
    )
    return fig

