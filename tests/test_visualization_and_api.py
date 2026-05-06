import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from causal_forecasting.api import create_counterfactual_app
from causal_forecasting.counterfactuals import CounterfactualEngine
from causal_forecasting.forecasting import CausalForecastingEngine
from causal_forecasting.visualization import (
    create_counterfactual_figure,
    create_interactive_dag,
    create_model_comparison_figure,
    write_table,
)


def _fit_engine():
    df = pd.DataFrame(
        {
            "x": np.arange(12, dtype=float),
            "y": np.arange(12, dtype=float) * 2 + 1,
        }
    )
    engine = CausalForecastingEngine(
        [("x", "y", 1.0)],
        ["x", "y"],
        model_factory=LinearRegression,
    )
    engine.fit_structural_equations(df)
    return engine


def test_plot_builders_create_expected_traces():
    result = {
        "factual": {"y": [1.0, 2.0]},
        "counterfactual": {"y": [1.5, 2.5]},
        "confidence_interval": {
            "lower": {"y": [1.2, 2.2]},
            "upper": {"y": [1.8, 2.8]},
        },
    }

    cf_fig = create_counterfactual_figure(result, "y")
    comparison_fig = create_model_comparison_figure(
        [1.0, 2.0],
        {"Model A": [1.1, 2.1]},
        shock_index=0,
    )

    assert len(cf_fig.data) == 3
    assert len(comparison_fig.data) == 2


def test_create_interactive_dag_writes_html(tmp_path):
    engine = _fit_engine()
    output = create_interactive_dag(
        engine.G,
        np.array([[0.0, 0.9], [0.0, 0.0]]),
        ["x", "y"],
        tmp_path / "dag.html",
    )

    assert output.exists()
    assert "x" in output.read_text(encoding="utf-8")


def test_write_table_outputs_csv_and_markdown(tmp_path):
    table = pd.DataFrame({"model": ["A"], "mae": [1.0]})

    csv_artifact = write_table(table, tmp_path / "metrics.csv")
    md_artifact = write_table(table, tmp_path / "metrics.md")

    assert csv_artifact.path.exists()
    assert md_artifact.path.exists()
    assert csv_artifact.artifact_type == "table"


def test_counterfactual_api_serves_health_and_queries():
    engine = _fit_engine()
    counterfactual = CounterfactualEngine(engine, residual_scale=0.0)
    app = create_counterfactual_app(
        counterfactual,
        current_state_provider=lambda: {"x": 10.0, "y": 21.0},
    )
    client = app.test_client()

    health = client.get("/health")
    response = client.post(
        "/counterfactual",
        json={
            "intervention_variable": "x",
            "intervention_value": 5.0,
            "forecast_horizon": 1,
            "n_bootstrap": 10,
        },
    )

    assert health.json == {"status": "ok"}
    assert response.status_code == 200
    assert response.json["counterfactual"]["y"] == pytest.approx([11.0])
