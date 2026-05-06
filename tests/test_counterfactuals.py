import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from causal_forecasting.counterfactuals import (
    PARAMETER_UNCERTAINTY_DISCLAIMER,
    CounterfactualEngine,
)
from causal_forecasting.forecasting import CausalForecastingEngine


def _fit_linear_engine():
    df = pd.DataFrame(
        {
            "x": np.arange(20, dtype=float),
            "y": np.arange(20, dtype=float) * 2.0 + 1.0,
            "z": np.arange(20, dtype=float) * 4.0 + 3.0,
        }
    )
    engine = CausalForecastingEngine(
        [("x", "y", 1.0), ("y", "z", 1.0)],
        ["x", "y", "z"],
        model_factory=LinearRegression,
    )
    engine.fit_structural_equations(df)
    return engine


def test_counterfactual_intervention_mutilates_incoming_edges_and_propagates():
    engine = _fit_linear_engine()
    counterfactual = CounterfactualEngine(engine, residual_scale=0.0)

    result = counterfactual.intervene(
        {"x": 10.0, "y": 21.0, "z": 43.0},
        "y",
        101.0,
        forecast_horizon=2,
        n_bootstrap=10,
    )

    assert result.removed_incoming_edges == [("x", "y")]
    assert result.factual["y"] == pytest.approx([21.0, 21.0])
    assert result.counterfactual["y"] == pytest.approx([101.0, 101.0])
    assert result.counterfactual["z"] == pytest.approx([203.0, 203.0])
    assert result.difference["z"] == pytest.approx([160.0, 160.0])
    assert result.confidence_interval["lower"]["z"] == pytest.approx([203.0, 203.0])
    assert result.disclaimer == PARAMETER_UNCERTAINTY_DISCLAIMER


def test_counterfactual_relative_intervention_uses_percentage_change():
    engine = _fit_linear_engine()
    counterfactual = CounterfactualEngine(engine, residual_scale=0.0)

    result = counterfactual.intervene_relative(
        {"x": 10.0, "y": 21.0, "z": 43.0},
        "x",
        -0.50,
        forecast_horizon=1,
        n_bootstrap=10,
    )

    assert result.intervention.variable == "x"
    assert result.intervention.value == 5.0
    assert result.counterfactual["x"] == [5.0]
    assert result.counterfactual["y"] == pytest.approx([11.0])


def test_counterfactual_result_serializes_for_api_use():
    engine = _fit_linear_engine()
    counterfactual = CounterfactualEngine(engine, residual_scale=0.0)

    result = counterfactual.intervene(
        {"x": 10.0, "y": 21.0, "z": 43.0},
        "x",
        12.0,
        forecast_horizon=1,
        n_bootstrap=10,
    ).to_dict()

    assert result["intervention"] == {"variable": "x", "value": 12.0, "horizon": 1}
    assert "counterfactual" in result
    assert "confidence_interval" in result
    assert "structural uncertainty" in result["disclaimer"]


def test_counterfactual_engine_requires_fitted_causal_engine():
    engine = CausalForecastingEngine([("x", "y", 1.0)], ["x", "y"])

    with pytest.raises(RuntimeError, match="fitted"):
        CounterfactualEngine(engine)


def test_counterfactual_rejects_unknown_intervention_variable():
    engine = _fit_linear_engine()
    counterfactual = CounterfactualEngine(engine)

    with pytest.raises(KeyError, match="Unknown intervention variable"):
        counterfactual.intervene({"x": 1.0, "y": 3.0, "z": 7.0}, "missing", 1.0)

