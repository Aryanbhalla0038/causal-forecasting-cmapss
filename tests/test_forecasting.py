import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from causal_forecasting.forecasting import (
    CausalForecastingEngine,
    ShockInjector,
    compare_forecasts,
    evaluate_forecast,
    recovery_time_to_baseline,
)


def test_causal_forecasting_engine_fits_and_forecasts_through_dag():
    df = pd.DataFrame(
        {
            "x": np.arange(20, dtype=float),
            "y": np.arange(20, dtype=float) * 2.0 + 1.0,
            "z": np.arange(20, dtype=float) * 4.0 + 3.0,
        }
    )
    engine = CausalForecastingEngine(
        [("x", "y", 0.95), ("y", "z", 0.90), ("x", "z", 0.20)],
        ["x", "y", "z"],
        model_factory=LinearRegression,
    )

    engine.fit_structural_equations(df)
    forecast = engine.forecast({"x": 20.0, "y": 41.0, "z": 83.0}, horizon=2)

    assert set(forecast) == {"x", "y", "z"}
    assert forecast["x"] == [20.0, 20.0]
    assert forecast["y"] == pytest.approx([41.0, 41.0])
    assert forecast["z"] == pytest.approx([83.0, 83.0])
    assert ("x", "z") not in engine.G.edges
    assert engine.structural_equation_quality().shape[0] == 3


def test_causal_forecasting_engine_rejects_cycles():
    with pytest.raises(ValueError, match="acyclic"):
        CausalForecastingEngine(
            [("x", "y", 1.0), ("y", "x", 1.0)],
            ["x", "y"],
        )


def test_shock_injector_step_gradual_and_pulse():
    series = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])

    step = ShockInjector.step_shock(series, 2, 0.5)
    gradual = ShockInjector.gradual_shock(series, 1, 0.4, transition_steps=2)
    pulse = ShockInjector.pulse_shock(series, 1, -0.2, duration=2)

    assert step.tolist() == [10.0, 10.0, 15.0, 15.0, 15.0]
    assert gradual.tolist() == [10.0, 12.0, 14.0, 14.0, 14.0]
    assert pulse.tolist() == [10.0, 8.0, 8.0, 10.0, 10.0]


def test_inject_dataframe_shock_changes_only_selected_variable():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [5.0, 5.0, 5.0]})

    shocked = ShockInjector.inject_dataframe_shock(df, "x", "step", 1, 1.0)

    assert shocked["x"].tolist() == [1.0, 4.0, 6.0]
    assert shocked["y"].tolist() == [5.0, 5.0, 5.0]


def test_evaluate_forecast_and_comparison_table():
    result_a = evaluate_forecast(
        [10.0, 20.0],
        [11.0, 19.0],
        model_name="A",
        period_name="post-shock",
    )
    result_b = evaluate_forecast(
        [10.0, 20.0],
        [14.0, 22.0],
        model_name="B",
        period_name="post-shock",
    )
    table = compare_forecasts([result_b, result_a])

    assert result_a.mae == pytest.approx(1.0)
    assert result_a.rmse == pytest.approx(1.0)
    assert result_a.mape == pytest.approx(7.5)
    assert table.iloc[0]["model"] == "A"


def test_recovery_time_to_baseline():
    assert recovery_time_to_baseline([5.0, 3.0, 1.05], 1.0, tolerance=0.1) == 2
    assert recovery_time_to_baseline([5.0, 3.0], 1.0, tolerance=0.1) is None
