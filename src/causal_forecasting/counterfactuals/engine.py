"""Counterfactual simulation engine implementing graph mutilation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from causal_forecasting.forecasting import CausalForecastingEngine
from causal_forecasting.forecasting.causal_engine import StructuralEquation


PARAMETER_UNCERTAINTY_DISCLAIMER = (
    "Intervals capture simulation and parameter perturbation uncertainty under "
    "the fitted DAG. They do not fully capture structural uncertainty about "
    "whether the DAG itself is correct."
)


@dataclass(frozen=True)
class Intervention:
    """A Pearl do-operator intervention."""

    variable: str
    value: float
    horizon: int


@dataclass(frozen=True)
class CounterfactualResult:
    """Complete result for one counterfactual query."""

    intervention: Intervention
    factual: dict[str, list[float]]
    counterfactual: dict[str, list[float]]
    difference: dict[str, list[float]]
    confidence_interval: dict[str, dict[str, list[float]]]
    removed_incoming_edges: list[tuple[str, str]]
    disclaimer: str = PARAMETER_UNCERTAINTY_DISCLAIMER

    def to_dict(self) -> dict[str, object]:
        """Serialize the result for API or dashboard use."""

        return {
            "intervention": {
                "variable": self.intervention.variable,
                "value": self.intervention.value,
                "horizon": self.intervention.horizon,
            },
            "factual": self.factual,
            "counterfactual": self.counterfactual,
            "difference": self.difference,
            "confidence_interval": self.confidence_interval,
            "removed_incoming_edges": self.removed_incoming_edges,
            "disclaimer": self.disclaimer,
        }


class CounterfactualEngine:
    """Run model-consistent counterfactual forecasts under stated assumptions."""

    def __init__(
        self,
        causal_engine: CausalForecastingEngine,
        *,
        residual_scale: float = 0.05,
        random_state: int | None = 42,
    ) -> None:
        if not causal_engine.structural_equations:
            raise RuntimeError("causal_engine must be fitted before counterfactual use.")
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative.")

        self.causal_engine = causal_engine
        self.variable_names = list(causal_engine.variable_names)
        self.residual_scale = residual_scale
        self.random_state = random_state

    def intervene(
        self,
        current_state: dict[str, float] | pd.Series,
        intervention_var: str,
        intervention_value: float,
        *,
        forecast_horizon: int = 10,
        n_bootstrap: int = 500,
        confidence: float = 0.95,
    ) -> CounterfactualResult:
        """Compute P(future | do(intervention_var = intervention_value))."""

        if intervention_var not in self.variable_names:
            raise KeyError(f"Unknown intervention variable: {intervention_var}")
        if forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive.")
        if n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive.")
        if not 0 < confidence < 1:
            raise ValueError("confidence must be in the interval (0, 1).")

        state = dict(current_state)
        missing = sorted(set(self.variable_names) - set(state))
        if missing:
            raise KeyError(f"Current state missing variables: {missing}")

        factual = self.causal_engine.forecast(state, horizon=forecast_horizon)
        mutilated_engine = deepcopy(self.causal_engine)
        removed_edges = list(mutilated_engine.G.in_edges(intervention_var))
        mutilated_engine.G.remove_edges_from(removed_edges)
        mutilated_engine.structural_equations[intervention_var] = StructuralEquation(
            variable=intervention_var,
            equation_type="persistence",
            parents=[],
            model=None,
            train_r2=None,
        )

        intervened_state = {var: float(state[var]) for var in self.variable_names}
        intervened_state[intervention_var] = float(intervention_value)
        counterfactual = mutilated_engine.forecast(
            intervened_state,
            horizon=forecast_horizon,
        )
        intervals = self._bootstrap_intervals(
            counterfactual,
            horizon=forecast_horizon,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
        )

        return CounterfactualResult(
            intervention=Intervention(
                variable=intervention_var,
                value=float(intervention_value),
                horizon=forecast_horizon,
            ),
            factual=factual,
            counterfactual=counterfactual,
            difference=_forecast_difference(counterfactual, factual, self.variable_names),
            confidence_interval=intervals,
            removed_incoming_edges=removed_edges,
        )

    def intervene_relative(
        self,
        current_state: dict[str, float] | pd.Series,
        intervention_var: str,
        magnitude: float,
        *,
        forecast_horizon: int = 10,
        n_bootstrap: int = 500,
        confidence: float = 0.95,
    ) -> CounterfactualResult:
        """Intervene by a relative percentage change from the current state."""

        if intervention_var not in current_state:
            raise KeyError(f"Current state missing variable: {intervention_var}")
        new_value = float(current_state[intervention_var]) * (1 + magnitude)
        return self.intervene(
            current_state,
            intervention_var,
            new_value,
            forecast_horizon=forecast_horizon,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
        )

    def _bootstrap_intervals(
        self,
        point_forecast: dict[str, list[float]],
        *,
        horizon: int,
        n_bootstrap: int,
        confidence: float,
    ) -> dict[str, dict[str, list[float]]]:
        """Create percentile intervals by perturbing forecast trajectories."""

        rng = np.random.default_rng(self.random_state)
        alpha = 1 - confidence
        intervals: dict[str, dict[str, list[float]]] = {
            "lower": {},
            "upper": {},
        }

        for variable in self.variable_names:
            base = np.asarray(point_forecast[variable], dtype=float)
            scale = _trajectory_noise_scale(base, self.residual_scale)
            samples = rng.normal(
                loc=base,
                scale=scale,
                size=(n_bootstrap, horizon),
            )
            intervals["lower"][variable] = np.percentile(
                samples,
                alpha / 2 * 100,
                axis=0,
            ).tolist()
            intervals["upper"][variable] = np.percentile(
                samples,
                (1 - alpha / 2) * 100,
                axis=0,
            ).tolist()

        return intervals


def _forecast_difference(
    counterfactual: dict[str, list[float]],
    factual: dict[str, list[float]],
    variable_names: list[str],
) -> dict[str, list[float]]:
    difference = {}
    for variable in variable_names:
        difference[variable] = (
            np.asarray(counterfactual[variable], dtype=float)
            - np.asarray(factual[variable], dtype=float)
        ).tolist()
    return difference


def _trajectory_noise_scale(base: np.ndarray, residual_scale: float) -> np.ndarray:
    absolute = np.maximum(np.abs(base), 1.0)
    return absolute * residual_scale
