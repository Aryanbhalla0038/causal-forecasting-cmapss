"""DAG-guided causal forecasting engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


ModelFactory = Callable[[], object]


@dataclass
class StructuralEquation:
    """Fitted mechanism for one variable in the causal graph."""

    variable: str
    equation_type: str
    parents: list[str]
    model: object | None
    train_r2: float | None = None


class CausalForecastingEngine:
    """Forecast by propagating values through reliable causal edges."""

    def __init__(
        self,
        dag_edges: list[tuple[str, str, float]] | list[tuple[str, str]],
        variable_names: list[str],
        *,
        confidence_threshold: float = 0.80,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self.variable_names = list(variable_names)
        self.confidence_threshold = confidence_threshold
        self.model_factory = model_factory or (
            lambda: GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=3,
                random_state=42,
            )
        )
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.variable_names)
        self.G.add_edges_from(self._filter_edges(dag_edges))
        if not nx.is_directed_acyclic_graph(self.G):
            raise ValueError("Causal forecasting requires an acyclic DAG.")
        self.structural_equations: dict[str, StructuralEquation] = {}

    def _filter_edges(
        self,
        dag_edges: list[tuple[str, str, float]] | list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        reliable_edges = []
        known = set(self.variable_names)
        for edge in dag_edges:
            if len(edge) == 2:
                cause, effect = edge
                confidence = 1.0
            elif len(edge) == 3:
                cause, effect, confidence = edge
            else:
                raise ValueError("dag_edges must contain (cause, effect[, confidence]).")

            if cause not in known or effect not in known:
                raise KeyError(f"Unknown DAG edge variable: {cause} -> {effect}")
            if float(confidence) >= self.confidence_threshold:
                reliable_edges.append((cause, effect))
        return reliable_edges

    def fit_structural_equations(self, df: pd.DataFrame) -> None:
        """Fit one structural equation per variable using its DAG parents."""

        self._validate_dataframe(df)
        self.structural_equations = {}

        for variable in nx.topological_sort(self.G):
            parents = list(self.G.predecessors(variable))
            if not parents:
                self.structural_equations[variable] = StructuralEquation(
                    variable=variable,
                    equation_type="persistence",
                    parents=[],
                    model=None,
                    train_r2=None,
                )
                continue

            X = df[parents].values
            y = df[variable].values
            model = self.model_factory()
            model.fit(X, y)
            prediction = np.asarray(model.predict(X))
            self.structural_equations[variable] = StructuralEquation(
                variable=variable,
                equation_type="structural",
                parents=parents,
                model=model,
                train_r2=float(r2_score(y, prediction)),
            )

    def forecast(
        self,
        current_state: dict[str, float] | pd.Series,
        *,
        horizon: int = 5,
    ) -> dict[str, list[float]]:
        """Forecast every variable for ``horizon`` steps."""

        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        if not self.structural_equations:
            raise RuntimeError("fit_structural_equations must be called before forecast.")

        state = self._validate_state(current_state)
        forecasts = {var: [state[var]] for var in self.variable_names}

        for _ in range(horizon):
            step_values: dict[str, float] = {}
            for variable in nx.topological_sort(self.G):
                equation = self.structural_equations[variable]
                if equation.equation_type == "persistence":
                    step_values[variable] = forecasts[variable][-1]
                else:
                    parent_values = np.array(
                        [[forecasts[parent][-1] for parent in equation.parents]]
                    )
                    step_values[variable] = float(equation.model.predict(parent_values)[0])

            for variable in self.variable_names:
                forecasts[variable].append(step_values[variable])

        return {variable: values[1:] for variable, values in forecasts.items()}

    def structural_equation_quality(self) -> pd.DataFrame:
        """Return R2 and parent metadata for report tables."""

        rows = []
        for variable in self.variable_names:
            equation = self.structural_equations.get(variable)
            if equation is None:
                continue
            rows.append(
                {
                    "variable": variable,
                    "type": equation.equation_type,
                    "parents": ", ".join(equation.parents),
                    "train_r2": equation.train_r2,
                }
            )
        return pd.DataFrame(rows)

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        missing = sorted(set(self.variable_names) - set(df.columns))
        if missing:
            raise KeyError(f"Missing dataframe columns: {missing}")
        if df[self.variable_names].isna().any().any():
            raise ValueError("Training dataframe contains missing values.")

    def _validate_state(self, current_state: dict[str, float] | pd.Series) -> dict[str, float]:
        state = dict(current_state)
        missing = sorted(set(self.variable_names) - set(state))
        if missing:
            raise KeyError(f"Current state missing variables: {missing}")
        return {var: float(state[var]) for var in self.variable_names}

