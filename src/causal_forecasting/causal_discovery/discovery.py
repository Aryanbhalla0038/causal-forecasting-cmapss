"""Wrappers around PC and FCI from causal-learn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from causal_forecasting.causal_discovery.graph_utils import (
    CausalEdge,
    extract_edges_from_matrix,
)


IndependenceTest = Literal["fisherz", "chisq", "gsq", "kci", "mv_fisherz"]


@dataclass(frozen=True)
class DiscoveryConfig:
    """Common settings for constraint-based causal discovery."""

    alpha: float = 0.05
    independence_test: IndependenceTest = "fisherz"
    stable: bool = True
    show_progress: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class DiscoveryResult:
    """Normalized output from a causal discovery algorithm."""

    algorithm: Literal["PC", "FCI"]
    graph_matrix: np.ndarray
    variable_names: list[str]
    edges: list[CausalEdge]
    raw_graph: object


def _as_numpy_and_names(
    data: pd.DataFrame | np.ndarray,
    variable_names: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    if isinstance(data, pd.DataFrame):
        values = data.values
        names = list(data.columns) if variable_names is None else variable_names
    else:
        values = np.asarray(data)
        if variable_names is None:
            names = [f"X{i}" for i in range(values.shape[1])]
        else:
            names = variable_names

    if values.ndim != 2:
        raise ValueError("Causal discovery data must be a 2D matrix.")
    if values.shape[1] != len(names):
        raise ValueError("variable_names length must match the number of columns.")
    if not np.isfinite(values).all():
        raise ValueError("Causal discovery data contains NaN or infinite values.")

    return values, names


def run_pc(
    data: pd.DataFrame | np.ndarray,
    *,
    variable_names: list[str] | None = None,
    config: DiscoveryConfig | None = None,
) -> DiscoveryResult:
    """Run stable-PC and return normalized edge records."""

    from causallearn.search.ConstraintBased.PC import pc

    config = config or DiscoveryConfig()
    values, names = _as_numpy_and_names(data, variable_names)

    cg = pc(
        values,
        alpha=config.alpha,
        indep_test=config.independence_test,
        stable=config.stable,
        uc_rule=0,
        uc_priority=2,
        verbose=config.verbose,
        show_progress=config.show_progress,
        node_names=names,
    )
    matrix = np.asarray(cg.G.graph)
    return DiscoveryResult(
        algorithm="PC",
        graph_matrix=matrix,
        variable_names=names,
        edges=extract_edges_from_matrix(matrix, names),
        raw_graph=cg.G,
    )


def run_fci(
    data: pd.DataFrame | np.ndarray,
    *,
    variable_names: list[str] | None = None,
    config: DiscoveryConfig | None = None,
    depth: int = -1,
    max_path_length: int = -1,
) -> DiscoveryResult:
    """Run FCI and return normalized PAG edge records."""

    from causallearn.search.ConstraintBased.FCI import fci

    config = config or DiscoveryConfig()
    values, names = _as_numpy_and_names(data, variable_names)

    graph, _ = fci(
        dataset=values,
        independence_test_method=config.independence_test,
        alpha=config.alpha,
        depth=depth,
        max_path_length=max_path_length,
        verbose=config.verbose,
        show_progress=config.show_progress,
        node_names=names,
    )
    matrix = np.asarray(graph.graph)
    return DiscoveryResult(
        algorithm="FCI",
        graph_matrix=matrix,
        variable_names=names,
        edges=extract_edges_from_matrix(matrix, names),
        raw_graph=graph,
    )

