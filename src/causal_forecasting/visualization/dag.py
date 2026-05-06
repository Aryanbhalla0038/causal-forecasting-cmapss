"""Interactive DAG visualization helpers."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
from pyvis.network import Network


def create_interactive_dag(
    graph: nx.DiGraph,
    stability_matrix: np.ndarray,
    variable_names: list[str],
    output_path: str | Path,
    *,
    min_confidence: float = 0.50,
) -> Path:
    """Create a browser-viewable causal DAG with confidence-weighted edges."""

    if not isinstance(graph, nx.DiGraph):
        raise TypeError("graph must be a networkx.DiGraph.")
    matrix = np.asarray(stability_matrix, dtype=float)
    if matrix.shape != (len(variable_names), len(variable_names)):
        raise ValueError("stability_matrix shape must match variable_names.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    net = Network(height="680px", width="100%", directed=True, notebook=False)
    net.barnes_hut(gravity=-2600, central_gravity=0.25, spring_length=150)

    for variable in variable_names:
        in_degree = graph.in_degree(variable)
        out_degree = graph.out_degree(variable)
        if in_degree == 0:
            color = "#1f9d55"
        elif out_degree == 0:
            color = "#c2410c"
        else:
            color = "#2563eb"
        net.add_node(
            variable,
            label=variable,
            color=color,
            title=f"In-degree: {in_degree}; Out-degree: {out_degree}",
        )

    index = {name: idx for idx, name in enumerate(variable_names)}
    for source, target in graph.edges:
        if source not in index or target not in index:
            continue
        freq = float(matrix[index[source], index[target]])
        if freq < min_confidence:
            continue
        color = "#111827" if freq >= 0.80 else "#6b7280"
        net.add_edge(
            source,
            target,
            width=max(1.0, freq * 5),
            color=color,
            title=f"Bootstrap confidence: {freq:.2f}",
        )

    net.save_graph(str(output))
    return output

