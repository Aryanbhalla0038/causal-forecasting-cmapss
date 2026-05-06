"""Graph parsing and PC-vs-FCI comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import cohen_kappa_score


EdgeType = Literal[
    "directed",
    "undirected",
    "bidirected",
    "circle",
    "circle_directed",
    "circle_tail",
    "unknown",
]


@dataclass(frozen=True, order=True)
class CausalEdge:
    """Human-readable edge extracted from a causal-learn graph matrix."""

    source: str
    target: str
    edge_type: EdgeType

    @property
    def unordered_pair(self) -> frozenset[str]:
        return frozenset((self.source, self.target))


def _classify_pair(endpoint_ab: int, endpoint_ba: int) -> tuple[EdgeType, bool]:
    """Classify endpoint codes for a pair A,B.

    causal-learn encodes endpoints in a matrix. In the common convention:
    -1 is a tail, 1 is an arrowhead, 2 is a circle endpoint, and 0 is no edge.
    For a directed A -> B edge, matrix[A, B] = -1 and matrix[B, A] = 1.
    """

    if endpoint_ab == 0 and endpoint_ba == 0:
        return "unknown", False
    if endpoint_ab == -1 and endpoint_ba == 1:
        return "directed", True
    if endpoint_ab == 1 and endpoint_ba == -1:
        return "directed", False
    if endpoint_ab == -1 and endpoint_ba == -1:
        return "undirected", True
    if endpoint_ab == 1 and endpoint_ba == 1:
        return "bidirected", True
    if endpoint_ab == 2 and endpoint_ba == 2:
        return "circle", True
    if endpoint_ab == 2 and endpoint_ba == 1:
        return "circle_directed", True
    if endpoint_ab == 1 and endpoint_ba == 2:
        return "circle_directed", False
    if endpoint_ab == 2 and endpoint_ba == -1:
        return "circle_tail", True
    if endpoint_ab == -1 and endpoint_ba == 2:
        return "circle_tail", False
    return "unknown", True


def extract_edges_from_matrix(
    graph_matrix: np.ndarray,
    variable_names: list[str],
) -> list[CausalEdge]:
    """Extract readable edge records from a causal-learn adjacency matrix."""

    matrix = np.asarray(graph_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("graph_matrix must be square.")
    if matrix.shape[0] != len(variable_names):
        raise ValueError("variable_names length must match graph_matrix shape.")

    edges: list[CausalEdge] = []
    n_vars = len(variable_names)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            edge_type, forward = _classify_pair(int(matrix[i, j]), int(matrix[j, i]))
            if edge_type == "unknown":
                continue

            if edge_type == "directed":
                source, target = (
                    (variable_names[i], variable_names[j])
                    if forward
                    else (variable_names[j], variable_names[i])
                )
            else:
                source, target = variable_names[i], variable_names[j]
                if not forward:
                    source, target = target, source

            edges.append(CausalEdge(source, target, edge_type))

    return sorted(edges)


def _edge_set(
    edges: list[CausalEdge],
    edge_type: EdgeType,
) -> set[tuple[str, str]]:
    return {(edge.source, edge.target) for edge in edges if edge.edge_type == edge_type}


def compare_pc_fci(
    pc_edges: list[CausalEdge],
    fci_edges: list[CausalEdge],
) -> dict[str, set[tuple[str, str]]]:
    """Compare PC-DAG and FCI-PAG outputs.

    High-confidence edges agree exactly in direction. Suspected confounded
    edges are directed in PC but bidirected in FCI for the same variable pair.
    """

    pc_directed = _edge_set(pc_edges, "directed")
    fci_directed = _edge_set(fci_edges, "directed")
    fci_bidirected_pairs = {
        edge.unordered_pair for edge in fci_edges if edge.edge_type == "bidirected"
    }

    high_confidence = pc_directed & fci_directed
    suspected_confounded = {
        edge for edge in pc_directed if frozenset(edge) in fci_bidirected_pairs
    }
    fci_exclusive = fci_directed - pc_directed
    pc_exclusive = pc_directed - fci_directed - suspected_confounded

    return {
        "high_confidence_edges": high_confidence,
        "suspected_confounded": suspected_confounded,
        "fci_exclusive": fci_exclusive,
        "pc_exclusive": pc_exclusive,
    }


def compute_algorithm_agreement(
    pc_adj_matrix: np.ndarray,
    fci_adj_matrix: np.ndarray,
) -> dict[str, float]:
    """Compute edge-presence agreement and Cohen's Kappa for PC vs FCI."""

    pc_matrix = np.asarray(pc_adj_matrix)
    fci_matrix = np.asarray(fci_adj_matrix)
    if pc_matrix.shape != fci_matrix.shape:
        raise ValueError("PC and FCI matrices must have the same shape.")
    if pc_matrix.ndim != 2 or pc_matrix.shape[0] != pc_matrix.shape[1]:
        raise ValueError("Adjacency matrices must be square.")

    n_vars = pc_matrix.shape[0]
    pc_flat = []
    fci_flat = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            pc_flat.append(int(pc_matrix[i, j] != 0 or pc_matrix[j, i] != 0))
            fci_flat.append(int(fci_matrix[i, j] != 0 or fci_matrix[j, i] != 0))

    agreement_rate = float(np.mean(np.equal(pc_flat, fci_flat)))
    if len(set(pc_flat)) == 1 and len(set(fci_flat)) == 1:
        kappa = 1.0 if pc_flat == fci_flat else 0.0
    else:
        kappa = float(cohen_kappa_score(pc_flat, fci_flat))

    return {"kappa": kappa, "agreement_rate": agreement_rate}

