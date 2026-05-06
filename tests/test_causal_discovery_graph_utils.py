import numpy as np
import pytest

from causal_forecasting.causal_discovery.graph_utils import (
    CausalEdge,
    compare_pc_fci,
    compute_algorithm_agreement,
    extract_edges_from_matrix,
)


def test_extract_edges_from_matrix_classifies_common_endpoint_patterns():
    names = ["x", "y", "z", "w"]
    matrix = np.array(
        [
            [0, -1, 1, 2],
            [1, 0, 1, 0],
            [1, 1, 0, -1],
            [2, 0, -1, 0],
        ]
    )

    edges = extract_edges_from_matrix(matrix, names)

    assert CausalEdge("x", "y", "directed") in edges
    assert CausalEdge("x", "z", "bidirected") in edges
    assert CausalEdge("x", "w", "circle") in edges
    assert CausalEdge("z", "w", "undirected") in edges


def test_extract_edges_from_matrix_reverses_directed_edge_when_needed():
    names = ["cause", "effect"]
    matrix = np.array([[0, 1], [-1, 0]])

    edges = extract_edges_from_matrix(matrix, names)

    assert edges == [CausalEdge("effect", "cause", "directed")]


def test_compare_pc_fci_reports_agreement_and_confounding():
    pc_edges = [
        CausalEdge("x", "y", "directed"),
        CausalEdge("z", "y", "directed"),
        CausalEdge("a", "b", "directed"),
    ]
    fci_edges = [
        CausalEdge("x", "y", "directed"),
        CausalEdge("y", "z", "bidirected"),
        CausalEdge("c", "b", "directed"),
    ]

    comparison = compare_pc_fci(pc_edges, fci_edges)

    assert comparison["high_confidence_edges"] == {("x", "y")}
    assert comparison["suspected_confounded"] == {("z", "y")}
    assert comparison["fci_exclusive"] == {("c", "b")}
    assert comparison["pc_exclusive"] == {("a", "b")}


def test_compute_algorithm_agreement_returns_kappa_and_rate():
    pc = np.array(
        [
            [0, -1, 0],
            [1, 0, -1],
            [0, 1, 0],
        ]
    )
    fci = np.array(
        [
            [0, 2, 0],
            [2, 0, 0],
            [0, 0, 0],
        ]
    )

    result = compute_algorithm_agreement(pc, fci)

    assert result["agreement_rate"] == pytest.approx(2 / 3)
    assert -1.0 <= result["kappa"] <= 1.0


def test_compute_algorithm_agreement_requires_matching_shapes():
    with pytest.raises(ValueError, match="same shape"):
        compute_algorithm_agreement(np.zeros((2, 2)), np.zeros((3, 3)))

