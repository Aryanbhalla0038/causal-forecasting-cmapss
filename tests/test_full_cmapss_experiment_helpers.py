import numpy as np
import pandas as pd

from scripts.run_full_cmapss_experiment import (
    base_stability_to_frame,
    break_cycles_by_confidence,
    collapse_stability_matrix,
    fast_temporal_bootstrap_stability,
    parse_lagged_name,
    temporal_edges_from_base_stability,
    temporal_edges_from_discovery,
    temporal_edges_from_confidence_report,
)


class Row:
    def __init__(self, cause, effect, frequency):
        self.cause = cause
        self.effect = effect
        self.bootstrap_frequency = frequency


class Edge:
    def __init__(self, source, target, edge_type):
        self.source = source
        self.target = target
        self.edge_type = edge_type


def test_parse_lagged_name():
    assert parse_lagged_name("s11_lag1") == ("s11", 1)
    assert parse_lagged_name("s4") == ("s4", 0)


def test_temporal_edges_only_allows_lagged_cause_to_current_effect():
    report = [
        Row("s11_lag1", "s4", 0.8),
        Row("s4", "s11_lag1", 0.9),
        Row("s11_lag1", "s11", 0.9),
        Row("s2_lag1", "s3", 0.4),
    ]

    assert temporal_edges_from_confidence_report(report, threshold=0.5) == [
        ("s11", "s4", 0.8)
    ]


def test_temporal_edges_from_discovery_filters_to_directed_temporal_edges():
    edges = [
        Edge("s11_lag1", "s4", "directed"),
        Edge("s4", "s11_lag1", "directed"),
        Edge("s2_lag1", "s3", "undirected"),
    ]

    assert temporal_edges_from_discovery(edges, default_confidence=0.5) == [
        ("s11", "s4", 0.5)
    ]


def test_break_cycles_by_confidence_removes_weaker_cycle_edge():
    edges = [("a", "b", 0.9), ("b", "c", 0.8), ("c", "a", 0.7)]

    assert break_cycles_by_confidence(edges) == [("a", "b", 0.9), ("b", "c", 0.8)]


def test_collapse_stability_matrix_respects_temporal_direction():
    stability = np.array(
        [
            [0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0, 0.0],
        ]
    )
    collapsed = collapse_stability_matrix(
        stability,
        ["s2", "s2_lag1", "s3", "s3_lag1"],
        ["s2", "s3"],
    )

    assert collapsed[0, 1] == 0.9
    assert collapsed[1, 0] == 0.8


def test_temporal_edges_from_base_stability_and_report_frame():
    stability = np.array([[0.0, 0.9], [0.4, 0.0]])

    edges = temporal_edges_from_base_stability(stability, ["s2", "s3"], threshold=0.5)
    frame = base_stability_to_frame(stability, ["s2", "s3"], threshold=0.5)

    assert edges == [("s2", "s3", 0.9)]
    assert frame.iloc[0]["used_in_simulation"] == np.bool_(True)


def test_fast_temporal_bootstrap_stability_finds_lagged_predictor():
    values = np.arange(50, dtype=float)
    lagged = pd.DataFrame(
        {
            "s2": values[1:],
            "s2_lag1": values[:-1],
            "s3": values[1:] * 2,
            "s3_lag1": values[:-1] * 2,
        }
    )

    stability = fast_temporal_bootstrap_stability(
        lagged,
        ["s2", "s3"],
        n_bootstrap=5,
        sample_fraction=0.8,
        min_abs_correlation=0.5,
        max_parents_per_target=1,
        random_state=1,
    )

    assert stability[0, 1] == 1.0
