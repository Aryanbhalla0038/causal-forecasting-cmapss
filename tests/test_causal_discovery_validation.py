import numpy as np

from causal_forecasting.causal_discovery.discovery import DiscoveryResult
from causal_forecasting.causal_discovery.graph_utils import CausalEdge
from causal_forecasting.causal_discovery.validation import (
    bootstrap_edge_stability,
    build_domain_validation_records,
    compute_edge_confidence_report,
    domain_validation_to_frame,
    edge_confidence_report_to_frame,
    reliable_edges_from_report,
)


def test_bootstrap_edge_stability_counts_directed_edges_from_runner():
    calls = {"count": 0}

    def runner(data, variable_names, config):
        calls["count"] += 1
        edges = [CausalEdge("x", "y", "directed")]
        if calls["count"] % 2 == 0:
            edges.append(CausalEdge("y", "z", "directed"))
        return DiscoveryResult(
            algorithm="PC",
            graph_matrix=np.zeros((3, 3)),
            variable_names=variable_names,
            edges=edges,
            raw_graph=None,
        )

    data = np.arange(60, dtype=float).reshape(20, 3)

    stability = bootstrap_edge_stability(
        data,
        variable_names=["x", "y", "z"],
        n_bootstrap=4,
        sample_fraction=0.5,
        random_state=1,
        discovery_runner=runner,
    )

    assert stability[0, 1] == 1.0
    assert stability[1, 2] == 0.5
    assert stability[2, 0] == 0.0


def test_compute_edge_confidence_report_labels_and_filters_edges():
    matrix = np.array(
        [
            [0.0, 0.9, 0.4],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0],
        ]
    )

    report = compute_edge_confidence_report(matrix, ["x", "y", "z"])

    assert [row.cause for row in report] == ["x", "y", "x"]
    assert [row.effect for row in report] == ["y", "z", "z"]
    assert [row.confidence for row in report] == ["HIGH", "MEDIUM", "LOW"]
    assert reliable_edges_from_report(report) == [("x", "y", 0.9)]

    frame = edge_confidence_report_to_frame(report)
    assert frame.loc[0, "used_in_simulation"] == np.bool_(True)


def test_build_domain_validation_records_creates_report_ready_rows():
    pc_edges = [
        CausalEdge("core_speed", "hpc_temp", "directed"),
        CausalEdge("setting1", "setting2", "directed"),
    ]
    fci_edges = [
        CausalEdge("core_speed", "hpc_temp", "directed"),
        CausalEdge("fuel_flow", "lpt_temp", "directed"),
    ]
    priors = {
        ("core_speed", "hpc_temp"): "CONFIRMED",
        ("setting1", "setting2"): "SUSPICIOUS",
        ("fuel_flow", "lpt_temp"): "PLAUSIBLE",
    }

    records = build_domain_validation_records(
        pc_edges,
        fci_edges,
        priors,
        evidence={("core_speed", "hpc_temp"): "CMAPSS thermodynamic prior"},
    )
    frame = domain_validation_to_frame(records)

    accepted = frame[frame["edge"] == "core_speed -> hpc_temp"].iloc[0]
    flagged = frame[frame["edge"] == "fuel_flow -> lpt_temp"].iloc[0]

    assert accepted["pc"] == "Y"
    assert accepted["fci"] == "Y"
    assert accepted["status"] == "ACCEPT"
    assert flagged["status"] == "FLAG"

