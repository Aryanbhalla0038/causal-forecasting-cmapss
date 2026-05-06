import numpy as np

from scripts.run_cmapss_pipeline import (
    _base_name,
    _collapse_lagged_edges,
    _collapse_stability_matrix,
)


def test_base_name_removes_lag_suffix():
    assert _base_name("s2_lag1") == "s2"
    assert _base_name("s3") == "s3"


def test_collapse_lagged_edges_keeps_best_cross_variable_confidence():
    collapsed = _collapse_lagged_edges(
        [
            ("s2_lag1", "s3", 0.7),
            ("s2", "s3_lag1", 0.9),
            ("s2_lag1", "s2", 1.0),
        ]
    )

    assert collapsed == [("s2", "s3", 0.9)]


def test_collapse_stability_matrix_maps_lagged_to_base_variables():
    stability = np.array(
        [
            [0.0, 0.0, 0.8],
            [0.0, 0.0, 0.9],
            [0.0, 0.0, 0.0],
        ]
    )

    collapsed = _collapse_stability_matrix(
        stability,
        ["s2", "s2_lag1", "s3"],
        ["s2", "s3"],
    )

    assert collapsed[0, 1] == 0.9

