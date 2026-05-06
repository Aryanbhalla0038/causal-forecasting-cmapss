"""Run the real-data CMAPSS pipeline for the project demo artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causal_forecasting.causal_discovery import (
    DiscoveryConfig,
    bootstrap_edge_stability,
    build_domain_validation_records,
    compare_pc_fci,
    compute_algorithm_agreement,
    compute_edge_confidence_report,
    domain_validation_to_frame,
    edge_confidence_report_to_frame,
    reliable_edges_from_report,
    run_fci,
    run_pc,
)
from causal_forecasting.counterfactuals import CounterfactualEngine
from causal_forecasting.data.ingestion import load_cmapss_train, select_cmapss_unit
from causal_forecasting.data.preprocessing import (
    create_lagged_features,
    make_stationary,
    normalize_dataframe,
    run_stationarity_tests,
    stationarity_results_to_frame,
    validate_time_series_frame,
)
from causal_forecasting.forecasting import (
    CausalForecastingEngine,
    ShockInjector,
    compare_forecasts,
    evaluate_forecast,
)
from causal_forecasting.visualization import (
    create_counterfactual_figure,
    create_interactive_dag,
    create_model_comparison_figure,
    write_html_compact,
    write_table,
)


RAW_PATH = Path("data/raw/train_FD001.txt")
PROCESSED_DIR = Path("data/processed/cmapss_fd001_unit1")
FIGURE_DIR = Path("reports/figures")


VARIABLES = ["s2", "s3", "s4", "s7", "s11", "s15"]
DOMAIN_PRIORS = {
    ("s2", "s3"): "PLAUSIBLE",
    ("s3", "s4"): "PLAUSIBLE",
    ("s7", "s11"): "UNCERTAIN",
    ("s11", "s15"): "PLAUSIBLE",
}


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CMAPSS FD001...")
    full_df = load_cmapss_train(RAW_PATH)
    unit_df = select_cmapss_unit(full_df, unit=1)
    selected = unit_df[VARIABLES]
    quality = validate_time_series_frame(selected)
    pd.DataFrame([quality]).to_csv(PROCESSED_DIR / "data_quality.csv", index=False)

    print("Running stationarity tests...")
    stationarity = run_stationarity_tests(selected)
    stationarity_table = stationarity_results_to_frame(stationarity)
    write_table(stationarity_table, PROCESSED_DIR / "stationarity.csv")

    stationary_df, differenced = make_stationary(selected, stationarity)
    scaled_df, _ = normalize_dataframe(stationary_df)
    lagged_df = create_lagged_features(scaled_df, max_lag=1, variables=VARIABLES)
    lagged_df.to_csv(PROCESSED_DIR / "lagged_scaled.csv")
    pd.DataFrame({"differenced_variable": differenced}).to_csv(
        PROCESSED_DIR / "differenced_variables.csv",
        index=False,
    )

    discovery_config = DiscoveryConfig(alpha=0.05, show_progress=False, verbose=False)
    print("Running PC and FCI on lagged variables...")
    pc_result = run_pc(lagged_df, config=discovery_config)
    fci_result = run_fci(lagged_df, config=discovery_config)
    comparison = compare_pc_fci(pc_result.edges, fci_result.edges)
    agreement = compute_algorithm_agreement(pc_result.graph_matrix, fci_result.graph_matrix)

    edge_rows = [
        {"algorithm": "PC", "source": e.source, "target": e.target, "type": e.edge_type}
        for e in pc_result.edges
    ] + [
        {"algorithm": "FCI", "source": e.source, "target": e.target, "type": e.edge_type}
        for e in fci_result.edges
    ]
    write_table(pd.DataFrame(edge_rows), PROCESSED_DIR / "discovered_edges.csv")
    write_table(pd.DataFrame([agreement]), PROCESSED_DIR / "algorithm_agreement.csv")
    write_table(_comparison_to_frame(comparison), PROCESSED_DIR / "pc_fci_comparison.csv")

    print("Running bootstrap edge stability...")
    stability = bootstrap_edge_stability(
        lagged_df,
        n_bootstrap=20,
        sample_fraction=0.80,
        config=discovery_config,
        random_state=42,
    )
    np.save(PROCESSED_DIR / "stability_matrix.npy", stability)
    confidence_report = compute_edge_confidence_report(
        stability,
        list(lagged_df.columns),
        threshold=0.80,
    )
    confidence_table = edge_confidence_report_to_frame(confidence_report)
    write_table(confidence_table, PROCESSED_DIR / "edge_confidence.csv")

    validation_records = build_domain_validation_records(
        pc_result.edges,
        fci_result.edges,
        DOMAIN_PRIORS,
        evidence={
            ("s2", "s3"): "Compressor temperature variables are physically coupled.",
            ("s3", "s4"): "Compressor and turbine temperature propagation is plausible.",
        },
    )
    write_table(domain_validation_to_frame(validation_records), PROCESSED_DIR / "domain_validation.csv")

    print("Fitting causal forecast engine...")
    train_size = int(len(scaled_df) * 0.70)
    train_df = scaled_df.iloc[:train_size]
    test_df = scaled_df.iloc[train_size:]
    base_edges = _collapse_lagged_edges(reliable_edges_from_report(confidence_report))
    edge_source = "bootstrap_reliable"
    if not base_edges:
        edge_source = "domain_prior_fallback"
        base_edges = [("s2", "s3", 1.0), ("s3", "s4", 1.0), ("s11", "s15", 1.0)]
    write_table(
        pd.DataFrame(
            [
                {
                    "cause": source,
                    "effect": target,
                    "internal_confidence": confidence,
                    "source": edge_source,
                    "scientific_note": (
                        "Used only for demonstration because bootstrap did not "
                        "produce high-confidence directed edges."
                        if edge_source == "domain_prior_fallback"
                        else "Passed bootstrap reliability threshold."
                    ),
                }
                for source, target, confidence in base_edges
            ]
        ),
        PROCESSED_DIR / "forecast_edges_used.csv",
    )

    engine = CausalForecastingEngine(
        base_edges,
        VARIABLES,
        confidence_threshold=0.60,
    )
    engine.fit_structural_equations(train_df)
    write_table(engine.structural_equation_quality(), PROCESSED_DIR / "structural_equations.csv")

    horizon = min(10, len(test_df))
    current_state = train_df.iloc[-1]
    causal_forecast = engine.forecast(current_state, horizon=horizon)
    target = "s4"
    shocked = ShockInjector.inject_dataframe_shock(
        test_df.iloc[:horizon].copy(),
        variable="s11",
        shock_type="step",
        shock_start_idx=0,
        magnitude=0.10,
    )
    persistence = [float(current_state[target])] * horizon
    metrics = compare_forecasts(
        [
            evaluate_forecast(
                shocked[target],
                persistence,
                model_name="Persistence Baseline",
                period_name="post-shock",
            ),
            evaluate_forecast(
                shocked[target],
                causal_forecast[target],
                model_name="Causal Engine",
                period_name="post-shock",
            ),
        ]
    )
    write_table(metrics, PROCESSED_DIR / "forecast_metrics.csv")

    print("Running counterfactual simulation...")
    counterfactual_engine = CounterfactualEngine(engine, residual_scale=0.05, random_state=42)
    counterfactual = counterfactual_engine.intervene_relative(
        current_state,
        intervention_var="s11",
        magnitude=0.10,
        forecast_horizon=horizon,
        n_bootstrap=200,
    ).to_dict()

    write_html_compact(
        create_counterfactual_figure(counterfactual, target),
        FIGURE_DIR / "counterfactual_s11_to_s4.html",
    )
    comparison_figure = create_model_comparison_figure(
        actual=shocked[target].tolist(),
        predictions={
            "Persistence Baseline": persistence,
            "Causal Engine": causal_forecast[target],
        },
        shock_index=0,
        title="CMAPSS FD001 Unit 1 Post-shock Forecast",
    )
    write_html_compact(
        comparison_figure,
        FIGURE_DIR / "model_comparison_fd001_unit1.html",
    )

    base_stability = _collapse_stability_matrix(stability, list(lagged_df.columns), VARIABLES)
    create_interactive_dag(
        engine.G,
        base_stability,
        VARIABLES,
        FIGURE_DIR / "causal_dag_fd001_unit1.html",
        min_confidence=0.0,
    )
    _write_run_summary(
        path=PROCESSED_DIR / "run_summary.md",
        quality=quality,
        differenced=differenced,
        agreement=agreement,
        edge_source=edge_source,
        metrics=metrics,
    )

    print("CMAPSS pipeline complete.")
    print(f"Artifacts: {PROCESSED_DIR}")
    print(f"Figures:   {FIGURE_DIR}")
    print(f"Agreement: {agreement}")
    print(f"Forecast metrics:\n{metrics.to_string(index=False)}")


def _comparison_to_frame(comparison: dict[str, set[tuple[str, str]]]) -> pd.DataFrame:
    rows = []
    for category, edges in comparison.items():
        for source, target in sorted(edges):
            rows.append({"category": category, "source": source, "target": target})
    return pd.DataFrame(rows or [{"category": "none", "source": "", "target": ""}])


def _base_name(variable: str) -> str:
    return variable.split("_lag", maxsplit=1)[0]


def _collapse_lagged_edges(edges: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    collapsed: dict[tuple[str, str], float] = {}
    for source, target, confidence in edges:
        base_source = _base_name(source)
        base_target = _base_name(target)
        if base_source == base_target:
            continue
        key = (base_source, base_target)
        collapsed[key] = max(collapsed.get(key, 0.0), confidence)
    return [(source, target, confidence) for (source, target), confidence in collapsed.items()]


def _collapse_stability_matrix(
    stability: np.ndarray,
    lagged_names: list[str],
    base_names: list[str],
) -> np.ndarray:
    collapsed = np.zeros((len(base_names), len(base_names)))
    base_index = {name: idx for idx, name in enumerate(base_names)}
    for i, source in enumerate(lagged_names):
        for j, target in enumerate(lagged_names):
            base_source = _base_name(source)
            base_target = _base_name(target)
            if base_source == base_target:
                continue
            if base_source in base_index and base_target in base_index:
                collapsed[base_index[base_source], base_index[base_target]] = max(
                    collapsed[base_index[base_source], base_index[base_target]],
                    stability[i, j],
                )
    return collapsed


def _write_run_summary(
    *,
    path: Path,
    quality: dict[str, object],
    differenced: list[str],
    agreement: dict[str, float],
    edge_source: str,
    metrics: pd.DataFrame,
) -> None:
    warning = ""
    if edge_source == "domain_prior_fallback":
        warning = (
            "\nScientific caution: bootstrap stability did not identify "
            "high-confidence directed edges on this compact single-unit run. "
            "Forecasting and counterfactual figures therefore use a documented "
            "domain-prior fallback graph for demonstration. Do not present those "
            "edges as discovered high-confidence causal claims.\n"
        )

    text = f"""# CMAPSS FD001 Unit 1 Run Summary

Dataset: `data/raw/train_FD001.txt`

Selected variables: `{", ".join(VARIABLES)}`

Rows: `{quality["rows"]}`

Differenced variables: `{", ".join(differenced) if differenced else "none"}`

PC/FCI agreement rate: `{agreement["agreement_rate"]:.3f}`

Cohen's Kappa: `{agreement["kappa"]:.3f}`

Forecast edge source: `{edge_source}`
{warning}
## Forecast Metrics

{metrics.to_string(index=False)}
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
