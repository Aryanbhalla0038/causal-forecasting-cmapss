"""Run a stronger multi-unit CMAPSS experiment.

Loops over every held-out engine and every shock scenario defined in
``docs/SHOCK_SCENARIOS.md`` so the headline metrics are means over many
engines, not anecdotes from a single one.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from causal_forecasting.causal_discovery import (
    DiscoveryConfig,
    compare_pc_fci,
    compute_algorithm_agreement,
    run_fci,
    run_pc,
)
from causal_forecasting.counterfactuals import CounterfactualEngine
from causal_forecasting.data.ingestion import load_cmapss_train
from causal_forecasting.data.preprocessing import (
    create_lagged_features,
    normalize_dataframe,
    run_stationarity_tests,
    stationarity_results_to_frame,
)
from causal_forecasting.forecasting import (
    CausalForecastingEngine,
    ShockInjector,
    arimax_forecast,
    compare_forecasts,
    evaluate_forecast,
    fit_arimax_baseline,
)
from causal_forecasting.visualization import (
    create_counterfactual_figure,
    create_interactive_dag,
    create_model_comparison_figure,
    write_html_compact,
    write_table,
)


DATASET_CHOICES = ("FD001", "FD002", "FD003", "FD004")
DATA_DIR = Path("data/raw")
PROCESSED_ROOT = Path("data/processed")
FIGURE_DIR = Path("reports/figures")
VARIABLES = ["s2", "s3", "s4", "s7", "s11", "s12", "s15", "s20"]
TARGET = "s4"
INTERVENTION = "s11"
MAX_LAG = 1
DISCOVERY_ROWS = 400
BOOTSTRAP_RUNS = 80
EDGE_THRESHOLD = 0.50
MIN_ABS_CORRELATION = 0.12
MAX_PARENTS_PER_TARGET = 2
HORIZON = 20
MIN_HORIZON = 5


# Shock scenarios mirror docs/SHOCK_SCENARIOS.md.
SHOCK_SCENARIOS = [
    {
        "name": "S1_step_s11_pos",
        "variable": "s11",
        "shock_type": "step",
        "magnitude": 0.10,
    },
    {
        "name": "S2_gradual_s11_pos",
        "variable": "s11",
        "shock_type": "gradual",
        "magnitude": 0.15,
        "transition_steps": 10,
    },
    {
        "name": "S3_pulse_s11_pos",
        "variable": "s11",
        "shock_type": "pulse",
        "magnitude": 0.20,
        "duration": 3,
    },
    {
        "name": "S4_step_s7_neg",
        "variable": "s7",
        "shock_type": "step",
        "magnitude": -0.10,
    },
    {
        "name": "S5_step_s2_pos",
        "variable": "s2",
        "shock_type": "step",
        "magnitude": 0.10,
    },
]
HEADLINE_SCENARIO = "S1_step_s11_pos"


def main(dataset_code: str = "FD001") -> pd.DataFrame:
    dataset_code = dataset_code.upper()
    if dataset_code not in DATASET_CHOICES:
        raise ValueError(f"dataset_code must be one of: {DATASET_CHOICES}")

    raw_path = DATA_DIR / f"train_{dataset_code}.txt"
    output_dir = PROCESSED_ROOT / f"cmapss_{dataset_code.lower()}_multiunit"
    output_dir.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {dataset_code} all units...")
    full_df = load_cmapss_train(raw_path).sort_values(["unit", "cycle"])
    selected = full_df[["unit", "cycle", *VARIABLES]].copy()
    train_units, test_units = _split_units(selected["unit"].unique(), train_fraction=0.70)
    train_raw = selected[selected["unit"].isin(train_units)].copy()
    test_raw = selected[selected["unit"].isin(test_units)].copy()

    print("Stationarity and per-unit differencing...")
    stationarity = run_stationarity_tests(train_raw[VARIABLES])
    stationarity_frame = stationarity_results_to_frame(stationarity)
    write_table(stationarity_frame, output_dir / "stationarity.csv")
    differenced_variables = [
        name for name, result in stationarity.items() if not result.stationary
    ]
    train_stationary = _difference_by_unit(train_raw, differenced_variables)
    test_stationary = _difference_by_unit(test_raw, differenced_variables)

    train_scaled, scaler = normalize_dataframe(train_stationary[VARIABLES])
    test_scaled_values = scaler.transform(test_stationary[VARIABLES].values)
    test_scaled = pd.DataFrame(
        test_scaled_values,
        columns=VARIABLES,
        index=test_stationary.index,
    )
    train_scaled = _with_unit_cycle(train_stationary, train_scaled)
    test_scaled = _with_unit_cycle(test_stationary, test_scaled)

    print("Creating pooled lag features without crossing unit boundaries...")
    lagged_train = _lag_by_unit(train_scaled, VARIABLES, max_lag=MAX_LAG)
    discovery_df = _sample_rows(lagged_train, max_rows=DISCOVERY_ROWS, random_state=42)
    discovery_df.to_csv(output_dir / "lagged_discovery_sample.csv", index=False)

    discovery_config = DiscoveryConfig(alpha=0.05, show_progress=False, verbose=False)
    print(f"Running PC and FCI on {len(discovery_df)} lagged rows...", flush=True)
    pc_result = run_pc(discovery_df, config=discovery_config)
    fci_result = run_fci(discovery_df, config=discovery_config)
    agreement = compute_algorithm_agreement(pc_result.graph_matrix, fci_result.graph_matrix)
    comparison = compare_pc_fci(pc_result.edges, fci_result.edges)
    write_table(pd.DataFrame([agreement]), output_dir / "algorithm_agreement.csv")
    write_table(_edge_frame(pc_result.edges, fci_result.edges), output_dir / "discovered_edges.csv")
    write_table(_comparison_to_frame(comparison), output_dir / "pc_fci_comparison.csv")

    print("Running fast temporal bootstrap stability...", flush=True)
    stability = fast_temporal_bootstrap_stability(
        lagged_train,
        VARIABLES,
        n_bootstrap=BOOTSTRAP_RUNS,
        sample_fraction=0.70,
        min_abs_correlation=MIN_ABS_CORRELATION,
        max_parents_per_target=MAX_PARENTS_PER_TARGET,
        random_state=7,
    )
    np.save(output_dir / "stability_matrix.npy", stability)
    confidence_table = base_stability_to_frame(
        stability,
        VARIABLES,
        threshold=EDGE_THRESHOLD,
    )
    write_table(confidence_table, output_dir / "edge_confidence_temporal.csv")
    temporal_edges = temporal_edges_from_base_stability(
        stability,
        VARIABLES,
        threshold=EDGE_THRESHOLD,
    )
    temporal_edges = break_cycles_by_confidence(temporal_edges)
    edge_source = "temporal_predictive_bootstrap"
    if not temporal_edges:
        edge_source = "pc_temporal_edges"
        temporal_edges = break_cycles_by_confidence(
            temporal_edges_from_discovery(pc_result.edges, default_confidence=EDGE_THRESHOLD)
        )
    if not temporal_edges:
        edge_source = "domain_prior_temporal_fallback"
        temporal_edges = [
            ("s2", "s3", 1.0),
            ("s3", "s4", 1.0),
            ("s11", "s15", 1.0),
        ]
    write_table(
        pd.DataFrame(temporal_edges, columns=["cause", "effect", "confidence"]),
        output_dir / "forecast_edges_used.csv",
    )

    print("Fitting causal engine on pooled training units...")
    engine = CausalForecastingEngine(
        temporal_edges,
        VARIABLES,
        confidence_threshold=EDGE_THRESHOLD,
    )
    engine.fit_structural_equations(train_scaled[VARIABLES])
    write_table(engine.structural_equation_quality(), output_dir / "structural_equations.csv")

    print("Fitting ARIMAX baseline...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        arimax = fit_arimax_baseline(
            train_scaled[VARIABLES].reset_index(drop=True),
            target_var=TARGET,
            exog_vars=[var for var in VARIABLES if var != TARGET],
            order=(1, 0, 1),
        )

    print(
        f"Evaluating across {len(test_units)} held-out engines x "
        f"{len(SHOCK_SCENARIOS)} shock scenarios...",
        flush=True,
    )
    per_unit_rows: list[dict] = []
    headline_first_unit_payload = None
    for unit in test_units:
        eval_df = (
            test_scaled[test_scaled["unit"] == unit]
            .reset_index(drop=True)
            .head(HORIZON + 1)
        )
        if len(eval_df) < MIN_HORIZON + 1:
            continue
        horizon = len(eval_df) - 1
        current_state = eval_df.loc[0, VARIABLES]
        actual_window = eval_df.loc[1:horizon, VARIABLES].reset_index(drop=True)

        causal_forecast = engine.forecast(current_state, horizon=horizon)
        persistence = [float(current_state[TARGET])] * horizon

        for scenario in SHOCK_SCENARIOS:
            shocked_window = _apply_scenario(actual_window, scenario)
            shocked_actual = shocked_window[TARGET].tolist()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arimax_pred = arimax_forecast(
                    arimax,
                    steps=horizon,
                    test_exog=shocked_window[[var for var in VARIABLES if var != TARGET]],
                )

            for model_name, predictions in [
                ("Persistence Baseline", persistence),
                ("ARIMAX", list(arimax_pred)),
                ("Causal Engine", causal_forecast[TARGET]),
            ]:
                evaluation = evaluate_forecast(
                    shocked_actual,
                    predictions,
                    model_name=model_name,
                    period_name="post-shock",
                )
                per_unit_rows.append(
                    {
                        "unit": int(unit),
                        "scenario": scenario["name"],
                        "model": evaluation.model,
                        "period": evaluation.period,
                        "mae": evaluation.mae,
                        "rmse": evaluation.rmse,
                        "mape": evaluation.mape,
                        "smape": evaluation.smape,
                    }
                )

            if (
                scenario["name"] == HEADLINE_SCENARIO
                and headline_first_unit_payload is None
            ):
                headline_first_unit_payload = {
                    "unit": int(unit),
                    "horizon": horizon,
                    "current_state": current_state,
                    "shocked_actual": shocked_actual,
                    "persistence": persistence,
                    "arimax": list(arimax_pred),
                    "causal": causal_forecast[TARGET],
                    "shocked_window": shocked_window,
                }

    per_unit = pd.DataFrame(per_unit_rows)
    write_table(per_unit, output_dir / "forecast_metrics_per_unit.csv")

    aggregated = (
        per_unit.groupby(["scenario", "model", "period"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mape_mean=("mape", "mean"),
            smape_mean=("smape", "mean"),
            smape_std=("smape", "std"),
            n_units=("unit", "nunique"),
        )
        .sort_values(["scenario", "mae_mean"], ignore_index=True)
    )
    write_table(aggregated, output_dir / "forecast_metrics_by_scenario.csv")

    # Headline metrics frame (back-compat: same columns as before, but values
    # are now means across all held-out engines for the headline scenario S1).
    headline_rows = []
    for _, row in aggregated[aggregated["scenario"] == HEADLINE_SCENARIO].iterrows():
        headline_rows.append(
            {
                "model": row["model"],
                "period": row["period"],
                "mae": row["mae_mean"],
                "rmse": row["rmse_mean"],
                "mape": row["mape_mean"],
                "smape": row["smape_mean"],
            }
        )
    metrics = pd.DataFrame(headline_rows).sort_values(
        ["period", "mae"], ignore_index=True
    )
    write_table(metrics, output_dir / "forecast_metrics.csv")

    print("Generating counterfactual and figures from headline first unit...")
    if headline_first_unit_payload is not None:
        counterfactual_engine = CounterfactualEngine(
            engine, residual_scale=0.05, random_state=42
        )
        counterfactual = counterfactual_engine.intervene_relative(
            headline_first_unit_payload["current_state"],
            INTERVENTION,
            0.10,
            forecast_horizon=headline_first_unit_payload["horizon"],
            n_bootstrap=300,
        ).to_dict()
        write_html_compact(
            create_counterfactual_figure(counterfactual, TARGET),
            FIGURE_DIR / f"counterfactual_multiunit_{dataset_code.lower()}.html",
        )
        write_html_compact(
            create_model_comparison_figure(
                headline_first_unit_payload["shocked_actual"],
                {
                    "Persistence Baseline": headline_first_unit_payload["persistence"],
                    "ARIMAX": headline_first_unit_payload["arimax"],
                    "Causal Engine": headline_first_unit_payload["causal"],
                },
                shock_index=0,
                title=(
                    f"{dataset_code} held-out unit "
                    f"{headline_first_unit_payload['unit']} - {HEADLINE_SCENARIO}"
                ),
            ),
            FIGURE_DIR / f"model_comparison_multiunit_{dataset_code.lower()}.html",
        )

    create_interactive_dag(
        engine.G,
        stability,
        VARIABLES,
        FIGURE_DIR / f"causal_dag_multiunit_{dataset_code.lower()}.html",
        min_confidence=0.0,
    )
    _write_summary(
        output_dir=output_dir,
        dataset_code=dataset_code,
        train_units=train_units,
        test_units=test_units,
        differenced=differenced_variables,
        agreement=agreement,
        temporal_edges=temporal_edges,
        edge_source=edge_source,
        metrics=metrics,
        aggregated=aggregated,
    )
    print("Multi-unit CMAPSS experiment complete.")
    print(f"Artifacts: {output_dir}")
    print(f"Figures:   {FIGURE_DIR}")
    print(metrics.to_string(index=False))
    return metrics


def _apply_scenario(actual_window: pd.DataFrame, scenario: dict) -> pd.DataFrame:
    kwargs = {
        "df": actual_window,
        "variable": scenario["variable"],
        "shock_type": scenario["shock_type"],
        "shock_start_idx": 0,
        "magnitude": scenario["magnitude"],
    }
    if "duration" in scenario:
        kwargs["duration"] = scenario["duration"]
    if "transition_steps" in scenario:
        kwargs["transition_steps"] = scenario["transition_steps"]
    return ShockInjector.inject_dataframe_shock(**kwargs)


def _split_units(units: np.ndarray, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    units = np.array(sorted(int(unit) for unit in units))
    split = int(len(units) * train_fraction)
    return units[:split], units[split:]


def _difference_by_unit(df: pd.DataFrame, differenced_variables: list[str]) -> pd.DataFrame:
    frames = []
    for _, group in df.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        unit_frame = group.copy()
        for variable in differenced_variables:
            unit_frame[variable] = unit_frame[variable].diff()
        frames.append(unit_frame.dropna(subset=differenced_variables))
    return pd.concat(frames, ignore_index=True)


def _with_unit_cycle(meta: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    output = values.reset_index(drop=True).copy()
    output["unit"] = meta["unit"].to_numpy()
    output["cycle"] = meta["cycle"].to_numpy()
    return output


def _lag_by_unit(df: pd.DataFrame, variables: list[str], max_lag: int) -> pd.DataFrame:
    frames = []
    for _, group in df.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        lagged = create_lagged_features(
            group[variables].reset_index(drop=True),
            max_lag=max_lag,
            variables=variables,
        )
        frames.append(lagged.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def _sample_rows(df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)


def parse_lagged_name(name: str) -> tuple[str, int]:
    if "_lag" not in name:
        return name, 0
    base, lag = name.rsplit("_lag", maxsplit=1)
    return base, int(lag)


def temporal_edges_from_confidence_report(report, threshold: float) -> list[tuple[str, str, float]]:
    collapsed: dict[tuple[str, str], float] = {}
    for row in report:
        if row.bootstrap_frequency < threshold:
            continue
        source_base, source_lag = parse_lagged_name(row.cause)
        target_base, target_lag = parse_lagged_name(row.effect)
        if source_base == target_base:
            continue
        if source_lag <= target_lag:
            continue
        key = (source_base, target_base)
        collapsed[key] = max(collapsed.get(key, 0.0), row.bootstrap_frequency)
    return [
        (source, target, confidence)
        for (source, target), confidence in sorted(
            collapsed.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
    ]


def temporal_edges_from_discovery(edges, default_confidence: float) -> list[tuple[str, str, float]]:
    collapsed: dict[tuple[str, str], float] = {}
    for edge in edges:
        if edge.edge_type != "directed":
            continue
        source_base, source_lag = parse_lagged_name(edge.source)
        target_base, target_lag = parse_lagged_name(edge.target)
        if source_base == target_base or source_lag <= target_lag:
            continue
        collapsed[(source_base, target_base)] = default_confidence
    return [(source, target, confidence) for (source, target), confidence in collapsed.items()]


def fast_temporal_bootstrap_stability(
    lagged_df: pd.DataFrame,
    variables: list[str],
    *,
    n_bootstrap: int,
    sample_fraction: float,
    min_abs_correlation: float,
    max_parents_per_target: int,
    random_state: int,
) -> np.ndarray:
    """Estimate fast lagged predictive stability for production graph selection."""

    rng = np.random.default_rng(random_state)
    n_samples = max(10, int(len(lagged_df) * sample_fraction))
    counts = np.zeros((len(variables), len(variables)), dtype=float)

    for _ in range(n_bootstrap):
        sample = lagged_df.sample(
            n=n_samples,
            replace=True,
            random_state=int(rng.integers(0, 1_000_000)),
        )
        for target_idx, target in enumerate(variables):
            scored_sources = []
            for source_idx, source in enumerate(variables):
                if source == target:
                    continue
                source_col = f"{source}_lag1"
                corr = sample[source_col].corr(sample[target])
                if pd.notna(corr) and abs(float(corr)) >= min_abs_correlation:
                    scored_sources.append((abs(float(corr)), source_idx))
            scored_sources.sort(reverse=True)
            for _, source_idx in scored_sources[:max_parents_per_target]:
                counts[source_idx, target_idx] += 1
    return counts / n_bootstrap


def base_stability_to_frame(
    stability: np.ndarray,
    variables: list[str],
    *,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for source_idx, source in enumerate(variables):
        for target_idx, target in enumerate(variables):
            if source == target:
                continue
            frequency = float(stability[source_idx, target_idx])
            if frequency <= 0:
                continue
            rows.append(
                {
                    "cause": source,
                    "effect": target,
                    "bootstrap_frequency": frequency,
                    "confidence": "HIGH"
                    if frequency >= 0.80
                    else "MEDIUM"
                    if frequency >= 0.50
                    else "LOW",
                    "used_in_simulation": frequency >= threshold,
                    "method": "lag1_predictive_correlation",
                }
            )
    frame = pd.DataFrame(
        rows,
        columns=[
            "cause",
            "effect",
            "bootstrap_frequency",
            "confidence",
            "used_in_simulation",
            "method",
        ],
    )
    if frame.empty:
        return frame
    return frame.sort_values(
        ["bootstrap_frequency", "cause", "effect"],
        ascending=[False, True, True],
        ignore_index=True,
    )


def temporal_edges_from_base_stability(
    stability: np.ndarray,
    variables: list[str],
    *,
    threshold: float,
) -> list[tuple[str, str, float]]:
    edges = []
    for source_idx, source in enumerate(variables):
        for target_idx, target in enumerate(variables):
            if source == target:
                continue
            confidence = float(stability[source_idx, target_idx])
            if confidence >= threshold:
                edges.append((source, target, confidence))
    return sorted(edges, key=lambda edge: (-edge[2], edge[0], edge[1]))


def break_cycles_by_confidence(edges: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    import networkx as nx

    graph = nx.DiGraph()
    kept: list[tuple[str, str, float]] = []
    for source, target, confidence in sorted(edges, key=lambda edge: -edge[2]):
        graph.add_edge(source, target)
        if nx.is_directed_acyclic_graph(graph):
            kept.append((source, target, confidence))
        else:
            graph.remove_edge(source, target)
    return kept


def collapse_stability_matrix(
    stability: np.ndarray,
    lagged_names: list[str],
    base_names: list[str],
) -> np.ndarray:
    collapsed = np.zeros((len(base_names), len(base_names)))
    base_index = {name: idx for idx, name in enumerate(base_names)}
    for i, source in enumerate(lagged_names):
        for j, target in enumerate(lagged_names):
            source_base, source_lag = parse_lagged_name(source)
            target_base, target_lag = parse_lagged_name(target)
            if source_base == target_base or source_lag <= target_lag:
                continue
            if source_base in base_index and target_base in base_index:
                collapsed[base_index[source_base], base_index[target_base]] = max(
                    collapsed[base_index[source_base], base_index[target_base]],
                    stability[i, j],
                )
    return collapsed


def _edge_frame(pc_edges, fci_edges) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"algorithm": "PC", "source": e.source, "target": e.target, "type": e.edge_type}
            for e in pc_edges
        ]
        + [
            {"algorithm": "FCI", "source": e.source, "target": e.target, "type": e.edge_type}
            for e in fci_edges
        ]
    )


def _comparison_to_frame(comparison: dict[str, set[tuple[str, str]]]) -> pd.DataFrame:
    rows = []
    for category, edges in comparison.items():
        for source, target in sorted(edges):
            rows.append({"category": category, "source": source, "target": target})
    return pd.DataFrame(rows or [{"category": "none", "source": "", "target": ""}])


def _write_summary(
    *,
    output_dir: Path,
    dataset_code: str,
    train_units: np.ndarray,
    test_units: np.ndarray,
    differenced: list[str],
    agreement: dict[str, float],
    temporal_edges: list[tuple[str, str, float]],
    edge_source: str,
    metrics: pd.DataFrame,
    aggregated: pd.DataFrame,
) -> None:
    summary = f"""# {dataset_code} Multi-unit Experiment Summary

Train units: `{len(train_units)}` engines (`{train_units[0]}` to `{train_units[-1]}`)

Held-out units: `{len(test_units)}` engines (`{test_units[0]}` to `{test_units[-1]}`)

Variables: `{", ".join(VARIABLES)}`

Differenced variables: `{", ".join(differenced) if differenced else "none"}`

Discovery sample rows: `{DISCOVERY_ROWS}`

Bootstrap runs: `{BOOTSTRAP_RUNS}`

Max parents per target: `{MAX_PARENTS_PER_TARGET}`

Temporal edge threshold: `{EDGE_THRESHOLD:.2f}`

PC/FCI agreement rate: `{agreement["agreement_rate"]:.3f}`

Cohen's Kappa: `{agreement["kappa"]:.3f}`

Forecast graph source: `{edge_source}`

Headline scenario: `{HEADLINE_SCENARIO}`

## Temporal Edges Used

{pd.DataFrame(temporal_edges, columns=["cause", "effect", "confidence"]).to_string(index=False)}

## Headline Forecast Metrics (mean over all held-out engines, scenario {HEADLINE_SCENARIO})

{metrics.to_string(index=False)}

## All Scenarios (mean +/- std over held-out engines)

{aggregated.to_string(index=False)}
"""
    (output_dir / "run_summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a CMAPSS multi-unit experiment.")
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default="FD001",
        help="CMAPSS subset to run.",
    )
    args = parser.parse_args()
    main(args.dataset)
