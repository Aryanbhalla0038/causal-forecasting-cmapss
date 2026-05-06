"""Streamlit dashboard for the causal forecasting demo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.linear_model import LinearRegression

from causal_forecasting.counterfactuals import CounterfactualEngine
from causal_forecasting.forecasting import (
    CausalForecastingEngine,
    ShockInjector,
    compare_forecasts,
    evaluate_forecast,
)
from causal_forecasting.visualization import (
    create_counterfactual_figure,
    create_model_comparison_figure,
)


DATASET_CHOICES = ("FD001", "FD002", "FD003", "FD004")
PROCESSED_ROOT = "data/processed"
REAL_FIGURE_DIR = "reports/figures"


def _detect_graph_source(summary_text: str) -> str:
    for tag in (
        "temporal_predictive_bootstrap",
        "pc_temporal_edges",
        "domain_prior_temporal_fallback",
    ):
        if tag in summary_text:
            return tag
    return "unknown"


@st.cache_resource
def build_demo_system():
    cycles = np.arange(1, 121, dtype=float)
    fuel_flow = 0.8 + cycles * 0.01
    core_speed = 100 + fuel_flow * 18
    hpc_temp = 420 + core_speed * 1.35
    lpt_temp = 260 + hpc_temp * 0.55
    df = pd.DataFrame(
        {
            "fuel_flow": fuel_flow,
            "core_speed": core_speed,
            "hpc_temp": hpc_temp,
            "lpt_temp": lpt_temp,
        }
    )
    edges = [
        ("fuel_flow", "core_speed", 0.92),
        ("core_speed", "hpc_temp", 0.91),
        ("hpc_temp", "lpt_temp", 0.88),
    ]
    engine = CausalForecastingEngine(
        edges,
        list(df.columns),
        model_factory=LinearRegression,
    )
    engine.fit_structural_equations(df.iloc[:90])
    counterfactual_engine = CounterfactualEngine(
        engine,
        residual_scale=0.03,
        random_state=42,
    )
    return df, engine, counterfactual_engine


def main() -> None:
    st.set_page_config(
        page_title="Causal Forecasting Lab",
        page_icon="",
        layout="wide",
    )
    st.title("Causal Forecasting Lab")

    real_tab, demo_tab = st.tabs(["CMAPSS Real Experiments", "Interactive Demo"])
    with real_tab:
        dataset_code = st.selectbox(
            "Dataset",
            DATASET_CHOICES,
            index=0,
            help="Pick the CMAPSS subset whose multi-unit artifacts to display.",
        )
        render_real_experiment(dataset_code)
    with demo_tab:
        render_interactive_demo()


def render_real_experiment(dataset_code: str = "FD001") -> None:
    output_dir = f"{PROCESSED_ROOT}/cmapss_{dataset_code.lower()}_multiunit"
    metrics_path = f"{output_dir}/forecast_metrics.csv"
    edges_path = f"{output_dir}/forecast_edges_used.csv"
    equations_path = f"{output_dir}/structural_equations.csv"
    confidence_path = f"{output_dir}/edge_confidence_temporal.csv"
    summary_path = f"{output_dir}/run_summary.md"

    try:
        metrics = pd.read_csv(metrics_path)
        edges = pd.read_csv(edges_path)
        equations = pd.read_csv(equations_path)
        confidence = pd.read_csv(confidence_path)
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = handle.read()
    except FileNotFoundError:
        st.info(
            f"Run `python scripts/run_full_cmapss_experiment.py --dataset {dataset_code}` "
            f"to generate {dataset_code} artifacts."
        )
        return

    st.subheader("Held-out Shock Forecast")
    cols = st.columns(3)
    best = metrics.sort_values("mae").iloc[0]
    cols[0].metric("Best Model", best["model"])
    cols[1].metric("Best MAE", f"{best['mae']:.4f}")
    cols[2].metric("Best RMSE", f"{best['rmse']:.4f}")

    provenance = _detect_graph_source(summary)
    badge_color = {
        "temporal_predictive_bootstrap": "green",
        "pc_temporal_edges": "orange",
        "domain_prior_temporal_fallback": "red",
    }.get(provenance, "gray")
    st.markdown(
        f"**Forecast graph source:** :{badge_color}[`{provenance}`]"
    )

    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.caption(
        "Headline metrics are means over all held-out engines on the S1 step "
        "shock scenario. MAE and RMSE are primary; sMAPE is reported because "
        "MAPE is unstable on standardized targets that can be close to zero."
    )

    st.subheader("Forecast Graph")
    left, right = st.columns([0.45, 0.55])
    with left:
        st.dataframe(edges, use_container_width=True, hide_index=True)
    with right:
        st.dataframe(equations, use_container_width=True, hide_index=True)

    st.subheader("Edge Stability")
    st.dataframe(confidence.head(20), use_container_width=True, hide_index=True)

    st.subheader("Figures")
    suffix = dataset_code.lower()
    for title, path in [
        ("Causal DAG", f"{REAL_FIGURE_DIR}/causal_dag_multiunit_{suffix}.html"),
        ("Counterfactual Simulation", f"{REAL_FIGURE_DIR}/counterfactual_multiunit_{suffix}.html"),
        ("Model Comparison", f"{REAL_FIGURE_DIR}/model_comparison_multiunit_{suffix}.html"),
    ]:
        st.markdown(f"**{title}**")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                components.html(handle.read(), height=520, scrolling=True)
        except FileNotFoundError:
            st.warning(f"Missing figure: `{path}`")

    scenario_path = f"{output_dir}/forecast_metrics_by_scenario.csv"
    try:
        scenarios = pd.read_csv(scenario_path)
        st.subheader("All Shock Scenarios (mean +/- std over held-out engines)")
        st.dataframe(scenarios, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        pass

    with st.expander("Run Summary"):
        st.markdown(summary)


def render_interactive_demo() -> None:

    df, engine, counterfactual_engine = build_demo_system()
    variable_names = list(df.columns)
    current_state = df.iloc[89]

    left, right = st.columns([0.28, 0.72])
    with left:
        st.subheader("Intervention")
        intervention_var = st.selectbox("Variable", variable_names, index=0)
        magnitude = st.slider("Change", -50, 50, -15, 5, format="%d%%")
        horizon = st.slider("Horizon", 1, 20, 10)
        target_var = st.selectbox("Target", variable_names, index=len(variable_names) - 1)
        st.caption(
            "Charts update automatically when you change a slider. The button "
            "below force-rebuilds with the current widget state."
        )
        if st.button("Re-run", type="primary", use_container_width=True):
            st.session_state["_demo_run_token"] = (
                st.session_state.get("_demo_run_token", 0) + 1
            )

        st.subheader("Structural Equations")
        st.dataframe(
            engine.structural_equation_quality(),
            use_container_width=True,
            hide_index=True,
        )

    result = counterfactual_engine.intervene_relative(
        current_state,
        intervention_var,
        magnitude / 100,
        forecast_horizon=horizon,
        n_bootstrap=300,
    )
    result_payload = result.to_dict()

    with right:
        st.plotly_chart(
            create_counterfactual_figure(result_payload, target_var),
            use_container_width=True,
        )

        shocked = ShockInjector.inject_dataframe_shock(
            df.iloc[90:110].reset_index(drop=True),
            variable=intervention_var,
            shock_type="step",
            shock_start_idx=0,
            magnitude=magnitude / 100,
        )
        actual = shocked[target_var].tolist()
        causal_values = result_payload["counterfactual"][target_var]
        persistence = [float(current_state[target_var])] * len(causal_values)
        st.plotly_chart(
            create_model_comparison_figure(
                actual[: len(causal_values)],
                {
                    "Persistence Baseline": persistence,
                    "Causal Engine": causal_values,
                },
                shock_index=0,
            ),
            use_container_width=True,
        )

        table = compare_forecasts(
            [
                evaluate_forecast(
                    actual[: len(causal_values)],
                    persistence,
                    model_name="Persistence Baseline",
                    period_name="post-shock",
                ),
                evaluate_forecast(
                    actual[: len(causal_values)],
                    causal_values,
                    model_name="Causal Engine",
                    period_name="post-shock",
                ),
            ]
        )
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption(result_payload["disclaimer"])


if __name__ == "__main__":
    main()
