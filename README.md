# Causal Discovery and Counterfactual Simulation in Multivariate Forecasting



This repository implements Architecture 2 in staged parts. The first stage
contains the scientifically defensible data foundation:

- NASA CMAPSS-style data ingestion
- time-series quality checks
- Augmented Dickey-Fuller stationarity testing
- first-order differencing for non-stationary variables
- standard scaling with reusable scaler metadata
- lag embedding for time-series causal discovery

Later stages will add PC/FCI causal discovery, DAG validation, bootstrap edge
stability, causal forecasting, shock injection, counterfactual simulation, and
the Streamlit visualization dashboard.

## Project Layout

```text
src/causal_forecasting/
  config.py              Project configuration dataclasses
  data/
    ingestion.py         Dataset loaders
    preprocessing.py     Quality checks, stationarity, lag features
  causal_discovery/
    discovery.py         PC and FCI wrappers
    graph_utils.py       Edge extraction and PC-vs-FCI comparison
    validation.py        Bootstrap stability and domain validation tables
  forecasting/
    causal_engine.py     DAG-guided structural-equation forecasting
    baselines.py         ARIMAX and LSTM baseline definitions
    shock.py             Controlled shock injection
    evaluation.py        MAE, RMSE, MAPE, recovery metrics
  counterfactuals/
    engine.py            do-operator interventions and uncertainty intervals
  visualization/
    dag.py               Interactive causal DAG HTML export
    plots.py             Counterfactual and model comparison charts
    reporting.py         Report table artifact writers
  api/
    counterfactual_api.py Flask counterfactual endpoint factory
apps/
  streamlit_dashboard.py Demo dashboard
tests/
  test_preprocessing.py  Unit tests for the data foundation
```

## Recommended Dataset

Use the NASA CMAPSS turbofan degradation dataset. Put raw files under:

```text
data/raw/
```

For example:

```text
data/raw/train_FD001.txt
```

## Part 1 Example

```python
from pathlib import Path

from causal_forecasting.data.ingestion import load_cmapss_train
from causal_forecasting.data.preprocessing import (
    create_lagged_features,
    make_stationary,
    normalize_dataframe,
    run_stationarity_tests,
    validate_time_series_frame,
)

df = load_cmapss_train(Path("data/raw/train_FD001.txt"))
validate_time_series_frame(df)

sensor_cols = [col for col in df.columns if col.startswith("s")]
stationarity = run_stationarity_tests(df[sensor_cols])
stationary_df, differenced = make_stationary(df[sensor_cols], stationarity)
scaled_df, scaler = normalize_dataframe(stationary_df)
lagged_df = create_lagged_features(scaled_df, max_lag=5)
```

## Part 2 Example

```python
from causal_forecasting.causal_discovery import (
    compare_pc_fci,
    compute_algorithm_agreement,
    run_fci,
    run_pc,
)

pc_result = run_pc(lagged_df)
fci_result = run_fci(lagged_df)

comparison = compare_pc_fci(pc_result.edges, fci_result.edges)
agreement = compute_algorithm_agreement(
    pc_result.graph_matrix,
    fci_result.graph_matrix,
)
```

Interpretation rules:

- PC and FCI agree on direction: high-confidence causal claim
- PC directed edge but FCI bidirected edge: suspected hidden confounding
- PC-only or FCI-only directed edge: investigate with bootstrap stability

## Scientific Reporting Notes

Part 1 produces the artifacts needed for the report's first methodological
tables:

- data quality summary
- ADF statistic and p-value per variable
- list of differenced variables
- lag embedding configuration

Do not run causal discovery on raw non-stationary time series.

Part 2 produces the PC-vs-FCI comparison table and Cohen's Kappa agreement
metric required for the dual-algorithm strategy.

## Part 3 Example

```python
from causal_forecasting.causal_discovery import (
    bootstrap_edge_stability,
    build_domain_validation_records,
    compute_edge_confidence_report,
    domain_validation_to_frame,
    edge_confidence_report_to_frame,
    reliable_edges_from_report,
)

stability = bootstrap_edge_stability(
    lagged_df,
    n_bootstrap=100,
    sample_fraction=0.80,
)
confidence_report = compute_edge_confidence_report(
    stability,
    variable_names=list(lagged_df.columns),
    threshold=0.80,
)
reliable_edges = reliable_edges_from_report(confidence_report)
confidence_table = edge_confidence_report_to_frame(confidence_report)

domain_records = build_domain_validation_records(
    pc_result.edges,
    fci_result.edges,
    priors={
        ("core_speed", "hpc_temp"): "CONFIRMED",
        ("fuel_flow", "lpt_temp"): "PLAUSIBLE",
    },
)
domain_table = domain_validation_to_frame(domain_records)
```

The stability matrix convention is:

```text
stability[source_index, target_index] = bootstrap frequency for source -> target
```

Only edges with `used_in_simulation = True` should move forward into
forecasting and counterfactual simulation.

## Part 4 Example

```python
from causal_forecasting.forecasting import (
    CausalForecastingEngine,
    ShockInjector,
    compare_forecasts,
    evaluate_forecast,
    fit_arimax_baseline,
)

engine = CausalForecastingEngine(
    reliable_edges,
    variable_names=list(train_df.columns),
    confidence_threshold=0.80,
)
engine.fit_structural_equations(train_df)

current_state = train_df.iloc[-1]
causal_forecast = engine.forecast(current_state, horizon=10)
equation_quality = engine.structural_equation_quality()

shocked_test = ShockInjector.inject_dataframe_shock(
    test_df,
    variable="fuel_flow",
    shock_type="step",
    shock_start_idx=0,
    magnitude=-0.15,
)

arimax = fit_arimax_baseline(
    train_df,
    target_var="hpc_temp",
    exog_vars=["fuel_flow", "core_speed"],
)
baseline_prediction = arimax.forecast(
    steps=len(shocked_test),
    exog=shocked_test[["fuel_flow", "core_speed"]],
)

results = compare_forecasts([
    evaluate_forecast(
        shocked_test["hpc_temp"],
        baseline_prediction,
        model_name="ARIMAX",
        period_name="post-shock",
    ),
    evaluate_forecast(
        shocked_test["hpc_temp"].iloc[:10],
        causal_forecast["hpc_temp"],
        model_name="Causal Engine",
        period_name="post-shock",
    ),
])
```

Part 4 produces the forecasting comparison table used for pre-shock and
post-shock experiments.

## Part 5 Example

```python
from causal_forecasting.counterfactuals import CounterfactualEngine

counterfactual_engine = CounterfactualEngine(
    engine,
    residual_scale=0.05,
    random_state=42,
)

result = counterfactual_engine.intervene_relative(
    current_state=train_df.iloc[-1],
    intervention_var="fuel_flow",
    magnitude=-0.15,
    forecast_horizon=10,
    n_bootstrap=500,
)

payload = result.to_dict()
```

The counterfactual engine implements graph mutilation by removing incoming
edges to the intervention variable, setting the intervened value, then
propagating through the fitted structural equations.

Report wording: counterfactual outputs are model-consistent estimates under
stated assumptions, not proof of what must happen in the real system.

## Part 6 Example

Run the dashboard:

```powershell
streamlit run apps/streamlit_dashboard.py
```

Create DAG and report artifacts:

```python
from causal_forecasting.visualization import (
    create_interactive_dag,
    create_model_comparison_figure,
    write_table,
)

create_interactive_dag(
    engine.G,
    stability,
    variable_names=list(train_df.columns),
    output_path="reports/figures/causal_dag.html",
)

write_table(results, "reports/forecast_metrics.csv")
```

Serve counterfactual queries:

```python
from causal_forecasting.api import create_counterfactual_app

app = create_counterfactual_app(
    counterfactual_engine,
    current_state_provider=lambda: train_df.iloc[-1].to_dict(),
)
app.run(port=5000)
```

## Real CMAPSS Run

After placing the NASA CMAPSS files in `data/raw/`, generate real-data
artifacts with:

```powershell
& 'C:\Users\HP\AppData\Local\Python\bin\python.exe' scripts/run_cmapss_pipeline.py
```

Outputs are written to:

- `data/processed/cmapss_fd001_unit1/`
- `reports/figures/`

Read `data/processed/cmapss_fd001_unit1/run_summary.md` before presenting the
results. If the script reports `domain_prior_fallback`, the demo graph is a
documented fallback for forecasting/counterfactual demonstration, not a
bootstrap-confirmed discovered causal graph.

For the stronger real experiment, run:

```powershell
& 'C:\Users\HP\AppData\Local\Python\bin\python.exe' scripts/run_full_cmapss_experiment.py --dataset FD001
```

Run all four CMAPSS subsets:

```powershell
& 'C:\Users\HP\AppData\Local\Python\bin\python.exe' scripts/run_all_cmapss_experiments.py
```

This uses all FD001 training engines, splits by engine unit, compares against
ARIMAX, and writes multi-unit artifacts to:

```text
data/processed/cmapss_fd001_multiunit/
reports/figures/
```

The Streamlit dashboard reads these real artifacts by default.
