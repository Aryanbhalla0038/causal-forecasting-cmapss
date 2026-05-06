# Implementation Stages

This project will be built in parts so each scientific layer remains testable.

## Part 1 - Data Foundation

Status: implemented

- CMAPSS ingestion
- datetime indexing
- data quality validation
- ADF stationarity testing
- first-order differencing
- normalization
- lag feature construction
- preprocessing unit tests

## Part 2 - Causal Discovery

Status: implemented

- PC algorithm wrapper
- FCI algorithm wrapper
- edge extraction utilities
- PC vs FCI comparison
- Cohen's Kappa agreement metric

## Part 3 - DAG Validation

Status: implemented

- bootstrap edge stability
- confidence labels
- domain validation table structure
- report-ready causal graph summaries

## Part 4 - Forecasting and Shock Evaluation

Status: implemented

- causal structural equation forecasting
- ARIMAX baseline
- LSTM baseline
- shock injection module
- pre-shock and post-shock metric tables

## Part 5 - Counterfactual Engine

Status: implemented

- do-operator graph mutilation
- factual vs counterfactual forecast comparison
- bootstrap confidence intervals
- API endpoint

## Part 6 - Dashboard and Final Reporting

Status: implemented

- interactive DAG visualization
- Streamlit counterfactual sliders
- model comparison panel
- experiment tables and figures

## Remaining Integration Work

- NASA CMAPSS data is present under `data/raw/`
- Single-unit FD001 pipeline is available at `scripts/run_cmapss_pipeline.py`
- Multi-unit FD001 real experiment is available at `scripts/run_full_cmapss_experiment.py`
- Final report tables are generated under `data/processed/cmapss_fd001_multiunit/`
- Final figures are generated under `reports/figures/`
- Tune shock scenarios with domain justification
- Prepare presentation screenshots and demo script

## Current Best Experiment

Single-dataset command:

```powershell
& 'C:\Users\HP\AppData\Local\Python\bin\python.exe' scripts/run_full_cmapss_experiment.py --dataset FD001
```

All-datasets command:

```powershell
& 'C:\Users\HP\AppData\Local\Python\bin\python.exe' scripts/run_all_cmapss_experiments.py
```

Current FD001 output (mean over all 30 held-out engines, scenario S1):

- Train units: 70 engines
- Held-out units: 30 engines
- Baselines: Persistence and ARIMAX
- Forecast graph source: temporal predictive bootstrap
- ARIMAX MAE: 0.378 (sees shocked exogenous variables)
- Causal Engine MAE: 0.431 (forecasts from unshocked t=0)
- Persistence MAE: 0.568

Dashboard:

```text
http://localhost:8501
```

Scientific note: the multi-unit production graph uses temporally constrained
lagged predictive bootstrap stability. PC and FCI remain diagnostic comparison
algorithms, reported separately through `algorithm_agreement.csv`,
`discovered_edges.csv`, and `pc_fci_comparison.csv`.

## Perfection Phase Progress

1. Multi-unit FD001 real experiment: complete
2. Dataset-parameterized FD001-FD004 runner: complete
3. All-datasets orchestrator: complete (`data/processed/cmapss_all_datasets_summary.csv`)
4. Stronger shock scenario documentation: complete (`docs/SHOCK_SCENARIOS.md`)
5. Final report generator: complete (`scripts/generate_final_report.py`, output `reports/final_report.md` and `reports/final_metrics.csv`)
6. Dashboard dataset selector: complete (FD001-FD004 selector in `apps/streamlit_dashboard.py`)
7. Presentation/demo script: complete (`docs/DEMO_SCRIPT.md`)

## Cross-Dataset Headline (multi-engine evaluation, scenario S1)

Means over **all held-out engines** (not just one) on the S1 step shock,
with sMAPE replacing MAPE as the scaled error metric:

- FD001: ARIMAX `0.378` < Causal Engine `0.431` < Persistence `0.568` (Causal beats Persistence by ~24%)
- FD002: ARIMAX `0.083` < Causal `1.126` ≈ Persistence `1.130` (multi-regime weakness)
- FD003: **Causal Engine `0.375`** < Persistence `0.435` < ARIMAX `0.568` (Causal wins outright)
- FD004: ARIMAX `0.080` < Causal `1.063` ≈ Persistence `1.070` (multi-regime weakness)

Interpretation: with the new across-all-engines evaluation the Causal
Engine still beats Persistence on FD001 and FD003, the two single-regime
subsets where the bootstrap-discovered DAG has real structure. ARIMAX is
strongest on FD002/FD004 because it consumes the shocked exogenous
variables directly; the causal engine and persistence both forecast from
the unshocked initial state. This is documented as a limitation in
`reports/final_report.md`.

## Improvements Applied (Round 2)

- **Multi-engine evaluation**: every held-out engine is scored, not one anecdote.
- **All five shock scenarios run**: see `forecast_metrics_by_scenario.csv` per dataset and the per-unit drilldown in `forecast_metrics_per_unit.csv`.
- **sMAPE alongside MAPE**: stable percentage error for standardized targets.
- **Plotly figures via CDN**: each HTML dropped from ~4.8 MB to ~10 KB (`include_plotlyjs="cdn"` in `write_html_compact`).
- **Dashboard provenance badge**: shows whether the active forecast graph came from `temporal_predictive_bootstrap`, `pc_temporal_edges`, or `domain_prior_temporal_fallback`.
- **Dashboard scenario table**: `forecast_metrics_by_scenario.csv` is rendered next to the headline metrics.
- **Bug fix**: `base_stability_to_frame` no longer crashes on empty bootstrap results (FD002 case).
