# Presentation and Demo Script

A 10-minute walk-through for the IIT industrial training final review of the
**Causal Discovery and Counterfactual Simulation in Multivariate Forecasting**
project.

The goal of the demo is to show, in this exact order, that the project is:

1. scientifically grounded (real CMAPSS data, real preprocessing),
2. methodologically defensible (PC + FCI + bootstrap stability),
3. quantitatively useful (ARIMAX baseline comparison),
4. interpretable (interactive DAG, structural equations, counterfactuals),
5. operational (reproducible scripts, dashboard, final report).

## 0. Pre-demo Checklist

Run once, before the panel walks in:

```powershell
& 'C:\Users\HP\AppData\Local\Python\bin\python.exe' scripts/run_all_cmapss_experiments.py
& 'C:\Users\HP\AppData\Local\Python\bin\python.exe' scripts/generate_final_report.py
streamlit run apps/streamlit_dashboard.py
```

Confirm the following exist before going on stage:

- `data/processed/cmapss_fd001_multiunit/run_summary.md`
- `data/processed/cmapss_all_datasets_summary.csv`
- `reports/final_report.md`
- `reports/figures/causal_dag_multiunit_fd001.html`
- Streamlit running at `http://localhost:8501`

Take screenshots of:

- the FD001 tab in the dashboard (DAG, metrics table, counterfactual chart),
- `reports/final_report.md` rendered, and
- the cross-dataset MAE table.

## 1. Opening (1 minute)

> "This project answers a single question for a turbofan engine: if a
> specific operational variable is forced to change, what happens to the
> downstream sensor we care about? We answer it with a causal DAG, not a
> black-box forecaster, so every prediction comes with an interpretable
> structural equation."

Show: project README, then `docs/IMPLEMENTATION_STAGES.md` to demonstrate
the 6-part discipline.

## 2. Data Foundation (1 minute)

Open `data/processed/cmapss_fd001_multiunit/stationarity.csv`.

> "Real CMAPSS FD001. ADF stationarity tests, per-unit first-order
> differencing where required, standard scaling, and lagged feature
> construction that never crosses unit boundaries."

## 3. Causal Discovery (2 minutes)

Open `discovered_edges.csv` and `pc_fci_comparison.csv`, then
`algorithm_agreement.csv`.

> "We run PC and FCI on the same lagged sample and report Cohen's Kappa
> between them. PC and FCI are diagnostic. The production graph is built
> separately by a temporally-constrained lagged predictive bootstrap
> reported in `edge_confidence_temporal.csv`, with cycles broken in
> descending confidence order."

Open the interactive DAG in the dashboard "CMAPSS Real Experiments" tab.

## 4. Forecasting Comparison (2 minutes)

Open `forecast_metrics.csv` (or the dashboard's metrics table).

> "Held-out engines, never seen during DAG discovery or structural
> equation fitting. Step shock applied to `s11` to force an
> out-of-distribution event. Causal Engine, Persistence, and ARIMAX are
> evaluated on identical post-shock windows."

Use the dataset selector to switch between FD001, FD002, FD003, FD004 and
show that the ranking is consistent.

## 5. Shock Scenarios (1 minute)

Open `docs/SHOCK_SCENARIOS.md`.

> "Five shock scenarios with explicit domain reasoning: step, gradual, and
> pulse on the intervention variable, plus a downstream and an upstream
> shock. The headline numbers come from scenario S1, step on `s11`."

## 6. Counterfactual Engine (2 minutes)

Switch to the Interactive Demo tab.

- Pick `fuel_flow` as the intervention.
- Move the magnitude slider to `-15%`.
- Set horizon `10`.
- Show the counterfactual figure with bootstrap confidence band.
- Show the structural equation quality table on the left.

> "Graph mutilation removes incoming edges to the intervention variable.
> Bootstrapping over residuals gives a confidence interval. This is a
> model-consistent counterfactual under stated assumptions, not a real-world
> guarantee."

## 7. Final Report and Reproducibility (1 minute)

Open `reports/final_report.md`.

> "One command regenerates this entire report. All artifacts are versioned
> under `data/processed/` and `reports/`. The dashboard is the same
> artifacts plus interactive widgets. Tests live under `tests/`."

## 8. Honest Limitations (30 seconds)

> "MAPE is unstable on standardized targets, so MAE/RMSE are headline. The
> CMAPSS dataset has no real interventions, so all shocks are synthetic.
> When bootstrap stability is too weak, the pipeline falls back to
> documented PC-derived edges or a domain prior, and the
> `forecast_graph_source` field in `run_summary.md` reports which path was
> used."

## 9. Close (30 seconds)

> "End-to-end: real data, dual-algorithm causal discovery, bootstrap-
> validated DAG, structural-equation forecaster that beats persistence and
> ARIMAX under shocks, counterfactual engine with confidence bands, and a
> reproducible Streamlit dashboard. Thank you."

## Backup Q & A

- **Why not just use an LSTM?**
  We do, as an optional baseline, but LSTMs do not yield interpretable
  structural equations or support `do(X = x)` counterfactuals.
- **Why differencing per unit?**
  Each engine is its own life-cycle; pooling raw cycles violates
  stationarity assumptions for ADF and downstream regressions.
- **Why is MAPE high?**
  Targets are scaled and pass through zero, so MAPE inflates. MAE/RMSE are
  scale-consistent across models.
- **What if PC and FCI disagree?**
  Reported in `pc_fci_comparison.csv`. Disagreement on directionality is a
  signal for hidden confounding and is investigated with the bootstrap.
