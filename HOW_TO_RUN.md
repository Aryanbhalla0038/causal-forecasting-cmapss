# How to Run This Project

This guide is the single source of truth for running the
**Causal Discovery and Counterfactual Simulation in Multivariate
Forecasting** project end-to-end on Windows.

All commands assume the repository root `c:\Users\HP\Desktop\arch2`
as the working directory and use the local Python at
`C:\Users\HP\AppData\Local\Python\bin\python.exe`. Replace that path
with `python` if your PATH already points to a 3.11+ interpreter.

---

## 0. Quickstart (Single Command)

From the repository root, run:

```powershell
.\run_all.ps1
```

This installs the package, runs FD001-FD004, builds the final report,
runs the tests, and launches the Streamlit dashboard.

Useful flags:

```powershell
.\run_all.ps1 -Clean             # wipe prior artifacts first
.\run_all.ps1 -SkipDashboard     # produce artifacts only, no UI
.\run_all.ps1 -SkipTests         # skip pytest
.\run_all.ps1 -Python python     # use a different interpreter
```

If PowerShell blocks the script the first time, allow it for this
session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The remaining sections describe each step manually if you prefer to
run them one at a time.

---

## 1. One-Time Setup

Install the project and its dependencies in editable mode:

```powershell
& 'C:\Users\HP\AppData\Local\Python\bin\python.exe' -m pip install -e .
```

The CMAPSS data is already shipped under `data/raw/`:

- `train_FD001.txt` ... `train_FD004.txt`
- `test_FD001.txt`  ... `test_FD004.txt`
- `RUL_FD001.txt`   ... `RUL_FD004.txt`
- `readme.txt`, `Damage Propagation Modeling.pdf`

No manual download is required.

---

## 2. Run the Full Pipeline (Recommended)

Three commands produce every artifact, build the consolidated report,
and open the live dashboard:

```powershell
& 'C:\Users\HP\AppData\Local\Python\bin\python.exe' -m scripts.run_all_cmapss_experiments
& 'C:\Users\HP\AppData\Local\Python\bin\python.exe' -m scripts.generate_final_report
streamlit run apps/streamlit_dashboard.py
```

- Step 1 takes about 3-5 minutes. It runs FD001-FD004, evaluates every
  held-out engine, and applies all five shock scenarios from
  `docs/SHOCK_SCENARIOS.md`.
- Step 2 is instant. It writes `reports/final_report.md` and
  `reports/final_metrics.csv`.
- Step 3 opens the dashboard at `http://localhost:8501`.

---

## 3. Run a Single Dataset

```powershell
& 'C:\Users\HP\AppData\Local\Python\bin\python.exe' -m scripts.run_full_cmapss_experiment --dataset FD001
```

Replace `FD001` with `FD002`, `FD003`, or `FD004`.

---

## 4. Run the Smaller Single-Unit Smoke Pipeline

For a quick sanity check on FD001 unit 1 only:

```powershell
& 'C:\Users\HP\AppData\Local\Python\bin\python.exe' -m scripts.run_cmapss_pipeline
```

---

## 5. Run the Tests

```powershell
& 'C:\Users\HP\AppData\Local\Python\bin\python.exe' -m pytest tests -q
```

Expected: **39 passed**.

---

## 6. Where the Outputs Land

### Per-dataset CSV artifacts

`data/processed/cmapss_<fd>_multiunit/`

| File | Purpose |
|------|---------|
| `forecast_metrics.csv` | headline mean over held-out engines, scenario S1 |
| `forecast_metrics_by_scenario.csv` | mean +/- std for all 5 shock scenarios |
| `forecast_metrics_per_unit.csv` | per-engine drilldown |
| `edge_confidence_temporal.csv` | bootstrap edge stability table |
| `forecast_edges_used.csv` | DAG actually used by the engine |
| `structural_equations.csv` | fitted equation R^2 per node |
| `algorithm_agreement.csv` | PC vs FCI Cohen's Kappa |
| `discovered_edges.csv` | PC and FCI raw edges |
| `pc_fci_comparison.csv` | edge-level PC/FCI comparison |
| `stationarity.csv` | ADF results per variable |
| `run_summary.md` | narrative summary for that dataset |

### Cross-dataset summary

`data/processed/cmapss_all_datasets_summary.csv`

### Consolidated report

`reports/final_report.md`
`reports/final_metrics.csv`

### Interactive figures

`reports/figures/*.html`  (~10 KB each, open in any browser)

- `causal_dag_multiunit_<fd>.html`
- `counterfactual_multiunit_<fd>.html`
- `model_comparison_multiunit_<fd>.html`

---

## 7. Using the Dashboard

### CMAPSS Real Experiments tab

1. Pick a dataset from the selector: `FD001`, `FD002`, `FD003`, `FD004`.
2. Read the headline metrics (Best Model, Best MAE, Best RMSE).
3. Check the **provenance badge**:
   - green `temporal_predictive_bootstrap` = bootstrap-confirmed graph
   - orange `pc_temporal_edges` = PC-derived fallback graph
   - red `domain_prior_temporal_fallback` = domain prior fallback
4. Inspect the discovered DAG, the structural equations, and edge stability.
5. Scroll to the All Shock Scenarios table for the full S1-S5 breakdown.
6. Expand `Run Summary` for the markdown summary.

### Interactive Demo tab

1. Pick an intervention variable.
2. Move the magnitude slider (-50% to +50%).
3. Set the forecast horizon.
4. Click **Re-run**.
5. Compare the factual vs counterfactual chart and the baseline comparison.

---

## 8. Reproducibility Recipe

To regenerate everything from scratch:

```powershell
Remove-Item data\processed\cmapss_*_multiunit -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item reports\figures\*multiunit* -Force -ErrorAction SilentlyContinue
& 'C:\Users\HP\AppData\Local\Python\bin\python.exe' -m scripts.run_all_cmapss_experiments
& 'C:\Users\HP\AppData\Local\Python\bin\python.exe' -m scripts.generate_final_report
& 'C:\Users\HP\AppData\Local\Python\bin\python.exe' -m pytest tests -q
streamlit run apps/streamlit_dashboard.py
```

---

## 9. Troubleshooting

- **`ModuleNotFoundError: causal_forecasting`**
  Install the package with `pip install -e .` (Step 1).
- **`streamlit: command not found`**
  Install streamlit into the same interpreter you used for Step 1, or run
  `python -m streamlit run apps/streamlit_dashboard.py`.
- **`KeyError: 'unit'` when loading CMAPSS**
  The training file in `data/raw/` must be the original NASA whitespace-
  separated `.txt`, not a re-saved CSV.
- **Plotly figures won't render in the dashboard**
  Each HTML uses CDN-loaded `plotly.js`; the browser needs internet access
  the first time it loads a figure.
- **First run is slow**
  Most of the time is per-unit ARIMAX forecasting and the bootstrap loop.
  No GPU is required.

---

## 10. Related Documents

- `README.md` - project overview and code examples per stage.
- `docs/IMPLEMENTATION_STAGES.md` - status of every implementation part.
- `docs/SHOCK_SCENARIOS.md` - five shock scenarios with domain reasoning.
- `docs/DEMO_SCRIPT.md` - 10-minute presentation walkthrough.
- `reports/final_report.md` - generated cross-dataset report.
