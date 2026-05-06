"""Generate the final consolidated project report.

Reads per-dataset multi-unit artifacts under ``data/processed`` and the
all-datasets summary, then writes a single human-readable Markdown report
plus a consolidated metrics CSV under ``reports/``.

Usage:

    python scripts/generate_final_report.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_full_cmapss_experiment import DATASET_CHOICES


PROCESSED_ROOT = Path("data/processed")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
SUMMARY_CSV = PROCESSED_ROOT / "cmapss_all_datasets_summary.csv"
REPORT_PATH = REPORTS_DIR / "final_report.md"
CONSOLIDATED_METRICS = REPORTS_DIR / "final_metrics.csv"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    sections = []
    for dataset_code in DATASET_CHOICES:
        dataset_dir = PROCESSED_ROOT / f"cmapss_{dataset_code.lower()}_multiunit"
        metrics_path = dataset_dir / "forecast_metrics.csv"
        summary_path = dataset_dir / "run_summary.md"
        edges_path = dataset_dir / "forecast_edges_used.csv"
        confidence_path = dataset_dir / "edge_confidence_temporal.csv"

        if not metrics_path.exists():
            sections.append(
                f"## {dataset_code}\n\n"
                f"Artifacts missing at `{dataset_dir}`. Run "
                f"`python scripts/run_full_cmapss_experiment.py --dataset {dataset_code}` "
                f"to generate them.\n"
            )
            continue

        metrics = pd.read_csv(metrics_path)
        for record in metrics.to_dict(orient="records"):
            rows.append({"dataset": dataset_code, **record})

        edges_text = (
            pd.read_csv(edges_path).to_string(index=False)
            if edges_path.exists()
            else "(missing)"
        )
        confidence_text = (
            pd.read_csv(confidence_path).head(10).to_string(index=False)
            if confidence_path.exists()
            else "(missing)"
        )
        summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""

        scenario_path = dataset_dir / "forecast_metrics_by_scenario.csv"
        scenario_text = (
            pd.read_csv(scenario_path).to_string(index=False)
            if scenario_path.exists()
            else "(missing)"
        )

        best = metrics.sort_values("mae").iloc[0]
        sections.append(
            f"## {dataset_code}\n\n"
            f"- Best model on headline scenario: **{best['model']}**\n"
            f"- Best MAE: `{best['mae']:.4f}` | Best RMSE: `{best['rmse']:.4f}` | "
            f"sMAPE: `{best['smape']:.2f}%`\n\n"
            f"### Headline Forecast Metrics (mean over all held-out engines, S1)\n\n"
            f"```\n{metrics.to_string(index=False)}\n```\n\n"
            f"### All Shock Scenarios\n\n"
            f"```\n{scenario_text}\n```\n\n"
            f"### Forecast Graph Used\n\n"
            f"```\n{edges_text}\n```\n\n"
            f"### Top Edge Stability\n\n"
            f"```\n{confidence_text}\n```\n\n"
            f"### Run Summary\n\n"
            f"{summary_text}\n"
        )

    consolidated = pd.DataFrame(rows)
    if not consolidated.empty:
        consolidated.to_csv(CONSOLIDATED_METRICS, index=False)

    if SUMMARY_CSV.exists():
        cross_table = pd.read_csv(SUMMARY_CSV).to_string(index=False)
    elif not consolidated.empty:
        cross_table = consolidated.to_string(index=False)
    else:
        cross_table = "(no metrics available)"

    figure_lines = []
    if FIGURES_DIR.exists():
        for figure in sorted(FIGURES_DIR.glob("*.html")):
            figure_lines.append(f"- `{figure.as_posix()}`")
    figure_block = "\n".join(figure_lines) if figure_lines else "(no figures found)"

    report = (
        "# Final Report: Causal Discovery and Counterfactual Simulation in "
        "Multivariate Forecasting\n\n"
        "IIT Industrial Training Final Project - AIML.\n\n"
        "This report consolidates the multi-unit CMAPSS experiments produced by "
        "`scripts/run_full_cmapss_experiment.py` and "
        "`scripts/run_all_cmapss_experiments.py`.\n\n"
        "## Cross-Dataset Comparison\n\n"
        f"```\n{cross_table}\n```\n\n"
        "Headline metrics: MAE and RMSE on the held-out post-shock window. "
        "MAPE is reported but is unstable on standardized targets that pass "
        "through zero.\n\n"
        "## Shock Scenarios\n\n"
        "See `docs/SHOCK_SCENARIOS.md` for the full catalogue and the "
        "domain reasoning behind each scenario. The headline results above "
        "use scenario S1 (step shock on `s11`).\n\n"
        "## Per-Dataset Results\n\n"
        + "\n".join(sections)
        + "\n## Generated Figures\n\n"
        f"{figure_block}\n\n"
        "## Reproduction\n\n"
        "```\n"
        "python scripts/run_all_cmapss_experiments.py\n"
        "python scripts/generate_final_report.py\n"
        "streamlit run apps/streamlit_dashboard.py\n"
        "```\n"
    )

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    if not consolidated.empty:
        print(f"Wrote {CONSOLIDATED_METRICS}")


if __name__ == "__main__":
    main()
