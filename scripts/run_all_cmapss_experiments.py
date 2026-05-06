"""Run FD001-FD004 experiments and build one comparison table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_full_cmapss_experiment import DATASET_CHOICES, main as run_one


OUTPUT_PATH = Path("data/processed/cmapss_all_datasets_summary.csv")


def main() -> pd.DataFrame:
    rows = []
    for dataset_code in DATASET_CHOICES:
        metrics = run_one(dataset_code)
        for row in metrics.to_dict(orient="records"):
            rows.append({"dataset": dataset_code, **row})

    summary = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
