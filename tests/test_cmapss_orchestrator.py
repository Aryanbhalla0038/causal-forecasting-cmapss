import pandas as pd

from scripts import run_all_cmapss_experiments


def test_run_all_orchestrator_collects_dataset_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(run_all_cmapss_experiments, "DATASET_CHOICES", ("FD001", "FD002"))
    monkeypatch.setattr(
        run_all_cmapss_experiments,
        "OUTPUT_PATH",
        tmp_path / "summary.csv",
    )

    def fake_run_one(dataset_code):
        return pd.DataFrame(
            [
                {
                    "model": "Causal Engine",
                    "period": "post-shock",
                    "mae": 1.0,
                    "rmse": 2.0,
                    "mape": 3.0,
                }
            ]
        )

    monkeypatch.setattr(run_all_cmapss_experiments, "run_one", fake_run_one)

    summary = run_all_cmapss_experiments.main()

    assert summary["dataset"].tolist() == ["FD001", "FD002"]
    assert (tmp_path / "summary.csv").exists()
