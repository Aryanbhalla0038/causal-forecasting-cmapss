"""Forecast evaluation metrics and comparison tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass(frozen=True)
class ForecastEvaluation:
    """Metric result for one model and evaluation period."""

    model: str
    period: str
    mae: float
    rmse: float
    mape: float
    smape: float = 0.0


def evaluate_forecast(
    actual: np.ndarray | pd.Series | list[float],
    predicted: np.ndarray | pd.Series | list[float],
    *,
    model_name: str,
    period_name: str,
) -> ForecastEvaluation:
    """Compute MAE, RMSE, MAPE, and sMAPE for a forecast.

    sMAPE (symmetric MAPE) is reported alongside MAPE because MAPE explodes on
    standardized targets that pass through zero. sMAPE bounds the metric to
    [0, 200%] and treats over- and under-prediction symmetrically.
    """

    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("actual and predicted must have the same shape.")
    if y_true.size == 0:
        raise ValueError("actual and predicted must not be empty.")

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denominator = np.where(np.isclose(y_true, 0.0), np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denominator)) * 100)

    symmetric_denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    symmetric_denom = np.where(np.isclose(symmetric_denom, 0.0), np.nan, symmetric_denom)
    smape = float(np.nanmean(np.abs(y_true - y_pred) / symmetric_denom) * 100)

    return ForecastEvaluation(
        model=model_name,
        period=period_name,
        mae=mae,
        rmse=rmse,
        mape=mape,
        smape=smape,
    )


def compare_forecasts(results: list[ForecastEvaluation]) -> pd.DataFrame:
    """Convert metric objects into a sorted comparison dataframe."""

    return pd.DataFrame(
        [
            {
                "model": result.model,
                "period": result.period,
                "mae": result.mae,
                "rmse": result.rmse,
                "mape": result.mape,
                "smape": result.smape,
            }
            for result in results
        ]
    ).sort_values(["period", "mae"], ignore_index=True)


def recovery_time_to_baseline(
    rolling_errors: np.ndarray | pd.Series | list[float],
    baseline_error: float,
    *,
    tolerance: float = 0.10,
) -> int | None:
    """Return first step where error recovers near baseline accuracy."""

    if baseline_error < 0:
        raise ValueError("baseline_error must be non-negative.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    threshold = baseline_error * (1 + tolerance)
    for idx, error in enumerate(np.asarray(rolling_errors, dtype=float)):
        if error <= threshold:
            return idx
    return None

