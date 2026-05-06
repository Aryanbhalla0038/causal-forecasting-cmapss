"""Preprocessing utilities for time-series causal discovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class StationarityResult:
    """ADF stationarity test result for one variable."""

    variable: str
    adf_statistic: float
    p_value: float
    stationary: bool
    used_observations: int


def validate_time_series_frame(
    df: pd.DataFrame,
    *,
    require_datetime_index: bool = True,
    allow_missing: bool = False,
) -> dict[str, object]:
    """Validate basic time-series quality requirements.

    Returns a compact summary that can be logged or inserted into the report.
    """

    if df.empty:
        raise ValueError("Dataframe is empty.")

    if require_datetime_index and not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex for temporal ordering.")

    duplicate_timestamps = int(df.index.duplicated().sum())
    if duplicate_timestamps:
        raise ValueError(f"Found {duplicate_timestamps} duplicate timestamps.")

    missing_values = int(df.isna().sum().sum())
    if missing_values and not allow_missing:
        raise ValueError(f"Found {missing_values} missing values.")

    monotonic = bool(df.index.is_monotonic_increasing)
    if not monotonic:
        raise ValueError("Datetime index must be sorted in increasing order.")

    inferred_frequency = None
    if isinstance(df.index, pd.DatetimeIndex):
        inferred_frequency = pd.infer_freq(df.index)

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": missing_values,
        "duplicate_timestamps": duplicate_timestamps,
        "inferred_frequency": inferred_frequency,
    }


def test_stationarity(
    series: pd.Series,
    *,
    variable_name: str | None = None,
    significance: float = 0.05,
) -> StationarityResult:
    """Run the Augmented Dickey-Fuller test for one numeric series."""

    clean = series.dropna()
    name = variable_name or str(series.name)
    if len(clean) < 8:
        raise ValueError(f"Not enough observations for ADF test: {name}")
    if np.isclose(clean.var(), 0.0):
        raise ValueError(f"ADF test is undefined for constant variable: {name}")

    result = adfuller(clean, autolag="AIC")
    adf_statistic = float(result[0])
    p_value = float(result[1])
    return StationarityResult(
        variable=name,
        adf_statistic=adf_statistic,
        p_value=p_value,
        stationary=p_value < significance,
        used_observations=int(result[3]),
    )


def run_stationarity_tests(
    df: pd.DataFrame,
    *,
    significance: float = 0.05,
) -> dict[str, StationarityResult]:
    """Run ADF stationarity tests for all columns in a dataframe."""

    results = {}
    for col in df.columns:
        results[col] = test_stationarity(
            df[col],
            variable_name=col,
            significance=significance,
        )
    return results


def stationarity_results_to_frame(
    results: dict[str, StationarityResult],
) -> pd.DataFrame:
    """Convert stationarity results into a report-ready dataframe."""

    return pd.DataFrame(
        [
            {
                "variable": result.variable,
                "adf_statistic": result.adf_statistic,
                "p_value": result.p_value,
                "stationary": result.stationary,
                "used_observations": result.used_observations,
            }
            for result in results.values()
        ]
    )


def make_stationary(
    df: pd.DataFrame,
    stationarity_results: dict[str, StationarityResult],
) -> tuple[pd.DataFrame, list[str]]:
    """Apply first-order differencing to variables that fail ADF."""

    stationary_df = df.copy()
    differenced_vars = []

    for col, result in stationarity_results.items():
        if col not in stationary_df.columns:
            raise KeyError(f"Stationarity result references missing column: {col}")
        if not result.stationary:
            stationary_df[col] = stationary_df[col].diff()
            differenced_vars.append(col)

    return stationary_df.dropna(), differenced_vars


def normalize_dataframe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standard-scale a dataframe and return the fitted scaler."""

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)
    return scaled_df, scaler


def create_lagged_features(
    df: pd.DataFrame,
    *,
    max_lag: int = 5,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Create lag-embedded features for time-series causal discovery.

    Each variable at t, t-1, ..., t-max_lag becomes a separate node. This lets
    constraint-based algorithms reason over temporal precedence instead of
    mixing contemporaneous and lagged effects.
    """

    if max_lag < 0:
        raise ValueError("max_lag must be non-negative.")

    selected = variables or list(df.columns)
    missing = sorted(set(selected) - set(df.columns))
    if missing:
        raise KeyError(f"Unknown variables for lag construction: {missing}")

    lagged_df = pd.DataFrame(index=df.index[max_lag:])
    for var in selected:
        for lag in range(max_lag + 1):
            col_name = f"{var}_lag{lag}" if lag > 0 else var
            lagged_df[col_name] = df[var].shift(lag).iloc[max_lag:]

    return lagged_df.dropna()

