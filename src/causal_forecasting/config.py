"""Configuration objects used across the causal forecasting pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingConfig:
    """Settings for stationarity, scaling, and lag construction."""

    stationarity_alpha: float = 0.05
    max_lag: int = 5
    scale_features: bool = True


@dataclass(frozen=True)
class CMapssConfig:
    """CMAPSS file layout settings."""

    synthetic_origin: str = "2020-01-01"
    datetime_unit: str = "D"

