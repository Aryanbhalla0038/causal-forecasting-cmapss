"""Controlled shock injection for out-of-distribution evaluation."""

from __future__ import annotations

import pandas as pd


class ShockInjector:
    """Inject controlled disturbances into time-series variables."""

    @staticmethod
    def step_shock(
        series: pd.Series,
        shock_start_idx: int,
        shock_magnitude: float,
        shock_duration: int | None = None,
    ) -> pd.Series:
        """Apply a temporary or permanent step change."""

        ShockInjector._validate_start(series, shock_start_idx)
        if shock_duration is not None and shock_duration <= 0:
            raise ValueError("shock_duration must be positive when provided.")

        shocked = series.copy()
        end_idx = (
            min(len(series), shock_start_idx + shock_duration)
            if shock_duration is not None
            else len(series)
        )
        shocked.iloc[shock_start_idx:end_idx] *= 1 + shock_magnitude
        return shocked

    @staticmethod
    def gradual_shock(
        series: pd.Series,
        shock_start_idx: int,
        final_magnitude: float,
        transition_steps: int = 10,
    ) -> pd.Series:
        """Apply a gradual drift until a final magnitude is reached."""

        ShockInjector._validate_start(series, shock_start_idx)
        if transition_steps <= 0:
            raise ValueError("transition_steps must be positive.")

        shocked = series.copy()
        for offset in range(transition_steps):
            idx = shock_start_idx + offset
            if idx >= len(series):
                break
            scale = 1 + final_magnitude * ((offset + 1) / transition_steps)
            shocked.iloc[idx] *= scale
        tail_start = shock_start_idx + transition_steps
        if tail_start < len(series):
            shocked.iloc[tail_start:] *= 1 + final_magnitude
        return shocked

    @staticmethod
    def pulse_shock(
        series: pd.Series,
        shock_start_idx: int,
        magnitude: float,
        duration: int = 3,
    ) -> pd.Series:
        """Apply a short spike that returns to the original trajectory."""

        return ShockInjector.step_shock(
            series,
            shock_start_idx=shock_start_idx,
            shock_magnitude=magnitude,
            shock_duration=duration,
        )

    @staticmethod
    def inject_dataframe_shock(
        df: pd.DataFrame,
        variable: str,
        shock_type: str,
        shock_start_idx: int,
        magnitude: float,
        *,
        duration: int | None = None,
        transition_steps: int = 10,
    ) -> pd.DataFrame:
        """Return a copy of a dataframe with one shocked variable."""

        if variable not in df.columns:
            raise KeyError(f"Unknown shock variable: {variable}")

        shocked = df.copy()
        if shock_type == "step":
            shocked[variable] = ShockInjector.step_shock(
                df[variable], shock_start_idx, magnitude, duration
            )
        elif shock_type == "gradual":
            shocked[variable] = ShockInjector.gradual_shock(
                df[variable], shock_start_idx, magnitude, transition_steps
            )
        elif shock_type == "pulse":
            shocked[variable] = ShockInjector.pulse_shock(
                df[variable], shock_start_idx, magnitude, duration or 3
            )
        else:
            raise ValueError("shock_type must be one of: step, gradual, pulse.")
        return shocked

    @staticmethod
    def _validate_start(series: pd.Series, shock_start_idx: int) -> None:
        if len(series) == 0:
            raise ValueError("series must not be empty.")
        if shock_start_idx < 0 or shock_start_idx >= len(series):
            raise IndexError("shock_start_idx is outside the series.")

