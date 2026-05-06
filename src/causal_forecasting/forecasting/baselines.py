"""Correlational forecasting baselines."""

from __future__ import annotations

import pandas as pd


def fit_arimax_baseline(
    train_data: pd.DataFrame,
    target_var: str,
    exog_vars: list[str],
    *,
    order: tuple[int, int, int] = (1, 1, 1),
):
    """Fit an ARIMAX/SARIMAX baseline for one target variable."""

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    _validate_arimax_inputs(train_data, target_var, exog_vars)
    model = SARIMAX(
        endog=train_data[target_var],
        exog=train_data[exog_vars] if exog_vars else None,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def arimax_forecast(
    fitted_model,
    *,
    steps: int,
    test_exog: pd.DataFrame | None = None,
) -> pd.Series:
    """Forecast from a fitted ARIMAX/SARIMAX model."""

    if steps <= 0:
        raise ValueError("steps must be positive.")
    return fitted_model.forecast(steps=steps, exog=test_exog)


def _validate_arimax_inputs(
    train_data: pd.DataFrame,
    target_var: str,
    exog_vars: list[str],
) -> None:
    missing = sorted({target_var, *exog_vars} - set(train_data.columns))
    if missing:
        raise KeyError(f"Missing ARIMAX columns: {missing}")
    if train_data[[target_var, *exog_vars]].isna().any().any():
        raise ValueError("ARIMAX training data contains missing values.")


try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - depends on optional dependency.
    torch = None
    nn = None


if nn is not None:

    class LSTMBaseline(nn.Module):
        """LSTM correlational baseline architecture.

        Training loops are intentionally kept outside this class so experiments
        can control sequence length, validation split, and early stopping.
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            output_size: int = 1,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            effective_dropout = dropout if num_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=effective_dropout,
            )
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])

else:

    class LSTMBaseline:  # pragma: no cover - exercised only without torch.
        """Placeholder that explains the missing optional dependency."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "PyTorch is required for LSTMBaseline. Install with "
                "`pip install .[deep-learning]`."
            )

