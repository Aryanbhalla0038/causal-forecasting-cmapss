import numpy as np
import pandas as pd
import pytest

from causal_forecasting.data.preprocessing import (
    StationarityResult,
    create_lagged_features,
    make_stationary,
    normalize_dataframe,
    validate_time_series_frame,
)


def test_validate_time_series_frame_accepts_clean_datetime_index():
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0]},
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )

    summary = validate_time_series_frame(df)

    assert summary["rows"] == 3
    assert summary["columns"] == 1
    assert summary["missing_values"] == 0


def test_validate_time_series_frame_rejects_duplicate_timestamps():
    index = pd.to_datetime(["2020-01-01", "2020-01-01"])
    df = pd.DataFrame({"x": [1.0, 2.0]}, index=index)

    with pytest.raises(ValueError, match="duplicate timestamps"):
        validate_time_series_frame(df)


def test_make_stationary_differences_only_non_stationary_columns():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    df = pd.DataFrame({"x": [1.0, 3.0, 6.0, 10.0], "y": [5.0, 4.0, 5.0, 4.0]}, index=index)
    results = {
        "x": StationarityResult("x", -1.0, 0.8, False, 4),
        "y": StationarityResult("y", -4.0, 0.01, True, 4),
    }

    stationary, differenced = make_stationary(df, results)

    assert differenced == ["x"]
    assert stationary["x"].tolist() == [2.0, 3.0, 4.0]
    assert stationary["y"].tolist() == [4.0, 5.0, 4.0]


def test_normalize_dataframe_returns_zero_mean_columns():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]}, index=index)

    scaled, scaler = normalize_dataframe(df)

    assert np.allclose(scaled.mean().values, [0.0, 0.0])
    assert hasattr(scaler, "inverse_transform")


def test_create_lagged_features_uses_expected_names_and_values():
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame({"x": [10, 11, 12, 13, 14], "y": [20, 21, 22, 23, 24]}, index=index)

    lagged = create_lagged_features(df, max_lag=2, variables=["x"])

    assert lagged.columns.tolist() == ["x", "x_lag1", "x_lag2"]
    assert lagged.iloc[0].tolist() == [12, 11, 10]
    assert lagged.iloc[-1].tolist() == [14, 13, 12]

