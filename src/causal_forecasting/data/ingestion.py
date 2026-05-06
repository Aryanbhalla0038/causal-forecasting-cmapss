"""Dataset ingestion helpers.

The initial implementation targets the NASA CMAPSS turbofan degradation
dataset because its sensor relationships have a defensible physical grounding.
"""

from pathlib import Path

import pandas as pd

from causal_forecasting.config import CMapssConfig


CMAPSS_COLUMNS = (
    ["unit", "cycle", "setting1", "setting2", "setting3"]
    + [f"s{i}" for i in range(1, 22)]
)


def load_cmapss_train(
    file_path: str | Path,
    config: CMapssConfig | None = None,
) -> pd.DataFrame:
    """Load a CMAPSS training split with a synthetic datetime index.

    CMAPSS files are whitespace-separated and include trailing empty columns in
    many distributions. Those columns are removed after loading.
    """

    config = config or CMapssConfig()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CMAPSS file not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMNS,
        index_col=False,
        engine="python",
    )
    df = df.dropna(axis=1, how="all")

    origin = pd.Timestamp(config.synthetic_origin)
    df["timestamp"] = pd.to_datetime(
        df["cycle"],
        unit=config.datetime_unit,
        origin=origin,
    )
    return df.set_index("timestamp").sort_index()


def select_cmapss_unit(df: pd.DataFrame, unit: int) -> pd.DataFrame:
    """Select one engine unit and restore a unique temporal index."""

    if "unit" not in df.columns:
        raise KeyError("CMAPSS dataframe must include a 'unit' column.")
    unit_df = df[df["unit"] == unit].copy()
    if unit_df.empty:
        raise ValueError(f"Unit {unit} was not found in the CMAPSS dataframe.")
    return unit_df.sort_values("cycle")
