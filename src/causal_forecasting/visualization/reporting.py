"""Report artifact helpers for tables and generated figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ExperimentArtifact:
    """A file generated for report or presentation use."""

    name: str
    path: Path
    artifact_type: str


def write_table(
    table: pd.DataFrame,
    output_path: str | Path,
    *,
    index: bool = False,
) -> ExperimentArtifact:
    """Write a report table as CSV or Markdown based on file extension."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        table.to_csv(path, index=index)
    elif suffix in {".md", ".markdown"}:
        path.write_text(_to_markdown(table, index=index), encoding="utf-8")
    else:
        raise ValueError("Supported table formats are .csv and .md.")
    return ExperimentArtifact(name=path.stem, path=path, artifact_type="table")


def _to_markdown(table: pd.DataFrame, *, index: bool = False) -> str:
    """Write a small GitHub-style Markdown table without optional deps."""

    frame = table.reset_index() if index else table.copy()
    columns = [str(col) for col in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in frame.columns) + " |")
    return "\n".join(rows) + "\n"
