"""DAG validation, bootstrap stability, and report table helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd

from causal_forecasting.causal_discovery.discovery import (
    DiscoveryConfig,
    DiscoveryResult,
    run_pc,
)
from causal_forecasting.causal_discovery.graph_utils import CausalEdge


ConfidenceLabel = Literal["HIGH", "MEDIUM", "LOW"]
DomainPrior = Literal["CONFIRMED", "PLAUSIBLE", "UNCERTAIN", "SUSPICIOUS"]
ValidationStatus = Literal["ACCEPT", "FLAG", "REJECT", "INVESTIGATE"]


@dataclass(frozen=True)
class EdgeConfidence:
    """Confidence assigned to an edge from bootstrap stability."""

    cause: str
    effect: str
    bootstrap_frequency: float
    confidence: ConfidenceLabel
    used_in_simulation: bool


@dataclass(frozen=True)
class DomainValidationRecord:
    """One report-ready row for domain validation of a causal edge."""

    cause: str
    effect: str
    pc_present: bool
    fci_present: bool
    domain_prior: DomainPrior
    status: ValidationStatus
    evidence: str = ""
    notes: str = ""


DiscoveryRunner = Callable[[np.ndarray, list[str], DiscoveryConfig], DiscoveryResult]


def confidence_label(
    frequency: float,
    *,
    high_threshold: float = 0.80,
    medium_threshold: float = 0.50,
) -> ConfidenceLabel:
    """Convert a bootstrap frequency into a qualitative confidence label."""

    if frequency < 0 or frequency > 1:
        raise ValueError("frequency must be between 0 and 1.")
    if frequency >= high_threshold:
        return "HIGH"
    if frequency >= medium_threshold:
        return "MEDIUM"
    return "LOW"


def bootstrap_edge_stability(
    data: pd.DataFrame | np.ndarray,
    *,
    variable_names: list[str] | None = None,
    n_bootstrap: int = 100,
    sample_fraction: float = 0.80,
    config: DiscoveryConfig | None = None,
    random_state: int | None = 42,
    discovery_runner: DiscoveryRunner | None = None,
) -> np.ndarray:
    """Estimate directed-edge stability by bootstrapping discovery runs.

    The returned matrix uses rows as causes and columns as effects:
    ``stability[source_index, target_index] = edge frequency``.
    """

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in the interval (0, 1].")

    if isinstance(data, pd.DataFrame):
        values = data.values
        names = list(data.columns) if variable_names is None else variable_names
    else:
        values = np.asarray(data)
        if variable_names is None:
            names = [f"X{i}" for i in range(values.shape[1])]
        else:
            names = variable_names

    if values.ndim != 2:
        raise ValueError("data must be a 2D matrix.")
    if values.shape[1] != len(names):
        raise ValueError("variable_names length must match data columns.")

    config = config or DiscoveryConfig()
    runner = discovery_runner or _run_pc_for_bootstrap
    rng = np.random.default_rng(random_state)
    n_samples = max(2, int(len(values) * sample_fraction))

    edge_counts = np.zeros((len(names), len(names)), dtype=float)
    successful_runs = 0

    for _ in range(n_bootstrap):
        sample_indices = rng.choice(len(values), size=n_samples, replace=True)
        subsample = values[sample_indices]

        try:
            result = runner(subsample, names, config)
        except Exception:
            continue

        successful_runs += 1
        name_to_index = {name: idx for idx, name in enumerate(names)}
        for edge in result.edges:
            if edge.edge_type != "directed":
                continue
            source_idx = name_to_index[edge.source]
            target_idx = name_to_index[edge.target]
            edge_counts[source_idx, target_idx] += 1

    if successful_runs == 0:
        raise RuntimeError("All bootstrap causal discovery runs failed.")

    return edge_counts / successful_runs


def _run_pc_for_bootstrap(
    data: np.ndarray,
    variable_names: list[str],
    config: DiscoveryConfig,
) -> DiscoveryResult:
    return run_pc(data, variable_names=variable_names, config=config)


def compute_edge_confidence_report(
    stability_matrix: np.ndarray,
    variable_names: list[str],
    *,
    threshold: float = 0.80,
    include_zero_frequency: bool = False,
) -> list[EdgeConfidence]:
    """Create a confidence report from a source-by-target stability matrix."""

    matrix = np.asarray(stability_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("stability_matrix must be square.")
    if matrix.shape[0] != len(variable_names):
        raise ValueError("variable_names length must match matrix shape.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    report: list[EdgeConfidence] = []
    for source_idx, cause in enumerate(variable_names):
        for target_idx, effect in enumerate(variable_names):
            if source_idx == target_idx:
                continue
            freq = float(matrix[source_idx, target_idx])
            if freq == 0 and not include_zero_frequency:
                continue
            report.append(
                EdgeConfidence(
                    cause=cause,
                    effect=effect,
                    bootstrap_frequency=freq,
                    confidence=confidence_label(freq),
                    used_in_simulation=freq >= threshold,
                )
            )

    return sorted(
        report,
        key=lambda item: (-item.bootstrap_frequency, item.cause, item.effect),
    )


def edge_confidence_report_to_frame(
    report: list[EdgeConfidence],
) -> pd.DataFrame:
    """Convert edge confidence records into a report-ready dataframe."""

    return pd.DataFrame(
        [
            {
                "cause": row.cause,
                "effect": row.effect,
                "bootstrap_frequency": row.bootstrap_frequency,
                "confidence": row.confidence,
                "used_in_simulation": row.used_in_simulation,
            }
            for row in report
        ]
    )


def build_domain_validation_records(
    pc_edges: list[CausalEdge],
    fci_edges: list[CausalEdge],
    priors: dict[tuple[str, str], DomainPrior],
    *,
    evidence: dict[tuple[str, str], str] | None = None,
    notes: dict[tuple[str, str], str] | None = None,
) -> list[DomainValidationRecord]:
    """Build report rows comparing discovered edges to domain priors."""

    evidence = evidence or {}
    notes = notes or {}
    pc_directed = {
        (edge.source, edge.target) for edge in pc_edges if edge.edge_type == "directed"
    }
    fci_directed = {
        (edge.source, edge.target) for edge in fci_edges if edge.edge_type == "directed"
    }
    all_pairs = sorted(set(priors) | pc_directed | fci_directed)

    records = []
    for pair in all_pairs:
        pc_present = pair in pc_directed
        fci_present = pair in fci_directed
        prior = priors.get(pair, "UNCERTAIN")
        records.append(
            DomainValidationRecord(
                cause=pair[0],
                effect=pair[1],
                pc_present=pc_present,
                fci_present=fci_present,
                domain_prior=prior,
                status=_infer_validation_status(pc_present, fci_present, prior),
                evidence=evidence.get(pair, ""),
                notes=notes.get(pair, ""),
            )
        )

    return records


def _infer_validation_status(
    pc_present: bool,
    fci_present: bool,
    prior: DomainPrior,
) -> ValidationStatus:
    if pc_present and fci_present and prior in {"CONFIRMED", "PLAUSIBLE"}:
        return "ACCEPT"
    if pc_present and fci_present and prior == "SUSPICIOUS":
        return "INVESTIGATE"
    if pc_present != fci_present:
        return "FLAG"
    if pc_present and fci_present:
        return "FLAG"
    return "INVESTIGATE" if prior == "CONFIRMED" else "FLAG"


def domain_validation_to_frame(
    records: list[DomainValidationRecord],
) -> pd.DataFrame:
    """Convert domain validation rows into a paper-ready dataframe."""

    return pd.DataFrame(
        [
            {
                "edge": f"{row.cause} -> {row.effect}",
                "pc": "Y" if row.pc_present else "N",
                "fci": "Y" if row.fci_present else "N",
                "domain_prior": row.domain_prior,
                "status": row.status,
                "evidence": row.evidence,
                "notes": row.notes,
            }
            for row in records
        ]
    )


def reliable_edges_from_report(
    report: list[EdgeConfidence],
) -> list[tuple[str, str, float]]:
    """Return edges that passed the simulation threshold."""

    return [
        (row.cause, row.effect, row.bootstrap_frequency)
        for row in report
        if row.used_in_simulation
    ]

