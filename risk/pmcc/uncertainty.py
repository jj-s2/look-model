"""Bootstrap uncertainty summaries and conservative PMCC refusal gates."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Sequence

import numpy as np

from .schema import EvidenceTier


@dataclass(frozen=True)
class UncertaintySummary:
    lower: tuple[float, ...]
    median: tuple[float, ...]
    upper: tuple[float, ...]
    width_72h: float
    abstained: bool
    reasons: tuple[str, ...]


class UncertaintyGate:
    """Summarize bootstrap forecasts and abstain when reliability is inadequate."""

    def evaluate(
        self,
        forecasts: Sequence[Sequence[float]],
        coverage: float,
        quality: float,
        evidence_tier: EvidenceTier,
        release_mode: bool,
    ) -> UncertaintySummary:
        matrix = _forecast_matrix(forecasts)
        coverage = _unit_interval(coverage, "coverage")
        quality = _unit_interval(quality, "quality")
        if not isinstance(evidence_tier, EvidenceTier):
            try:
                evidence_tier = EvidenceTier(evidence_tier)
            except (TypeError, ValueError) as error:
                raise ValueError("evidence_tier must be an EvidenceTier") from error
        if not isinstance(release_mode, bool):
            raise ValueError("release_mode must be a boolean")

        lower = tuple(float(value) for value in np.quantile(matrix, 0.1, axis=0))
        median = tuple(float(value) for value in np.quantile(matrix, 0.5, axis=0))
        upper = tuple(float(value) for value in np.quantile(matrix, 0.9, axis=0))
        width_72h = upper[2] - lower[2]
        reasons: list[str] = []
        if coverage < 0.5:
            reasons.append("insufficient_coverage")
        if quality < 0.5:
            reasons.append("insufficient_quality")
        if width_72h > 0.35:
            reasons.append("wide_72h_interval")
        if release_mode and evidence_tier in {EvidenceTier.SYNTHETIC_RESEARCH, EvidenceTier.OFFLINE_FIXTURE}:
            reasons.append("non_release_evidence")
        return UncertaintySummary(lower, median, upper, width_72h, bool(reasons), tuple(reasons))


def _forecast_matrix(forecasts: Sequence[Sequence[float]]) -> np.ndarray:
    rows = tuple(tuple(row) for row in forecasts)
    if len(rows) < 5:
        raise ValueError("forecasts must contain at least five bootstrap members")
    if any(len(row) != 7 for row in rows):
        raise ValueError("each forecast member must contain seven cumulative risks")
    for row in rows:
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError("forecasts must contain finite probabilities in [0, 1]")
    return np.asarray(rows, dtype=float)


def _unit_interval(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return float(value)
