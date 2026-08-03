"""Pure discrete-time survival arithmetic used by PMCC models."""
from __future__ import annotations

from dataclasses import dataclass
from math import expm1, isfinite, log1p
from numbers import Real
from typing import Sequence


_HORIZON_DAYS = {"24h": 1, "72h": 3, "7d": 7}


@dataclass(frozen=True)
class SurvivalLabel:
    event_day: int | None
    censor_day: int

    def __post_init__(self) -> None:
        if isinstance(self.censor_day, bool) or not isinstance(self.censor_day, int) or not 1 <= self.censor_day <= 7:
            raise ValueError("censor_day must be in [1, 7]")
        if self.event_day is not None and (
            isinstance(self.event_day, bool)
            or not isinstance(self.event_day, int)
            or not 1 <= self.event_day <= self.censor_day
        ):
            raise ValueError("event_day must be within the observed censoring window")


def hazards_to_cumulative(hazards: Sequence[float]) -> tuple[float, ...]:
    """Convert seven conditional day hazards into cumulative event risks."""
    checked = _validate_hazards(hazards)
    log_survival = 0.0
    cumulative: list[float] = []
    for hazard in checked:
        if hazard == 1.0:
            log_survival = float("-inf")
        elif log_survival != float("-inf"):
            log_survival += log1p(-hazard)
        value = 1.0 if log_survival == float("-inf") else -expm1(log_survival)
        cumulative.append(min(1.0, max(0.0, value)))
    return tuple(cumulative)


def cumulative_risk_for_horizons(hazards: Sequence[float]) -> dict[str, float]:
    """Return 24-hour, 72-hour and seven-day cumulative risk values."""
    cumulative = hazards_to_cumulative(hazards)
    return {name: cumulative[day - 1] for name, day in _HORIZON_DAYS.items()}


def _validate_hazards(hazards: Sequence[float]) -> tuple[float, ...]:
    values = tuple(hazards)
    if len(values) != 7:
        raise ValueError("hazards must contain exactly seven probabilities")
    checked: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
            raise ValueError("hazards must be finite probabilities")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("hazards must be within [0, 1]")
        checked.append(float(value))
    return tuple(checked)
