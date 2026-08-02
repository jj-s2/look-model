"""Conservative daily summaries for physiology records."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Sequence

from .base import PhysiologyRecord


_HEART_RATE_RANGE = (30.0, 220.0)
_RESPIRATORY_RATE_RANGE = (4.0, 40.0)
_ASLEEP_STAGES = {"light", "deep", "rem", "asleep"}
_AWAKE_STAGES = {"awake", "wake"}


@dataclass(frozen=True)
class DailyPhysiologySummary:
    resting_heart_rate_median: float | None
    respiratory_rate_median: float | None
    sleep_duration_hours: float
    nighttime_awakenings: int
    coverage: float
    valid_heart_rate_count: int
    valid_respiratory_rate_count: int
    invalid_measurement_count: int
    quality_reasons: tuple[str, ...]


def summarize_daily(records: Sequence[PhysiologyRecord]) -> DailyPhysiologySummary:
    heart_rates: list[float] = []
    respiratory_rates: list[float] = []
    invalid_count = 0
    valid_records = 0
    sleep_samples = 0
    awakenings = 0
    previous_asleep = False
    for record in records:
        record_valid = False
        if record.heart_rate is not None:
            if _in_range(record.heart_rate, _HEART_RATE_RANGE):
                heart_rates.append(record.heart_rate)
                record_valid = True
            else:
                invalid_count += 1
        if record.respiratory_rate is not None:
            if _in_range(record.respiratory_rate, _RESPIRATORY_RATE_RANGE):
                respiratory_rates.append(record.respiratory_rate)
                record_valid = True
            else:
                invalid_count += 1
        stage = record.sleep_stage.lower() if record.sleep_stage else None
        if stage in _ASLEEP_STAGES:
            sleep_samples += 1
            previous_asleep = True
            record_valid = True
        elif stage in _AWAKE_STAGES:
            if previous_asleep:
                awakenings += 1
            previous_asleep = False
            record_valid = True
        elif stage is not None:
            invalid_count += 1
        valid_records += int(record_valid)
    reasons: list[str] = []
    if not records:
        reasons.append("no_records")
    if invalid_count:
        reasons.append("invalid_measurements")
    if not valid_records and records:
        reasons.append("no_valid_measurements")
    return DailyPhysiologySummary(
        resting_heart_rate_median=float(median(heart_rates)) if heart_rates else None,
        respiratory_rate_median=float(median(respiratory_rates)) if respiratory_rates else None,
        sleep_duration_hours=float(sleep_samples),
        nighttime_awakenings=awakenings,
        coverage=valid_records / len(records) if records else 0.0,
        valid_heart_rate_count=len(heart_rates),
        valid_respiratory_rate_count=len(respiratory_rates),
        invalid_measurement_count=invalid_count,
        quality_reasons=tuple(reasons),
    )


def _in_range(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]
