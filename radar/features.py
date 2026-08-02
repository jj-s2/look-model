"""Conservative daily summaries for physiology records."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Sequence

from .base import PhysiologyBatch, PhysiologyQuality, PhysiologyRecord


_HEART_RATE_RANGE = (30.0, 220.0)
_RESPIRATORY_RATE_RANGE = (4.0, 40.0)
_ASLEEP_STAGES = {"light", "deep", "rem", "asleep"}
_AWAKE_STAGES = {"awake", "wake"}


@dataclass(frozen=True)
class DailyPhysiologySummary:
    heart_rate_median: float | None
    respiratory_rate_median: float | None
    sleep_duration_hours: float
    nighttime_awakenings: int
    coverage: float
    valid_heart_rate_count: int
    valid_respiratory_rate_count: int
    invalid_measurement_count: int
    quality_reasons: tuple[str, ...]
    demo: bool
    source_quality: PhysiologyQuality | None


def summarize_daily(records: Sequence[PhysiologyRecord] | PhysiologyBatch) -> DailyPhysiologySummary:
    """Summarize a source batch without losing its demo or availability status.

    Each valid asleep stage is treated as one hourly sample. Consumers must use
    the attached quality reason instead of presenting it as device-reported
    sleep duration.
    """
    source_quality = records.quality if isinstance(records, PhysiologyBatch) else None
    source_records = records.records if isinstance(records, PhysiologyBatch) else records
    heart_rates: list[float] = []
    respiratory_rates: list[float] = []
    invalid_count = 0
    valid_records = 0
    sleep_samples = 0
    awakenings = 0
    previous_asleep = False
    for record in source_records:
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
    if not source_records:
        reasons.append("no_records")
    if invalid_count:
        reasons.append("invalid_measurements")
    if not valid_records and source_records:
        reasons.append("no_valid_measurements")
    if sleep_samples:
        reasons.append("sleep_duration_estimated_from_hourly_samples")
    if source_quality and source_quality.reason:
        reasons.append(source_quality.reason)
    return DailyPhysiologySummary(
        heart_rate_median=float(median(heart_rates)) if heart_rates else None,
        respiratory_rate_median=float(median(respiratory_rates)) if respiratory_rates else None,
        sleep_duration_hours=float(sleep_samples),
        nighttime_awakenings=awakenings,
        coverage=valid_records / len(source_records) if source_records else 0.0,
        valid_heart_rate_count=len(heart_rates),
        valid_respiratory_rate_count=len(respiratory_rates),
        invalid_measurement_count=invalid_count,
        quality_reasons=tuple(reasons),
        demo=source_quality.demo if source_quality else False,
        source_quality=source_quality,
    )


def _in_range(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]
