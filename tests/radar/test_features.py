from __future__ import annotations

from datetime import datetime, timezone

from radar.base import PhysiologyRecord
from radar.features import summarize_daily


def record(**values: object) -> PhysiologyRecord:
    return PhysiologyRecord(
        timestamp=datetime(2026, 8, 1, 12, tzinfo=timezone.utc),
        heart_rate=values.get("heart_rate"),
        respiratory_rate=values.get("respiratory_rate"),
        sleep_stage=values.get("sleep_stage"),
        source_fields={},
    )


def test_out_of_range_measurement_is_excluded_and_counted() -> None:
    summary = summarize_daily([record(heart_rate=400), record(heart_rate=72)])

    assert summary.valid_heart_rate_count == 1
    assert summary.invalid_measurement_count == 1
    assert summary.resting_heart_rate_median == 72.0


def test_summary_reports_quality_and_sleep_fields_from_valid_records() -> None:
    summary = summarize_daily(
        [
            record(heart_rate=60, respiratory_rate=12, sleep_stage="light"),
            record(heart_rate=80, respiratory_rate=16, sleep_stage="awake"),
            record(heart_rate=70, respiratory_rate=14, sleep_stage="deep"),
        ]
    )

    assert summary.respiratory_rate_median == 14.0
    assert summary.sleep_duration_hours == 2.0
    assert summary.nighttime_awakenings == 1
    assert summary.coverage == 1.0
    assert summary.quality_reasons == ()
