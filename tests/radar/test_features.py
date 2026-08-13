from __future__ import annotations

from datetime import datetime, timezone

from radar.base import PhysiologyBatch, PhysiologyQuality, PhysiologyRecord
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
    assert summary.heart_rate_median == 72.0


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
    assert summary.quality_reasons == ("sleep_duration_estimated_from_hourly_samples",)


def test_summary_carries_demo_provenance_from_a_batch() -> None:
    summary = summarize_daily(
        PhysiologyBatch((record(heart_rate=72),), PhysiologyQuality(available=True, demo=True, reason="demo_data"))
    )

    assert summary.demo is True
    assert summary.source_quality == PhysiologyQuality(available=True, demo=True, reason="demo_data")


def test_sleep_duration_is_explicitly_marked_as_hourly_sample_estimate() -> None:
    summary = summarize_daily([record(sleep_stage="deep")])

    assert summary.sleep_duration_hours == 1.0
    assert "sleep_duration_estimated_from_hourly_samples" in summary.quality_reasons


def test_out_of_range_respiratory_rate_is_excluded_and_counted() -> None:
    summary = summarize_daily([record(respiratory_rate=60), record(respiratory_rate=14)])

    assert summary.valid_respiratory_rate_count == 1
    assert summary.respiratory_rate_median == 14.0
    assert summary.invalid_measurement_count == 1


def test_coverage_counts_records_with_any_valid_measurement() -> None:
    summary = summarize_daily(
        [record(heart_rate=72), record(heart_rate=400, respiratory_rate=60), record(respiratory_rate=14)]
    )

    assert summary.coverage == 2 / 3
    assert summary.invalid_measurement_count == 2


def test_empty_summary_has_no_coverage_and_explains_the_gap() -> None:
    summary = summarize_daily([])

    assert summary.coverage == 0.0
    assert summary.quality_reasons == ("no_records",)
