"""Offline integration contracts for the monitoring release workflow."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from core.events import DataQuality, EventType, SensorEvent, Source
from pipeline.live_service import LiveMonitoringService
from scripts.benchmark_live_pipeline import benchmark_live_pipeline
from scripts.generate_evaluation_report import ReleaseGateError, build_report, generate_evaluation_report


NOW = datetime(2026, 8, 2, 9, tzinfo=timezone.utc)


class CameraFallSource:
    name = "camera"

    def poll(self, now: datetime):
        return [
            SensorEvent(
                timestamp=NOW,
                source=Source.VISION,
                event_type=EventType.FALL_EVENT,
                payload={"subject_id": "fixture-person", "confirmed": True},
                quality=DataQuality(True, 0.95, True, "controlled offline fixture"),
            )
        ]


class UnavailableRadarSource:
    name = "radar"

    def poll(self, now: datetime):
        return [
            SensorEvent(
                timestamp=NOW,
                source=Source.RADAR,
                event_type=EventType.AVAILABILITY,
                payload={"modality": "radar"},
                quality=DataQuality(False, 0.0, True, "offline fixture unavailable"),
            )
        ]


def test_monitoring_flow_survives_radar_unavailable() -> None:
    service = LiveMonitoringService([CameraFallSource(), UnavailableRadarSource()], clock=lambda: NOW)

    snapshot = service.step()

    fall = next(item for item in snapshot.decisions if item.kind == "fall_event")
    assert snapshot.radar_health == "offline"
    assert fall.quality == "vision_only"
    assert fall.level == "critical"


def test_evaluation_report_fails_release_gate_below_recall_threshold() -> None:
    with pytest.raises(ReleaseGateError, match="recall"):
        build_report(
            {
                "fall_f1": 0.92,
                "fall_recall": 0.87,
                "p95_latency_seconds": 1.2,
                "false_alarms_per_hour": 0.2,
            }
        )


def test_benchmark_records_missing_local_input_as_unavailable(tmp_path) -> None:
    report = benchmark_live_pipeline(tmp_path / "not-present.mp4", duration_seconds=1)

    assert report.input_status == "unavailable"
    assert report.release_gate_passed is False
    assert report.p95_latency_seconds is None
    assert report.dataset == "unavailable"
    assert report.threshold is None


def test_report_merge_keeps_classification_metrics_when_benchmark_marks_them_unavailable(tmp_path) -> None:
    evaluation = tmp_path / "evaluation.json"
    benchmark = tmp_path / "benchmark.json"
    evaluation.write_text(json.dumps({"overall_argmax": {
        "fall_f1": 0.91, "fall_recall": 0.90, "fall_precision": 0.92,
        "tn": 8, "fp": 1, "fn": 1, "tp": 10,
    }}), encoding="utf-8")
    benchmark.write_text(json.dumps({
        "kind": "pipeline_benchmark", "fall_f1": None, "fall_recall": None,
        "p95_latency_seconds": 1.1, "false_alarms_per_hour": 0.1,
    }), encoding="utf-8")

    report = generate_evaluation_report([evaluation, benchmark])

    assert "| Fall F1 | 0.91" in report
    assert "## Release gate: PASS" in report
