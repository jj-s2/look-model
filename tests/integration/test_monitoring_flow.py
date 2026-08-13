"""Offline integration contracts for the monitoring release workflow."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest

from core.events import DataQuality, EventType, SensorEvent, Source
from pipeline.live_service import LiveMonitoringService
from pipeline.live_service import SourceBatch
from risk.phase_model.schema import Phase, PhaseModelOutput, PoseObservation
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


def test_report_rejects_metrics_without_release_provenance(tmp_path) -> None:
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

    with pytest.raises(ReleaseGateError, match="release"):
        generate_evaluation_report([evaluation, benchmark])


def test_report_requires_passing_full_inference_benchmark(tmp_path) -> None:
    evaluation = tmp_path / "evaluation.json"
    benchmark = tmp_path / "benchmark.json"
    evaluation.write_text(json.dumps({
        "kind": "classification_evaluation", "release_id": "release-2026-08-02",
        "overall_argmax": {"fall_f1": 0.91, "fall_recall": 0.90, "fall_precision": 0.92,
                           "tn": 8, "fp": 1, "fn": 1, "tp": 10},
    }), encoding="utf-8")
    benchmark.write_text(json.dumps({
        "kind": "pipeline_benchmark", "release_id": "release-2026-08-02",
        "pipeline_scope": "capture_decode_only", "release_gate_passed": True,
        "p95_latency_seconds": 1.1, "false_alarms_per_hour": 0.1,
    }), encoding="utf-8")

    with pytest.raises(ReleaseGateError, match="full_inference"):
        generate_evaluation_report([evaluation, benchmark])


def test_report_rejects_classification_and_benchmark_from_different_releases(tmp_path) -> None:
    evaluation = tmp_path / "evaluation.json"
    benchmark = tmp_path / "benchmark.json"
    evaluation.write_text(json.dumps({
        "kind": "classification_evaluation", "release_id": "release-a",
        "overall_argmax": {"fall_f1": 0.91, "fall_recall": 0.90, "fall_precision": 0.92,
                           "tn": 8, "fp": 1, "fn": 1, "tp": 10},
    }), encoding="utf-8")
    benchmark.write_text(json.dumps({
        "kind": "pipeline_benchmark", "release_id": "release-b",
        "pipeline_scope": "full_inference", "release_gate_passed": True,
        "p95_latency_seconds": 1.1, "false_alarms_per_hour": 0.1,
    }), encoding="utf-8")

    with pytest.raises(ReleaseGateError, match="release_id values do not match"):
        generate_evaluation_report([evaluation, benchmark])


def test_report_renders_fall_and_adl_confusion_matrices() -> None:
    report = build_report(
        {
            "release_id": "release-2026-08-02",
            "fall_f1": 0.91,
            "fall_recall": 0.90,
            "fall_precision": 0.92,
            "p95_latency_seconds": 1.1,
            "false_alarms_per_hour": 0.1,
            "pipeline_scope": "full_inference",
            "benchmark_release_gate_passed": True,
            "per_class_confusion_matrix": {
                "fall": {"tn": 8, "fp": 1, "fn": 1, "tp": 10},
                "adl": {"tn": 10, "fp": 1, "fn": 1, "tp": 8},
            },
        }
    )

    assert "| fall | 8 | 1 | 1 | 10 |" in report
    assert "| adl | 10 | 1 | 1 | 8 |" in report


def test_rgpc_event_id_is_stable_across_confirmation_and_recovery() -> None:
    class FakeDecoder:
        def __init__(self):
            self.config = type("Config", (), {"confirm_seconds": .1, "recovery_seconds": .1})()
            self.calls = 0

        def update(self, decision):
            self.calls += 1
            from risk.phase_model.event_state import EventTransition
            if self.calls == 1:
                return EventTransition("suspected", "fall-000001", None, "pending", decision.timestamp)
            if self.calls == 2:
                return EventTransition("confirmed", "fall-000001", "fall_confirmed", "confirmed", decision.timestamp)
            return EventTransition("recovered", "fall-000001", "fall_recovered", "recovered", decision.timestamp)

    class FakePredictor:
        def predict(self, window):
            return PhaseModelOutput(
                phase_probs=(1.0, 0, 0, 0, 0, 0), fall_event_prob=.9,
                prefall_prob=0, recovery_prob=0, quality_score=.95,
                embedding_version="e", model_version="r1", phase=Phase.DESCENDING,
                fall_decision=1,
            )

    from pipeline.vision_phase_source import _RGPCPhaseService
    service = _RGPCPhaseService(FakePredictor(), FakeDecoder())
    service.buffer = type("Buffer", (), {"append": lambda self, observation: object()})()
    def obs(t):
        return PoseObservation(
            timestamp=NOW + timedelta(seconds=t), tracking_id="elder-1",
            keypoints=((1.0, 2.0),) * 17, scores=(.9,) * 17,
            visible_mask=(True,) * 17, bbox=(0, 0, 20, 40), frame_size=(20, 40),
            stream_fresh=True,
        )
    first = service.observe(obs(0.0))
    second = service.observe(obs(1.0))
    third = service.observe(obs(2.0))
    confirmed = next(item for item in second if item.event_type is EventType.FALL_EVENT)
    recovered = next(item for item in third if item.event_type is EventType.FALL_EVENT)
    assert confirmed.payload["event_id"] == recovered.payload["event_id"] == "fall-000001"
    assert confirmed.payload["confirmed"] is True
    assert recovered.payload["recovered"] is True
