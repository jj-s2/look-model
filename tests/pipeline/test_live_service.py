from datetime import datetime, timedelta, timezone
import time

from core.events import DataQuality, EventType, SensorEvent, Source
from pipeline.live_service import LiveMonitoringService, SourceBatch
from storage.clip_buffer import CircularClipBuffer


NOW = datetime(2026, 8, 2, 9, tzinfo=timezone.utc)


class FakeSource:
    name = "camera"

    def poll(self, now: datetime):
        return [
            SensorEvent(
                timestamp=NOW,
                source=Source.VISION,
                event_type=EventType.FALL_EVENT,
                payload={"subject_id": "demo-person", "confirmed": True},
                quality=DataQuality(True, 0.95, True, "demo_fixture"),
            )
        ]


def test_live_service_emits_decision_from_fake_sources() -> None:
    service = LiveMonitoringService(event_sources=[FakeSource()], clock=lambda: NOW)

    snapshot = service.step()

    assert snapshot.camera_health == "healthy"
    assert snapshot.demo is True
    assert any(item.kind == "fall_event" for item in snapshot.decisions)


def test_source_failure_is_degraded_without_stopping_other_sources() -> None:
    class BrokenSource:
        name = "radar"

        def poll(self, now: datetime):
            raise RuntimeError("fixture unavailable")

    service = LiveMonitoringService(event_sources=[BrokenSource(), FakeSource()], clock=lambda: NOW)

    snapshot = service.step()

    assert snapshot.radar_health == "degraded"
    assert any(item.kind == "fall_event" for item in snapshot.decisions)


def test_timed_out_source_returns_degraded_snapshot_without_blocking_other_sources() -> None:
    class SlowRadar:
        name = "radar"

        def poll(self, now: datetime):
            time.sleep(.2)
            return []

    started = time.monotonic()
    snapshot = LiveMonitoringService(
        [SlowRadar(), FakeSource()], clock=lambda: NOW, poll_timeout_seconds=.01
    ).step()

    assert time.monotonic() - started < .1
    assert snapshot.radar_health == "degraded"
    assert "radar: timeout" in snapshot.source_errors
    assert any(item.kind == "fall_event" for item in snapshot.decisions)


def test_unavailable_vision_event_marks_camera_offline() -> None:
    class OfflineCamera:
        name = "camera"

        def poll(self, now: datetime):
            return [
                SensorEvent(
                    timestamp=now, source=Source.VISION, event_type=EventType.AVAILABILITY,
                    payload={}, quality=DataQuality(False, 0.0, False, "camera_unavailable"),
                )
            ]

    snapshot = LiveMonitoringService([OfflineCamera()], clock=lambda: NOW).step()

    assert snapshot.camera_health == "offline"


def test_service_buffers_frames_and_exports_only_confirmed_fall_window(tmp_path) -> None:
    buffer = CircularClipBuffer(tmp_path, recording_opt_in=True, pre_seconds=10)
    samples = iter(
        [
            SourceBatch((), frame="before", frame_timestamp=NOW - timedelta(seconds=10)),
            SourceBatch((
                SensorEvent(
                    timestamp=NOW, source=Source.VISION, event_type=EventType.FALL_EVENT,
                    payload={"confirmed": True}, quality=DataQuality(True, .9, False),
                ),
            ), frame="event", frame_timestamp=NOW),
            SourceBatch((), frame="after", frame_timestamp=NOW + timedelta(seconds=20)),
        ]
    )

    class Camera:
        name = "camera"

        def poll(self, now: datetime):
            return next(samples)

    service = LiveMonitoringService([Camera()], clip_buffer=buffer, clock=lambda: NOW)
    service.step()
    event_snapshot = service.step()
    service.step()

    assert event_snapshot.latest_frame == "event"
    clip = buffer.completed_clips[0]
    assert clip.exists()


def test_unconfirmed_fall_does_not_export_a_clip(tmp_path) -> None:
    buffer = CircularClipBuffer(tmp_path, recording_opt_in=True)

    class Camera:
        name = "camera"

        def poll(self, now: datetime):
            return SourceBatch((
                SensorEvent(
                    timestamp=now, source=Source.VISION, event_type=EventType.FALL_EVENT,
                    payload={"confirmed": False}, quality=DataQuality(True, .9, False),
                ),
            ), frame="event", frame_timestamp=now)

    LiveMonitoringService([Camera()], clip_buffer=buffer, clock=lambda: NOW).step()

    assert buffer.completed_clips == ()


def test_retention_and_alert_history_failures_are_isolated() -> None:
    class BrokenDispatcher:
        def dispatch(self, decision):
            raise OSError("disk unavailable")

        def recent_alerts(self, limit=None):
            raise OSError("disk unavailable")

    class TrackingRetention:
        def __init__(self):
            self.called = False

        def prune(self, now):
            self.called = True
            raise OSError("disk unavailable")

    retention = TrackingRetention()
    snapshot = LiveMonitoringService(
        [FakeSource()], dispatcher=BrokenDispatcher(), retention_policy=retention, clock=lambda: NOW
    ).step()

    assert retention.called is True
    assert snapshot.alert_history == ()
    assert "alerts: unavailable" in snapshot.source_errors
    assert "alert history: unavailable" in snapshot.source_errors
    assert "retention: unavailable" in snapshot.source_errors


def test_malformed_batch_event_is_dropped_and_other_components_continue() -> None:
    class MalformedCamera:
        name = "camera"

        def poll(self, now: datetime):
            return SourceBatch(("bad",))

    class RecordingDispatcher:
        def __init__(self):
            self.history_requested = False

        def dispatch(self, decision):
            raise AssertionError("bad batch must not create a decision")

        def recent_alerts(self, limit=None):
            self.history_requested = True
            return []

    class RecordingRetention:
        def __init__(self):
            self.called = False

        def prune(self, now):
            self.called = True
            return []

    dispatcher = RecordingDispatcher()
    retention = RecordingRetention()
    snapshot = LiveMonitoringService(
        [MalformedCamera()], dispatcher=dispatcher, retention_policy=retention, clock=lambda: NOW
    ).step()

    assert snapshot.camera_health == "degraded"
    assert snapshot.decisions == ()
    assert "camera: invalid data" in snapshot.source_errors
    assert dispatcher.history_requested is True
    assert retention.called is True


def test_live_service_marks_model_abstention_degraded_and_keeps_events() -> None:
    abstained = SensorEvent(
        timestamp=NOW, source=Source.VISION, event_type=EventType.POSE,
        payload={"tracking_id": "elder-1", "quality_mode": "abstained"},
        quality=DataQuality(False, .2, False, "model_reliability_gate"),
    )
    result = LiveMonitoringService(clock=lambda: NOW).step(SourceBatch((abstained,)))
    assert result.system_health == "degraded"
    assert result.events == (abstained,)
    assert not any(event.event_type is EventType.FALL_EVENT for event in result.events)


def test_live_service_preserves_release_event_id_for_clip_confirmation(tmp_path) -> None:
    class RecordingClip:
        def __init__(self):
            self.ids = []

        def on_frame(self, frame, timestamp):
            return None

        def confirm_event(self, event_id):
            self.ids.append(event_id)

    clip = RecordingClip()
    event = SensorEvent(
        timestamp=NOW, source=Source.VISION, event_type=EventType.FALL_EVENT,
        payload={"subject_id": "elder-1", "event_id": "fall-000007", "confirmed": True},
        quality=DataQuality(True, .95, False),
    )
    LiveMonitoringService(clip_buffer=clip, clock=lambda: NOW).step(SourceBatch((event,), frame="f"))
    assert clip.ids == ["fall-000007"]


def test_live_service_does_not_dispatch_rgpc_suspected_transition() -> None:
    class RecordingDispatcher:
        def __init__(self):
            self.calls = 0

        def dispatch(self, decision):
            self.calls += 1

        def recent_alerts(self, limit=None):
            return []

    event = SensorEvent(
        timestamp=NOW, source=Source.VISION, event_type=EventType.FALL_EVENT,
        payload={"subject_id": "elder-1", "event_id": "fall-000001", "event_state": "suspected", "confirmed": False},
        quality=DataQuality(True, .9, False),
    )
    dispatcher = RecordingDispatcher()
    LiveMonitoringService(dispatcher=dispatcher, clock=lambda: NOW).step(SourceBatch((event,)))
    assert dispatcher.calls == 0


def test_live_service_downgrades_non_mapping_payload_without_crashing() -> None:
    malformed = SensorEvent(
        timestamp=NOW, source=Source.VISION, event_type=EventType.POSE,
        payload=None,  # type: ignore[arg-type]
        quality=DataQuality(True, .9, False),
    )
    result = LiveMonitoringService(clock=lambda: NOW).step(SourceBatch((malformed,)))
    assert result.camera_health == "degraded"
    assert result.events == ()
    assert "vision: invalid data" in result.source_errors
