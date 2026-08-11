from datetime import datetime, timedelta, timezone
import hashlib

from core.events import EventType
from pipeline.vision_phase_source import VisionPhaseSource
from vision.pose_pipeline import PoseFrameResult
from risk.phase_model.release_config import RGPCReleaseConfig, write_release_config
from risk.phase_model.schema import PhaseModelOutput, Phase, PoseObservation


NOW = datetime(2026, 8, 2, 9, tzinfo=timezone.utc)
FRAME = object()


class FakeStream:
    name = "ezviz"

    def __init__(self, samples):
        self.samples = iter(samples)

    def read_frame(self):
        return next(self.samples)


class FakePosePipeline:
    def process(self, frame, timestamp):
        return PoseFrameResult(1, timestamp, [[0, 0, 20, 40]], [[[1.0, 2.0]] * 17], [[.9] * 17])


class FakeTracker:
    def track(self, result):
        return [{"tracking_id": "elder-1", "index": 0}]


class FakePhaseService:
    def __init__(self):
        self.observations = []

    def observe(self, observation):
        self.observations.append(observation)
        return ()


def test_failed_read_returns_availability_without_inference():
    phase = FakePhaseService()
    source = VisionPhaseSource(FakeStream([(False, None)]), FakePosePipeline(), FakeTracker(), phase)
    batch = source.poll(NOW)
    assert batch.frame is None
    assert batch.events[0].event_type is EventType.AVAILABILITY
    assert batch.events[0].quality.available is False
    assert phase.observations == []


def test_frame_flows_to_pose_and_phase_service():
    phase = FakePhaseService()
    source = VisionPhaseSource(FakeStream([(True, FRAME)]), FakePosePipeline(), FakeTracker(), phase)
    batch = source.poll(NOW)
    assert batch.frame is FRAME
    assert phase.observations
    assert batch.events[0].event_type is EventType.POSE


def test_frame_uses_stream_video_timestamp_when_available():
    """Recorded-video PTS must drive temporal windows instead of decode speed."""
    phase = FakePhaseService()

    class TimestampedStream(FakeStream):
        def timestamp_for_frame(self, wall_clock):
            assert wall_clock == NOW
            return NOW + timedelta(seconds=0.04)

    source = VisionPhaseSource(
        TimestampedStream([(True, FRAME)]), FakePosePipeline(), FakeTracker(), phase
    )

    batch = source.poll(NOW)

    assert phase.observations[0].timestamp == NOW + timedelta(seconds=0.04)
    assert batch.frame_timestamp == NOW + timedelta(seconds=0.04)


def _release(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = RGPCReleaseConfig(
        schema_version="rgpc.release.v1", release_id="r1", model_sha256=digest,
        dataset_sha256="a" * 64, split_sha256="b" * 64, temperature=1.0,
        fall_threshold=.7, reliability_threshold=.6, confirm_seconds=.8,
        recovery_seconds=2.0, cooldown_seconds=1.0, minimum_coverage=.8,
    )
    write_release_config(config, tmp_path / "release_config.json")
    return tmp_path


def test_live_rgpc_source_loads_event_timers_from_release(tmp_path, monkeypatch):
    release = _release(tmp_path)

    class FakePredictor:
        @classmethod
        def from_release(cls, release_dir, *, device):
            assert release_dir == release
            assert device == "cpu"
            return cls()

        def predict(self, window):
            return PhaseModelOutput(
                phase_probs=(1.0, 0, 0, 0, 0, 0), fall_event_prob=.1,
                prefall_prob=0, recovery_prob=0, quality_score=.99,
                embedding_version="e", model_version="r1", phase=Phase.NORMAL_ADL,
                fall_decision=0,
            )

    monkeypatch.setattr("pipeline.vision_phase_source.RGPredictor", FakePredictor)
    source = VisionPhaseSource.from_rgpc_release(release, device="cpu")
    assert source.decoder.config.confirm_seconds == .8
    assert source.decoder.config.recovery_seconds == 2.0


def test_rgpc_abstention_maps_to_pose_quality_reason(tmp_path, monkeypatch):
    release = _release(tmp_path)

    class FakePredictor:
        @classmethod
        def from_release(cls, release_dir, *, device):
            return cls()

        def predict(self, window):
            return PhaseModelOutput(
                phase_probs=(1.0, 0, 0, 0, 0, 0), fall_event_prob=.9,
                prefall_prob=0, recovery_prob=0, quality_score=.2,
                embedding_version="e", model_version="r1", phase=Phase.NORMAL_ADL,
                fall_decision=None,
            )

    monkeypatch.setattr("pipeline.vision_phase_source.RGPredictor", FakePredictor)
    source = VisionPhaseSource.from_rgpc_release(release, device="cpu")
    source.phase_service.buffer = type("Buffer", (), {"append": lambda self, observation: object()})()
    observation = PoseObservation(
        timestamp=NOW, tracking_id="elder-1", keypoints=((1.0, 2.0),) * 17,
        scores=(.9,) * 17, visible_mask=(True,) * 17, bbox=(0, 0, 20, 40),
        frame_size=(20, 40), stream_fresh=True,
    )
    events = source.phase_service.observe(observation)
    assert events[0].event_type is EventType.POSE
    assert events[0].quality.reason == "model_reliability_gate"
    assert events[0].quality.available is False
