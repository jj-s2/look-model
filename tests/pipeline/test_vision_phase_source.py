from datetime import datetime, timezone

from core.events import EventType
from pipeline.vision_phase_source import VisionPhaseSource
from vision.pose_pipeline import PoseFrameResult


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
