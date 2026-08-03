from datetime import datetime, timezone

from risk.phase_model.schema import Phase, PhaseModelOutput, PoseObservation
from risk.phase_model.service import PhaseRiskService


class OneShotBuffer:
    def __init__(self, window):
        self.window = window

    def append(self, observation):
        return self.window


class FixedPredictor:
    def __init__(self, output):
        self.output = output

    def predict(self, window):
        return self.output


def observation(score=0.9):
    return PoseObservation(
        timestamp=datetime.now(timezone.utc), tracking_id="elder-1",
        keypoints=((1.0, 1.0),) * 17, scores=(score,) * 17,
        visible_mask=(score >= 0.5,) * 17, bbox=(0, 0, 10, 20),
        frame_size=(640, 480), stream_fresh=True,
    )


def test_service_low_quality_does_not_confirm_fall():
    class EmptyBuffer:
        def append(self, item):
            return None
    service = PhaseRiskService(FixedPredictor(None), EmptyBuffer(), clock=lambda: datetime.now(timezone.utc))
    output = service.observe(observation(0.49))
    assert all(item.quality.confidence <= 0.49 for item in output)
    assert not any(item.event_type.value == "fall_event" and item.payload.get("confirmed") for item in output)


def test_service_emits_prefall_warning_after_smoothing():
    output = PhaseModelOutput((0.01, 0.91, 0.02, 0.02, 0.02, 0.02), 0.1, 0.8, 0.1, 0.9, "e1", "m1", phase=Phase.PREFALL_ABNORMAL)
    item = observation()
    class Buffer:
        def append(self, item):
            return type("Window", (), {"short": (item,), "long": (item,)})()
    service = PhaseRiskService(FixedPredictor(output), Buffer(), clock=lambda: item.timestamp)
    service.observe(item)
    events = service.observe(item)
    assert any(event.event_type.value == "prefall_warning" for event in events)
