from dataclasses import FrozenInstanceError

import pytest

from risk.phase_model.continuous_replay import ReplayFrame, ReplayResult, replay_stream
from risk.phase_model.event_state import EventDecoder, EventDecoderConfig, FrameDecision


def _decoder():
    return EventDecoder(EventDecoderConfig(0.6, 0.5, 1.0, 2.0))


def _normal_decision(timestamp):
    return FrameDecision(float(timestamp), 0.1, "normal", True)


def test_replay_rejects_nonmonotonic_timestamps_before_prediction():
    frames = [ReplayFrame(1.0, "x"), ReplayFrame(0.5, "y")]
    seen = []

    def predictor(frame):
        seen.append(frame.timestamp)
        return _normal_decision(frame.timestamp)

    with pytest.raises(ValueError, match="strictly increasing"):
        replay_stream(frames, predictor=predictor, decoder=_decoder())
    assert seen == [1.0]


def test_replay_payload_excludes_clip_boundary_and_truth_metadata():
    frame = ReplayFrame(
        0.0,
        {
            "pose": [1],
            "clip_id": "hidden",
            "clip_boundary": True,
            "truth_event_id": "t1",
            "truth_start": 0.0,
            "truth_end": 1.0,
            "nested": {"truth_event_id": "also-hidden", "x": 2},
        },
        truth_event_id="t1",
        truth_start=0.0,
        truth_end=1.0,
    )
    seen = []

    def predictor(model_frame):
        seen.append(model_frame)
        return _normal_decision(model_frame.timestamp)

    replay_stream([frame], predictor=predictor, decoder=_decoder())
    assert seen[0].payload == {"pose": [1], "nested": {"x": 2}}
    assert seen[0].truth_event_id is None
    assert seen[0].truth_start is None
    assert seen[0].truth_end is None
    assert frame.payload["clip_id"] == "hidden"


def test_replay_persists_abstentions_and_derives_alerts():
    frames = [
        ReplayFrame(0.0, {"pose": 0}),
        ReplayFrame(0.6, {"pose": 1}),
        ReplayFrame(0.7, {"pose": 2}),
        ReplayFrame(1.8, {"pose": 3}),
    ]

    def predictor(frame):
        if frame.timestamp == 0.7:
            return FrameDecision(frame.timestamp, 0.9, "descent_or_impact", False)
        if frame.timestamp < 0.7:
            return FrameDecision(frame.timestamp, 0.9, "descent_or_impact", True)
        return _normal_decision(frame.timestamp)

    result = replay_stream(frames, predictor=predictor, decoder=_decoder())
    assert isinstance(result, ReplayResult)
    assert len(result.transitions) == 4
    assert result.transitions[2].state == "abstain"
    assert result.abstained_frames == 1
    assert result.total_frames == 4
    assert result.coverage == pytest.approx(3 / 4)
    assert [alert.emitted for alert in result.alerts] == ["fall_confirmed"]


def test_replay_alerts_include_confirmed_and_recovered_transitions():
    frames = [
        ReplayFrame(0.0, "fall"),
        ReplayFrame(0.5, "fall"),
        ReplayFrame(0.6, "normal"),
        ReplayFrame(1.5, "normal"),
        ReplayFrame(1.6, "normal"),
    ]

    def predictor(frame):
        if frame.timestamp < 0.6:
            return FrameDecision(frame.timestamp, 0.9, "descent_or_impact", True)
        return _normal_decision(frame.timestamp)

    result = replay_stream(frames, predictor=predictor, decoder=_decoder())
    assert [alert.emitted for alert in result.alerts] == [
        "fall_confirmed",
        "fall_recovered",
    ]
    assert result.alerts[0].event_id == result.alerts[1].event_id


def test_replay_is_deterministic_and_does_not_expose_mutable_input():
    payload = {"pose": [1, 2], "clip_id": "secret"}
    frame = ReplayFrame(0.0, payload)
    seen = []

    def predictor(model_frame):
        seen.append(model_frame.payload)
        model_frame.payload["pose"].append(99)
        return _normal_decision(model_frame.timestamp)

    first = replay_stream([frame], predictor=predictor, decoder=_decoder())
    second = replay_stream([frame], predictor=lambda f: _normal_decision(f.timestamp), decoder=_decoder())
    assert first.transitions == second.transitions
    assert payload == {"pose": [1, 2], "clip_id": "secret"}
    assert frame.payload == {"pose": [1, 2], "clip_id": "secret"}


def test_replay_validates_empty_stream_and_contracts():
    result = replay_stream([], predictor=lambda frame: _normal_decision(frame.timestamp), decoder=_decoder())
    assert result.transitions == ()
    assert result.alerts == ()
    assert result.coverage == 0.0
    assert result.abstained_frames == 0
    assert result.total_frames == 0
    with pytest.raises(ValueError, match="timestamp"):
        ReplayFrame(float("nan"), None)
    with pytest.raises(FrozenInstanceError):
        result.total_frames = 1
