from datetime import datetime, timezone
import json
import subprocess
import sys

import pytest

from risk.phase_model.schema import Phase, PhaseModelOutput, PoseObservation


def valid_output(**changes):
    values = dict(
        phase_probs=(0.7, 0.1, 0.1, 0.05, 0.03, 0.02),
        fall_event_prob=0.18,
        prefall_prob=0.1,
        recovery_prob=0.02,
        quality_score=0.9,
        embedding_version="padtfs-v1",
        model_version="fixture-v1",
    )
    values.update(changes)
    return PhaseModelOutput(**values)


def test_phase_output_is_normalized_and_versioned():
    output = valid_output()
    assert sum(output.phase_probs) == pytest.approx(1.0)
    assert output.phase is Phase.NORMAL_ADL
    assert json.loads(output.to_json())["schema_version"] == "padtfs.phase_output.v1"
    assert output.fall_decision == 0


def test_phase_output_distinguishes_legacy_decision_from_explicit_abstention():
    legacy = valid_output(fall_event_prob=0.8)
    assert legacy.fall_decision == 1
    payload = legacy.to_dict()
    payload["fall_decision"] = None
    abstained = PhaseModelOutput.from_dict(payload)
    assert abstained.fall_decision is None
    omitted = dict(payload)
    omitted.pop("fall_decision")
    assert PhaseModelOutput.from_dict(omitted).fall_decision == 1


@pytest.mark.parametrize("value", [-0.1, 1.1, True])
def test_probabilities_reject_invalid_values(value):
    with pytest.raises(ValueError):
        valid_output(fall_event_prob=value)


def test_phase_probs_must_have_six_finite_values():
    with pytest.raises(ValueError):
        valid_output(phase_probs=(1.0, 0.0))
    with pytest.raises(ValueError):
        valid_output(phase_probs=(float("nan"), 0.0, 0.0, 0.0, 0.0, 1.0))


def test_pose_observation_requires_timezone_and_mask():
    kwargs = dict(
        tracking_id="resident-1",
        keypoints=((0.0, 0.0),) * 17,
        scores=(0.9,) * 17,
        visible_mask=(True,) * 17,
        bbox=(0.0, 0.0, 10.0, 20.0),
        frame_size=(640, 480),
        stream_fresh=True,
    )
    with pytest.raises(ValueError):
        PoseObservation(timestamp=datetime(2026, 8, 3), **kwargs)
    with pytest.raises(ValueError):
        PoseObservation(timestamp=datetime.now(timezone.utc), visible_mask=(True,), **{k: v for k, v in kwargs.items() if k != "visible_mask"})


def test_pose_observation_round_trip_preserves_mask():
    observation = PoseObservation(
        timestamp=datetime(2026, 8, 3, tzinfo=timezone.utc),
        tracking_id="resident-1",
        keypoints=((1.0, 2.0),) * 17,
        scores=(0.9,) * 17,
        visible_mask=(True,) * 16 + (False,),
        bbox=(0.0, 0.0, 10.0, 20.0),
        frame_size=(640, 480),
        stream_fresh=True,
    )
    assert PoseObservation.from_dict(observation.to_dict()) == observation


def test_schema_import_does_not_require_numpy_or_torch():
    code = "import risk.phase_model.schema; print('ok')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    assert result.stdout.strip() == "ok"
