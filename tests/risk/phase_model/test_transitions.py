from datetime import datetime, timedelta, timezone

from risk.phase_model.schema import Phase, PhaseModelOutput
from risk.phase_model.transitions import PhaseSmoother, allowed_transition, transition_penalty


def output_for(phase: Phase, confidence: float = 0.91) -> PhaseModelOutput:
    probs = [0.01] * len(Phase)
    probs[list(Phase).index(phase)] = confidence
    remainder = (1.0 - confidence) / (len(Phase) - 1)
    probs = [remainder if value == 0.01 else value for value in probs]
    return PhaseModelOutput(tuple(probs), 0.9 if phase in (Phase.IMPACT, Phase.FALLEN) else 0.1, 0.8 if phase is Phase.PREFALL_ABNORMAL else 0.1, 0.1, 0.9, "e1", "m1", phase=phase)


def test_allowed_and_forbidden_transitions():
    assert allowed_transition(Phase.NORMAL_ADL, Phase.PREFALL_ABNORMAL)
    assert allowed_transition(Phase.DESCENDING, Phase.IMPACT)
    assert allowed_transition(Phase.FALLEN, Phase.RECOVERING)
    assert not allowed_transition(Phase.NORMAL_ADL, Phase.FALLEN)
    assert not allowed_transition(Phase.IMPACT, Phase.PREFALL_ABNORMAL)


def test_transition_penalty_counts_illegal_mass():
    probabilities = [[0.0] * len(Phase) for _ in Phase]
    probabilities[list(Phase).index(Phase.NORMAL_ADL)][list(Phase).index(Phase.FALLEN)] = 0.7
    assert transition_penalty(probabilities) == 0.7


def test_single_noisy_fallen_frame_is_not_confirmed():
    smoother = PhaseSmoother(min_persistence=2)
    timestamp = datetime.now(timezone.utc)
    result = smoother.update(output_for(Phase.FALLEN), timestamp)
    assert not result.confirmed_fall


def test_descending_then_impact_confirms_fall_and_two_prefall_warns():
    smoother = PhaseSmoother(min_persistence=2)
    timestamp = datetime.now(timezone.utc)
    assert not smoother.update(output_for(Phase.PREFALL_ABNORMAL), timestamp).prefall_warning
    assert smoother.update(output_for(Phase.PREFALL_ABNORMAL), timestamp + timedelta(seconds=1)).prefall_warning
    smoother.update(output_for(Phase.DESCENDING), timestamp + timedelta(seconds=2))
    result = smoother.update(output_for(Phase.IMPACT), timestamp + timedelta(seconds=3))
    assert result.confirmed_fall
