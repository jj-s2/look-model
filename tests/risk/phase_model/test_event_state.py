from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from risk.phase_model.event_state import (
    EventDecoder,
    EventDecoderConfig,
    EventTransition,
    FrameDecision,
)


def _frame(t, p, phase="normal", reliable=True):
    return FrameDecision(float(t), float(p), phase, reliable)


def test_confirmation_depends_on_elapsed_seconds_not_frame_count():
    decoder = EventDecoder(EventDecoderConfig(0.6, 0.8, 2.0, 10.0))
    assert decoder.update(_frame(0.0, 0.8, "descent_or_impact")).state == "suspected"
    assert decoder.update(_frame(0.2, 0.8, "descent_or_impact")).state == "suspected"
    result = decoder.update(_frame(0.81, 0.8, "descent_or_impact"))
    assert result.state == "confirmed"
    assert result.emitted == "fall_confirmed"
    assert result.event_id == "fall-000001"


def test_abstention_clears_unconfirmed_evidence_and_never_confirms():
    decoder = EventDecoder(EventDecoderConfig(0.6, 0.8, 2.0, 10.0))
    decoder.update(_frame(0.0, 0.9, "descent_or_impact"))
    result = decoder.update(_frame(1.0, 0.9, "descent_or_impact", reliable=False))
    assert result.state == "abstain"
    assert result.emitted is None
    assert decoder.update(_frame(1.1, 0.9, "descent_or_impact")).state == "suspected"


def test_confirmed_event_emits_once_until_recovery():
    decoder = EventDecoder(EventDecoderConfig(0.6, 0.5, 1.0, 2.0))
    decoder.update(_frame(0.0, 0.9, "descent_or_impact"))
    first = decoder.update(_frame(0.6, 0.9, "postfall_or_recovery"))
    second = decoder.update(_frame(0.8, 0.9, "postfall_or_recovery"))
    assert first.emitted == "fall_confirmed"
    assert second.emitted is None
    assert first.event_id == second.event_id


def test_postfall_recovery_is_time_based_and_emits_once():
    decoder = EventDecoder(EventDecoderConfig(0.6, 0.5, 1.0, 2.0))
    decoder.update(_frame(0.0, 0.9, "descent_or_impact"))
    decoder.update(_frame(0.5, 0.9, "postfall_or_recovery"))
    assert decoder.update(_frame(0.6, 0.2, "normal")).state == "confirmed"
    assert decoder.update(_frame(1.4, 0.2, "normal")).state == "confirmed"
    recovered = decoder.update(_frame(1.61, 0.2, "normal"))
    assert recovered.state == "recovered"
    assert recovered.emitted == "fall_recovered"
    assert decoder.update(_frame(2.0, 0.2, "normal")).emitted is None


def test_cooldown_blocks_new_suspicion_until_elapsed():
    decoder = EventDecoder(EventDecoderConfig(0.6, 0.1, 0.5, 2.0))
    decoder.update(_frame(0.0, 0.9, "descent_or_impact"))
    assert decoder.update(_frame(0.1, 0.9, "descent_or_impact")).state == "confirmed"
    decoder.update(_frame(0.2, 0.2, "normal"))
    recovered = decoder.update(_frame(0.7, 0.2, "normal"))
    assert recovered.emitted == "fall_recovered"
    assert decoder.update(_frame(1.0, 0.9, "descent_or_impact")).state == "recovered"
    next_event = decoder.update(_frame(2.8, 0.9, "descent_or_impact"))
    assert next_event.state == "suspected"
    assert next_event.event_id == "fall-000002"


def test_validation_rejects_nonfinite_or_nonmonotonic_inputs():
    with pytest.raises(ValueError, match="fall_threshold"):
        EventDecoderConfig(nan, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="confirm_seconds"):
        EventDecoderConfig(0.5, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="finite number"):
        FrameDecision(0.0, inf, "normal", True)

    decoder = EventDecoder(EventDecoderConfig(0.5, 1.0, 1.0, 1.0))
    decoder.update(_frame(1.0, 0.1))
    with pytest.raises(ValueError, match="strictly increasing"):
        decoder.update(_frame(1.0, 0.1))
    with pytest.raises(ValueError, match="strictly increasing"):
        decoder.update(_frame(0.5, 0.1))


def test_contracts_are_frozen_and_event_transition_is_detached():
    config = EventDecoderConfig(0.5, 1.0, 1.0, 1.0)
    frame = _frame(0.0, 0.1)
    transition = EventTransition("monitoring", None, None, "monitoring", 0.0)
    with pytest.raises(FrozenInstanceError):
        config.fall_threshold = 0.7
    with pytest.raises(FrozenInstanceError):
        frame.phase = "descent_or_impact"
    with pytest.raises(FrozenInstanceError):
        transition.state = "confirmed"


def test_low_reliability_cannot_confirm_even_after_elapsed_time():
    decoder = EventDecoder(EventDecoderConfig(0.6, 0.1, 1.0, 1.0))
    assert decoder.update(_frame(0.0, 0.9, "descent_or_impact", reliable=False)).state == "abstain"
    assert decoder.update(_frame(1.0, 0.9, "descent_or_impact", reliable=False)).state == "abstain"
    assert decoder.update(_frame(1.1, 0.9, "normal", reliable=True)).state == "monitoring"
