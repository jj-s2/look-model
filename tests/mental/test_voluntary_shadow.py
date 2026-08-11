from datetime import datetime, timedelta, timezone

import pytest

from mental.contracts import VoluntaryCheckin
from mental.voluntary_shadow import ShadowModelConfig, VoluntaryShadowModel


TZ = timezone(timedelta(hours=8))


def checkin(*, transcript: str = "今天感觉还可以", model_version: str = "unavailable", asr_confidence: float | None = 0.95, audio_quality: float | None = 0.9) -> VoluntaryCheckin:
    return VoluntaryCheckin(
        session_id="session-1",
        subject_alias="senior-1",
        started_at=datetime(2026, 8, 12, 10, tzinfo=TZ),
        consented=True,
        prompt_ids=("mood", "energy"),
        answers={"mood": transcript},
        transcript=transcript,
        asr_confidence=asr_confidence,
        audio_quality=audio_quality,
        model_version=model_version,
    )


def test_shadow_defaults_to_research_only_and_abstains_without_loaded_artifacts() -> None:
    result = VoluntaryShadowModel().assess(checkin())
    assert result.abstained is True
    assert result.promoted is False
    assert result.research_only is True
    assert "research_model_unavailable" in result.abstention_reasons


def test_short_answer_and_low_asr_are_abstained_without_imputation() -> None:
    model = VoluntaryShadowModel(
        config=ShadowModelConfig(min_text_chars=4, min_asr_confidence=0.8),
        text_predictor=lambda text: 0.4,
        audio_predictor=lambda _: 0.4,
    )
    assert "text_too_short" in model.assess(checkin(transcript="好")).abstention_reasons
    assert "asr_confidence_low" in model.assess(checkin(asr_confidence=0.2)).abstention_reasons


def test_version_mismatch_and_required_missing_audio_are_abstained() -> None:
    config = ShadowModelConfig(model_version="macbert-v1", required_modalities=("text", "audio"))
    model = VoluntaryShadowModel(config=config, text_predictor=lambda _: 0.4)
    mismatch = model.assess(checkin(model_version="other-v2"))
    missing = model.assess(checkin(model_version="macbert-v1", audio_quality=None, asr_confidence=None))
    assert "model_version_mismatch" in mismatch.abstention_reasons
    assert "audio_missing" in missing.abstention_reasons


def test_text_and_audio_are_fused_as_one_voluntary_episode_with_stability_metadata() -> None:
    model = VoluntaryShadowModel(
        config=ShadowModelConfig(model_version="macbert-v1"),
        text_predictor=lambda _: 0.7,
        audio_predictor=lambda _: 0.6,
    )
    result = model.assess(checkin(model_version="macbert-v1"))
    assert result.abstained is False
    assert result.combined_score == pytest.approx(0.65)
    assert result.coverage == pytest.approx(1.0)
    assert result.evidence_domains == ("voluntary_episode",)
    assert result.counterfactual_stability is not None
    assert result.promoted is False


def test_predictor_must_return_finite_probability() -> None:
    model = VoluntaryShadowModel(text_predictor=lambda _: float("nan"))
    result = model.assess(checkin(audio_quality=None, asr_confidence=None))
    assert result.abstained is True
    assert "text_prediction_invalid" in result.abstention_reasons


def test_conflicting_text_and_audio_are_not_treated_as_independent_confirmation() -> None:
    model = VoluntaryShadowModel(
        text_predictor=lambda _: 0.95,
        audio_predictor=lambda _: 0.1,
    )
    result = model.assess(checkin())
    assert result.abstained is True
    assert "modalities_conflict" in result.abstention_reasons
