from datetime import date, datetime, timedelta, timezone

import pytest

from core.events import EventType, Source
from mental.contracts import (
    DailyWellbeingObservation,
    DeliveryScope,
    ModalityEvidence,
    SensitiveExpressionCandidate,
    VoluntaryCheckin,
    WellbeingAssessmentEvent,
)


TZ = timezone(timedelta(hours=8))


def test_daily_observation_rejects_invalid_domain_values() -> None:
    with pytest.raises(ValueError, match="sleep_hours"):
        DailyWellbeingObservation("alias-1", date(2026, 8, 12), sleep_hours=25.0)
    with pytest.raises(ValueError, match="steps"):
        DailyWellbeingObservation("alias-1", date(2026, 8, 12), steps=-1.0)
    with pytest.raises(ValueError, match="quality"):
        DailyWellbeingObservation(
            "alias-1", date(2026, 8, 12), feature_quality={"steps": 1.2}
        )


def test_daily_observation_detaches_quality_and_requires_alias() -> None:
    quality = {"steps": 0.8}
    observation = DailyWellbeingObservation(
        " alias-1 ", date(2026, 8, 12), steps=3000.0, feature_quality=quality
    )
    quality["steps"] = 0.0
    assert observation.subject_alias == "alias-1"
    assert observation.feature_quality == {"steps": 0.8}
    with pytest.raises(ValueError, match="subject_alias"):
        DailyWellbeingObservation("  ", date(2026, 8, 12))


def test_voluntary_checkin_requires_explicit_consent_and_no_raw_media_field() -> None:
    checkin = VoluntaryCheckin(
        session_id="session-1",
        subject_alias="alias-1",
        started_at=datetime(2026, 8, 12, 10, tzinfo=TZ),
        consented=True,
        prompt_ids=("mood", "energy"),
        answers={"mood": "还可以"},
        asr_confidence=0.91,
    )
    assert checkin.consented is True
    assert not hasattr(checkin, "audio_bytes")
    with pytest.raises(ValueError, match="consent"):
        VoluntaryCheckin(
            "session-2", "alias-1", checkin.started_at, False, ("mood",), {}
        )


def test_modality_evidence_bounds_and_detaches_codes() -> None:
    codes = ["asr_ok"]
    evidence = ModalityEvidence(
        modality="text", score=0.4, available=True, reliability=0.8,
        uncertainty=0.2, evidence_codes=codes
    )
    codes.append("mutated")
    assert evidence.evidence_codes == ("asr_ok",)
    with pytest.raises(ValueError, match="reliability"):
        ModalityEvidence("audio", score=0.2, available=True, reliability=1.1, uncertainty=0.1)


def test_assessment_defaults_to_external_forbidden_and_non_diagnostic() -> None:
    event = WellbeingAssessmentEvent(
        event_id="wb-1",
        subject_alias="alias-1",
        timestamp=datetime(2026, 8, 12, 10, tzinfo=TZ),
        state="observe",
        action="local_record_only",
        evidence_domains=("behavior",),
        evidence_codes=("stable_baseline",),
        coverage=0.9,
        quality=0.8,
        uncertainty=0.2,
        baseline_state="operational",
        model_version="rules_v1",
    )
    assert event.delivery_scope is DeliveryScope.EXTERNAL_FORBIDDEN
    assert event.is_diagnosis is False
    sensor_event = event.to_sensor_event()
    assert sensor_event.source is Source.SCREENING
    assert sensor_event.event_type is EventType.WELLBEING_CHANGE
    assert sensor_event.payload["delivery_scope"] == "external_forbidden"
    assert sensor_event.payload["is_diagnosis"] is False


def test_abstained_assessment_is_unavailable_and_cannot_be_critical() -> None:
    event = WellbeingAssessmentEvent(
        event_id="wb-2",
        subject_alias="alias-1",
        timestamp=datetime(2026, 8, 12, 10, tzinfo=TZ),
        state="abstained",
        action="local_record_only",
        evidence_domains=(),
        evidence_codes=(),
        coverage=0.1,
        quality=0.2,
        uncertainty=0.9,
        baseline_state="provisional",
        abstention_reasons=("insufficient_coverage",),
        model_version="rules_v1",
    )
    converted = event.to_sensor_event()
    assert converted.quality.available is False
    assert converted.quality.reason == "insufficient_coverage"
    assert converted.payload["fall_critical_eligible"] is False


def test_sensitive_expression_is_human_review_only_and_explicitly_non_external() -> None:
    candidate = SensitiveExpressionCandidate(
        event_id="safe-1",
        subject_alias="alias-1",
        timestamp=datetime(2026, 8, 12, 10, tzinfo=TZ),
        source="asr_hypothesis",
        transcript_excerpt="我最近有点害怕",
        asr_confidence=0.62,
        consent_scope="voluntary_checkin",
        retention_expiry=datetime(2026, 8, 12, 12, tzinfo=TZ),
    )
    assert candidate.requires_human_review is True
    assert candidate.external_dispatch_allowed is False
    with pytest.raises(ValueError, match="timezone-aware"):
        SensitiveExpressionCandidate(
            "safe-2", "alias-1", datetime(2026, 8, 12, 10), "direct_typed_text", "x"
        )
