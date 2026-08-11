from datetime import datetime, timedelta, timezone

from mental.contracts import DeliveryScope, WellbeingAssessmentEvent
from mental.safety_gate import SafetyGate, SafetyGateConfig
from mental.voluntary_shadow import ShadowAssessment


TZ = timezone(timedelta(hours=8))


def behavior_event() -> WellbeingAssessmentEvent:
    return WellbeingAssessmentEvent(
        event_id="behavior-1", subject_alias="senior-1", timestamp=datetime(2026, 8, 12, tzinfo=TZ),
        state="observe", action="local_record_only", evidence_domains=("activity",),
        evidence_codes=("stable_baseline",), coverage=1.0, quality=0.9, uncertainty=0.1,
        baseline_state="operational", model_version="pace_behavior_v1",
    )


def shadow(*, score: float = 0.4, stability: float = 1.0, reasons: tuple[str, ...] = ()) -> ShadowAssessment:
    return ShadowAssessment(
        session_id="session-1", subject_alias="senior-1", combined_score=score,
        coverage=1.0, uncertainty=0.1, evidence_domains=("voluntary_episode",),
        evidence_codes=("text_ok", "audio_ok"), counterfactual_stability=stability,
        model_version="research_shadow_v1", model_mode="research_shadow",
        abstention_reasons=reasons,
    )


def test_shadow_result_never_promotes_or_becomes_external() -> None:
    result = SafetyGate().evaluate(shadow(score=0.8), behavior_event())
    assert result.promoted is False
    assert result.event.delivery_scope is DeliveryScope.EXTERNAL_FORBIDDEN
    assert result.event.model_mode == "research_shadow"
    assert result.event.is_diagnosis is False
    assert result.event.action == "local_record_only"


def test_conflict_or_counterfactual_instability_abstains() -> None:
    gate = SafetyGate(SafetyGateConfig(min_counterfactual_stability=0.8))
    conflict = gate.evaluate(shadow(reasons=("modalities_conflict",)), behavior_event())
    unstable = gate.evaluate(shadow(stability=0.2), behavior_event())
    assert conflict.event.state == "abstained"
    assert "modalities_conflict" in conflict.event.abstention_reasons
    assert unstable.event.state == "abstained"
    assert "counterfactual_instability" in unstable.event.abstention_reasons


def test_existing_abstention_is_preserved_and_never_upgraded() -> None:
    result = SafetyGate().evaluate(shadow(reasons=("audio_missing",)), behavior_event())
    assert result.event.state == "abstained"
    assert result.event.action == "local_record_only"
    assert result.event.quality == 0.0


def test_high_shadow_score_is_screening_only_and_carries_human_review_candidate() -> None:
    result = SafetyGate().evaluate(
        shadow(score=0.9), behavior_event(), sensitive_expression="我不想活了"
    )
    assert result.event.state == "screening_concern"
    assert result.human_review_required is True
    assert result.external_dispatch_allowed is False
    assert result.sensitive_expression is not None
    assert result.sensitive_expression.requires_human_review is True
