"""Reliability, abstention, and human-review gate for shadow evidence."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from math import isfinite

from mental.contracts import SensitiveExpressionCandidate, WellbeingAssessmentEvent
from mental.voluntary_shadow import ShadowAssessment


@dataclass(frozen=True)
class SafetyGateConfig:
    screening_threshold: float = 0.75
    min_coverage: float = 0.5
    min_counterfactual_stability: float = 0.8
    model_version: str = "research_shadow_v1"

    def __post_init__(self) -> None:
        for name in ("screening_threshold", "min_coverage", "min_counterfactual_stability"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("model_version must be a non-empty string")
        object.__setattr__(self, "model_version", self.model_version.strip())


@dataclass(frozen=True)
class SafetyGateResult:
    event: WellbeingAssessmentEvent
    promoted: bool
    human_review_required: bool
    external_dispatch_allowed: bool
    sensitive_expression: SensitiveExpressionCandidate | None = None


class SafetyGate:
    _SELF_HARM_MARKERS = ("不想活", "想死", "自杀", "结束生命", "活着没意思")

    def __init__(self, config: SafetyGateConfig | None = None) -> None:
        self.config = config or SafetyGateConfig()

    def evaluate(
        self,
        evidence: ShadowAssessment,
        behavior_event: WellbeingAssessmentEvent,
        *,
        sensitive_expression: str | None = None,
    ) -> SafetyGateResult:
        if not isinstance(evidence, ShadowAssessment) or not isinstance(behavior_event, WellbeingAssessmentEvent):
            raise ValueError("evidence and behavior_event have invalid types")
        reasons = list(evidence.abstention_reasons)
        if evidence.model_version != self.config.model_version:
            reasons.append("model_version_mismatch")
        if evidence.coverage < self.config.min_coverage:
            reasons.append("coverage_insufficient")
        if evidence.counterfactual_stability is not None and evidence.counterfactual_stability < self.config.min_counterfactual_stability:
            reasons.append("counterfactual_instability")
        if behavior_event.abstained:
            reasons.append("behavior_abstained")
        candidate = self._sensitive_candidate(behavior_event, sensitive_expression)
        if reasons or evidence.combined_score is None:
            return self._result(behavior_event, evidence, "abstained", reasons or ["evidence_unavailable"], candidate)
        state = "screening_concern" if evidence.combined_score >= self.config.screening_threshold else "observe"
        return self._result(behavior_event, evidence, state, (), candidate)

    def _result(
        self,
        behavior_event: WellbeingAssessmentEvent,
        evidence: ShadowAssessment,
        state: str,
        reasons: list[str] | tuple[str, ...],
        candidate: SensitiveExpressionCandidate | None,
    ) -> SafetyGateResult:
        human_review = candidate is not None
        action = "enqueue_human_review" if human_review else "local_record_only"
        event = WellbeingAssessmentEvent(
            event_id=f"shadow-{evidence.session_id}",
            subject_alias=evidence.subject_alias,
            timestamp=behavior_event.timestamp,
            state=state,
            action=action,
            evidence_domains=evidence.evidence_domains,
            evidence_codes=tuple(dict.fromkeys((*evidence.evidence_codes, *reasons))),
            coverage=0.0 if state == "abstained" else evidence.coverage,
            quality=0.0 if state == "abstained" else max(0.0, 1.0 - evidence.uncertainty),
            uncertainty=1.0 if state == "abstained" else evidence.uncertainty,
            baseline_state=behavior_event.baseline_state,
            abstention_reasons=tuple(dict.fromkeys(reasons)),
            consent_scope="voluntary_checkin",
            model_version=evidence.model_version,
            model_mode="research_shadow",
            is_diagnosis=False,
            fall_critical_eligible=False,
            demo=behavior_event.demo,
        )
        return SafetyGateResult(
            event=event,
            promoted=False,
            human_review_required=human_review,
            external_dispatch_allowed=False,
            sensitive_expression=candidate,
        )

    def _sensitive_candidate(self, behavior_event: WellbeingAssessmentEvent, text: str | None) -> SensitiveExpressionCandidate | None:
        if not isinstance(text, str) or not text.strip() or not any(marker in text for marker in self._SELF_HARM_MARKERS):
            return None
        return SensitiveExpressionCandidate(
            event_id=f"review-{behavior_event.event_id}",
            subject_alias=behavior_event.subject_alias,
            timestamp=behavior_event.timestamp,
            source="direct_typed_text",
            transcript_excerpt=text.strip()[:200],
            consent_scope="voluntary_checkin",
            retention_expiry=behavior_event.timestamp + timedelta(days=7),
        )
