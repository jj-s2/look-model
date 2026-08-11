"""Strict, consent-aware contracts for the wellbeing screening lane."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Mapping

from core.events import DataQuality, EventType, SensorEvent, Source


class DeliveryScope(str, Enum):
    EXTERNAL_FORBIDDEN = "external_forbidden"
    LOCAL_ONLY = "local_only"
    INTERNAL_HUMAN_REVIEW_ONLY = "internal_human_review_only"


_STATES = {
    "baseline_forming", "observe", "invite_candidate", "screening_concern",
    "abstained", "human_review_required",
}
_ACTIONS = {
    "local_record_only", "local_invite_short_checkin", "local_offer_gds",
    "enqueue_human_review",
}
_SOURCES = {"direct_typed_text", "direct_button", "asr_hypothesis"}


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _optional_finite(value: object | None, field_name: str) -> float | None:
    return None if value is None else _finite_number(value, field_name)


def _bounded(value: object, field_name: str, lower: float = 0.0, upper: float = 1.0) -> float:
    result = _finite_number(value, field_name)
    if not lower <= result <= upper:
        raise ValueError(f"{field_name} must be in [{lower:g}, {upper:g}]")
    return result


def _optional_bounded(value: object | None, field_name: str) -> float | None:
    return None if value is None else _bounded(value, field_name)


def _timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _alias(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _string_tuple(values: object, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a sequence of strings")
    try:
        result = tuple(values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{field_name} must be a sequence of strings") from error
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return tuple(value.strip() for value in result)


@dataclass(frozen=True)
class DailyWellbeingObservation:
    subject_alias: str
    local_day: date
    sleep_hours: float | None = None
    sleep_regularity: float | None = None
    steps: float | None = None
    feature_quality: Mapping[str, float] = field(default_factory=dict)
    excluded_reason: str | None = None
    provenance_ids: tuple[str, ...] = ()
    demo: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_alias", _alias(self.subject_alias, "subject_alias"))
        if not isinstance(self.local_day, date) or isinstance(self.local_day, datetime):
            raise ValueError("local_day must be a date")
        sleep = _optional_finite(self.sleep_hours, "sleep_hours")
        if sleep is not None and not 0.0 <= sleep <= 24.0:
            raise ValueError("sleep_hours must be in [0, 24]")
        regularity = _optional_bounded(self.sleep_regularity, "sleep_regularity")
        steps = _optional_finite(self.steps, "steps")
        if steps is not None and steps < 0.0:
            raise ValueError("steps must be non-negative")
        if not isinstance(self.feature_quality, Mapping):
            raise ValueError("feature_quality must be a mapping")
        quality: dict[str, float] = {}
        for key, value in self.feature_quality.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("feature_quality keys must be non-empty strings")
            quality[key.strip()] = _bounded(value, "feature_quality value")
        if not isinstance(self.demo, bool):
            raise ValueError("demo must be boolean")
        object.__setattr__(self, "sleep_hours", sleep)
        object.__setattr__(self, "sleep_regularity", regularity)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "feature_quality", MappingProxyType(quality))
        object.__setattr__(self, "provenance_ids", _string_tuple(self.provenance_ids, "provenance_ids"))
        if self.excluded_reason is not None and not isinstance(self.excluded_reason, str):
            raise ValueError("excluded_reason must be a string or None")


@dataclass(frozen=True)
class VoluntaryCheckin:
    session_id: str
    subject_alias: str
    started_at: datetime
    consented: bool
    prompt_ids: tuple[str, ...]
    answers: Mapping[str, str]
    transcript: str | None = None
    asr_confidence: float | None = None
    audio_quality: float | None = None
    model_version: str = "unavailable"
    demo: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _alias(self.session_id, "session_id"))
        object.__setattr__(self, "subject_alias", _alias(self.subject_alias, "subject_alias"))
        _timestamp(self.started_at, "started_at")
        if self.consented is not True:
            raise ValueError("voluntary check-in requires explicit consent")
        prompts = _string_tuple(self.prompt_ids, "prompt_ids")
        if not 1 <= len(prompts) <= 3:
            raise ValueError("prompt_ids must contain one to three prompts")
        if not isinstance(self.answers, Mapping):
            raise ValueError("answers must be a mapping")
        answers: dict[str, str] = {}
        for key, value in self.answers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("answers must map strings to strings")
            answers[key.strip()] = value.strip()
        if self.transcript is not None and not isinstance(self.transcript, str):
            raise ValueError("transcript must be a string or None")
        object.__setattr__(self, "prompt_ids", prompts)
        object.__setattr__(self, "answers", MappingProxyType(answers))
        object.__setattr__(self, "asr_confidence", _optional_bounded(self.asr_confidence, "asr_confidence"))
        object.__setattr__(self, "audio_quality", _optional_bounded(self.audio_quality, "audio_quality"))
        object.__setattr__(self, "model_version", _alias(self.model_version, "model_version"))
        if not isinstance(self.demo, bool):
            raise ValueError("demo must be boolean")


@dataclass(frozen=True)
class ModalityEvidence:
    modality: str
    score: float | None
    available: bool
    reliability: float
    uncertainty: float
    evidence_codes: tuple[str, ...] = ()
    episode_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "modality", _alias(self.modality, "modality"))
        object.__setattr__(self, "score", _optional_bounded(self.score, "score"))
        if not isinstance(self.available, bool):
            raise ValueError("available must be boolean")
        object.__setattr__(self, "reliability", _bounded(self.reliability, "reliability"))
        object.__setattr__(self, "uncertainty", _bounded(self.uncertainty, "uncertainty"))
        object.__setattr__(self, "evidence_codes", _string_tuple(self.evidence_codes, "evidence_codes"))
        if self.episode_id is not None:
            object.__setattr__(self, "episode_id", _alias(self.episode_id, "episode_id"))


@dataclass(frozen=True)
class WellbeingAssessmentEvent:
    event_id: str
    subject_alias: str
    timestamp: datetime
    state: str
    action: str
    evidence_domains: tuple[str, ...]
    evidence_codes: tuple[str, ...]
    coverage: float
    quality: float
    uncertainty: float
    baseline_state: str
    abstention_reasons: tuple[str, ...] = ()
    consent_scope: str = "aggregate_only"
    model_version: str = "rules_v1"
    model_mode: str = "rules_v1"
    delivery_scope: DeliveryScope = DeliveryScope.EXTERNAL_FORBIDDEN
    is_diagnosis: bool = False
    fall_critical_eligible: bool = False
    demo: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _alias(self.event_id, "event_id"))
        object.__setattr__(self, "subject_alias", _alias(self.subject_alias, "subject_alias"))
        _timestamp(self.timestamp, "timestamp")
        object.__setattr__(self, "state", _alias(self.state, "state"))
        object.__setattr__(self, "action", _alias(self.action, "action"))
        if self.state not in _STATES:
            raise ValueError("invalid wellbeing state")
        if self.action not in _ACTIONS:
            raise ValueError("invalid wellbeing action")
        object.__setattr__(self, "evidence_domains", _string_tuple(self.evidence_domains, "evidence_domains"))
        object.__setattr__(self, "evidence_codes", _string_tuple(self.evidence_codes, "evidence_codes"))
        object.__setattr__(self, "abstention_reasons", _string_tuple(self.abstention_reasons, "abstention_reasons"))
        object.__setattr__(self, "coverage", _bounded(self.coverage, "coverage"))
        object.__setattr__(self, "quality", _bounded(self.quality, "quality"))
        object.__setattr__(self, "uncertainty", _bounded(self.uncertainty, "uncertainty"))
        object.__setattr__(self, "baseline_state", _alias(self.baseline_state, "baseline_state"))
        object.__setattr__(self, "consent_scope", _alias(self.consent_scope, "consent_scope"))
        object.__setattr__(self, "model_version", _alias(self.model_version, "model_version"))
        object.__setattr__(self, "model_mode", _alias(self.model_mode, "model_mode"))
        try:
            scope = self.delivery_scope if isinstance(self.delivery_scope, DeliveryScope) else DeliveryScope(self.delivery_scope)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid delivery_scope") from error
        object.__setattr__(self, "delivery_scope", scope)
        if self.is_diagnosis is not False or self.fall_critical_eligible is not False:
            raise ValueError("wellbeing events cannot be diagnostic or fall-critical")
        if not isinstance(self.demo, bool):
            raise ValueError("demo must be boolean")

    @property
    def abstained(self) -> bool:
        return self.state == "abstained"

    def to_sensor_event(self) -> SensorEvent:
        payload = {
            "event_id": self.event_id,
            "subject_id": self.subject_alias,
            "state": self.state,
            "action": self.action,
            "delivery_scope": self.delivery_scope.value,
            "evidence_domains": list(self.evidence_domains),
            "evidence_codes": list(self.evidence_codes),
            "coverage": self.coverage,
            "quality": self.quality,
            "uncertainty": self.uncertainty,
            "baseline_state": self.baseline_state,
            "abstention_reasons": list(self.abstention_reasons),
            "consent_scope": self.consent_scope,
            "model_version": self.model_version,
            "model_mode": self.model_mode,
            "is_diagnosis": False,
            "fall_critical_eligible": False,
            "sustained_change": self.state in {"invite_candidate", "screening_concern"},
        }
        return SensorEvent(
            timestamp=self.timestamp,
            source=Source.SCREENING,
            event_type=EventType.WELLBEING_CHANGE,
            payload=payload,
            quality=DataQuality(
                available=not self.abstained,
                confidence=self.quality,
                demo=self.demo,
                reason=self.abstention_reasons[0] if self.abstention_reasons else None,
            ),
        )


@dataclass(frozen=True)
class SensitiveExpressionCandidate:
    event_id: str
    subject_alias: str
    timestamp: datetime
    source: str
    transcript_excerpt: str
    asr_confidence: float | None = None
    consent_scope: str = "voluntary_checkin"
    retention_expiry: datetime | None = None
    requires_human_review: bool = True
    external_dispatch_allowed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _alias(self.event_id, "event_id"))
        object.__setattr__(self, "subject_alias", _alias(self.subject_alias, "subject_alias"))
        _timestamp(self.timestamp, "timestamp")
        object.__setattr__(self, "source", _alias(self.source, "source"))
        if self.source not in _SOURCES:
            raise ValueError("invalid sensitive-expression source")
        if not isinstance(self.transcript_excerpt, str) or not self.transcript_excerpt.strip():
            raise ValueError("transcript_excerpt must be non-empty")
        object.__setattr__(self, "asr_confidence", _optional_bounded(self.asr_confidence, "asr_confidence"))
        object.__setattr__(self, "consent_scope", _alias(self.consent_scope, "consent_scope"))
        if self.retention_expiry is not None:
            _timestamp(self.retention_expiry, "retention_expiry")
            if self.retention_expiry < self.timestamp:
                raise ValueError("retention_expiry must not precede timestamp")
        if self.requires_human_review is not True or self.external_dispatch_allowed is not False:
            raise ValueError("sensitive expressions require human review and forbid external dispatch")
