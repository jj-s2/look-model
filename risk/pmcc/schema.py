"""Strict, JSON-safe records for the uncertainty-aware PMCC pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping


class EvidenceTier(str, Enum):
    REAL_DEVICE_LONGITUDINAL = "real_device_longitudinal"
    REAL_PUBLIC = "real_public"
    SYNTHETIC_RESEARCH = "synthetic_research"
    OFFLINE_FIXTURE = "offline_fixture"


class BaselineState(str, Enum):
    POPULATION_ONLY = "population_only"
    BLENDED = "blended"
    PERSONAL = "personal"


class DecisionBand(str, Enum):
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"
    VERY_HIGH = "very_high"


class OutcomeType(str, Enum):
    CONFIRMED_FALL = "confirmed_fall"
    NEAR_FALL = "near_fall"
    NORMAL_ADL = "normal_adl"
    FALSE_ALARM = "false_alarm"
    UNKNOWN = "unknown"


def _subject_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("subject_id must be a non-empty string")
    return value


def _timestamp(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value


def _parse_timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO-8601 timestamp string")
    try:
        return _timestamp(datetime.fromisoformat(value), name)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 timestamp") from error


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite numeric value")
    return float(value)


def _unit_interval(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return number


def _mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _normalize_json_value(value: Any, name: str, active: set[int] | None = None) -> Any:
    """Validate provenance and retain an immutable, JSON-safe representation."""
    active = set() if active is None else active
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{name} must not contain non-finite floats")
        return value
    if isinstance(value, datetime):
        return _timestamp(value, name).isoformat()
    if isinstance(value, Enum):
        return _normalize_json_value(value.value, name, active)
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{name} must not contain cyclic mappings")
        active.add(identity)
        try:
            normalized = {}
            for key, nested in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{name} mapping keys must be strings")
                normalized[key] = _normalize_json_value(nested, name, active)
            return MappingProxyType(normalized)
        finally:
            active.remove(identity)
    if isinstance(value, (tuple, list)):
        identity = id(value)
        if identity in active:
            raise ValueError(f"{name} must not contain cyclic sequences")
        active.add(identity)
        try:
            return tuple(_normalize_json_value(item, name, active) for item in value)
        finally:
            active.remove(identity)
    raise ValueError(f"{name} contains unsupported value type {type(value).__name__}")


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _provenance(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _normalize_json_value(_mapping(value, "provenance"), "provenance")


def _required(data: Mapping[str, Any], field: str) -> Any:
    if field not in data:
        raise ValueError(f"data requires {field}")
    return data[field]


def _named_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    copied = _mapping(value, name)
    if any(not isinstance(key, str) or not key for key in copied):
        raise ValueError(f"{name} keys must be non-empty strings")
    return copied


def _enum(value: Any, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a valid {enum_type.__name__}")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a valid {enum_type.__name__}") from error


@dataclass(frozen=True)
class DailyObservation:
    subject_id: str
    observed_at: datetime
    features: Mapping[str, float | None]
    quality: Mapping[str, float]
    availability: Mapping[str, bool]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", _subject_id(self.subject_id))
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at, "observed_at"))
        features = _named_mapping(self.features, "features")
        for key, value in features.items():
            if value is not None:
                features[key] = _finite_number(value, f"features[{key!r}]")
        quality = _named_mapping(self.quality, "quality")
        for key, value in quality.items():
            quality[key] = _unit_interval(value, f"quality[{key!r}]")
        availability = _named_mapping(self.availability, "availability")
        for key, value in availability.items():
            if not isinstance(value, bool):
                raise ValueError(f"availability[{key!r}] must be a boolean")
        for key, value in quality.items():
            if availability.get(key) is False and value != 0.0:
                raise ValueError(f"quality[{key!r}] must be zero when unavailable")
        object.__setattr__(self, "features", MappingProxyType(features))
        object.__setattr__(self, "quality", MappingProxyType(quality))
        object.__setattr__(self, "availability", MappingProxyType(availability))
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "observed_at": self.observed_at.isoformat(),
            "features": dict(self.features),
            "quality": dict(self.quality),
            "availability": dict(self.availability),
            "provenance": _json_value(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DailyObservation":
        copied = _mapping(data, "data")
        return cls(
            subject_id=copied.get("subject_id"),
            observed_at=_parse_timestamp(_required(copied, "observed_at"), "observed_at"),
            features=_required(copied, "features"),
            quality=_required(copied, "quality"),
            availability=_required(copied, "availability"),
            provenance=copied.get("provenance", {}),
        )


@dataclass(frozen=True)
class ChangeEvent:
    subject_id: str
    occurred_at: datetime
    feature: str
    previous_value: float | None
    current_value: float | None
    confidence: float
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", _subject_id(self.subject_id))
        object.__setattr__(self, "occurred_at", _timestamp(self.occurred_at, "occurred_at"))
        if not isinstance(self.feature, str) or not self.feature:
            raise ValueError("feature must be a non-empty string")
        for name in ("previous_value", "current_value"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_number(value, name))
        object.__setattr__(self, "confidence", _unit_interval(self.confidence, "confidence"))
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        return {"subject_id": self.subject_id, "occurred_at": self.occurred_at.isoformat(), "feature": self.feature,
                "previous_value": self.previous_value, "current_value": self.current_value,
                "confidence": self.confidence, "provenance": _json_value(self.provenance)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChangeEvent":
        copied = _mapping(data, "data")
        return cls(copied.get("subject_id"), _parse_timestamp(_required(copied, "occurred_at"), "occurred_at"),
                   copied.get("feature"), copied.get("previous_value"), copied.get("current_value"),
                   copied.get("confidence"), copied.get("provenance", {}))


@dataclass(frozen=True)
class TemporalChain:
    subject_id: str
    created_at: datetime
    observations: tuple[DailyObservation, ...] = ()
    changes: tuple[ChangeEvent, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", _subject_id(self.subject_id))
        object.__setattr__(self, "created_at", _timestamp(self.created_at, "created_at"))
        observations = tuple(self.observations)
        changes = tuple(self.changes)
        if not all(isinstance(item, DailyObservation) for item in observations):
            raise ValueError("observations must contain DailyObservation records")
        if not all(isinstance(item, ChangeEvent) for item in changes):
            raise ValueError("changes must contain ChangeEvent records")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "changes", changes)
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        return {"subject_id": self.subject_id, "created_at": self.created_at.isoformat(),
                "observations": [item.to_dict() for item in self.observations],
                "changes": [item.to_dict() for item in self.changes], "provenance": _json_value(self.provenance)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TemporalChain":
        copied = _mapping(data, "data")
        return cls(copied.get("subject_id"), _parse_timestamp(_required(copied, "created_at"), "created_at"),
                   tuple(DailyObservation.from_dict(item) for item in copied.get("observations", ())),
                   tuple(ChangeEvent.from_dict(item) for item in copied.get("changes", ())),
                   copied.get("provenance", {}))


@dataclass(frozen=True)
class PMCCForecast:
    subject_id: str
    forecast_at: datetime
    horizon_days: int
    risk_score: float
    decision_band: DecisionBand
    confidence: float
    baseline_state: BaselineState
    temporal_chain: TemporalChain | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", _subject_id(self.subject_id))
        object.__setattr__(self, "forecast_at", _timestamp(self.forecast_at, "forecast_at"))
        if isinstance(self.horizon_days, bool) or not isinstance(self.horizon_days, int) or self.horizon_days < 1:
            raise ValueError("horizon_days must be a positive integer")
        object.__setattr__(self, "risk_score", _unit_interval(self.risk_score, "risk_score"))
        object.__setattr__(self, "confidence", _unit_interval(self.confidence, "confidence"))
        object.__setattr__(self, "decision_band", _enum(self.decision_band, DecisionBand, "decision_band"))
        object.__setattr__(self, "baseline_state", _enum(self.baseline_state, BaselineState, "baseline_state"))
        if self.temporal_chain is not None and not isinstance(self.temporal_chain, TemporalChain):
            raise ValueError("temporal_chain must be a TemporalChain or None")
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        return {"subject_id": self.subject_id, "forecast_at": self.forecast_at.isoformat(),
                "horizon_days": self.horizon_days, "risk_score": self.risk_score,
                "decision_band": self.decision_band.value, "confidence": self.confidence,
                "baseline_state": self.baseline_state.value,
                "temporal_chain": None if self.temporal_chain is None else self.temporal_chain.to_dict(),
                "provenance": _json_value(self.provenance)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PMCCForecast":
        copied = _mapping(data, "data")
        chain = copied.get("temporal_chain")
        return cls(copied.get("subject_id"), _parse_timestamp(_required(copied, "forecast_at"), "forecast_at"),
                   copied.get("horizon_days"), copied.get("risk_score"), copied.get("decision_band"),
                   copied.get("confidence"), copied.get("baseline_state"),
                   None if chain is None else TemporalChain.from_dict(chain), copied.get("provenance", {}))


@dataclass(frozen=True)
class OutcomeFeedback:
    subject_id: str
    recorded_at: datetime
    outcome: OutcomeType
    confidence: float
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_id", _subject_id(self.subject_id))
        object.__setattr__(self, "recorded_at", _timestamp(self.recorded_at, "recorded_at"))
        object.__setattr__(self, "outcome", _enum(self.outcome, OutcomeType, "outcome"))
        object.__setattr__(self, "confidence", _unit_interval(self.confidence, "confidence"))
        object.__setattr__(self, "provenance", _provenance(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        return {"subject_id": self.subject_id, "recorded_at": self.recorded_at.isoformat(),
                "outcome": self.outcome.value, "confidence": self.confidence,
                "provenance": _json_value(self.provenance)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OutcomeFeedback":
        copied = _mapping(data, "data")
        return cls(copied.get("subject_id"), _parse_timestamp(_required(copied, "recorded_at"), "recorded_at"),
                   copied.get("outcome"), copied.get("confidence"), copied.get("provenance", {}))
