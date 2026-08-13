"""PACE-Behavior: conservative personalized passive-wellbeing evidence.

The behavior lane is deliberately rules-first.  It consumes daily aggregates,
never raw audio/video, freezes anomalous observations out of the personal
baseline, and emits a local-only invitation candidate only after persistence.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from hashlib import sha256
from math import isfinite
from statistics import median
from types import MappingProxyType
from typing import Mapping

from mental.contracts import DailyWellbeingObservation, DeliveryScope, WellbeingAssessmentEvent


@dataclass(frozen=True)
class PACEBehaviorConfig:
    min_effective_days: int = 14
    provisional_days: int = 7
    persistence_window: int = 3
    persistence_required: int = 2
    min_quality: float = 0.5
    model_version: str = "pace_behavior_v1"

    def __post_init__(self) -> None:
        integer_fields = (
            ("min_effective_days", self.min_effective_days),
            ("provisional_days", self.provisional_days),
            ("persistence_window", self.persistence_window),
            ("persistence_required", self.persistence_required),
        )
        for name, value in integer_fields:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_effective_days < 14:
            raise ValueError("min_effective_days must be >= 14")
        if self.provisional_days > self.min_effective_days:
            raise ValueError("provisional_days must not exceed min_effective_days")
        if self.persistence_required > self.persistence_window:
            raise ValueError("persistence_required must not exceed persistence_window")
        if isinstance(self.min_quality, bool) or not isinstance(self.min_quality, (int, float)):
            raise ValueError("min_quality must be a finite number")
        quality = float(self.min_quality)
        if not isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("min_quality must be in [0, 1]")
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("model_version must be a non-empty string")
        object.__setattr__(self, "min_quality", quality)
        object.__setattr__(self, "model_version", self.model_version.strip())


class PACEBehaviorModel:
    """Personalized, abstaining behavior gate for one elderly subject."""

    _FEATURES = ("sleep_hours", "sleep_regularity", "steps")
    _DOMAINS = {
        "sleep_hours": "sleep",
        "sleep_regularity": "sleep",
        "steps": "activity",
    }

    def __init__(self, subject_alias: str, config: PACEBehaviorConfig | None = None) -> None:
        if not isinstance(subject_alias, str) or not subject_alias.strip():
            raise ValueError("subject_alias must be a non-empty string")
        self.subject_alias = subject_alias.strip()
        self.config = config or PACEBehaviorConfig()
        self._history: dict[str, list[float]] = {key: [] for key in self._FEATURES}
        self._recent_evidence: list[frozenset[str]] = []
        self._last_day: date | None = None

    @property
    def effective_days(self) -> Mapping[str, int]:
        """Return detached per-feature counts; callers cannot mutate state."""
        return MappingProxyType({key: len(values) for key, values in self._history.items() if values})

    def update(self, observation: DailyWellbeingObservation) -> WellbeingAssessmentEvent:
        if not isinstance(observation, DailyWellbeingObservation):
            raise ValueError("observation must be DailyWellbeingObservation")
        if observation.subject_alias != self.subject_alias:
            raise ValueError("observation subject_alias does not match model")
        if self._last_day is not None and observation.local_day <= self._last_day:
            raise ValueError("observations must be strictly chronological by local_day")

        values = self._values(observation)
        timestamp = datetime.combine(observation.local_day, time.min, tzinfo=timezone.utc)
        if observation.excluded_reason:
            self._last_day = observation.local_day
            return self._event(
                observation,
                timestamp,
                state="abstained",
                action="local_record_only",
                coverage=0.0,
                quality=0.0,
                uncertainty=1.0,
                evidence_domains=(),
                evidence_codes=(),
                abstention_reasons=(f"excluded:{observation.excluded_reason.strip()}",),
            )

        provided = [key for key in self._FEATURES if getattr(observation, key) is not None]
        low_quality = [key for key in provided if float(observation.feature_quality.get(key, 0.0)) < self.config.min_quality]
        if low_quality:
            self._last_day = observation.local_day
            return self._event(
                observation,
                timestamp,
                state="abstained",
                action="local_record_only",
                coverage=len(values) / len(self._FEATURES),
                quality=min((float(observation.feature_quality.get(key, 0.0)) for key in provided), default=0.0),
                uncertainty=1.0,
                evidence_domains=tuple(sorted({self._DOMAINS[key] for key in values})),
                evidence_codes=(),
                abstention_reasons=("low_quality", *[f"low_quality:{key}" for key in low_quality]),
            )
        if not values:
            self._last_day = observation.local_day
            return self._event(
                observation,
                timestamp,
                state="abstained",
                action="local_record_only",
                coverage=0.0,
                quality=0.0,
                uncertainty=1.0,
                evidence_domains=(),
                evidence_codes=(),
                abstention_reasons=("no_valid_observation",),
            )

        evidence = frozenset(self._detect_anomalies(values))
        baseline_frozen = bool(evidence)
        if not baseline_frozen:
            for key, value in values.items():
                self._history[key].append(value)
        self._recent_evidence.append(evidence)
        self._recent_evidence = self._recent_evidence[-self.config.persistence_window :]
        self._last_day = observation.local_day

        baseline_state = self._baseline_state()
        sustained_keys = {
            key
            for item in self._recent_evidence
            for key in item
            if sum(key in recent for recent in self._recent_evidence) >= self.config.persistence_required
        }
        sustained = baseline_state == "operational" and bool(sustained_keys)
        quality = sum(float(observation.feature_quality.get(key, 0.0)) for key in values) / len(values)
        uncertainty = min(1.0, (1.0 - quality) + (0.25 if baseline_state != "operational" else 0.0))
        codes = tuple(sorted(f"behavior:{key}" for key in evidence))
        if baseline_frozen:
            codes += ("baseline_frozen_on_anomaly",)
        if not codes:
            codes = ("stable_baseline" if baseline_state == "operational" else "baseline_forming",)
        domains = tuple(sorted({self._DOMAINS[key] for key in evidence}))
        return self._event(
            observation,
            timestamp,
            state="invite_candidate" if sustained else ("baseline_forming" if baseline_state == "cold_start" else "observe"),
            action="local_invite_short_checkin" if sustained else "local_record_only",
            coverage=len(values) / len(self._FEATURES),
            quality=quality,
            uncertainty=uncertainty,
            evidence_domains=domains,
            evidence_codes=codes,
            abstention_reasons=(),
            baseline_state=baseline_state,
        )
    def _values(self, observation: DailyWellbeingObservation) -> dict[str, float]:
        values: dict[str, float] = {}
        for key in self._FEATURES:
            value = getattr(observation, key)
            if value is None:
                continue
            numeric = float(value)
            if not isfinite(numeric):
                raise ValueError(f"{key} must be finite")
            if float(observation.feature_quality.get(key, 0.0)) >= self.config.min_quality:
                values[key] = numeric
        return values

    def _baseline_state(self) -> str:
        counts = [len(values) for values in self._history.values() if values]
        if not counts:
            return "cold_start"
        if min(counts) >= self.config.min_effective_days:
            return "operational"
        if max(counts) >= self.config.provisional_days:
            return "provisional"
        return "cold_start"

    def _detect_anomalies(self, values: Mapping[str, float]) -> tuple[str, ...]:
        anomalies: list[str] = []
        for key, value in values.items():
            history = self._history[key]
            if len(history) < self.config.min_effective_days:
                continue
            baseline = median(history[-self.config.min_effective_days :])
            deviation = abs(value - baseline)
            if key == "steps":
                changed = value < baseline * 0.7
            elif key == "sleep_hours":
                changed = deviation >= 1.5
            else:
                changed = value < baseline - 0.2
            if changed:
                anomalies.append(key)
        return tuple(anomalies)

    def _event(
        self,
        observation: DailyWellbeingObservation,
        timestamp: datetime,
        *,
        state: str,
        action: str,
        coverage: float,
        quality: float,
        uncertainty: float,
        evidence_domains: tuple[str, ...],
        evidence_codes: tuple[str, ...],
        abstention_reasons: tuple[str, ...],
        baseline_state: str | None = None,
    ) -> WellbeingAssessmentEvent:
        digest = sha256(f"{self.subject_alias}|{observation.local_day.isoformat()}|{self.config.model_version}".encode("utf-8")).hexdigest()[:16]
        return WellbeingAssessmentEvent(
            event_id=f"pace-{digest}",
            subject_alias=self.subject_alias,
            timestamp=timestamp,
            state=state,
            action=action,
            evidence_domains=evidence_domains,
            evidence_codes=evidence_codes,
            coverage=coverage,
            quality=quality,
            uncertainty=uncertainty,
            baseline_state=baseline_state or self._baseline_state(),
            abstention_reasons=abstention_reasons,
            consent_scope="aggregate_only",
            model_version=self.config.model_version,
            model_mode="rules_v1",
            delivery_scope=DeliveryScope.EXTERNAL_FORBIDDEN,
            is_diagnosis=False,
            fall_critical_eligible=False,
            demo=observation.demo,
        )
