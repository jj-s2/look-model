"""Timestamp-based event decoding for continuous RG-PCNet monitoring.

The decoder is intentionally independent from the legacy frame-count based
``FallStateMachine``.  It consumes calibrated frame decisions, gates all
state changes on reliability, and measures confirmation/recovery evidence in
seconds rather than in observations.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Final


_STATES: Final[frozenset[str]] = frozenset(
    {"monitoring", "suspected", "confirmed", "postfall", "recovered", "abstain"}
)
_TRIGGER_PHASE: Final[str] = "descent_or_impact"
_POSTFALL_PHASE: Final[str] = "postfall_or_recovery"


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _probability(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


@dataclass(frozen=True)
class EventDecoderConfig:
    """Frozen thresholds shared by offline replay and live decoding."""

    fall_threshold: float
    confirm_seconds: float
    recovery_seconds: float
    cooldown_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "fall_threshold", _probability(self.fall_threshold, "fall_threshold"))
        confirm = _finite_number(self.confirm_seconds, "confirm_seconds")
        recovery = _finite_number(self.recovery_seconds, "recovery_seconds")
        cooldown = _finite_number(self.cooldown_seconds, "cooldown_seconds")
        if confirm <= 0.0:
            raise ValueError("confirm_seconds must be greater than zero")
        if recovery <= 0.0:
            raise ValueError("recovery_seconds must be greater than zero")
        if cooldown < 0.0:
            raise ValueError("cooldown_seconds must be non-negative")
        object.__setattr__(self, "confirm_seconds", confirm)
        object.__setattr__(self, "recovery_seconds", recovery)
        object.__setattr__(self, "cooldown_seconds", cooldown)


@dataclass(frozen=True)
class FrameDecision:
    """One calibrated model decision presented to the decoder."""

    timestamp: float
    fall_probability: float
    phase: str
    reliable: bool

    def __post_init__(self) -> None:
        timestamp = _finite_number(self.timestamp, "timestamp")
        probability = _probability(self.fall_probability, "fall_probability")
        if not isinstance(self.phase, str) or not self.phase:
            raise ValueError("phase must be a non-empty string")
        if not isinstance(self.reliable, bool):
            raise ValueError("reliable must be a boolean")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "fall_probability", probability)


@dataclass(frozen=True)
class EventTransition:
    """Immutable externally visible state transition."""

    state: str
    event_id: str | None
    emitted: str | None
    reason: str
    timestamp: float

    def __post_init__(self) -> None:
        if self.state not in _STATES:
            raise ValueError(f"state must be one of {sorted(_STATES)}")
        if self.event_id is not None and not isinstance(self.event_id, str):
            raise ValueError("event_id must be a string or None")
        if self.emitted is not None and not isinstance(self.emitted, str):
            raise ValueError("emitted must be a string or None")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")
        object.__setattr__(self, "timestamp", _finite_number(self.timestamp, "timestamp"))


class EventDecoder:
    """Decode reliable frame decisions into one-shot fall events.

    ``update`` is deliberately the only mutating operation.  Each call must
    have a finite timestamp strictly later than the previous call.  A
    reliability abstention discards only unconfirmed evidence; an already
    confirmed event remains latched until recovery.
    """

    def __init__(self, config: EventDecoderConfig) -> None:
        if not isinstance(config, EventDecoderConfig):
            raise TypeError("config must be an EventDecoderConfig")
        self.config = config
        self._state = "monitoring"
        self._resume_state = "monitoring"
        self._last_timestamp: float | None = None
        self._suspicion_started_at: float | None = None
        self._recovery_started_at: float | None = None
        self._cooldown_until: float | None = None
        self._active_event_id: str | None = None
        self._sequence = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def active_event_id(self) -> str | None:
        return self._active_event_id

    def update(self, decision: FrameDecision) -> EventTransition:
        if not isinstance(decision, FrameDecision):
            raise TypeError("decision must be a FrameDecision")
        timestamp = decision.timestamp
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        self._last_timestamp = timestamp

        if not decision.reliable:
            return self._abstain(timestamp)

        if self._state == "abstain":
            self._state = self._resume_state

        if self._state == "recovered":
            if self._cooldown_until is not None and timestamp < self._cooldown_until:
                return self._transition("recovered", None, "cooldown_active", timestamp)
            self._state = "monitoring"
            self._cooldown_until = None

        trigger = self._is_trigger(decision)
        if self._state == "monitoring":
            if trigger:
                self._begin_suspicion(timestamp)
                return self._transition("suspected", None, "temporal_evidence_started", timestamp)
            return self._transition("monitoring", None, "monitoring", timestamp)

        if self._state == "suspected":
            if not trigger:
                self._clear_suspicion()
                self._state = "monitoring"
                return self._transition("monitoring", None, "evidence_interrupted", timestamp)
            assert self._suspicion_started_at is not None
            if self._duration_reached(timestamp, self._suspicion_started_at, self.config.confirm_seconds):
                self._state = "confirmed"
                self._recovery_started_at = None
                return self._transition("confirmed", "fall_confirmed", "confirmation_duration_reached", timestamp)
            return self._transition("suspected", None, "confirmation_pending", timestamp)

        if self._state in {"confirmed", "postfall"}:
            if decision.phase == _POSTFALL_PHASE:
                self._state = "postfall"
                self._recovery_started_at = None
                return self._transition("postfall", None, "postfall_evidence", timestamp)
            if self._is_recovery_frame(decision):
                if self._recovery_started_at is None:
                    self._recovery_started_at = timestamp
                    return self._transition(self._state, None, "recovery_started", timestamp)
                if self._duration_reached(timestamp, self._recovery_started_at, self.config.recovery_seconds):
                    self._state = "recovered"
                    self._cooldown_until = timestamp + self.config.cooldown_seconds
                    self._recovery_started_at = None
                    return self._transition("recovered", "fall_recovered", "recovery_duration_reached", timestamp)
                return self._transition(self._state, None, "recovery_pending", timestamp)
            self._recovery_started_at = None
            return self._transition(self._state, None, "event_latched", timestamp)

        # This branch is defensive: all public states are handled above.
        self._state = "monitoring"
        self._clear_suspicion()
        return self._transition("monitoring", None, "monitoring", timestamp)

    def _abstain(self, timestamp: float) -> EventTransition:
        if self._state in {"monitoring", "suspected"}:
            self._clear_suspicion()
            self._state = "abstain"
            self._resume_state = "monitoring"
            event_id = None
        else:
            self._resume_state = self._state
            self._state = "abstain"
            event_id = self._active_event_id
        return self._transition("abstain", None, "reliability_gate", timestamp, event_id=event_id)

    def _begin_suspicion(self, timestamp: float) -> None:
        self._state = "suspected"
        self._suspicion_started_at = timestamp
        self._recovery_started_at = None
        self._sequence += 1
        self._active_event_id = f"fall-{self._sequence:06d}"

    def _clear_suspicion(self) -> None:
        self._suspicion_started_at = None
        if self._state == "suspected":
            self._active_event_id = None

    def _is_trigger(self, decision: FrameDecision) -> bool:
        return decision.fall_probability >= self.config.fall_threshold or decision.phase == _TRIGGER_PHASE

    @staticmethod
    def _is_recovery_frame(decision: FrameDecision) -> bool:
        return decision.phase == "normal"

    @staticmethod
    def _duration_reached(timestamp: float, started_at: float, duration: float) -> bool:
        # Account only for a few ulps of binary representation; all timers
        # remain timestamp based and no frame-count approximation is used.
        return timestamp - started_at + 1e-12 >= duration

    def _transition(
        self,
        state: str,
        emitted: str | None,
        reason: str,
        timestamp: float,
        *,
        event_id: str | None = None,
    ) -> EventTransition:
        return EventTransition(
            state=state,
            event_id=self._active_event_id if event_id is None else event_id,
            emitted=emitted,
            reason=reason,
            timestamp=timestamp,
        )
