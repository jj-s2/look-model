"""One-step, local-only orchestration for monitoring demo components."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol, Sequence

from alerts.dispatcher import AlertDispatcher
from core.events import EventType, SensorEvent, Source
from fusion.decision_engine import DecisionEngine, RiskDecision
from storage.clip_buffer import CircularClipBuffer


class EventSource(Protocol):
    name: str

    def poll(self, now: datetime) -> Sequence[SensorEvent]:
        """Return currently available events without performing required network I/O."""


@dataclass(frozen=True)
class ServiceSnapshot:
    collected_at: datetime
    camera_health: str
    radar_health: str
    decisions: tuple[RiskDecision, ...]
    alert_history: tuple[dict[str, object], ...]
    demo: bool
    latest_frame: object | None = None
    source_errors: tuple[str, ...] = ()


class LiveMonitoringService:
    """Run independent local sources once, isolating failures to their component."""

    def __init__(
        self,
        event_sources: Sequence[EventSource] = (),
        *,
        decision_engine: DecisionEngine | None = None,
        dispatcher: AlertDispatcher | None = None,
        clip_buffer: CircularClipBuffer | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.event_sources = tuple(event_sources)
        self.decision_engine = decision_engine or DecisionEngine()
        self.dispatcher = dispatcher
        self.clip_buffer = clip_buffer
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.last_snapshot: ServiceSnapshot | None = None

    def step(self) -> ServiceSnapshot:
        """Collect one bounded iteration; one faulty source cannot halt other modules."""
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        events: list[SensorEvent] = []
        source_state: dict[str, str] = {"camera": "offline", "radar": "offline"}
        errors: list[str] = []
        for source in self.event_sources:
            name = getattr(source, "name", "source")
            try:
                batch = source.poll(now)
            except Exception:
                source_state[name] = "degraded"
                errors.append(f"{name}: unavailable")
                continue
            events.extend(batch)
            source_state[name] = self._health_from_events(name, batch)
        decisions = tuple(self.decision_engine.evaluate(events, now))
        if self.dispatcher is not None:
            for decision in decisions:
                try:
                    self.dispatcher.dispatch(decision)
                except Exception:
                    errors.append("alerts: unavailable")
        if self.clip_buffer is not None:
            for decision in decisions:
                if decision.kind == "fall_event" and not decision.recovery_confirmed:
                    self.clip_buffer.confirm_event(self._event_id(decision))
        history = tuple(self.dispatcher.recent_alerts(limit=20)) if self.dispatcher is not None else ()
        snapshot = ServiceSnapshot(
            collected_at=now,
            camera_health=source_state["camera"],
            radar_health=source_state["radar"],
            decisions=decisions,
            alert_history=history,
            demo=any(event.quality.demo for event in events),
            source_errors=tuple(errors),
        )
        self.last_snapshot = snapshot
        return snapshot

    @staticmethod
    def _health_from_events(name: str, events: Sequence[SensorEvent]) -> str:
        relevant = [event for event in events if event.source.value == name]
        if not relevant:
            return "healthy" if name == "camera" else "offline"
        return "healthy" if any(event.quality.available for event in relevant) else "degraded"

    @staticmethod
    def _event_id(decision: RiskDecision) -> str:
        timestamp = decision.timestamp.isoformat() if decision.timestamp else "unknown"
        return f"{decision.subject_id}-{timestamp}"
