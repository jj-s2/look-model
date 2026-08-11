"""Bounded, local-only orchestration for monitoring components."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, Sequence

from alerts.dispatcher import AlertDispatcher
from core.events import EventType, SensorEvent, Source
from fusion.decision_engine import DecisionEngine, RiskDecision
from storage.clip_buffer import CircularClipBuffer
from storage.retention import RetentionPolicy


class EventSource(Protocol):
    name: str

    def poll(self, now: datetime) -> Sequence[SensorEvent] | "SourceBatch":
        """Return local source observations for one bounded service iteration."""


@dataclass(frozen=True)
class SourceBatch:
    """Source observations plus an optional current frame kept only in memory."""

    events: Sequence[SensorEvent]
    frame: object | None = None
    frame_timestamp: datetime | None = None


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
    events: tuple[SensorEvent, ...] = ()
    system_health: str = "healthy"


class LiveMonitoringService:
    """Run independent components through bounded worker calls.

    Timed-out workers are left in their boundary rather than blocking this loop;
    an in-flight source is not resubmitted until it completes, avoiding unbounded
    queued work if a device library hangs.
    """

    def __init__(
        self,
        event_sources: Sequence[EventSource] = (),
        *,
        decision_engine: DecisionEngine | None = None,
        dispatcher: AlertDispatcher | None = None,
        clip_buffer: CircularClipBuffer | None = None,
        retention_policy: RetentionPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
        poll_timeout_seconds: float = .25,
        component_timeout_seconds: float = .25,
    ) -> None:
        if poll_timeout_seconds <= 0 or component_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        self.event_sources = tuple(event_sources)
        self.decision_engine = decision_engine or DecisionEngine()
        self.dispatcher = dispatcher
        self.clip_buffer = clip_buffer
        self.retention_policy = retention_policy
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.poll_timeout_seconds = poll_timeout_seconds
        self.component_timeout_seconds = component_timeout_seconds
        self._executor = ThreadPoolExecutor(max_workers=max(4, len(self.event_sources) + 4), thread_name_prefix="monitor")
        self._source_futures: dict[int, Future[Any]] = {}
        self._incoming_event_ids: dict[tuple[str, datetime], str] = {}
        self.last_snapshot: ServiceSnapshot | None = None

    def close(self) -> None:
        """Release idle worker resources; callers may use this during app shutdown."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def step(self, incoming: Sequence[SensorEvent] | SourceBatch | None = None) -> ServiceSnapshot:
        """Collect one iteration without allowing a device or storage call to block it."""
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        events: list[SensorEvent] = []
        source_state: dict[str, str] = {"camera": "offline", "radar": "offline"}
        errors: list[str] = []
        latest_frame: object | None = None
        batch_records: list[tuple[str, SourceBatch, int]] = []
        if incoming is not None:
            try:
                batch, invalid_event_count = self._normalise_batch(incoming)
            except (TypeError, ValueError):
                errors.append("vision: invalid data")
            else:
                batch_records.append(("vision", batch, invalid_event_count))
        for source in () if incoming is not None else self.event_sources:
            name = str(getattr(source, "name", "source"))
            modality = self._modality_for(name)
            future = self._source_futures.get(id(source))
            if future is None:
                future = self._executor.submit(source.poll, now)
                self._source_futures[id(source)] = future
            try:
                raw_batch = future.result(timeout=self.poll_timeout_seconds)
            except TimeoutError:
                if modality is not None:
                    source_state[modality] = "degraded"
                errors.append(f"{name}: timeout")
                continue
            except Exception:
                self._source_futures.pop(id(source), None)
                if modality is not None:
                    source_state[modality] = "degraded"
                errors.append(f"{name}: unavailable")
                continue
            self._source_futures.pop(id(source), None)
            try:
                batch, invalid_event_count = self._normalise_batch(raw_batch)
            except (TypeError, ValueError):
                if modality is not None:
                    source_state[modality] = "degraded"
                errors.append(f"{name}: invalid data")
                continue
            batch_records.append((name, batch, invalid_event_count))
        for name, batch, invalid_event_count in batch_records:
            modality = self._modality_for(name)
            events.extend(batch.events)
            for event in batch.events:
                event_id = event.payload.get("event_id")
                subject = event.payload.get("subject_id", event.payload.get("tracking_id"))
                if isinstance(event_id, str) and event_id and isinstance(subject, str) and subject:
                    self._incoming_event_ids[(subject, event.timestamp)] = event_id
            if modality is not None:
                source_state[modality] = "degraded" if invalid_event_count else self._health_from_events(modality, batch.events)
            if invalid_event_count:
                errors.append(f"{name}: invalid data")
            if batch.frame is not None:
                latest_frame = batch.frame
                if self.clip_buffer is not None:
                    self._run_component(
                        "clip buffer", self.clip_buffer.on_frame, batch.frame, batch.frame_timestamp or now, errors=errors
                    )
        evaluated = self._run_component("decision engine", self.decision_engine.evaluate, events, now, errors=errors, default=[])
        decisions = tuple(evaluated)
        if self.dispatcher is not None:
            for decision in decisions:
                if self._should_dispatch(decision, events):
                    self._run_component("alerts", self.dispatcher.dispatch, decision, errors=errors)
        if self.clip_buffer is not None:
            for decision in decisions:
                if decision.kind == "fall_event" and decision.level == "critical" and not decision.recovery_confirmed:
                    self._run_component("clip buffer", self.clip_buffer.confirm_event, self._event_id(decision), errors=errors)
        history: tuple[dict[str, object], ...] = ()
        if self.dispatcher is not None:
            history = tuple(self._run_component("alert history", self.dispatcher.recent_alerts, errors=errors, limit=20, default=[]))
        if self.retention_policy is not None:
            self._run_component("retention", self.retention_policy.prune, now, errors=errors)
        snapshot = ServiceSnapshot(
            collected_at=now,
            camera_health=source_state["camera"],
            radar_health=source_state["radar"],
            decisions=decisions,
            alert_history=history,
            demo=any(event.quality.demo for event in events),
            latest_frame=latest_frame,
            source_errors=tuple(errors),
            events=tuple(events),
            system_health=self._system_health(source_state, events, errors),
        )
        self.last_snapshot = snapshot
        return snapshot

    def _run_component(
        self, label: str, function: Callable[..., Any], *args: Any, errors: list[str], default: Any = None, **kwargs: Any
    ) -> Any:
        """Call a component through the worker boundary and downgrade its failure."""
        future = self._executor.submit(function, *args, **kwargs)
        try:
            return future.result(timeout=self.component_timeout_seconds)
        except TimeoutError:
            errors.append(f"{label}: timeout")
            return default
        except Exception:
            errors.append(f"{label}: unavailable")
            return default

    @staticmethod
    def _normalise_batch(raw_batch: Sequence[SensorEvent] | SourceBatch) -> tuple[SourceBatch, int]:
        if isinstance(raw_batch, SourceBatch):
            batch = raw_batch
        elif isinstance(raw_batch, (str, bytes)):
            raise TypeError("source batch must contain sensor events")
        else:
            batch = SourceBatch(tuple(raw_batch))
        valid_events = tuple(event for event in batch.events if isinstance(event, SensorEvent))
        invalid_event_count = len(batch.events) - len(valid_events)
        return SourceBatch(valid_events, batch.frame, batch.frame_timestamp), invalid_event_count

    @staticmethod
    def _modality_for(name: str) -> str | None:
        normalized = name.lower()
        if normalized in {"camera", "vision"}:
            return "camera"
        if normalized == "radar":
            return "radar"
        return None

    @staticmethod
    def _health_from_events(modality: str, events: Sequence[SensorEvent]) -> str:
        source = Source.VISION if modality == "camera" else Source.RADAR
        relevant = [event for event in events if event.source is source]
        if not relevant:
            return "degraded"
        if any(event.quality.available for event in relevant):
            return "healthy"
        if any(event.event_type is EventType.AVAILABILITY for event in relevant):
            return "offline"
        return "degraded"

    def _event_id(self, decision: RiskDecision) -> str:
        if decision.timestamp is not None:
            incoming = self._incoming_event_ids.get((decision.subject_id, decision.timestamp))
            if incoming is not None:
                return incoming
        timestamp = decision.timestamp.isoformat() if decision.timestamp else "unknown"
        return f"{decision.subject_id}-{timestamp}"

    @staticmethod
    def _should_dispatch(decision: RiskDecision, events: Sequence[SensorEvent]) -> bool:
        """Do not alert on RG-PCNet suspected/abstained evidence."""
        for event in events:
            if event.event_type is not EventType.FALL_EVENT:
                continue
            subject = str(event.payload.get("subject_id", event.payload.get("tracking_id", "unknown")))
            if subject != decision.subject_id or event.timestamp != decision.timestamp:
                continue
            state = event.payload.get("event_state")
            if state in {"suspected", "abstain"}:
                return False
            if "event_id" in event.payload and not bool(event.payload.get("confirmed")) and not bool(event.payload.get("recovered")):
                return False
        return True

    @staticmethod
    def _system_health(source_state: dict[str, str], events: Sequence[SensorEvent], errors: Sequence[str]) -> str:
        if errors or any(state != "healthy" for state in source_state.values() if state != "offline"):
            return "degraded"
        if any(
            event.source is Source.VISION
            and event.event_type is EventType.POSE
            and not event.quality.available
            and event.quality.reason == "model_reliability_gate"
            for event in events
        ):
            return "degraded"
        return "healthy"
