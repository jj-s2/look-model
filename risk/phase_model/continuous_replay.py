"""Deterministic replay of timestamped model streams.

The replay boundary is deliberately separate from model inference.  Frames may
carry truth and clip metadata for an evaluator, but a model receives a fresh
frame view with those fields removed from its payload and attributes.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Callable, Iterable, Mapping

from .event_state import EventDecoder, EventTransition, FrameDecision


_BOUNDARY_KEYS = frozenset(
    {
        "clip_id",
        "clip_boundary",
        "truth_event_id",
        "truth_start",
        "truth_end",
        "truth",
        "ground_truth",
    }
)
_ALERT_EMISSIONS = frozenset({"fall_confirmed", "fall_recovered"})


def _finite_timestamp(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("timestamp must be a finite number")
    timestamp = float(value)
    if not isfinite(timestamp):
        raise ValueError("timestamp must be a finite number")
    return timestamp


def _optional_timestamp(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number or None")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number or None")
    return number


def _strip_boundary_metadata(value: object) -> object:
    """Deep-copy a payload while removing evaluator-only mapping keys."""

    if isinstance(value, Mapping):
        return {
            key: _strip_boundary_metadata(item)
            for key, item in value.items()
            if not (isinstance(key, str) and key in _BOUNDARY_KEYS)
        }
    if isinstance(value, list):
        return [_strip_boundary_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_boundary_metadata(item) for item in value)
    if isinstance(value, set):
        return {_strip_boundary_metadata(item) for item in value}
    return deepcopy(value)


@dataclass(frozen=True)
class ReplayFrame:
    """One chronological sample and evaluator-only annotations.

    ``payload`` is copied on construction.  The copy protects the evaluator's
    source record when a predictor mutates its model-visible frame.  The
    optional truth/clip fields are never supplied to the predictor.
    """

    timestamp: float
    payload: Any
    truth_event_id: str | None = None
    truth_start: float | None = None
    truth_end: float | None = None
    clip_id: str | None = None
    clip_boundary: bool | None = None
    evaluator_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _finite_timestamp(self.timestamp))
        if self.truth_event_id is not None and not isinstance(self.truth_event_id, str):
            raise ValueError("truth_event_id must be a string or None")
        if self.clip_id is not None and not isinstance(self.clip_id, str):
            raise ValueError("clip_id must be a string or None")
        if self.clip_boundary is not None and not isinstance(self.clip_boundary, bool):
            raise ValueError("clip_boundary must be a boolean or None")
        object.__setattr__(self, "truth_start", _optional_timestamp(self.truth_start, "truth_start"))
        object.__setattr__(self, "truth_end", _optional_timestamp(self.truth_end, "truth_end"))
        object.__setattr__(self, "payload", deepcopy(self.payload))
        if self.evaluator_metadata is not None:
            if not isinstance(self.evaluator_metadata, Mapping):
                raise ValueError("evaluator_metadata must be a mapping or None")
            object.__setattr__(self, "evaluator_metadata", deepcopy(dict(self.evaluator_metadata)))

    def model_view(self) -> "ReplayFrame":
        """Return a detached frame containing no evaluator-side metadata."""

        return ReplayFrame(self.timestamp, _strip_boundary_metadata(self.payload))


@dataclass(frozen=True)
class ReplayResult:
    """Replay audit result, including every decoder transition."""

    transitions: tuple[EventTransition, ...]
    alerts: tuple[EventTransition, ...]
    coverage: float
    abstained_frames: int
    total_frames: int

    def __post_init__(self) -> None:
        transitions = tuple(self.transitions)
        alerts = tuple(self.alerts)
        if not all(isinstance(item, EventTransition) for item in transitions):
            raise TypeError("transitions must contain EventTransition values")
        if not all(isinstance(item, EventTransition) for item in alerts):
            raise TypeError("alerts must contain EventTransition values")
        if isinstance(self.total_frames, bool) or not isinstance(self.total_frames, int) or self.total_frames < 0:
            raise ValueError("total_frames must be a non-negative integer")
        if isinstance(self.abstained_frames, bool) or not isinstance(self.abstained_frames, int):
            raise ValueError("abstained_frames must be a non-negative integer")
        if self.abstained_frames < 0 or self.abstained_frames > self.total_frames:
            raise ValueError("abstained_frames must not exceed total_frames")
        if isinstance(self.coverage, bool) or not isinstance(self.coverage, Real):
            raise ValueError("coverage must be a finite number in [0, 1]")
        coverage = float(self.coverage)
        if not isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError("coverage must be a finite number in [0, 1]")
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "alerts", alerts)
        object.__setattr__(self, "coverage", coverage)


def _predict(predictor: object, frame: ReplayFrame) -> FrameDecision:
    if callable(predictor):
        decision = predictor(frame)
    else:
        method = getattr(predictor, "predict", None)
        if not callable(method):
            raise TypeError("predictor must be callable or expose predict(frame)")
        decision = method(frame)
    if not isinstance(decision, FrameDecision):
        raise TypeError("predictor must return a FrameDecision")
    if decision.timestamp != frame.timestamp:
        raise ValueError("prediction timestamp must match replay frame timestamp")
    return decision


def replay_stream(
    frames: Iterable[ReplayFrame],
    predictor: Callable[[ReplayFrame], FrameDecision] | object,
    decoder: EventDecoder,
) -> ReplayResult:
    """Replay chronological frames through a predictor and event decoder.

    A frame is checked for strict timestamp ordering before prediction.  Every
    decision, including reliability-gated abstentions, is retained in
    ``transitions``.  Alerts are the decoder transitions that emit confirmed or
    recovered fall events.
    """

    if not isinstance(decoder, EventDecoder):
        raise TypeError("decoder must be an EventDecoder")
    transitions: list[EventTransition] = []
    alerts: list[EventTransition] = []
    previous_timestamp: float | None = None
    abstained = 0

    for frame in frames:
        if not isinstance(frame, ReplayFrame):
            raise TypeError("frames must contain ReplayFrame values")
        if previous_timestamp is not None and frame.timestamp <= previous_timestamp:
            raise ValueError("replay timestamps must be strictly increasing")
        previous_timestamp = frame.timestamp
        decision = _predict(predictor, frame.model_view())
        transition = decoder.update(decision)
        transitions.append(transition)
        if not decision.reliable:
            abstained += 1
        if transition.emitted in _ALERT_EMISSIONS:
            alerts.append(transition)

    total = len(transitions)
    coverage = 0.0 if total == 0 else (total - abstained) / total
    return ReplayResult(tuple(transitions), tuple(alerts), coverage, abstained, total)
