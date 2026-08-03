"""Build registered, explicitly non-causal temporal PMCC associations."""
from __future__ import annotations

from datetime import datetime
from math import exp
from typing import Sequence

from .schema import ChangeEvent, TemporalChain


_MAX_GAP_HOURS = 72.0
_EDGES = {
    ("sleep", "activity"): 1.0,
    ("sleep", "gait"): 1.0,
    ("activity", "sit_to_stand"): 1.0,
    ("sit_to_stand", "near_fall"): 1.0,
    ("gait", "near_fall"): 1.0,
    ("physiology", "sleep"): 1.0,
}


class TemporalChainBuilder:
    """Construct only pre-registered temporal associations, never causal claims."""

    def build(self, events: Sequence[ChangeEvent]) -> tuple[TemporalChain, ...]:
        if not all(isinstance(event, ChangeEvent) for event in events):
            raise ValueError("events must contain ChangeEvent records")
        _require_strictly_chronological(
            tuple(event.occurred_at for event in events), "events"
        )
        by_subject: dict[str, list[ChangeEvent]] = {}
        for event in events:
            by_subject.setdefault(event.subject_id, []).append(event)

        chains: list[TemporalChain] = []
        for subject_id, subject_events in by_subject.items():
            for start_index, event in enumerate(subject_events):
                chains.extend(self._extend(subject_id, subject_events, (event,), start_index + 1))
        return tuple(chains)

    def _extend(
        self,
        subject_id: str,
        events: Sequence[ChangeEvent],
        path: tuple[ChangeEvent, ...],
        next_index: int,
    ) -> list[TemporalChain]:
        tail = path[-1]
        extensions: list[tuple[int, ChangeEvent]] = []
        for index in range(next_index, len(events)):
            candidate = events[index]
            gap = _gap_hours(tail.occurred_at, candidate.occurred_at)
            if gap > _MAX_GAP_HOURS:
                break
            if gap < 0 or not _registered(tail.feature, candidate.feature):
                continue
            extensions.append((index, candidate))
        if not extensions:
            return [self._chain(subject_id, path)] if len(path) >= 2 else []
        chains: list[TemporalChain] = []
        for index, candidate in extensions:
            chains.extend(self._extend(subject_id, events, path + (candidate,), index + 1))
        return chains

    @staticmethod
    def _chain(subject_id: str, events: tuple[ChangeEvent, ...]) -> TemporalChain:
        gaps = tuple(_gap_hours(left.occurred_at, right.occurred_at) for left, right in zip(events, events[1:]))
        edge_scores = []
        for left, right, gap in zip(events, events[1:], gaps):
            weight = _EDGES[(_node(left.feature), _node(right.feature))]
            edge_scores.append(
                weight * _persistence(left) * _persistence(right) * _quality(left) * _quality(right) * exp(-gap / 48.0)
            )
        score = max(0.0, min(1.0, _combine(edge_scores)))
        node_ids = tuple(_event_id(event) for event in events)
        return TemporalChain(
            subject_id=subject_id,
            created_at=events[-1].occurred_at,
            changes=events,
            provenance={
                "association_only": True,
                "node_ids": node_ids,
                "gap_hours": gaps,
                "edge_scores": tuple(edge_scores),
                "score": score,
            },
        )


def _combine(edge_scores: Sequence[float]) -> float:
    result = 1.0
    for score in edge_scores:
        result *= score
    return result


def _gap_hours(left: datetime, right: datetime) -> float:
    return (right - left).total_seconds() / 3600.0


def _registered(left: str, right: str) -> bool:
    return (_node(left), _node(right)) in _EDGES


def _node(feature: str) -> str:
    name = feature.lower().replace("-", "_")
    if "sleep" in name:
        return "sleep"
    if "sit_to_stand" in name or "sit2stand" in name:
        return "sit_to_stand"
    if "near_fall" in name:
        return "near_fall"
    if any(token in name for token in ("gait", "trunk_sway", "stride", "balance")):
        return "gait"
    if any(token in name for token in ("activity", "step", "sedentary")):
        return "activity"
    if any(token in name for token in ("heart", "pressure", "spo2", "physiology", "respir")):
        return "physiology"
    return name


def _persistence(event: ChangeEvent) -> float:
    return _bounded(event.provenance.get("persistence", 1.0))


def _quality(event: ChangeEvent) -> float:
    return _bounded(event.provenance.get("quality", event.confidence))


def _bounded(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _event_id(event: ChangeEvent) -> str:
    value = event.provenance.get("event_id")
    return value if isinstance(value, str) and value else f"{event.feature}:{event.occurred_at.isoformat()}"


def _require_strictly_chronological(timestamps: tuple[datetime, ...], name: str) -> None:
    if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{name} must be supplied in strictly chronological order")
