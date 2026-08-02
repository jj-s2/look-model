"""Local JSONL alert dispatcher with recovery-gated fall rearming."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path

from alerts.models import DispatchResult
from fusion.decision_engine import RiskDecision


class AlertDispatcher:
    """Persist UI-readable alerts and suppress duplicate incidents.

    This first version deliberately performs no network I/O.  A caller may
    configure a webhook URL in a later delivery adapter, while this dispatcher
    remains safe to exercise with fake events and writes only local JSONL.
    """

    def __init__(self, path: Path, cooldown: timedelta = timedelta(minutes=15)) -> None:
        if cooldown <= timedelta(0):
            raise ValueError("cooldown must be positive")
        self.path = Path(path)
        self.cooldown = cooldown
        self._last_sent: dict[str, datetime] = {}
        self._active_fall_levels: dict[str, str] = {}
        self._restore_state()

    def dispatch(self, decision: RiskDecision) -> DispatchResult:
        key = self._dedupe_key(decision)
        if self._is_recovery(decision):
            self._active_fall_levels.pop(decision.subject_id, None)
            self._last_sent.pop(key, None)
            self._append(decision, sent=False)
            return DispatchResult(False, "recovery rearmed fall detection", key)
        is_escalation = False
        if decision.kind == "fall_event":
            active_level = self._active_fall_levels.get(decision.subject_id)
            if active_level is not None:
                is_escalation = self._LEVEL_RANK[decision.level] > self._LEVEL_RANK[active_level]
                if not is_escalation:
                    return DispatchResult(False, "fall remains active until confirmed recovery", key)
        last_sent = self._last_sent.get(key)
        if not is_escalation and last_sent is not None and abs(decision.timestamp - last_sent) < self.cooldown:
            return DispatchResult(False, "duplicate within cooldown", key)
        self._append(decision, sent=True)
        self._last_sent[key] = decision.timestamp
        if decision.kind == "fall_event":
            self._active_fall_levels[decision.subject_id] = decision.level
        return DispatchResult(True, "written to local JSONL", key)

    def _dedupe_key(self, decision: RiskDecision) -> str:
        timestamp = decision.timestamp
        if timestamp is None:
            raise ValueError("decision timestamp is required for deduplication")
        return f"{decision.kind}:{decision.subject_id}"

    @staticmethod
    def _is_recovery(decision: RiskDecision) -> bool:
        return decision.kind == "fall_event" and decision.recovery_confirmed

    def recent_alerts(self, limit: int | None = None) -> list[dict[str, object]]:
        """Read local JSONL history for the UI without contacting any service."""
        records: list[dict[str, object]] = []
        if not self.path.exists():
            return records
        with self.path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        return records[-limit:] if limit is not None else records

    def _restore_state(self) -> None:
        for record in self.recent_alerts():
            kind = record.get("kind")
            subject = record.get("subject_id")
            timestamp = record.get("timestamp")
            if not isinstance(kind, str) or not isinstance(subject, str) or not isinstance(timestamp, str):
                continue
            try:
                recorded_at = datetime.fromisoformat(timestamp)
            except ValueError:
                continue
            key = f"{kind}:{subject}"
            if kind == "fall_event" and record.get("recovery_confirmed") is True:
                self._active_fall_levels.pop(subject, None)
                self._last_sent.pop(key, None)
            elif record.get("sent") is True:
                previous = self._last_sent.get(key)
                if previous is None or recorded_at > previous:
                    self._last_sent[key] = recorded_at
                if kind == "fall_event":
                    level = record.get("level")
                    if isinstance(level, str) and level in self._LEVEL_RANK:
                        self._active_fall_levels[subject] = level

    def _append(self, decision: RiskDecision, *, sent: bool) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "kind": decision.kind, "level": decision.level, "score": decision.score,
            "reasons": list(decision.reasons), "quality": decision.quality,
            "recommended_action": decision.recommended_action, "subject_id": decision.subject_id,
            "timestamp": decision.timestamp.isoformat() if decision.timestamp else None,
            "recovery_confirmed": decision.recovery_confirmed, "sent": sent,
        }
        with self.path.open("a", encoding="utf-8") as destination:
            json.dump(record, destination, ensure_ascii=False, separators=(",", ":"))
            destination.write("\n")

    _LEVEL_RANK = {"info": 0, "watch": 1, "warning": 2, "critical": 3}
