"""Local JSONL alert dispatcher with recovery-gated fall rearming."""

from __future__ import annotations

from datetime import timedelta
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
        self._sent_keys: set[str] = set()
        self._active_fall_subjects: set[str] = set()

    def dispatch(self, decision: RiskDecision) -> DispatchResult:
        key = self._dedupe_key(decision)
        if self._is_recovery(decision):
            self._active_fall_subjects.discard(decision.subject_id)
            self._sent_keys = {known for known in self._sent_keys if not known.startswith(f"fall_event:{decision.subject_id}:")}
            return DispatchResult(False, "recovery rearmed fall detection", key)
        if decision.kind == "fall_event" and decision.subject_id in self._active_fall_subjects:
            return DispatchResult(False, "fall remains active until confirmed recovery", key)
        if key in self._sent_keys:
            return DispatchResult(False, "duplicate within cooldown", key)
        self._append(decision)
        self._sent_keys.add(key)
        if decision.kind == "fall_event":
            self._active_fall_subjects.add(decision.subject_id)
        return DispatchResult(True, "written to local JSONL", key)

    def _dedupe_key(self, decision: RiskDecision) -> str:
        timestamp = decision.timestamp
        if timestamp is None:
            raise ValueError("decision timestamp is required for deduplication")
        seconds = int(timestamp.timestamp())
        bucket = seconds // int(self.cooldown.total_seconds())
        return f"{decision.kind}:{decision.subject_id}:{bucket}"

    @staticmethod
    def _is_recovery(decision: RiskDecision) -> bool:
        return decision.kind == "fall_event" and any("confirmed recovery" in reason.lower() for reason in decision.reasons)

    def _append(self, decision: RiskDecision) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "kind": decision.kind, "level": decision.level, "score": decision.score,
            "reasons": list(decision.reasons), "quality": decision.quality,
            "recommended_action": decision.recommended_action, "subject_id": decision.subject_id,
            "timestamp": decision.timestamp.isoformat() if decision.timestamp else None,
        }
        with self.path.open("a", encoding="utf-8") as destination:
            json.dump(record, destination, ensure_ascii=False, separators=(",", ":"))
            destination.write("\n")
