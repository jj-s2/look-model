"""Offline SDNL1-like JSONL source used only for visibly marked demos."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .base import PhysiologyBatch, PhysiologyQuality, PhysiologyRecord


class DemoJsonlPhysiologySource:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def poll(self, start: datetime, end: datetime) -> PhysiologyBatch:
        records: list[PhysiologyRecord] = []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return PhysiologyBatch((), PhysiologyQuality(False, True, "demo_fixture_unavailable"))
        for line in lines:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(raw, Mapping):
                record = _record_from_mapping(raw)
                if record is not None and start <= record.timestamp < end:
                    records.append(record)
        return PhysiologyBatch(tuple(records), PhysiologyQuality(True, True, "demo_data"))


def _record_from_mapping(raw: Mapping[str, Any]) -> PhysiologyRecord | None:
    timestamp = _parse_iso8601(raw.get("timestamp"))
    if timestamp is None:
        return None
    return PhysiologyRecord(
        timestamp=timestamp,
        heart_rate=_number(raw.get("heart_rate")),
        respiratory_rate=_number(raw.get("respiratory_rate")),
        sleep_stage=raw.get("sleep_stage") if isinstance(raw.get("sleep_stage"), str) else None,
        source_fields=dict(raw),
    )


def _parse_iso8601(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
