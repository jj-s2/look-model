"""Configured SDNL1 endpoint adapter.

EZVIZ's public camera API does not presently provide a documented SDNL1 health
endpoint.  Therefore this adapter is inert until both a user-configured URL and
an explicit field mapping are supplied.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from .base import PhysiologyBatch, PhysiologyQuality, PhysiologyRecord
from .demo_jsonl import _number, _parse_iso8601


class EzvizPhysiologySource:
    def __init__(
        self,
        data_url: str | None,
        client: Any | None = None,
        field_mapping: Mapping[str, Mapping[str, str]] | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._data_url = data_url
        self._client = client
        self._field_mapping = dict(field_mapping or {})
        self._timeout = timeout

    def poll(self, start: datetime, end: datetime) -> PhysiologyBatch:
        if not self._data_url:
            return PhysiologyBatch((), PhysiologyQuality(False, False, "sdnl1_api_not_configured"))
        if not self._mapping_is_usable():
            return PhysiologyBatch((), PhysiologyQuality(False, False, "sdnl1_field_mapping_not_configured"))
        try:
            response = self._get_client().get(
                self._data_url,
                params={"start": start.isoformat(), "end": end.isoformat()},
                timeout=self._timeout,
            )
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            payload = response.json()
        except Exception:
            return PhysiologyBatch((), PhysiologyQuality(False, False, "sdnl1_request_failed"))
        raw_records = payload.get("data", []) if isinstance(payload, Mapping) else payload
        if not isinstance(raw_records, list):
            return PhysiologyBatch((), PhysiologyQuality(False, False, "sdnl1_invalid_response"))
        records = [record for item in raw_records if isinstance(item, Mapping) if (record := self._record(item))]
        return PhysiologyBatch(
            tuple(record for record in records if start <= record.timestamp < end),
            PhysiologyQuality(True, False),
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - normal environments include requests
            raise RuntimeError("requests is required only for a configured SDNL1 endpoint") from exc
        return requests

    def _mapping_is_usable(self) -> bool:
        timestamp = self._field_mapping.get("timestamp", {})
        return isinstance(timestamp.get("field"), str) and timestamp.get("format") in {"iso8601", "iso"}

    def _record(self, raw: Mapping[str, Any]) -> PhysiologyRecord | None:
        timestamp_config = self._field_mapping["timestamp"]
        timestamp = _parse_iso8601(raw.get(timestamp_config["field"]))
        if timestamp is None:
            return None
        return PhysiologyRecord(
            timestamp=timestamp,
            heart_rate=self._mapped_number(raw, "heart_rate", "bpm"),
            respiratory_rate=self._mapped_number(raw, "respiratory_rate", "breaths_per_min"),
            sleep_stage=self._mapped_stage(raw),
            source_fields=dict(raw),
        )

    def _mapped_number(self, raw: Mapping[str, Any], field: str, expected_unit: str) -> float | None:
        config = self._field_mapping.get(field)
        if not config or config.get("unit") != expected_unit:
            return None
        return _number(raw.get(config.get("field")))

    def _mapped_stage(self, raw: Mapping[str, Any]) -> str | None:
        config = self._field_mapping.get("sleep_stage")
        if not config or config.get("unit") != "stage":
            return None
        value = raw.get(config.get("field"))
        return value if isinstance(value, str) else None
