from __future__ import annotations

import json
from datetime import datetime, timezone

from radar.demo_jsonl import DemoJsonlPhysiologySource
from radar.ezviz_source import EzvizPhysiologySource


START = datetime(2026, 8, 1, tzinfo=timezone.utc)
END = datetime(2026, 8, 2, tzinfo=timezone.utc)


class NoNetworkClient:
    def get(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("an unconfigured source must not make a request")


def test_real_source_reports_unavailable_without_configured_endpoint() -> None:
    batch = EzvizPhysiologySource(data_url=None, client=NoNetworkClient()).poll(START, END)

    assert batch.quality.available is False
    assert batch.quality.demo is False
    assert batch.quality.reason == "sdnl1_api_not_configured"


def test_demo_source_marks_every_batch_as_demo(tmp_path) -> None:
    fixture = tmp_path / "physiology.jsonl"
    fixture.write_text(
        json.dumps({"timestamp": "2026-08-01T12:00:00+00:00", "heart_rate": 72}) + "\n",
        encoding="utf-8",
    )

    batch = DemoJsonlPhysiologySource(fixture).poll(START, END)

    assert batch.quality.available is True
    assert batch.quality.demo is True
    assert batch.records[0].heart_rate == 72.0


def test_real_source_keeps_unknown_fields_for_audit_without_parsing_them() -> None:
    class Response:
        def json(self):
            return [{"at": "2026-08-01T12:00:00+00:00", "pulse": 71, "secret_score": 99}]

    class Client:
        def get(self, url: str, **kwargs: object) -> Response:
            return Response()

    source = EzvizPhysiologySource(
        data_url="https://example.invalid/sdnl1",
        client=Client(),
        field_mapping={
            "timestamp": {"field": "at", "format": "iso8601"},
            "heart_rate": {"field": "pulse", "unit": "bpm"},
        },
    )

    record = source.poll(START, END).records[0]

    assert record.heart_rate == 71.0
    assert record.respiratory_rate is None
    assert record.source_fields["secret_score"] == 99


def test_real_source_rejects_a_declared_measurement_with_the_wrong_unit() -> None:
    class Response:
        def json(self):
            return [{"at": "2026-08-01T12:00:00+00:00", "pulse": 71}]

    class Client:
        def get(self, url: str, **kwargs: object) -> Response:
            return Response()

    batch = EzvizPhysiologySource(
        "https://example.invalid/sdnl1",
        client=Client(),
        field_mapping={
            "timestamp": {"field": "at", "format": "iso8601"},
            "heart_rate": {"field": "pulse", "unit": "kilograms"},
        },
    ).poll(START, END)

    assert batch.records[0].heart_rate is None
    assert batch.records[0].source_fields["pulse"] == 71
