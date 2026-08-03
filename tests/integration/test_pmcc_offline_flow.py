from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from core.events import DataQuality, EventType, SensorEvent, Source
from fusion.decision_engine import DecisionEngine
from risk.pmcc.schema import DailyObservation
from risk.pmcc.service import PMCCService


UTC = timezone.utc


def test_pmcc_offline_flow_does_not_change_confirmed_fall_event_contract() -> None:
    service = PMCCService()
    as_of = date(2026, 8, 1)
    for offset in range(14):
        day = as_of - timedelta(days=13 - offset)
        service.observe(DailyObservation(
            "resident-1", datetime.combine(day, datetime.min.time(), tzinfo=UTC),
            {"steps": 1200.0, "trunk_sway": 0.3}, {"steps": 0.9, "trunk_sway": 0.9},
            {"steps": True, "trunk_sway": True}, {"evidence_tier": "real_device_longitudinal", "promoted": True},
        ))

    forecast = service.forecast("resident-1", as_of)
    fall = SensorEvent(
        timestamp=datetime(2026, 8, 1, tzinfo=UTC), source=Source.VISION, event_type=EventType.FALL_EVENT,
        payload={"subject_id": "resident-1", "confirmed": True},
        quality=DataQuality(available=True, confidence=0.9, demo=True),
    )

    decision = DecisionEngine().evaluate([fall], fall.timestamp)[0]
    assert forecast.provenance["forecast_kind"] == "fall_forecast"
    assert forecast.provenance["decision"] != "critical"
    assert decision.kind == "fall_event"
    assert decision.level == "critical"
