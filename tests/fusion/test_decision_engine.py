from datetime import datetime, timezone

import pytest

from core.events import DataQuality, EventType, SensorEvent, Source
from fusion.decision_engine import DecisionEngine


NOW = datetime(2026, 8, 2, 9, tzinfo=timezone.utc)


def event(
    source: Source,
    event_type: EventType,
    payload: dict[str, object],
    *,
    available: bool = True,
    confidence: float = 0.9,
) -> SensorEvent:
    return SensorEvent(
        timestamp=NOW,
        source=source,
        event_type=event_type,
        payload=payload,
        quality=DataQuality(
            available=available,
            confidence=confidence,
            demo=True,
            reason="test fixture" if not available else None,
        ),
    )


@pytest.fixture
def engine() -> DecisionEngine:
    return DecisionEngine()


CONFIRMED_FALL = event(Source.VISION, EventType.FALL_EVENT, {"subject_id": "elder-1", "confirmed": True})
SUSTAINED_WELLBEING_CHANGE = event(
    Source.SCREENING, EventType.WELLBEING_CHANGE, {"subject_id": "elder-1", "sustained_change": True}
)
RADAR_UNAVAILABLE = event(Source.RADAR, EventType.AVAILABILITY, {"modality": "radar"}, available=False)


def test_fall_event_and_wellbeing_change_remain_separate(engine: DecisionEngine) -> None:
    decisions = engine.evaluate([CONFIRMED_FALL, SUSTAINED_WELLBEING_CHANGE], NOW)

    assert {item.kind for item in decisions} == {"fall_event", "wellbeing_change"}


def test_missing_radar_lowers_quality_but_does_not_block_fall(engine: DecisionEngine) -> None:
    decisions = engine.evaluate([CONFIRMED_FALL, RADAR_UNAVAILABLE], NOW)
    fall = next(item for item in decisions if item.kind == "fall_event")

    assert fall.level == "critical"
    assert fall.quality == "vision_only"
    assert len(fall.reasons) >= 2
    assert any("radar" in reason for reason in fall.reasons)


@pytest.mark.parametrize(("available", "confidence"), [(False, 0.9), (True, 0.49)])
def test_confirmed_fall_with_insufficient_source_quality_is_not_critical(
    engine: DecisionEngine, available: bool, confidence: float
) -> None:
    confirmed = event(
        Source.VISION,
        EventType.FALL_EVENT,
        {"subject_id": "elder-1", "confirmed": True},
        available=available,
        confidence=confidence,
    )

    fall = engine.evaluate([confirmed], NOW)[0]

    assert fall.level != "critical"
    assert fall.quality == "degraded"
    assert any("insufficient" in reason for reason in fall.reasons)


def test_high_forecast_score_with_low_source_confidence_is_not_critical(engine: DecisionEngine) -> None:
    forecast = event(
        Source.VISION,
        EventType.FALL_FORECAST,
        {"subject_id": "elder-1", "score": 0.99},
        confidence=0.49,
    )

    decision = engine.evaluate([forecast], NOW)[0]

    assert decision.level != "critical"
    assert decision.quality == "degraded"
    assert any("insufficient" in reason for reason in decision.reasons)


def test_vision_fall_without_radar_evidence_is_vision_only(engine: DecisionEngine) -> None:
    fall = engine.evaluate([CONFIRMED_FALL], NOW)[0]

    assert fall.quality == "vision_only"


def test_matching_available_radar_evidence_enables_multimodal_fall(engine: DecisionEngine) -> None:
    radar = event(Source.RADAR, EventType.PHYSIOLOGY, {"subject_id": "elder-1", "motion_anomaly": True})

    fall = engine.evaluate([CONFIRMED_FALL, radar], NOW)[0]

    assert fall.quality == "multimodal"


def test_other_subject_radar_evidence_does_not_enable_multimodal_fall(engine: DecisionEngine) -> None:
    radar = event(Source.RADAR, EventType.PHYSIOLOGY, {"subject_id": "elder-2", "motion_anomaly": True})

    fall = engine.evaluate([CONFIRMED_FALL, radar], NOW)[0]

    assert fall.quality == "vision_only"


@pytest.mark.parametrize(
    ("source_event", "expected_level"),
    [
        (event(Source.VISION, EventType.FALL_FORECAST, {"subject_id": "elder-1", "score": 0.1}), "info"),
        (event(Source.VISION, EventType.FALL_FORECAST, {"subject_id": "elder-1", "score": 0.45}), "watch"),
        (event(Source.VISION, EventType.FALL_FORECAST, {"subject_id": "elder-1", "score": 0.75}), "warning"),
        (CONFIRMED_FALL, "critical"),
    ],
)
def test_decisions_use_all_four_risk_levels(engine: DecisionEngine, source_event: SensorEvent, expected_level: str) -> None:
    decision = engine.evaluate([source_event], NOW)[0]

    assert decision.level == expected_level
    assert len(decision.reasons) >= 2


def test_physiology_only_does_not_create_wellbeing_diagnosis(engine: DecisionEngine) -> None:
    physiology = event(Source.RADAR, EventType.PHYSIOLOGY, {"subject_id": "elder-1", "sleep_hours": 3.0})

    assert engine.evaluate([physiology], NOW) == []
