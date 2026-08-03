from datetime import datetime, timezone

from alerts.delivery.ezviz import EzvizDelivery, EzvizDeliveryCapability
from fusion.decision_engine import RiskDecision


def test_unverified_speaker_capability_never_sends():
    class FakeClient:
        calls = []
    capability = EzvizDeliveryCapability("CN", "CS-C6c", False, False, True, None, "")
    result = EzvizDelivery(client=FakeClient(), capability=capability).send(RiskDecision("fall_event", "critical", 1.0, (), "vision_only", "check", "elder", datetime.now(timezone.utc)))
    assert result.sent is False
    assert result.reason == "ezviz_delivery_not_verified"
    assert FakeClient.calls == []
