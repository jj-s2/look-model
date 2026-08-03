from datetime import datetime, timezone

from alerts.delivery.base import DeliveryResult
from alerts.delivery.webhook import WebhookDelivery
from alerts.dispatcher import AlertDispatcher
from fusion.decision_engine import RiskDecision


NOW = datetime(2026, 8, 2, 9, tzinfo=timezone.utc)
WARNING = RiskDecision("prefall_warning", "warning", .8, ("safe reason",), "vision_only", "check in", "elder-1", NOW)


class RaisingClient:
    def post(self, *args, **kwargs):
        raise OSError("network down")


class RaisingDelivery:
    def send(self, decision):
        raise OSError("network down")


def test_webhook_failure_is_reported_without_raising():
    result = WebhookDelivery("https://alerts.example.test", client=RaisingClient(), timeout=2.0).send(WARNING)
    assert result.sent is False
    assert result.reason == "webhook_request_failed"


def test_webhook_payload_uses_subject_alias_and_no_secrets():
    class Client:
        def __init__(self): self.payload = None
        def post(self, url, **kwargs): self.payload = kwargs["json"]; return type("Response", (), {"raise_for_status": lambda self: None})()
    client = Client()
    result = WebhookDelivery("https://alerts.example.test", client=client).send(WARNING)
    assert result.sent
    assert client.payload["subject_id"] != "elder-1"
    assert "reasons" in client.payload


def test_dispatcher_persists_before_delivery(tmp_path):
    dispatcher = AlertDispatcher(tmp_path / "alerts.jsonl", delivery=RaisingDelivery())
    result = dispatcher.dispatch(WARNING)
    assert result.sent is True
    record = dispatcher.recent_alerts()[0]
    assert record["sent"] is True
    assert record["delivery"][0]["sent"] is False
