"""Conservative EZVIZ capability-gated delivery boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fusion.decision_engine import RiskDecision

from .base import DeliveryResult


@dataclass(frozen=True)
class EzvizDeliveryCapability:
    account_region: str
    device_model: str
    app_notification_supported: bool
    two_way_audio_supported: bool
    proactive_audio_supported: bool
    verified_at: datetime | None
    evidence_source: str

    @property
    def verified(self) -> bool:
        return self.verified_at is not None and bool(self.evidence_source.strip())


class EzvizDelivery:
    def __init__(self, *, client: Any, capability: EzvizDeliveryCapability) -> None:
        self.client = client
        self.capability = capability

    def send(self, decision: RiskDecision) -> DeliveryResult:
        attempted_at = datetime.now(timezone.utc)
        if not self.capability.verified or not self.capability.app_notification_supported:
            return DeliveryResult("ezviz", False, "ezviz_delivery_not_verified", attempted_at)
        method = getattr(self.client, "send_notification", None)
        if not callable(method):
            return DeliveryResult("ezviz", False, "ezviz_notification_api_unavailable", attempted_at)
        try:
            method(kind=decision.kind, level=decision.level, score=decision.score, action=decision.recommended_action)
            return DeliveryResult("ezviz", True, "ezviz_notification_sent", attempted_at)
        except Exception:
            return DeliveryResult("ezviz", False, "ezviz_notification_failed", attempted_at)
