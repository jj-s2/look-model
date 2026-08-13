from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from fusion.decision_engine import RiskDecision

from .base import DeliveryResult


class DashboardDelivery:
    def __init__(self, publish: Callable[[RiskDecision], None]) -> None:
        self.publish = publish

    def send(self, decision: RiskDecision) -> DeliveryResult:
        try:
            self.publish(decision)
            return DeliveryResult("dashboard", True, "dashboard_published", datetime.now(timezone.utc))
        except Exception:
            return DeliveryResult("dashboard", False, "dashboard_publish_failed", datetime.now(timezone.utc))
