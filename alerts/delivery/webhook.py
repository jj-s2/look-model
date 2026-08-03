from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Any

from fusion.decision_engine import RiskDecision

from .base import DeliveryResult


class WebhookDelivery:
    def __init__(self, url: str, *, client: Any | None = None, timeout: float = 2.0) -> None:
        if not url.startswith("https://"):
            raise ValueError("webhook URL must use HTTPS")
        self.url = url
        self.client = client
        self.timeout = timeout

    def send(self, decision: RiskDecision) -> DeliveryResult:
        attempted_at = datetime.now(timezone.utc)
        payload = {
            "kind": decision.kind, "level": decision.level, "score": decision.score,
            "reasons": list(decision.reasons),
            "subject_id": "subject-" + hashlib.sha256(decision.subject_id.encode()).hexdigest()[:10],
            "timestamp": decision.timestamp.isoformat() if decision.timestamp else None,
            "recommended_action": decision.recommended_action,
        }
        try:
            client = self.client
            if client is None:
                import requests
                client = requests
            response = client.post(self.url, json=payload, timeout=self.timeout)
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            return DeliveryResult("webhook", True, "webhook_sent", attempted_at)
        except Exception:
            return DeliveryResult("webhook", False, "webhook_request_failed", attempted_at)
