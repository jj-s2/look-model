from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, Sequence

from fusion.decision_engine import RiskDecision


@dataclass(frozen=True)
class DeliveryResult:
    channel: str
    sent: bool
    reason: str
    attempted_at: datetime

    def to_dict(self) -> dict[str, object]:
        return {"channel": self.channel, "sent": self.sent, "reason": self.reason, "attempted_at": self.attempted_at.isoformat()}


class DeliveryAdapter(Protocol):
    def send(self, decision: RiskDecision) -> DeliveryResult: ...


class CompositeDelivery:
    def __init__(self, adapters: Sequence[DeliveryAdapter]) -> None:
        self.adapters = tuple(adapters)

    def send(self, decision: RiskDecision) -> tuple[DeliveryResult, ...]:
        results = []
        for adapter in self.adapters:
            try:
                results.append(adapter.send(decision))
            except Exception:
                results.append(DeliveryResult(type(adapter).__name__, False, "delivery_adapter_failed", datetime.now(timezone.utc)))
        return tuple(results)
