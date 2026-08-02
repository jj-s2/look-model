"""Serializable alert-delivery results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DispatchResult:
    sent: bool
    reason: str
    dedupe_key: str
