"""Typed values returned by the EZVIZ Open Platform."""

from dataclasses import dataclass
from typing import Any, Literal, Mapping


TalkMode = Literal["none", "full_duplex", "half_duplex", "unknown"]


@dataclass(frozen=True)
class AccessToken:
    """An EZVIZ access token and its epoch-millisecond expiry."""

    value: str
    expires_at_ms: int | None = None

    def __repr__(self) -> str:
        return "AccessToken(value='***', expires_at_ms=%r)" % self.expires_at_ms


@dataclass(frozen=True)
class EzvizDevice:
    """A device returned by the EZVIZ device list endpoint."""

    serial: str
    model: str | None
    online: bool | None
    channel_count: int | None
    talk_mode: TalkMode
    capabilities: Mapping[str, Any]
