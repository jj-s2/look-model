"""Dynamic EZVIZ stream-address providers for frame input adapters."""

from __future__ import annotations

from .ezviz_client import EzvizClient


class EzvizLiveUrlProvider:
    """Refresh a short-lived EZVIZ playback address on demand.

    The provider is intentionally callable so it can be passed directly to
    :class:`vision.input_adapter.EzvizStreamAdapter`, which refreshes the
    address after a failed read. Credentials and device identifiers are never
    included in its representation.
    """

    def __init__(
        self,
        client: EzvizClient,
        device_serial: str,
        *,
        channel_no: int = 1,
        device_code: str | None = None,
        protocol: int = 2,
        quality: int = 2,
    ) -> None:
        self._client = client
        self._device_serial = device_serial
        self._channel_no = channel_no
        self._device_code = device_code
        self._protocol = protocol
        self._quality = quality

    def __call__(self) -> str:
        return self._client.get_live_address(
            device_serial=self._device_serial,
            channel_no=self._channel_no,
            device_code=self._device_code,
            protocol=self._protocol,
            quality=self._quality,
        )

    def __repr__(self) -> str:
        return "EzvizLiveUrlProvider(device='***', channel_no={!r}, protocol={!r}, quality={!r})".format(
            self._channel_no, self._protocol, self._quality
        )
