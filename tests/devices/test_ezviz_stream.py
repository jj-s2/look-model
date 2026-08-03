from __future__ import annotations

from devices.ezviz_stream import EzvizLiveUrlProvider


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_live_address(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        return "https://open.ys7.com/v3/openlive/redacted.m3u8"


def test_live_url_provider_refreshes_address_with_stream_parameters() -> None:
    client = FakeClient()
    provider = EzvizLiveUrlProvider(
        client, "SERIAL", channel_no=1, device_code="ABC123", protocol=2, quality=2
    )

    assert provider() == "https://open.ys7.com/v3/openlive/redacted.m3u8"
    assert client.calls == [
        {
            "device_serial": "SERIAL",
            "channel_no": 1,
            "device_code": "ABC123",
            "protocol": 2,
            "quality": 2,
        }
    ]


def test_live_url_provider_repr_redacts_device_identity() -> None:
    provider = EzvizLiveUrlProvider(FakeClient(), "SERIAL", device_code="ABC123")

    rendered = repr(provider)

    assert "SERIAL" not in rendered
    assert "ABC123" not in rendered
    assert "EzvizLiveUrlProvider" in rendered
