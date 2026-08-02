from __future__ import annotations

import pytest

from devices.ezviz_client import EzvizApiError, EzvizClient


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


class FakeSession:
    def __init__(self) -> None:
        self._responses: list[dict[str, object]] = []
        self.last_call: dict[str, object] | None = None

    def queue(self, payload: dict[str, object]) -> None:
        self._responses.append(payload)

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.last_call = {"url": url, **kwargs}
        return FakeResponse(self._responses.pop(0))


@pytest.fixture
def fake_session() -> FakeSession:
    return FakeSession()


def test_token_request_uses_form_body_and_never_repr_secret(fake_session: FakeSession) -> None:
    fake_session.queue({"code": "200", "data": {"accessToken": "token", "expireTime": 1999999999999}})
    client = EzvizClient("key", "secret", session=fake_session)

    token = client.get_access_token()

    assert token.value == "token"
    assert fake_session.last_call["data"] == {"appKey": "key", "appSecret": "secret"}
    assert fake_session.last_call["url"] == "https://open.ys7.com/api/lapp/token/get"
    assert "secret" not in repr(client)


def test_live_address_is_extracted_from_success_response(fake_session: FakeSession) -> None:
    fake_session.queue({"code": "200", "data": {"url": "ezopen://example/live"}})
    client = EzvizClient("key", "secret", session=fake_session, access_token="token")

    assert client.get_live_address("SERIAL", 1) == "ezopen://example/live"
    assert fake_session.last_call["data"] == {"accessToken": "token", "deviceSerial": "SERIAL", "channelNo": 1}


def test_list_devices_maps_talk_mode_and_keeps_raw_capabilities(fake_session: FakeSession) -> None:
    fake_session.queue(
        {
            "code": "200",
            "data": [
                {
                    "deviceSerial": "ABCDEF1234",
                    "deviceName": "C6C",
                    "status": 1,
                    "channelNumber": 2,
                    "supportTalk": 3,
                    "customCapability": "raw-value",
                }
            ],
        }
    )
    client = EzvizClient("key", "secret", session=fake_session, access_token="token")

    device = client.list_devices()[0]

    assert device.talk_mode == "half_duplex"
    assert device.capabilities["customCapability"] == "raw-value"
    assert fake_session.last_call["url"] == "https://open.ys7.com/api/lapp/device/list"


@pytest.mark.parametrize(
    ("support_talk", "expected"),
    [(0, "none"), (1, "full_duplex"), (3, "half_duplex"), (99, "unknown"), (None, "unknown")],
)
def test_talk_mode_mapping_handles_known_and_missing_values(
    fake_session: FakeSession, support_talk: int | None, expected: str
) -> None:
    payload = {"deviceSerial": "ABCDEF1234"}
    if support_talk is not None:
        payload["supportTalk"] = support_talk
    fake_session.queue({"code": "200", "data": [payload]})

    client = EzvizClient("key", "secret", session=fake_session, access_token="token")

    assert client.list_devices()[0].talk_mode == expected


def test_api_error_is_redacted(fake_session: FakeSession) -> None:
    fake_session.queue({"code": "10002", "msg": "invalid secret access-token"})
    client = EzvizClient("key", "secret", session=fake_session, access_token="access-token")

    with pytest.raises(EzvizApiError) as exc_info:
        client.get_access_token()

    assert exc_info.value.code == "10002"
    assert exc_info.value.endpoint == "/api/lapp/token/get"
    assert "secret" not in str(exc_info.value)
    assert "access-token" not in str(exc_info.value)
