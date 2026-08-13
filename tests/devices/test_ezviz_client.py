from __future__ import annotations

import pytest

from devices.ezviz_client import EzvizApiError, EzvizClient
from devices.models import AccessToken


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


class FakeSession:
    def __init__(self) -> None:
        self._responses: list[dict[str, object]] = []
        self.last_call: dict[str, object] | None = None
        self.call_count = 0

    def queue(self, payload: dict[str, object]) -> None:
        self._responses.append(payload)

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.call_count += 1
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
    client = EzvizClient(
        "key", "secret", session=fake_session, access_token=AccessToken("token", 9_999_999_999_999)
    )

    assert client.get_live_address("SERIAL", 1) == "ezopen://example/live"
    assert fake_session.last_call["data"] == {"accessToken": "token", "deviceSerial": "SERIAL", "channelNo": 1}


def test_live_address_sends_device_encryption_code_when_configured(fake_session: FakeSession) -> None:
    fake_session.queue({"code": "200", "data": {"url": "https://open.ys7.com/v3/openlive/test.m3u8"}})
    client = EzvizClient(
        "key", "secret", session=fake_session, access_token=AccessToken("token", 9_999_999_999_999)
    )

    client.get_live_address("SERIAL", 1, device_code="ABC123")

    assert fake_session.last_call["data"] == {
        "accessToken": "token",
        "deviceSerial": "SERIAL",
        "channelNo": 1,
        "code": "ABC123",
    }


def test_live_address_can_request_hls_for_frame_consumers(fake_session: FakeSession) -> None:
    fake_session.queue({"code": "200", "data": {"url": "https://open.ys7.com/v3/openlive/test.m3u8"}})
    client = EzvizClient(
        "key", "secret", session=fake_session, access_token=AccessToken("token", 9_999_999_999_999)
    )

    client.get_live_address("SERIAL", 1, device_code="ABC123", protocol=2, quality=2)

    assert fake_session.last_call["data"] == {
        "accessToken": "token",
        "deviceSerial": "SERIAL",
        "channelNo": 1,
        "code": "ABC123",
        "protocol": 2,
        "quality": 2,
    }


def test_package_activation_uses_json_body_and_access_token_header(fake_session: FakeSession) -> None:
    fake_session.queue(
        {
            "meta": {"code": 200, "message": "操作成功"},
            "data": [{"packageDeviceId": "PACKAGE", "activeCode": 0, "activeMessage": ""}],
        }
    )
    client = EzvizClient(
        "key", "secret", session=fake_session, access_token=AccessToken("token", 9_999_999_999_999)
    )

    result = client.activate_device_package("PACKAGE", "SERIAL", 1)

    assert result.activated is True
    assert result.active_code == 0
    assert fake_session.last_call["url"] == "https://open.ys7.com/api/v3/mall/device/package/code/active"
    assert fake_session.last_call["headers"] == {"accessToken": "token", "Content-Type": "application/json"}
    assert fake_session.last_call["json"] == [
        {"packageDeviceId": "PACKAGE", "deviceSerial": "SERIAL", "channelNo": "1"}
    ]


def test_encode_type_change_uses_config_form_endpoint(fake_session: FakeSession) -> None:
    fake_session.queue({"code": "200", "msg": "操作成功", "data": None})
    client = EzvizClient(
        "key", "secret", session=fake_session, access_token=AccessToken("token", 9_999_999_999_999)
    )

    client.change_encode_type("SERIAL", 1, "H264")

    assert fake_session.last_call["url"] == "https://open.ys7.com/api/lapp/device/encodeType/change"
    assert fake_session.last_call["data"] == {
        "accessToken": "token",
        "deviceSerial": "SERIAL",
        "channelNo": 1,
        "encodeType": "H264",
    }


def test_list_devices_uses_official_fields_and_keeps_raw_capabilities(fake_session: FakeSession) -> None:
    fake_session.queue(
        {
            "code": "200",
            "data": [
                {
                    "deviceSerial": "ABCDEF1234",
                    "deviceName": "ABCDEF1234 hidden serial",
                    "deviceType": "CS-C6C",
                    "status": 2,
                    "cameraNum": 2,
                    "supportTalk": 3,
                    "customCapability": "raw-value",
                }
            ],
        }
    )
    client = EzvizClient("key", "secret", session=fake_session, access_token=AccessToken("token", 9_999_999_999_999))

    device = client.list_devices()[0]

    assert device.talk_mode == "half_duplex"
    assert device.model == "CS-C6C"
    assert device.online is False
    assert device.channel_count == 2
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

    client = EzvizClient("key", "secret", session=fake_session, access_token=AccessToken("token", 9_999_999_999_999))

    assert client.list_devices()[0].talk_mode == expected


def test_list_devices_without_talk_capability_reports_unknown(fake_session: FakeSession) -> None:
    fake_session.queue({"code": "200", "data": [{"deviceSerial": "ABCDEF1234", "deviceType": "CS-C6C"}]})
    client = EzvizClient("key", "secret", session=fake_session, access_token=AccessToken("token", 9_999_999_999_999))

    assert client.list_devices()[0].talk_mode == "unknown"


def test_valid_access_token_is_reused_until_safety_margin(fake_session: FakeSession) -> None:
    client = EzvizClient(
        "key", "secret", session=fake_session, access_token=AccessToken("injected", 70_000), now_ms=lambda: 1_000
    )
    fake_session.queue({"code": "200", "data": []})
    fake_session.queue({"code": "200", "data": []})

    client.list_devices()
    client.list_devices()

    assert fake_session.call_count == 2
    assert fake_session.last_call["data"] == {"accessToken": "injected"}


def test_expired_access_token_is_refreshed_before_device_request(fake_session: FakeSession) -> None:
    client = EzvizClient(
        "key", "secret", session=fake_session, access_token=AccessToken("expired", 30_000), now_ms=lambda: 1_000
    )
    fake_session.queue({"code": "200", "data": {"accessToken": "fresh", "expireTime": 90_000}})
    fake_session.queue({"code": "200", "data": []})

    client.list_devices()

    assert fake_session.call_count == 2
    assert fake_session.last_call["data"] == {"accessToken": "fresh"}


def test_access_token_without_expiry_is_refreshed_before_use(fake_session: FakeSession) -> None:
    client = EzvizClient("key", "secret", session=fake_session, access_token="unbounded", now_ms=lambda: 1_000)
    fake_session.queue({"code": "200", "data": {"accessToken": "fresh", "expireTime": 90_000}})
    fake_session.queue({"code": "200", "data": []})

    client.list_devices()

    assert fake_session.last_call["data"] == {"accessToken": "fresh"}


def test_api_error_is_redacted(fake_session: FakeSession) -> None:
    fake_session.queue({"code": "10002", "msg": "invalid secret access-token"})
    client = EzvizClient("key", "secret", session=fake_session, access_token="access-token")

    with pytest.raises(EzvizApiError) as exc_info:
        client.get_access_token()

    assert exc_info.value.code == "10002"
    assert exc_info.value.endpoint == "/api/lapp/token/get"
    assert "secret" not in str(exc_info.value)
    assert "access-token" not in str(exc_info.value)
