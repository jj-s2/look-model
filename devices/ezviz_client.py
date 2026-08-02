"""Small, redaction-safe client for the EZVIZ Open Platform."""

from __future__ import annotations

from typing import Any, Mapping

import requests

from .models import AccessToken, EzvizDevice, TalkMode


BASE_URL = "https://open.ys7.com"
TOKEN_ENDPOINT = "/api/lapp/token/get"
DEVICE_LIST_ENDPOINT = "/api/lapp/device/list"
LIVE_ADDRESS_ENDPOINT = "/api/lapp/v2/live/address/get"
_TALK_MODE_BY_VALUE: dict[int, TalkMode] = {0: "none", 1: "full_duplex", 3: "half_duplex"}


class EzvizApiError(RuntimeError):
    """A server-side EZVIZ error without the originating request body."""

    def __init__(self, code: str, message: str, endpoint: str) -> None:
        self.code = code
        self.message = message
        self.endpoint = endpoint
        super().__init__(f"EZVIZ {endpoint} failed (code={code}): {message}")


class EzvizClient:
    """Call documented EZVIZ endpoints while keeping credentials out of reprs."""

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        session: requests.Session | Any | None = None,
        access_token: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._app_key = app_key
        self._app_secret = app_secret
        self._session = requests.Session() if session is None else session
        self._access_token = access_token
        self._timeout = timeout

    def __repr__(self) -> str:
        return f"EzvizClient(app_key='***', credential='***', access_token={'***' if self._access_token else None}, timeout={self._timeout!r})"

    def get_access_token(self) -> AccessToken:
        data = self._post(TOKEN_ENDPOINT, {"appKey": self._app_key, "appSecret": self._app_secret})
        token = data.get("accessToken")
        if not isinstance(token, str) or not token:
            raise EzvizApiError("invalid_response", "missing access token", TOKEN_ENDPOINT)
        expiry = data.get("expireTime")
        access_token = AccessToken(value=token, expires_at_ms=expiry if isinstance(expiry, int) else None)
        self._access_token = access_token.value
        return access_token

    def list_devices(self) -> list[EzvizDevice]:
        data = self._post(DEVICE_LIST_ENDPOINT, {"accessToken": self._require_access_token()})
        if not isinstance(data, list):
            raise EzvizApiError("invalid_response", "missing device list", DEVICE_LIST_ENDPOINT)
        return [self._device_from_payload(item) for item in data if isinstance(item, Mapping)]

    def get_live_address(self, device_serial: str, channel_no: int = 1) -> str:
        if channel_no < 1:
            raise ValueError("channel_no must be at least 1")
        data = self._post(
            LIVE_ADDRESS_ENDPOINT,
            {"accessToken": self._require_access_token(), "deviceSerial": device_serial, "channelNo": channel_no},
            secrets=(device_serial,),
        )
        url = data.get("url")
        if not isinstance(url, str) or not url:
            raise EzvizApiError("invalid_response", "missing live address", LIVE_ADDRESS_ENDPOINT)
        return url

    def _require_access_token(self) -> str:
        return self._access_token or self.get_access_token().value

    def _post(self, endpoint: str, body: dict[str, object], secrets: tuple[str, ...] = ()) -> Any:
        response = self._session.post(f"{BASE_URL}{endpoint}", data=body, timeout=self._timeout)
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise EzvizApiError("invalid_response", "invalid JSON response", endpoint) from exc
        if not isinstance(payload, Mapping):
            raise EzvizApiError("invalid_response", "invalid response", endpoint)
        if str(payload.get("code")) != "200":
            message = str(payload.get("msg") or payload.get("message") or "request failed")
            raise EzvizApiError(str(payload.get("code", "unknown")), self._redact(message, *secrets), endpoint)
        return payload.get("data")

    def _redact(self, message: str, *additional_secrets: str) -> str:
        for value in (self._app_key, self._app_secret, self._access_token, *additional_secrets):
            if value:
                message = message.replace(value, "***")
        return message

    @staticmethod
    def _device_from_payload(payload: Mapping[str, Any]) -> EzvizDevice:
        support_talk = payload.get("supportTalk", payload.get("support_talk"))
        talk_mode = _TALK_MODE_BY_VALUE.get(support_talk, "unknown") if isinstance(support_talk, int) else "unknown"
        status = payload.get("status")
        return EzvizDevice(
            serial=str(payload.get("deviceSerial", "")),
            model=_first_string(payload, "deviceName", "model", "deviceModel"),
            online=_online_status(status),
            channel_count=_first_int(payload, "channelNumber", "channelNo", "channelCount"),
            talk_mode=talk_mode,
            capabilities=dict(payload),
        )


def _first_string(payload: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def _first_int(payload: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int):
            return value
    return None


def _online_status(status: Any) -> bool | None:
    if status in (1, "1", True):
        return True
    if status in (0, "0", False):
        return False
    return None
