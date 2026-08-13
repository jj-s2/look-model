"""Small, redaction-safe client for the EZVIZ Open Platform."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Mapping

import requests

from .models import AccessToken, EzvizDevice, EzvizPackageActivation, TalkMode


BASE_URL = "https://open.ys7.com"
TOKEN_ENDPOINT = "/api/lapp/token/get"
DEVICE_LIST_ENDPOINT = "/api/lapp/device/list"
LIVE_ADDRESS_ENDPOINT = "/api/lapp/v2/live/address/get"
PACKAGE_ACTIVATE_ENDPOINT = "/api/v3/mall/device/package/code/active"
ENCODE_TYPE_CHANGE_ENDPOINT = "/api/lapp/device/encodeType/change"
_TALK_MODE_BY_VALUE: dict[int, TalkMode] = {0: "none", 1: "full_duplex", 3: "half_duplex"}
_TOKEN_SAFETY_MARGIN_MS = 60_000


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
        access_token: AccessToken | str | None = None,
        timeout: float = 10.0,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        self._app_key = app_key
        self._app_secret = app_secret
        self._session = requests.Session() if session is None else session
        self._access_token = access_token if isinstance(access_token, AccessToken) else (
            AccessToken(access_token) if access_token else None
        )
        self._timeout = timeout
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))

    def __repr__(self) -> str:
        return f"EzvizClient(app_key='***', credential='***', access_token={'***' if self._access_token else None}, timeout={self._timeout!r})"

    def get_access_token(self) -> AccessToken:
        data = self._post(TOKEN_ENDPOINT, {"appKey": self._app_key, "appSecret": self._app_secret})
        token = data.get("accessToken")
        if not isinstance(token, str) or not token:
            raise EzvizApiError("invalid_response", "missing access token", TOKEN_ENDPOINT)
        expiry = data.get("expireTime")
        access_token = AccessToken(value=token, expires_at_ms=expiry if isinstance(expiry, int) else None)
        self._access_token = access_token
        return access_token

    def list_devices(self) -> list[EzvizDevice]:
        data = self._post(DEVICE_LIST_ENDPOINT, {"accessToken": self._require_access_token()})
        if not isinstance(data, list):
            raise EzvizApiError("invalid_response", "missing device list", DEVICE_LIST_ENDPOINT)
        return [self._device_from_payload(item) for item in data if isinstance(item, Mapping)]

    def get_live_address(
        self,
        device_serial: str,
        channel_no: int = 1,
        device_code: str | None = None,
        protocol: int | None = None,
        quality: int | None = None,
    ) -> str:
        if channel_no < 1:
            raise ValueError("channel_no must be at least 1")
        if protocol is not None and protocol not in (1, 2, 3, 4):
            raise ValueError("protocol must be one of 1 (ezopen), 2 (hls), 3 (rtmp), or 4 (flv)")
        if quality is not None and quality not in (1, 2):
            raise ValueError("quality must be 1 (hd) or 2 (smooth)")
        body: dict[str, object] = {
            "accessToken": self._require_access_token(),
            "deviceSerial": device_serial,
            "channelNo": channel_no,
        }
        if device_code:
            # The Open Platform REST API calls the device verification code `code`.
            body["code"] = device_code
        if protocol is not None:
            body["protocol"] = protocol
        if quality is not None:
            body["quality"] = quality
        data = self._post(
            LIVE_ADDRESS_ENDPOINT,
            body,
            secrets=(device_serial, device_code or ""),
        )
        url = data.get("url")
        if not isinstance(url, str) or not url:
            raise EzvizApiError("invalid_response", "missing live address", LIVE_ADDRESS_ENDPOINT)
        return url

    def activate_device_package(
        self, package_device_id: str, device_serial: str, channel_no: int = 1
    ) -> EzvizPackageActivation:
        """Bind one competition package activation code to one device channel."""
        if not package_device_id:
            raise ValueError("package_device_id must not be empty")
        if not device_serial:
            raise ValueError("device_serial must not be empty")
        if channel_no < 1:
            raise ValueError("channel_no must be at least 1")
        response = self._session.post(
            f"{BASE_URL}{PACKAGE_ACTIVATE_ENDPOINT}",
            headers={"accessToken": self._require_access_token(), "Content-Type": "application/json"},
            json=[
                {
                    "packageDeviceId": package_device_id,
                    "deviceSerial": device_serial,
                    "channelNo": str(channel_no),
                }
            ],
            timeout=self._timeout,
        )
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise EzvizApiError("invalid_response", "invalid JSON response", PACKAGE_ACTIVATE_ENDPOINT) from exc
        if not isinstance(payload, Mapping):
            raise EzvizApiError("invalid_response", "invalid response", PACKAGE_ACTIVATE_ENDPOINT)
        meta = payload.get("meta")
        if not isinstance(meta, Mapping) or str(meta.get("code")) != "200":
            message = str(meta.get("message") if isinstance(meta, Mapping) else "request failed")
            raise EzvizApiError(
                str(meta.get("code", "unknown") if isinstance(meta, Mapping) else "unknown"),
                self._redact(message, package_device_id, device_serial),
                PACKAGE_ACTIVATE_ENDPOINT,
            )
        data = payload.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], Mapping):
            raise EzvizApiError("invalid_response", "missing activation result", PACKAGE_ACTIVATE_ENDPOINT)
        result = data[0]
        active_code = result.get("activeCode")
        if not isinstance(active_code, int):
            raise EzvizApiError("invalid_response", "missing activation status", PACKAGE_ACTIVATE_ENDPOINT)
        return EzvizPackageActivation(
            package_device_id=str(result.get("packageDeviceId") or package_device_id),
            active_code=active_code,
            active_message=str(result.get("activeMessage") or ""),
        )

    def change_encode_type(self, device_serial: str, channel_no: int = 1, encode_type: str = "H264") -> None:
        """Change a device channel's video encoding for standard-stream consumers."""
        if not device_serial:
            raise ValueError("device_serial must not be empty")
        if channel_no < 1:
            raise ValueError("channel_no must be at least 1")
        normalized = encode_type.upper()
        if normalized not in {"H264", "H265"}:
            raise ValueError("encode_type must be H264 or H265")
        self._post(
            ENCODE_TYPE_CHANGE_ENDPOINT,
            {
                "accessToken": self._require_access_token(),
                "deviceSerial": device_serial,
                "channelNo": channel_no,
                "encodeType": normalized,
            },
            secrets=(device_serial,),
        )

    def _require_access_token(self) -> str:
        if self._access_token is not None and self._token_is_valid(self._access_token):
            return self._access_token.value
        return self.get_access_token().value

    def _token_is_valid(self, token: AccessToken) -> bool:
        """Do not reuse a token whose expiry is unknown or within the safety window."""
        return token.expires_at_ms is not None and token.expires_at_ms - self._now_ms() > _TOKEN_SAFETY_MARGIN_MS

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
        token_value = self._access_token.value if self._access_token else None
        for value in (self._app_key, self._app_secret, token_value, *additional_secrets):
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
            model=_first_string(payload, "deviceType", "model", "deviceModel"),
            online=_online_status(status),
            channel_count=_first_int(payload, "cameraNum", "channelNumber", "channelNo", "channelCount"),
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
    if status in (0, "0", 2, "2", False):
        return False
    return None
