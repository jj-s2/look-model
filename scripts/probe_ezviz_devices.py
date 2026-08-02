"""Print a redaction-safe inventory of configured EZVIZ devices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.settings import Settings
from devices.ezviz_client import EzvizApiError, EzvizClient
from devices.models import EzvizDevice


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-fixture", type=Path, help="JSON device response to render without API access")
    return parser.parse_args()


def _print_device(device: EzvizDevice) -> None:
    state = "available" if device.online is True else "unavailable"
    channels = device.channel_count if device.channel_count is not None else "unknown"
    model = device.model or "unknown"
    print(f"{state}: model={model} online={device.online} channels={channels} talk={device.talk_mode} serial=***{device.serial[-4:]}")


def _offline_devices(path: Path) -> list[EzvizDevice]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("offline fixture must contain a device list")
    return [EzvizClient._device_from_payload(item) for item in entries if isinstance(item, dict)]


def main() -> int:
    args = _parse_args()
    if args.offline_fixture:
        try:
            devices = _offline_devices(args.offline_fixture)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"offline fixture unavailable: {exc}", file=sys.stderr)
            return 2
        for device in devices:
            _print_device(device)
        return 0

    settings = Settings.from_env()
    missing = [name for name, value in (("EZVIZ_APP_KEY", settings.ezviz_app_key), ("EZVIZ_APP_SECRET", settings.ezviz_app_secret)) if not value]
    if missing:
        print(f"missing required configuration: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        devices = EzvizClient(settings.ezviz_app_key, settings.ezviz_app_secret).list_devices()
    except EzvizApiError as exc:
        print(f"EZVIZ probe unavailable: endpoint={exc.endpoint} code={exc.code}", file=sys.stderr)
        return 1
    for device in devices:
        _print_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
