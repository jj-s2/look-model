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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline-fixture", nargs="?", const=Path("tests/fixtures/ezviz_offline_unavailable.json"), type=Path,
        help="render the bundled (or supplied) unavailable fixture without API access",
    )
    parser.add_argument("--write-report", type=Path, help="write a redaction-safe device capability Markdown report")
    return parser.parse_args(argv)


def _print_device(device: EzvizDevice) -> None:
    state = "available" if device.online is True else "unavailable"
    channels = device.channel_count if device.channel_count is not None else "unknown"
    model = _safe_display(device.model or "unknown", device.serial)
    serial = f"***{device.serial[-4:]}" if len(device.serial) > 4 else "***"
    print(f"{state}: model={model} online={device.online} channels={channels} talk={device.talk_mode} serial={serial}")


def _safe_display(value: str, serial: str) -> str:
    """Prevent device metadata from echoing a serial number into console output."""
    return value.replace(serial, "***") if serial else value


def _offline_devices(path: Path) -> list[EzvizDevice]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("offline fixture must contain a device list")
    return [EzvizClient._device_from_payload(item) for item in entries if isinstance(item, dict)]


def _write_report(path: Path, devices: list[EzvizDevice], *, fixture: bool) -> None:
    """Write only observed capability state; fixture mode is never live evidence."""
    lines = [
        "# Device capability report", "",
        "## Evidence status", "",
        "- **unavailable**: this report was generated from an offline fixture; no network request was made.",
        "- No real C6c, live stream, intercom, or SDNL1 data-field validation is claimed.",
        "- A live probe may replace a capability only with an observed API response; missing fields remain `unavailable`.",
        "", "## Device inventory", "",
        "| Device/model | Online | Live stream | Intercom | SDNL1 data fields | Evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for device in devices:
        model = _safe_display(device.model or "unknown", device.serial)
        online = "available" if device.online is True and not fixture else "unavailable"
        stream = "unavailable" if fixture else "unavailable (not probed by inventory endpoint)"
        talk = "unavailable" if fixture or device.talk_mode == "unknown" else device.talk_mode
        sdnl1 = "unavailable"
        lines.append(f"| {model} | {online} | {stream} | {talk} | {sdnl1} | {'offline fixture' if fixture else 'live inventory'} |")
    if not devices:
        lines.append("| unavailable | unavailable | unavailable | unavailable | unavailable | no device response |")
    lines.extend([
        "", "## Privacy", "",
        "Device serial numbers are intentionally omitted; only a terminal four-digit mask may appear in console output.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None, session: object | None = None) -> int:
    args = _parse_args(argv)
    if args.offline_fixture:
        try:
            devices = _offline_devices(args.offline_fixture)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"offline fixture unavailable: {exc}", file=sys.stderr)
            return 2
        for device in devices:
            _print_device(device)
        if args.write_report:
            _write_report(args.write_report, devices, fixture=True)
        return 0

    settings = Settings.from_env()
    missing = [name for name, value in (("EZVIZ_APP_KEY", settings.ezviz_app_key), ("EZVIZ_APP_SECRET", settings.ezviz_app_secret)) if not value]
    if missing:
        print(f"missing required configuration: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        devices = EzvizClient(settings.ezviz_app_key, settings.ezviz_app_secret, session=session).list_devices()
    except EzvizApiError as exc:
        print(f"EZVIZ probe unavailable: endpoint={exc.endpoint} code={exc.code}", file=sys.stderr)
        return 1
    for device in devices:
        _print_device(device)
    if args.write_report:
        _write_report(args.write_report, devices, fixture=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
