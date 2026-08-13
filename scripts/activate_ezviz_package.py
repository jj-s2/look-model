"""Activate one competition package code for a configured EZVIZ channel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.settings import Settings
from devices.ezviz_client import EzvizApiError, EzvizClient


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-code", action="append", required=True, help="competition package activation code")
    parser.add_argument("--serial", help="device serial; defaults to EZVIZ_DEVICE_SERIAL")
    parser.add_argument("--channel", type=int, default=1, help="device channel number")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, client: Any | None = None) -> int:
    args = _parse_args(argv)
    settings = Settings.from_env()
    serial = args.serial or settings.ezviz_device_serial
    missing = [
        name
        for name, value in (
            ("EZVIZ_APP_KEY", settings.ezviz_app_key),
            ("EZVIZ_APP_SECRET", settings.ezviz_app_secret),
            ("EZVIZ_DEVICE_SERIAL", serial),
        )
        if not value
    ]
    if missing:
        print(f"missing required configuration: {', '.join(missing)}", file=sys.stderr)
        return 2
    if args.channel < 1:
        print("--channel must be at least 1", file=sys.stderr)
        return 2

    activation_client = client or EzvizClient(settings.ezviz_app_key or "", settings.ezviz_app_secret or "")
    for index, package_code in enumerate(args.package_code, 1):
        try:
            result = activation_client.activate_device_package(package_code, serial, args.channel)
        except EzvizApiError as exc:
            print(f"candidate_{index}: api_failed code={exc.code}")
            continue
        if result.activated:
            print(f"activation=success candidate={index} active_code={result.active_code}")
            return 0
        print(f"candidate_{index}: rejected active_code={result.active_code}")
    print("activation=failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
