"""Read a small number of frames from the configured EZVIZ camera."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.settings import Settings
from devices.ezviz_client import EzvizApiError, EzvizClient
from devices.ezviz_stream import EzvizLiveUrlProvider
from vision.input_adapter import EzvizStreamAdapter


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=3, help="number of frames to read")
    parser.add_argument("--channel", type=int, default=1, help="EZVIZ channel number")
    parser.add_argument("--quality", type=int, choices=(1, 2), default=2, help="1=HD, 2=smooth")
    parser.add_argument("--save-first-frame", type=Path, help="optional path for the first BGR frame")
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    client: Any | None = None,
    capture_factory: Callable[[str], Any] | None = None,
) -> int:
    args = _parse_args(argv)
    if args.frames < 1:
        print("--frames must be at least 1", file=sys.stderr)
        return 2
    settings = Settings.from_env()
    missing = [
        name
        for name, value in (
            ("EZVIZ_APP_KEY", settings.ezviz_app_key),
            ("EZVIZ_APP_SECRET", settings.ezviz_app_secret),
            ("EZVIZ_DEVICE_SERIAL", settings.ezviz_device_serial),
        )
        if not value
    ]
    if missing:
        print(f"missing required configuration: {', '.join(missing)}", file=sys.stderr)
        return 2

    stream_client = client or EzvizClient(settings.ezviz_app_key, settings.ezviz_app_secret)
    provider = EzvizLiveUrlProvider(
        stream_client,
        settings.ezviz_device_serial,
        channel_no=args.channel,
        device_code=settings.ezviz_device_code,
        protocol=2,
        quality=args.quality,
    )
    adapter = EzvizStreamAdapter(provider, capture_factory=capture_factory, max_retries=2)
    first_frame = None
    count = 0
    try:
        for _ in range(args.frames):
            frame = adapter.read_frame()
            if frame is None:
                print(f"stream_failed state={adapter.health.state} reason={adapter.health.reason or 'unknown'}", file=sys.stderr)
                return 1
            if first_frame is None:
                first_frame = frame
            count += 1
    except EzvizApiError as exc:
        print(f"EZVIZ stream unavailable: endpoint={exc.endpoint} code={exc.code}", file=sys.stderr)
        return 1
    finally:
        adapter.release()

    if args.save_first_frame:
        import cv2

        args.save_first_frame.parent.mkdir(parents=True, exist_ok=True)
        if first_frame is None or not cv2.imwrite(str(args.save_first_frame), first_frame):
            print("failed to save first frame", file=sys.stderr)
            return 1
    shape = tuple(int(value) for value in first_frame.shape) if first_frame is not None else ()
    print(f"live_transport=success count={count} shape={shape} health=healthy content_validation=not_performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
