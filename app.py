"""Run the local monitoring service with optional, explicitly marked demo fixtures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from alerts.dispatcher import AlertDispatcher
from core.events import DataQuality, EventType, SensorEvent, Source
from pipeline.live_service import LiveMonitoringService
from ui.dashboard import build_dashboard


class DemoFixtureSource:
    """In-process fixture source; it never opens a device, file, or network connection."""

    name = "camera"

    def poll(self, now: datetime) -> list[SensorEvent]:
        return [
            SensorEvent(
                timestamp=now,
                source=Source.VISION,
                event_type=EventType.FALL_EVENT,
                payload={"subject_id": "demo-person", "confirmed": True, "demo": True},
                quality=DataQuality(True, 0.95, True, "demo_fixture"),
            )
        ]


def create_service(*, demo_fixtures: bool) -> LiveMonitoringService:
    sources = [DemoFixtureSource()] if demo_fixtures else []
    dispatcher = AlertDispatcher(Path("outputs") / "local_alerts.jsonl")
    return LiveMonitoringService(sources, dispatcher=dispatcher)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local, privacy-aware elderly monitoring demo")
    parser.add_argument("--demo-fixtures", action="store_true", help="use visibly marked in-process demo fixtures")
    parser.add_argument("--no-browser", action="store_true", help="run locally without launching a UI")
    parser.add_argument("--smoke-seconds", type=float, default=0, help="run bounded local smoke loop then exit")
    args = parser.parse_args()
    if args.smoke_seconds < 0:
        parser.error("--smoke-seconds must be non-negative")
    service = create_service(demo_fixtures=args.demo_fixtures)
    if args.smoke_seconds:
        deadline = time.monotonic() + args.smoke_seconds
        while True:
            snapshot = service.step()
            print(f"demo={str(snapshot.demo).lower()} camera={snapshot.camera_health} decisions={len(snapshot.decisions)}")
            if time.monotonic() >= deadline:
                return 0
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))
    if args.no_browser:
        snapshot = service.step()
        print(f"demo={str(snapshot.demo).lower()} camera={snapshot.camera_health} decisions={len(snapshot.decisions)}")
        return 0
    try:
        build_dashboard(service).launch(inbrowser=True)
    except RuntimeError as error:
        print(str(error))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
