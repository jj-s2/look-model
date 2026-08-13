"""Local UTF-8 JSONL persistence for sensor events."""

from datetime import datetime
import json
from pathlib import Path

from core.events import EventType, SensorEvent


class JsonlEventStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, event: SensorEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as event_file:
            json.dump(event.to_dict(), event_file, ensure_ascii=False, separators=(",", ":"))
            event_file.write("\n")
            event_file.flush()

    def query(
        self,
        start: datetime,
        end: datetime,
        event_types: set[EventType] | None = None,
    ) -> list[SensorEvent]:
        if not self.path.exists():
            return []

        events: list[SensorEvent] = []
        with self.path.open("r", encoding="utf-8") as event_file:
            for line_number, line in enumerate(event_file, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        raise ValueError("event must be an object")
                    event = SensorEvent.from_dict(data)
                except (json.JSONDecodeError, ValueError, TypeError) as error:
                    raise ValueError(f"invalid event data at line {line_number}") from error
                if start <= event.timestamp <= end and (
                    event_types is None or event.event_type in event_types
                ):
                    events.append(event)
        return events
