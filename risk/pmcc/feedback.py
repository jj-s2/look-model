"""Append-only local storage for audited PMCC outcome feedback."""
from __future__ import annotations

import json
from pathlib import Path

from .schema import OutcomeFeedback, OutcomeType


class FeedbackStore:
    """Append feedback envelopes without modifying earlier forecast records."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def append(self, feedback: OutcomeFeedback) -> None:
        if not isinstance(feedback, OutcomeFeedback):
            raise ValueError("feedback must be an OutcomeFeedback")
        envelope = {
            "record_type": "outcome_feedback",
            "feedback": feedback.to_dict(),
            "training_label": None if feedback.outcome is OutcomeType.UNKNOWN else feedback.outcome.value,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(envelope, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
            stream.write("\n")

    def read(self, forecast_id: str | None = None) -> tuple[OutcomeFeedback, ...]:
        if forecast_id is not None and (not isinstance(forecast_id, str) or not forecast_id):
            raise ValueError("forecast_id must be a non-empty string or None")
        if not self._path.exists():
            return ()
        records: list[OutcomeFeedback] = []
        with self._path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL at line {line_number}") from error
                if item.get("record_type") != "outcome_feedback":
                    continue
                raw_feedback = item.get("feedback")
                feedback = OutcomeFeedback.from_dict(raw_feedback)
                stored_forecast_id = feedback.provenance.get("forecast_id")
                if forecast_id is None or stored_forecast_id == forecast_id:
                    records.append(feedback)
        return tuple(records)
