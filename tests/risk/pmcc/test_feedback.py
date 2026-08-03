from __future__ import annotations

import json
from datetime import UTC, datetime

from risk.pmcc.feedback import FeedbackStore
from risk.pmcc.schema import OutcomeFeedback, OutcomeType


def _feedback(outcome: OutcomeType) -> OutcomeFeedback:
    return OutcomeFeedback("subject-1", datetime(2026, 8, 3, tzinfo=UTC), outcome, 0.9, {"forecast_id": "forecast-1"})


def test_append_preserves_existing_forecast_and_unknown_is_not_training_label(tmp_path) -> None:
    path = tmp_path / "feedback.jsonl"
    original = {"record_type": "fall_forecast", "forecast_id": "forecast-1"}
    path.write_text(json.dumps(original) + "\n", encoding="utf-8")
    store = FeedbackStore(path)
    store.append(_feedback(OutcomeType.UNKNOWN))

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines[0] == original
    assert lines[1]["training_label"] is None
    assert store.read("forecast-1") == (_feedback(OutcomeType.UNKNOWN),)


def test_feedback_store_appends_instead_of_rewriting(tmp_path) -> None:
    path = tmp_path / "feedback.jsonl"
    store = FeedbackStore(path)
    store.append(_feedback(OutcomeType.NORMAL_ADL))
    store.append(_feedback(OutcomeType.NEAR_FALL))
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
