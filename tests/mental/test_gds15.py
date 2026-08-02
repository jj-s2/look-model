import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mental.gds15 import GDS15


@pytest.fixture
def scale():
    return GDS15.from_json(Path("configs/screening/gds15_zh.json"))


def test_gds_resource_has_exactly_fifteen_items_and_source_metadata(scale):
    assert len(scale.items) == 15
    assert scale.source_url == "https://web.stanford.edu/~yesavage/GDS"
    assert scale.version


def test_gds_resource_locks_stanford_order_risk_answers_and_sources():
    config = json.loads(Path("configs/screening/gds15_zh.json").read_text(encoding="utf-8"))
    expected_risk_answers = [False, True, True, True, False, True, False, True, True, True, False, True, False, True, True]
    assert config["metadata"]["exact_order"] == [f"q{number:02d}" for number in range(1, 16)]
    assert [item["id"] for item in config["items"]] == config["metadata"]["exact_order"]
    assert [item["risk_answer"] for item in config["items"]] == expected_risk_answers
    assert all(item["question_en"] for item in config["items"])
    assert config["metadata"]["source_urls"][:2] == [
        "https://web.stanford.edu/~yesavage/GDS",
        "https://web.stanford.edu/~yesavage/Chinese.html",
    ]


def test_gds_requires_exactly_fifteen_answers(scale):
    with pytest.raises(ValueError, match="15 answers"):
        scale.score({"q01": True})


def test_gds_score_uses_each_items_risk_answer(scale):
    answers = {item.id: item.risk_answer for item in scale.items}

    result = scale.score(answers, completed_at=datetime(2026, 8, 2, tzinfo=timezone.utc))

    assert result.score == 15
    assert result.is_diagnosis is False
    assert result.disclaimer == GDS15.SCREENING_DISCLAIMER
    assert result.completed_at == datetime(2026, 8, 2, tzinfo=timezone.utc)


def test_gds_rejects_unknown_or_non_boolean_answer_ids(scale):
    answers = {item.id: False for item in scale.items}
    answers["other"] = True
    with pytest.raises(ValueError, match="exactly 15"):
        scale.score(answers)


@pytest.mark.parametrize(("score", "expected"), [(0, "lower_screening_risk"), (4, "lower_screening_risk"), (5, "elevated_screening_risk"), (8, "elevated_screening_risk"), (9, "high_screening_risk"), (15, "high_screening_risk")])
def test_gds_risk_band_boundaries(scale, score, expected):
    answers = {item.id: not item.risk_answer for item in scale.items}
    for item in scale.items[:score]:
        answers[item.id] = item.risk_answer
    assert scale.score(answers).risk_band == expected


def test_gds_rejects_naive_completion_timestamp(scale):
    answers = {item.id: item.risk_answer for item in scale.items}
    with pytest.raises(ValueError, match="timezone-aware"):
        scale.score(answers, completed_at=datetime(2026, 8, 2))
