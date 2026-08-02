from datetime import datetime, timezone
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

