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
    expected_items = [
        ("q01", "Are you basically satisfied with your life?", "你对自己的生活基本上满意吗？", False),
        ("q02", "Have you dropped many of your activities and interests?", "你是否已放弃了自己的很多活动和兴趣？", True),
        ("q03", "Do you feel that your life is empty?", "你是否觉得生活空虚？", True),
        ("q04", "Do you often get bored?", "你是否常常感到无聊？", True),
        ("q05", "Are you in good spirits most of the time?", "你是否常常感到精神不错，精神还可以？", False),
        ("q06", "Are you afraid that something bad is going to happen to you?", "你是否害怕会有不好的事情发生在你身上呢？", True),
        ("q07", "Do you feel happy most of the time?", "你大部分时间心情还可以吗？", False),
        ("q08", "Do you often feel helpless?", "你是否经常觉得无助？", True),
        ("q09", "Do you prefer to stay at home, rather than going out and doing new things?", "你是否宁愿留在家里，而不外出并做些新的事呢？", True),
        ("q10", "Do you feel you have more problems with memory than most?", "你是否觉得你的记忆力比多数人差？", True),
        ("q11", "Do you think it is wonderful to be alive now?", "你是否觉得活着真好，活着还不错？", False),
        ("q12", "Do you feel pretty worthless the way you are now?", "你是否觉得自己现在很没用呢？", True),
        ("q13", "Do you feel full of energy?", "你是否感到精力充足，精力足够应付日常的生活？", False),
        ("q14", "Do you feel that your situation is hopeless?", "你是否觉得自己的处境没有希望？", True),
        ("q15", "Do you think that most people are better off than you are?", "你是否觉得多数人都比你活得好吗？", True),
    ]
    actual_items = [
        (item["id"], item["question_en"], item["question_zh"], item["risk_answer"])
        for item in config["items"]
    ]
    assert actual_items == expected_items
    assert config["metadata"]["exact_order"] == [item[0] for item in expected_items]
    assert config["metadata"]["source_urls"] == [
        "https://web.stanford.edu/~yesavage/GDS",
        "https://web.stanford.edu/~yesavage/Chinese.html",
        "https://web.stanford.edu/~yesavage/Australian%20Chinese.pdf",
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
