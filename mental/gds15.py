"""Versioned GDS-15 scoring; this is a screening aid, never a diagnosis."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class GDS15Item:
    id: str
    question_zh: str
    options: tuple[str, str]
    risk_answer: bool


@dataclass(frozen=True)
class GDS15Result:
    score: int
    risk_band: str
    completed_at: datetime
    version: str
    is_diagnosis: bool
    disclaimer: str


class GDS15:
    """Loads a reviewed 15-item configuration and scores only its stable IDs."""

    SCREENING_DISCLAIMER = "本量表仅用于筛查，不构成诊断；如有担忧，请联系专业医疗人员。"

    def __init__(self, *, version: str, source_url: str, items: tuple[GDS15Item, ...]):
        if len(items) != 15:
            raise ValueError("GDS-15 configuration must contain exactly 15 items")
        if len({item.id for item in items}) != 15:
            raise ValueError("GDS-15 item IDs must be unique")
        self.version = version
        self.source_url = source_url
        self.items = items

    @classmethod
    def from_json(cls, path: str | Path) -> "GDS15":
        with Path(path).open(encoding="utf-8") as handle:
            config = json.load(handle)
        metadata = config.get("metadata", {})
        items = tuple(
            GDS15Item(
                id=str(item["id"]),
                question_zh=str(item["question_zh"]),
                options=tuple(item["options"]),
                risk_answer=bool(item["risk_answer"]),
            )
            for item in config.get("items", [])
        )
        if any(len(item.options) != 2 for item in items):
            raise ValueError("each GDS-15 item must have yes/no options")
        return cls(version=str(metadata["version"]), source_url=str(metadata["source_url"]), items=items)

    def score(self, answers: Mapping[str, bool], completed_at: datetime | None = None) -> GDS15Result:
        expected = {item.id for item in self.items}
        if len(answers) != 15 or set(answers) != expected:
            raise ValueError("GDS-15 requires exactly 15 answers with configured item IDs")
        if any(not isinstance(answer, bool) for answer in answers.values()):
            raise ValueError("GDS-15 answers must be boolean")
        score = sum(answers[item.id] == item.risk_answer for item in self.items)
        timestamp = completed_at or datetime.now(timezone.utc)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("completed_at must be timezone-aware")
        return GDS15Result(
            score=score,
            risk_band=self._risk_band(score),
            completed_at=timestamp,
            version=self.version,
            is_diagnosis=False,
            disclaimer=self.SCREENING_DISCLAIMER,
        )

    @staticmethod
    def _risk_band(score: int) -> str:
        if score <= 4:
            return "lower_screening_risk"
        if score <= 8:
            return "elevated_screening_risk"
        return "high_screening_risk"
