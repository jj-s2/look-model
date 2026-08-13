"""Stable, serializable clip records for unified fall datasets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping


PHASES = (
    "normal_adl",
    "prefall_abnormal",
    "descending",
    "impact",
    "fallen",
    "recovering",
)
COARSE_EVENTS = {"fall", "adl"}
SUPERVISION_FIELDS = {"phase", "fall_event", "prefall", "recovery", "none"}


class LabelSource(str, Enum):
    OFFICIAL = "official"
    DERIVED = "derived"
    HUMAN = "human"
    UNKNOWN = "unknown"


def stable_clip_id(
    dataset: str,
    dataset_version: str,
    subject_id: str,
    camera_id: str,
    media_path: str,
    start_sec: float,
    end_sec: float,
) -> str:
    """Return a path-root-independent identifier for one temporal clip."""
    normalized_path = media_path.replace("\\", "/")
    payload = "|".join(
        [
            str(dataset),
            str(dataset_version),
            str(subject_id),
            str(camera_id),
            normalized_path,
            f"{float(start_sec):.6f}",
            f"{float(end_sec):.6f}",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class UnifiedClip:
    clip_id: str
    dataset: str
    dataset_version: str
    subject_id: str
    camera_id: str
    event_group_id: str
    media_path: str
    start_sec: float
    end_sec: float
    phase: str | None
    coarse_event: str | None
    hard_negative: str | None
    supervision_mask: tuple[str, ...]
    source_label: str
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        text_fields = (
            "clip_id",
            "dataset",
            "dataset_version",
            "subject_id",
            "camera_id",
            "event_group_id",
            "media_path",
            "source_label",
        )
        for field_name in text_fields:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        if len(self.clip_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.clip_id.lower()
        ):
            raise ValueError("clip_id must be a SHA-256 hex string")
        if _is_absolute_media_path(self.media_path):
            raise ValueError("media_path must be relative")
        if (
            isinstance(self.start_sec, bool)
            or isinstance(self.end_sec, bool)
            or not math.isfinite(self.start_sec)
            or not math.isfinite(self.end_sec)
            or self.start_sec < 0
            or self.end_sec <= self.start_sec
        ):
            raise ValueError("clip interval must be finite and non-negative")
        if self.phase is not None and self.phase not in PHASES:
            raise ValueError(f"invalid phase: {self.phase}")
        if self.coarse_event is not None and self.coarse_event not in COARSE_EVENTS:
            raise ValueError(f"invalid coarse_event: {self.coarse_event}")
        if not isinstance(self.supervision_mask, tuple):
            raise ValueError("supervision_mask must be a tuple")
        if any(item not in SUPERVISION_FIELDS for item in self.supervision_mask):
            raise ValueError("supervision_mask contains an unknown field")
        if (
            self.phase is None
            and not self.supervision_mask
            and self.provenance.get("metadata_only") != "true"
        ):
            raise ValueError("unknown phase requires supervision_mask")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.provenance.items()
        ):
            raise ValueError("provenance keys and values must be strings")

    def to_dict(self) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "subject_id": self.subject_id,
            "camera_id": self.camera_id,
            "event_group_id": self.event_group_id,
            "media_path": self.media_path.replace("\\", "/"),
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "phase": self.phase,
            "coarse_event": self.coarse_event,
            "hard_negative": self.hard_negative,
            "supervision_mask": list(self.supervision_mask),
            "source_label": self.source_label,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "UnifiedClip":
        try:
            values = dict(raw)
            values["supervision_mask"] = tuple(values["supervision_mask"])
            values["provenance"] = dict(values["provenance"])
            return cls(**values)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid unified clip") from error

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


def _is_absolute_media_path(value: str) -> bool:
    path = Path(value)
    return path.is_absolute() or PureWindowsPath(value).is_absolute()
