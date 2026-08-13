"""Conservative source-label to unified-phase mappings."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping


@dataclass(frozen=True)
class LabelMapping:
    phase: str | None
    coarse_event: str | None
    hard_negative: str | None
    supervision_mask: tuple[str, ...]
    source_label: str
    mapping_reason: str


def map_source_label(
    dataset: str,
    label: str,
    metadata: Mapping[str, object],
) -> LabelMapping:
    """Map only labels with an explicit, reviewable source rule."""
    source_label = str(label)
    dataset_key = _canonical(dataset)
    label_key = _canonical(source_label)
    if dataset_key == "gmdcsa24":
        return _map_gmdcsa24(source_label, label_key)
    if dataset_key == "prevfall":
        return _map_prevfall(source_label, label_key)
    if dataset_key == "caucafall":
        return _map_caucafall(source_label, label_key)
    if dataset_key in {"upfall3dskeletons", "upfall3d"}:
        return _map_upfall(source_label, label_key)
    if dataset_key in {"ntu120", "nturgbd120"}:
        return _map_ntu120(source_label, label_key)
    return _unknown(source_label, "unmapped_dataset")


def _map_gmdcsa24(source: str, key: str) -> LabelMapping:
    if key == "adl":
        return _normal(source, "gmdcsa24_adl")
    if key in {"falling", "fallingfw", "fallingbw", "fallingsw"}:
        return _coarse_fall(source, "gmdcsa24_timed_fall_interval")
    return _unknown(source, "unmapped_source_label")


def _map_prevfall(source: str, key: str) -> LabelMapping:
    if key == "normal":
        return _normal(source, "prevfall_normal")
    if key == "abnormal":
        return LabelMapping(
            "prefall_abnormal",
            None,
            None,
            ("phase", "prefall"),
            source,
            "prevfall_abnormal",
        )
    if key == "fall":
        return _coarse_fall(source, "prevfall_coarse_fall")
    return _unknown(source, "unmapped_source_label")


def _map_caucafall(source: str, key: str) -> LabelMapping:
    hard_negatives = {
        "sitting": "sit_down",
        "sitdown": "sit_down",
        "hopping": "hopping",
        "hop": "hopping",
        "pickingupanobject": "pick_up",
        "pickupobject": "pick_up",
        "kneeling": "kneel",
        "kneel": "kneel",
        "walking": "walking",
        "walk": "walking",
    }
    if key in hard_negatives:
        return _normal(source, f"caucafall_{hard_negatives[key]}", hard_negatives[key])
    if key.startswith("fall") and key in {
        "fallforward",
        "fallbackwards",
        "fallleft",
        "fallright",
        "fallsitting",
    }:
        return _coarse_fall(source, "caucafall_coarse_fall")
    return _unknown(source, "unmapped_source_label")


def _map_upfall(source: str, key: str) -> LabelMapping:
    if key in {"impact", "label1"}:
        return LabelMapping(
            "impact",
            "fall",
            None,
            ("phase", "fall_event"),
            source,
            "upfall_explicit_impact",
        )
    return _unknown(source, "upfall_nonimpact_not_normalized")


def _map_ntu120(source: str, key: str) -> LabelMapping:
    if key in {"a42", "staggering"}:
        return LabelMapping(
            "prefall_abnormal",
            None,
            None,
            ("phase", "prefall"),
            source,
            "ntu120_staggering",
        )
    if key in {"a43", "fallingdown"}:
        return _coarse_fall(source, "ntu120_falling_down")
    hard_negatives = {
        "a5": "drop",
        "a6": "pick_up",
        "a8": "sit_down",
        "a9": "stand_up",
        "a80": "squat",
    }
    if key in hard_negatives:
        return _normal(source, f"ntu120_{hard_negatives[key]}", hard_negatives[key])
    return _unknown(source, "unmapped_source_label")


def _normal(source: str, reason: str, hard_negative: str | None = None) -> LabelMapping:
    return LabelMapping(
        "normal_adl",
        "adl",
        hard_negative,
        ("phase", "fall_event"),
        source,
        reason,
    )


def _coarse_fall(source: str, reason: str) -> LabelMapping:
    return LabelMapping(None, "fall", None, ("fall_event",), source, reason)


def _unknown(source: str, reason: str) -> LabelMapping:
    return LabelMapping(None, None, None, ("none",), source, reason)


def _canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())
