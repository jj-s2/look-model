"""Versioned, license-aware dataset metadata and annotation contracts."""

from .registry import DatasetEntry, DatasetRegistry, load_registry
from .labels import LabelMapping, map_source_label
from .schema import LabelSource, UnifiedClip, stable_clip_id

__all__ = [
    "DatasetEntry",
    "DatasetRegistry",
    "LabelSource",
    "LabelMapping",
    "UnifiedClip",
    "load_registry",
    "map_source_label",
    "stable_clip_id",
]
