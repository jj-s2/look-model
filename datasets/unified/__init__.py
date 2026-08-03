"""Versioned, license-aware dataset metadata and annotation contracts."""

from .registry import DatasetEntry, DatasetRegistry, load_registry
from .schema import LabelSource, UnifiedClip, stable_clip_id

__all__ = [
    "DatasetEntry",
    "DatasetRegistry",
    "LabelSource",
    "UnifiedClip",
    "load_registry",
    "stable_clip_id",
]
