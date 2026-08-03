"""Versioned, license-aware dataset metadata and annotation contracts."""

from .registry import DatasetEntry, DatasetRegistry, load_registry

__all__ = ["DatasetEntry", "DatasetRegistry", "load_registry"]
