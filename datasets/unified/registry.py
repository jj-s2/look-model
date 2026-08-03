"""Strict validation for the project's governed dataset registry."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterator
from urllib.parse import urlparse


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_FIELDS = {
    "name",
    "version",
    "official_source",
    "license",
    "license_status",
    "download_status",
    "expected_size_bytes",
    "sha256",
    "subjects",
    "split_strategy",
    "redistribution",
}
_VALID_LICENSE_STATUS = {"confirmed", "needs_confirmation", "unknown"}
_VALID_DOWNLOAD_STATUS = {"not_downloaded", "deferred", "completed", "failed"}


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    version: str
    official_source: str
    license: str
    license_status: str
    download_status: str
    expected_size_bytes: int | None
    sha256: str | None
    subjects: int | None
    split_strategy: str
    redistribution: str
    training_allowed: bool = True
    academic_only: bool = False
    role: tuple[str, ...] = ()
    raw_video_license: str | None = None

    @property
    def can_train(self) -> bool:
        return self.training_allowed and self.license_status == "confirmed"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "DatasetEntry":
        missing = sorted(_REQUIRED_FIELDS.difference(raw))
        if missing:
            raise ValueError(f"dataset entry missing fields: {', '.join(missing)}")
        name = _required_text(raw, "name")
        version = _required_text(raw, "version")
        official_source = _required_text(raw, "official_source")
        parsed = urlparse(official_source)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("official_source must be an HTTPS URL")
        license_name = _required_text(raw, "license")
        license_status = _required_text(raw, "license_status")
        if license_status not in _VALID_LICENSE_STATUS:
            raise ValueError(f"invalid license_status: {license_status}")
        download_status = _required_text(raw, "download_status")
        if download_status not in _VALID_DOWNLOAD_STATUS:
            raise ValueError(f"invalid download_status: {download_status}")
        expected_size = raw["expected_size_bytes"]
        if expected_size is not None and (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
        ):
            raise ValueError("expected_size_bytes must be a positive integer or null")
        sha256 = raw["sha256"]
        if sha256 is not None and (
            not isinstance(sha256, str) or not _SHA256.fullmatch(sha256.lower())
        ):
            raise ValueError("sha256 must be a 64-character lowercase hex string or null")
        if download_status == "completed" and not sha256:
            raise ValueError("completed dataset requires sha256")
        subjects = raw["subjects"]
        if subjects is not None and (
            isinstance(subjects, bool) or not isinstance(subjects, int) or subjects <= 0
        ):
            raise ValueError("subjects must be a positive integer or null")
        training_allowed = raw.get("training_allowed", True)
        if not isinstance(training_allowed, bool):
            raise ValueError("training_allowed must be boolean")
        role = raw.get("role", [])
        if not isinstance(role, list) or not all(isinstance(item, str) and item for item in role):
            raise ValueError("role must be a list of non-empty strings")
        return cls(
            name=name,
            version=version,
            official_source=official_source,
            license=license_name,
            license_status=license_status,
            download_status=download_status,
            expected_size_bytes=expected_size,
            sha256=sha256.lower() if sha256 else None,
            subjects=subjects,
            split_strategy=_required_text(raw, "split_strategy"),
            redistribution=_required_text(raw, "redistribution"),
            training_allowed=training_allowed,
            academic_only=bool(raw.get("academic_only", False)),
            role=tuple(role),
            raw_video_license=raw.get("raw_video_license"),
        )


@dataclass(frozen=True)
class DatasetRegistry:
    schema_version: str
    entries: tuple[DatasetEntry, ...]

    def __getitem__(self, name: str) -> DatasetEntry:
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise KeyError(name)

    def __iter__(self) -> Iterator[DatasetEntry]:
        return iter(self.entries)


def load_registry(path: Path) -> DatasetRegistry:
    """Load and validate a version 2.0 dataset manifest."""
    manifest_path = Path(path)
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read dataset manifest: {manifest_path}") from error
    if not isinstance(raw_manifest, dict) or raw_manifest.get("schema_version") != "2.0":
        raise ValueError("dataset manifest schema_version must be 2.0")
    raw_entries = raw_manifest.get("datasets")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("dataset manifest datasets must be a non-empty list")
    entries = tuple(
        DatasetEntry.from_mapping(entry)
        for entry in raw_entries
        if isinstance(entry, dict)
    )
    if len(entries) != len(raw_entries):
        raise ValueError("each dataset entry must be an object")
    names = [entry.name for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("dataset names must be unique")
    return DatasetRegistry(schema_version="2.0", entries=entries)


def _required_text(raw: dict[str, Any], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()
