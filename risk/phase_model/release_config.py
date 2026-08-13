"""Validated, versioned RG-PCNet release decision configuration."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real
from pathlib import Path
from typing import Any, Iterable


_SCHEMA_VERSION = "rgpc.release.v1"
_FIELD_NAMES = (
    "schema_version",
    "release_id",
    "model_sha256",
    "dataset_sha256",
    "split_sha256",
    "temperature",
    "fall_threshold",
    "reliability_threshold",
    "confirm_seconds",
    "recovery_seconds",
    "cooldown_seconds",
    "minimum_coverage",
)
_HEX_CHARACTERS = frozenset("0123456789abcdef")


def _number(value: object, name: str) -> Real | Decimal:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{name} must be a finite number")
    return value


def _finite_float(value: object, name: str) -> float:
    numeric = _number(value, name)
    try:
        normalized = float(numeric)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return 0.0 if normalized == 0.0 else normalized


def _closed_range(value: object, name: str, low: float, high: float) -> float:
    numeric = _number(value, name)
    try:
        in_range = low <= numeric <= high
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not in_range:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return _finite_float(numeric, name)


def _seconds(value: object, name: str, *, strictly_positive: bool) -> float:
    numeric = _number(value, name)
    try:
        invalid = numeric <= 0.0 if strictly_positive else numeric < 0.0
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if invalid:
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    normalized = _finite_float(numeric, name)
    if strictly_positive and normalized == 0.0:
        raise ValueError(f"{name} must be positive after float normalization")
    return normalized


def _sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX_CHARACTERS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class RGPCReleaseConfig:
    """Immutable release-time thresholds, timers, and provenance hashes."""

    schema_version: str
    release_id: str
    model_sha256: str
    dataset_sha256: str
    split_sha256: str
    temperature: float
    fall_threshold: float
    reliability_threshold: float
    confirm_seconds: float
    recovery_seconds: float
    cooldown_seconds: float
    minimum_coverage: float

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be exactly {_SCHEMA_VERSION!r}")
        if type(self.release_id) is not str or not self.release_id.strip():
            raise ValueError("release_id must be a non-empty string")

        release_id = self.release_id.strip()
        if any(0xD800 <= ord(character) <= 0xDFFF for character in release_id):
            raise ValueError("release_id must not contain isolated Unicode surrogates")
        object.__setattr__(self, "release_id", release_id)
        for name in ("model_sha256", "dataset_sha256", "split_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        object.__setattr__(
            self, "temperature", _closed_range(self.temperature, "temperature", 0.5, 5.0)
        )
        for name in (
            "fall_threshold",
            "reliability_threshold",
            "minimum_coverage",
        ):
            object.__setattr__(self, name, _closed_range(getattr(self, name), name, 0.0, 1.0))
        object.__setattr__(
            self,
            "confirm_seconds",
            _seconds(self.confirm_seconds, "confirm_seconds", strictly_positive=True),
        )
        for name in ("recovery_seconds", "cooldown_seconds"):
            object.__setattr__(
                self,
                name,
                _seconds(getattr(self, name), name, strictly_positive=False),
            )

    def to_dict(self) -> dict[str, str | float]:
        """Return a detached JSON-safe snapshot in frozen contract field order."""

        return {
            "schema_version": self.schema_version,
            "release_id": self.release_id,
            "model_sha256": self.model_sha256,
            "dataset_sha256": self.dataset_sha256,
            "split_sha256": self.split_sha256,
            "temperature": self.temperature,
            "fall_threshold": self.fall_threshold,
            "reliability_threshold": self.reliability_threshold,
            "confirm_seconds": self.confirm_seconds,
            "recovery_seconds": self.recovery_seconds,
            "cooldown_seconds": self.cooldown_seconds,
            "minimum_coverage": self.minimum_coverage,
        }


class _InvalidJSONConstant(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise _InvalidJSONConstant(value)


def _object_without_duplicates(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _target_path(path: os.PathLike[str] | str) -> Path:
    try:
        return Path(path)
    except (TypeError, ValueError) as exc:
        raise ValueError("release config path must be path-like") from exc


def load_release_config(path: os.PathLike[str] | str) -> RGPCReleaseConfig:
    """Load a strict UTF-8 JSON release decision contract from ``path``."""

    target = _target_path(path)
    try:
        document = target.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("release config must be valid UTF-8 JSON") from exc
    except OSError as exc:
        raise ValueError("unable to read release config") from exc

    try:
        payload = json.loads(
            document,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=Decimal,
        )
    except (json.JSONDecodeError, _InvalidJSONConstant) as exc:
        raise ValueError("invalid release config JSON") from exc

    if type(payload) is not dict:
        raise ValueError("release config root must be a JSON object")

    actual_fields = set(payload)
    expected_fields = set(_FIELD_NAMES)
    missing = sorted(expected_fields - actual_fields)
    unknown = sorted(actual_fields - expected_fields)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        raise ValueError(f"release config fields must match schema ({'; '.join(details)})")

    return RGPCReleaseConfig(**payload)


def write_release_config(
    config: RGPCReleaseConfig, path: os.PathLike[str] | str
) -> None:
    """Atomically write canonical JSON, creating missing parent directories."""

    if not isinstance(config, RGPCReleaseConfig):
        raise ValueError("config must be an RGPCReleaseConfig")
    target = _target_path(path)
    encoded = (
        json.dumps(
            # Dispatch through the base implementation so a subclass cannot
            # bypass the validated, versioned contract with an override.
            RGPCReleaseConfig.to_dict(config),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
