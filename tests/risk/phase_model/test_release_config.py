from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, fields
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import risk.phase_model.release_config as release_config_module
from risk.phase_model.release_config import (
    RGPCReleaseConfig,
    load_release_config,
    write_release_config,
)


FIELD_NAMES = (
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


def _config(**overrides: object) -> RGPCReleaseConfig:
    values: dict[str, object] = {
        "schema_version": "rgpc.release.v1",
        "release_id": "r1",
        "model_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "temperature": 1.5,
        "fall_threshold": 0.6,
        "reliability_threshold": 0.4,
        "confirm_seconds": 0.8,
        "recovery_seconds": 2.0,
        "cooldown_seconds": 10.0,
        "minimum_coverage": 0.6,
    }
    values.update(overrides)
    return RGPCReleaseConfig(**values)


def test_release_config_round_trip_is_exact(tmp_path: Path) -> None:
    path = tmp_path / "release_config.json"

    write_release_config(_config(), path)

    assert load_release_config(path) == _config()


def test_release_contract_has_exact_frozen_field_order_and_detached_dict() -> None:
    config = _config()

    assert tuple(field.name for field in fields(config)) == FIELD_NAMES
    assert tuple(config.to_dict()) == FIELD_NAMES
    snapshot = config.to_dict()
    snapshot["release_id"] = "mutated"
    assert config.release_id == "r1"
    with pytest.raises(FrozenInstanceError):
        config.release_id = "mutated"  # type: ignore[misc]


def test_release_id_is_canonicalized_by_stripping_outer_whitespace() -> None:
    config = _config(release_id="  发布 r1  ")

    assert config.release_id == "发布 r1"
    assert config.to_dict()["release_id"] == "发布 r1"


@pytest.mark.parametrize("release_id", ["", " ", "\t\r\n", 1, None, True])
def test_release_id_rejects_empty_or_non_string_values(release_id: object) -> None:
    with pytest.raises(ValueError, match="release_id"):
        _config(release_id=release_id)


@pytest.mark.parametrize("schema_version", ["rgpc.release.v2", "", 1, None, True])
def test_schema_version_must_match_exactly(schema_version: object) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        _config(schema_version=schema_version)


@pytest.mark.parametrize("field_name", ["model_sha256", "dataset_sha256", "split_sha256"])
@pytest.mark.parametrize(
    "invalid_hash",
    [
        "a" * 63,
        "a" * 65,
        "g" * 64,
        "A" * 64,
        "0" * 63 + "-",
        b"a" * 64,
        None,
    ],
)
def test_every_hash_rejects_wrong_case_length_charset_or_type(
    field_name: str, invalid_hash: object
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _config(**{field_name: invalid_hash})


@pytest.mark.parametrize(
    ("field_name", "low", "high"),
    [
        ("temperature", 0.5, 5.0),
        ("fall_threshold", 0.0, 1.0),
        ("reliability_threshold", 0.0, 1.0),
        ("minimum_coverage", 0.0, 1.0),
    ],
)
def test_closed_numeric_ranges_accept_both_exact_endpoints(
    field_name: str, low: float, high: float
) -> None:
    assert getattr(_config(**{field_name: low}), field_name) == low
    assert getattr(_config(**{field_name: high}), field_name) == high


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("temperature", math.nextafter(0.5, -math.inf)),
        ("temperature", math.nextafter(5.0, math.inf)),
        ("fall_threshold", math.nextafter(0.0, -math.inf)),
        ("fall_threshold", math.nextafter(1.0, math.inf)),
        ("reliability_threshold", math.nextafter(0.0, -math.inf)),
        ("reliability_threshold", math.nextafter(1.0, math.inf)),
        ("minimum_coverage", math.nextafter(0.0, -math.inf)),
        ("minimum_coverage", math.nextafter(1.0, math.inf)),
        ("confirm_seconds", 0.0),
        ("confirm_seconds", math.nextafter(0.0, -math.inf)),
        ("recovery_seconds", math.nextafter(0.0, -math.inf)),
        ("cooldown_seconds", math.nextafter(0.0, -math.inf)),
    ],
)
def test_every_numeric_boundary_rejects_the_first_out_of_range_value(
    field_name: str, invalid_value: float
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _config(**{field_name: invalid_value})


def test_seconds_contract_accepts_positive_confirm_and_zero_other_timers() -> None:
    smallest_positive = math.nextafter(0.0, math.inf)

    config = _config(
        confirm_seconds=smallest_positive,
        recovery_seconds=0,
        cooldown_seconds=Decimal("0"),
    )

    assert config.confirm_seconds == smallest_positive
    assert config.recovery_seconds == 0.0
    assert config.cooldown_seconds == 0.0


@pytest.mark.parametrize(
    "field_name",
    [
        "temperature",
        "fall_threshold",
        "reliability_threshold",
        "confirm_seconds",
        "recovery_seconds",
        "cooldown_seconds",
        "minimum_coverage",
    ],
)
@pytest.mark.parametrize("invalid_value", [True, False, "0.8", None, complex(1, 0)])
def test_every_numeric_field_rejects_non_real_values(
    field_name: str, invalid_value: object
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _config(**{field_name: invalid_value})


@pytest.mark.parametrize(
    "field_name",
    [
        "temperature",
        "fall_threshold",
        "reliability_threshold",
        "confirm_seconds",
        "recovery_seconds",
        "cooldown_seconds",
        "minimum_coverage",
    ],
)
@pytest.mark.parametrize(
    "invalid_value", [float("nan"), float("inf"), float("-inf"), Decimal("NaN")]
)
def test_every_numeric_field_rejects_non_finite_values(
    field_name: str, invalid_value: object
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _config(**{field_name: invalid_value})


def test_real_decimal_and_numpy_values_normalize_to_json_safe_builtin_floats() -> None:
    config = _config(
        temperature=Decimal("1.25"),
        fall_threshold=np.float32(0.5),
        reliability_threshold=Fraction(1, 4),
        confirm_seconds=np.int64(1),
        recovery_seconds=Decimal("2.5"),
        cooldown_seconds=np.float32(-0.0),
        minimum_coverage=np.int32(1),
    )

    for field_name in FIELD_NAMES[5:]:
        assert type(getattr(config, field_name)) is float
    assert config.cooldown_seconds == 0.0
    assert math.copysign(1.0, config.cooldown_seconds) == 1.0
    assert json.dumps(config.to_dict(), allow_nan=False)


def test_write_is_canonical_utf8_and_creates_missing_parent_directories(
    tmp_path: Path,
) -> None:
    path = tmp_path / "new" / "nested" / "release_config.json"
    config = _config(release_id="  发布 r1  ")
    expected = (
        "{\n"
        '  "confirm_seconds": 0.8,\n'
        '  "cooldown_seconds": 10.0,\n'
        f'  "dataset_sha256": "{"b" * 64}",\n'
        '  "fall_threshold": 0.6,\n'
        '  "minimum_coverage": 0.6,\n'
        f'  "model_sha256": "{"a" * 64}",\n'
        '  "recovery_seconds": 2.0,\n'
        '  "release_id": "发布 r1",\n'
        '  "reliability_threshold": 0.4,\n'
        '  "schema_version": "rgpc.release.v1",\n'
        f'  "split_sha256": "{"c" * 64}",\n'
        '  "temperature": 1.5\n'
        "}\n"
    ).encode("utf-8")

    write_release_config(config, path)

    assert path.read_bytes() == expected
    assert path.read_bytes().endswith(b"\n")
    assert not path.read_bytes().endswith(b"\n\n")


def test_write_replaces_atomically_and_cleans_temporary_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "release_config.json"
    original = b"existing release remains intact\n"
    path.write_bytes(original)

    observed_replace: list[tuple[Path, Path]] = []

    def fail_replace(source: object, target: object) -> None:
        observed_replace.append((Path(source), Path(target)))
        raise OSError("injected replace failure")

    monkeypatch.setattr(release_config_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        write_release_config(_config(), path)

    assert len(observed_replace) == 1
    temporary_path, replacement_target = observed_replace[0]
    assert replacement_target == path
    assert temporary_path.parent == path.parent
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_write_rejects_non_config_values_without_touching_target(tmp_path: Path) -> None:
    path = tmp_path / "release_config.json"
    path.write_bytes(b"old\n")

    with pytest.raises(ValueError, match="RGPCReleaseConfig"):
        write_release_config({"release_id": "r1"}, path)  # type: ignore[arg-type]

    assert path.read_bytes() == b"old\n"


@pytest.mark.parametrize("missing_field", FIELD_NAMES)
def test_load_rejects_every_missing_field(
    tmp_path: Path, missing_field: str
) -> None:
    payload = _config().to_dict()
    del payload[missing_field]
    path = tmp_path / "release_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing"):
        load_release_config(path)


def test_load_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = {**_config().to_dict(), "unexpected": "value"}
    path = tmp_path / "release_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown"):
        load_release_config(path)


def test_load_rejects_duplicate_json_keys_instead_of_using_the_last_value(
    tmp_path: Path,
) -> None:
    canonical = json.dumps(_config().to_dict(), ensure_ascii=False)
    duplicated = canonical.replace(
        '"release_id": "r1"', '"release_id": "r1", "release_id": "r2"'
    )
    path = tmp_path / "release_config.json"
    path.write_text(duplicated, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key: release_id"):
        load_release_config(path)


@pytest.mark.parametrize("document", ["[1, 2]", '"value"', "1", "true", "null"])
def test_load_rejects_non_object_json_roots(tmp_path: Path, document: str) -> None:
    path = tmp_path / "release_config.json"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_release_config(path)


@pytest.mark.parametrize("document", ["{", "", "not json", '{"temperature": NaN}'])
def test_load_rejects_malformed_or_nonstandard_json(
    tmp_path: Path, document: str
) -> None:
    path = tmp_path / "release_config.json"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="invalid release config JSON"):
        load_release_config(path)


def test_load_rejects_invalid_utf8_with_deterministic_value_error(tmp_path: Path) -> None:
    path = tmp_path / "release_config.json"
    path.write_bytes(b"{\xff}")

    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        load_release_config(path)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("schema_version", 1),
        ("release_id", []),
        ("model_sha256", None),
        ("temperature", "1.5"),
        ("fall_threshold", True),
    ],
)
def test_load_rejects_invalid_json_value_types_as_value_error(
    tmp_path: Path, field_name: str, invalid_value: object
) -> None:
    payload = _config().to_dict()
    payload[field_name] = invalid_value
    path = tmp_path / "release_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=field_name):
        load_release_config(path)
