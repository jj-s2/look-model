"""Run and aggregate the fixed-seed RG-PCNet experiment matrix.

This entry point never fabricates metrics.  A training/evaluation runner must
write ``run_manifest.json`` with the declared input/config hashes and a
successful status.  Existing successful artifacts are resumed only when those
hashes and the immutable :class:`~risk.phase_model.experiment_matrix.ExperimentRun`
specification match exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Iterable, Mapping, Sequence

if __package__ in {None, ""}:  # direct ``python scripts/...`` invocation
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk.phase_model.experiment_matrix import ExperimentRun, VARIANTS, build_experiment_matrix


_SHA256_HEX = frozenset("0123456789abcdef")
_REQUIRED_METRIC = "event_f1"
_FORBIDDEN_SOURCES = frozenset({"manual", "synthetic", "demo", "smoke_test"})


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON input: {path}") from exc


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(_canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def sha256_path(path: os.PathLike[str] | str) -> str:
    """Hash a regular file in streaming mode."""

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"hash input is not a regular file: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"unable to hash input: {source}") from exc
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(char in _SHA256_HEX for char in value)


def hash_input_paths(paths: Mapping[str, os.PathLike[str] | str]) -> dict[str, str]:
    """Hash frozen manifests/dataset locks used by every matrix run."""

    if not paths:
        raise ValueError("at least one immutable input path is required")
    output = {str(name): sha256_path(path) for name, path in sorted(paths.items())}
    return output


def _safe_run_dir(root: Path, run: ExperimentRun) -> Path:
    # run_id is generated from names but still reject path separators if a
    # future caller constructs an ExperimentRun manually.
    if Path(run.run_id).name != run.run_id or run.run_id in {"", ".", ".."}:
        raise ValueError("run_id contains a path separator")
    return root / "runs" / run.run_id


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / "run_manifest.json"


def _canonical_matrix(matrix: Sequence[ExperimentRun]) -> tuple[ExperimentRun, ...]:
    """Normalize caller-provided run order for deterministic execution/output."""

    return tuple(sorted(matrix, key=lambda run: run.run_id))


def _load_manifest(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"run manifest must be an object: {path}")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _validate_metrics(metrics: object, path: Path) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise ValueError(f"metrics must be an object: {path}")
    if _REQUIRED_METRIC not in metrics:
        raise ValueError(f"metrics missing {_REQUIRED_METRIC}: {path}")
    output: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"metric names must be non-empty strings: {path}")
        output[key] = _finite_number(value, f"metrics.{key}")
    return output


def _validate_hashes(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping of SHA-256 values")
    output: dict[str, str] = {}
    for key, digest in value.items():
        if not isinstance(key, str) or not _valid_sha256(digest):
            raise ValueError(f"invalid {name} entry: {key}")
        output[key] = digest
    return output


def _validate_manifest(run: ExperimentRun, manifest: Mapping[str, Any], expected_input_hashes: Mapping[str, str], path: Path) -> dict[str, Any]:
    if manifest.get("run_id") != run.run_id:
        raise ValueError(f"run_id mismatch in {path}")
    if manifest.get("status") != "success":
        raise ValueError(f"run artifact is not successful: {path}")
    for flag in ("synthetic", "demo", "manual_metrics"):
        # A declared flag must be an explicit false.  Treating strings such as
        # ``"true"`` as harmless would allow demo data to enter a release.
        if flag in manifest and manifest[flag] is not False and manifest[flag] is not None:
            raise ValueError(f"synthetic/demo/manual metrics are not accepted: {path}")
    source = manifest.get("metrics_source")
    if isinstance(source, str) and source.strip().lower() in _FORBIDDEN_SOURCES:
        raise ValueError(f"metrics source is not a real evaluation artifact: {path}")
    input_hashes = _validate_hashes(manifest.get("input_hashes"), "input_hashes")
    if dict(input_hashes) != dict(expected_input_hashes):
        raise ValueError(f"input hashes mismatch in {path}")
    if manifest.get("config_sha256") != run.config_sha256:
        raise ValueError(f"config hash mismatch in {path}")
    expected_spec = {
        "variant": run.variant,
        "seed": run.seed,
        "outer_subject": run.outer_subject,
        "train_datasets": list(run.train_datasets),
        "held_out_dataset": run.held_out_dataset,
        "config_overrides": dict(run.config_overrides),
    }
    actual_spec = {key: manifest.get(key) for key in expected_spec}
    if _canonical_bytes(actual_spec) != _canonical_bytes(expected_spec):
        raise ValueError(f"run specification mismatch in {path}")
    result = dict(manifest)
    result["input_hashes"] = input_hashes
    result["metrics"] = _validate_metrics(manifest.get("metrics"), path)
    for key in ("cross_domain", "corruption", "latency"):
        if key in result and result[key] is not None and not isinstance(result[key], Mapping):
            raise ValueError(f"{key} must be an object in {path}")
    return result


def _existing_success(run: ExperimentRun, run_dir: Path, input_hashes: Mapping[str, str]) -> dict[str, Any] | None:
    path = _manifest_path(run_dir)
    if not path.is_file():
        return None
    try:
        return _validate_manifest(run, _load_manifest(path), input_hashes, path)
    except ValueError:
        return None


def _publish_run(staging: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    moved = False
    try:
        if destination.exists():
            if not destination.is_dir():
                raise ValueError(f"run output must be a directory: {destination}")
            backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}.backup-", dir=destination.parent))
            backup.rmdir()
            os.replace(destination, backup)
            moved = True
        os.replace(staging, destination)
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    except Exception:
        if moved and backup is not None and backup.exists():
            if destination.exists():
                shutil.rmtree(destination)
            os.replace(backup, destination)
            backup = None
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists() and not moved:
            shutil.rmtree(backup)


def _run_one(
    run: ExperimentRun,
    *,
    output_root: Path,
    input_hashes: Mapping[str, str],
    runner: Callable[[ExperimentRun, Path], Mapping[str, Any] | None],
) -> tuple[dict[str, Any], bool]:
    destination = _safe_run_dir(output_root, run)
    existing = _existing_success(run, destination, input_hashes)
    if existing is not None:
        return existing, True
    staging = Path(tempfile.mkdtemp(prefix=f".{run.run_id}.staging-", dir=output_root / "runs"))
    try:
        returned = runner(run, staging)
        if returned is not None:
            if not isinstance(returned, Mapping):
                raise ValueError("matrix runner must return a mapping or None")
            manifest_path = _manifest_path(staging)
            if manifest_path.exists():
                raise ValueError("runner returned a manifest while also writing one")
            manifest_path.write_bytes(_canonical_bytes(returned))
        manifest = _validate_manifest(run, _load_manifest(_manifest_path(staging)), input_hashes, _manifest_path(staging))
        # Validate before replacing an existing run, so a failed candidate can
        # never destroy a previously valid artifact.
        _publish_run(staging, destination)
        return manifest, False
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _numeric_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        try:
            numeric = float(item)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(numeric):
            result[key] = numeric
    return result


def _mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: mean([row[key] for row in rows if key in row]) for key in keys if any(key in row for row in rows)}


def _std_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: pstdev([row[key] for row in rows if key in row]) if sum(key in row for row in rows) > 1 else 0.0 for key in keys}


def _group_summary(items: Sequence[tuple[str, Mapping[str, float]]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, float]]] = defaultdict(list)
    for group, metrics in items:
        grouped[str(group)].append(metrics)
    return {group: _mean_metrics(rows) for group, rows in sorted(grouped.items())}


def aggregate_matrix_artifacts(matrix: Sequence[ExperimentRun], manifests: Sequence[Mapping[str, Any]], *, input_hashes: Mapping[str, str], resumed_count: int = 0) -> dict[str, Any]:
    """Aggregate validated real artifacts into a traceable machine JSON."""

    matrix = _canonical_matrix(matrix)
    if len(matrix) != len(manifests):
        raise ValueError("matrix and artifact counts differ")
    expected_hashes = _validate_hashes(input_hashes, "input_hashes")
    expected_ids = [run.run_id for run in matrix]
    actual_ids = [str(manifest.get("run_id")) for manifest in manifests]
    if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != set(expected_ids):
        raise ValueError("missing or duplicate run artifacts")
    by_id = {str(item["run_id"]): item for item in manifests}
    rows: list[dict[str, Any]] = []
    for run in matrix:
        item = _validate_manifest(
            run,
            by_id[run.run_id],
            expected_hashes,
            Path(f"<in-memory:{run.run_id}>") ,
        )
        rows.append({
            "run": run.to_dict(),
            "status": item["status"],
            "input_hashes": dict(item["input_hashes"]),
            "config_sha256": item["config_sha256"],
            "metrics": dict(item["metrics"]),
            "cross_domain": dict(item.get("cross_domain") or {}),
            "corruption": dict(item.get("corruption") or {}),
            "latency": dict(item.get("latency") or {}),
        })
    metric_rows = [_numeric_mapping(row["metrics"]) for row in rows]
    per_subject: dict[str, dict[str, float]] = {}
    per_seed: dict[str, dict[str, float]] = {}
    for subject in sorted({run.outer_subject for run in matrix}):
        per_subject[subject] = _mean_metrics([metric_rows[index] for index, run in enumerate(matrix) if run.outer_subject == subject])
    for seed in sorted({run.seed for run in matrix}):
        per_seed[str(seed)] = _mean_metrics([metric_rows[index] for index, run in enumerate(matrix) if run.seed == seed])
    cross_domain_items: list[tuple[str, Mapping[str, float]]] = []
    corruption_items: list[tuple[str, Mapping[str, float]]] = []
    latency_items: list[tuple[str, Mapping[str, float]]] = []
    for row in rows:
        cross = row["cross_domain"]
        if cross:
            cross_domain_items.append((str(cross.get("dataset", "unknown")), _numeric_mapping(cross)))
        corruption = row["corruption"]
        if corruption:
            corruption_items.append((str(corruption.get("severity", "unknown")), _numeric_mapping(corruption)))
        latency = row["latency"]
        if latency:
            latency_items.append(("all", _numeric_mapping(latency)))
    spec_bytes = _canonical_bytes([run.to_dict() for run in matrix])
    result: dict[str, Any] = {
        "schema_version": "rgpc.experiment_matrix.v1",
        "status": "success",
        "run_count": len(rows),
        "resumed_count": resumed_count,
        "input_hashes": dict(expected_hashes),
        "matrix_sha256": hashlib.sha256(spec_bytes).hexdigest(),
        "runs": rows,
        "summaries": {
            "per_subject": per_subject,
            "per_seed": per_seed,
            "macro": _mean_metrics(metric_rows),
            "standard_deviation": _std_metrics(metric_rows),
            "cross_domain": _group_summary(cross_domain_items),
            "corruption": _group_summary(corruption_items),
            "latency": _group_summary(latency_items),
        },
    }
    return result


def run_experiment_matrix(
    matrix: Sequence[ExperimentRun],
    *,
    output_root: os.PathLike[str] | str,
    runner: Callable[[ExperimentRun, Path], Mapping[str, Any] | None],
    input_hashes: Mapping[str, str] | None = None,
    input_paths: Mapping[str, os.PathLike[str] | str] | None = None,
) -> dict[str, Any]:
    """Execute missing runs, resume hash-matching runs, and aggregate results."""

    matrix = _canonical_matrix(matrix)
    if not matrix:
        raise ValueError("experiment matrix must not be empty")
    if input_hashes is not None and input_paths is not None:
        raise ValueError("provide input_hashes or input_paths, not both")
    if input_hashes is None:
        if input_paths is None:
            raise ValueError("input_hashes or input_paths must not be empty")
        input_hashes = hash_input_paths(input_paths)
    expected_hashes = _validate_hashes(input_hashes, "input_hashes")
    root = Path(output_root)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    resumed_count = 0
    for run in matrix:
        manifest, resumed = _run_one(run, output_root=root, input_hashes=expected_hashes, runner=runner)
        manifests.append(manifest)
        resumed_count += int(resumed)
    result = aggregate_matrix_artifacts(matrix, manifests, input_hashes=expected_hashes, resumed_count=resumed_count)
    _write_json(root / "experiment_matrix.json", result)
    return result


# Stable aliases make the aggregation/execution contracts convenient for
# notebooks while the descriptive names remain the canonical API.
aggregate_experiment_matrix = aggregate_matrix_artifacts
run_matrix = run_experiment_matrix


def _subprocess_runner(command: str) -> Callable[[ExperimentRun, Path], None]:
    if not command.strip():
        raise ValueError("--runner-command is required; no synthetic runner is provided")
    tokens = shlex.split(command, posix=os.name != "nt")
    if not tokens:
        raise ValueError("--runner-command is empty")

    def run(run: ExperimentRun, run_dir: Path) -> None:
        values = {
            "run_dir": str(run_dir),
            "run_id": run.run_id,
            "variant": run.variant,
            "seed": str(run.seed),
            "outer_subject": run.outer_subject,
        }
        argv = [token.format(**values) for token in tokens]
        env = os.environ.copy()
        env.update({f"RGPC_{key.upper()}": value for key, value in values.items()})
        subprocess.run(argv, check=True, env=env)

    return run


def _manifest_subjects(path: Path) -> tuple[str, ...]:
    value = _read_json(path)
    if isinstance(value, list):
        subjects = value
    elif isinstance(value, Mapping):
        subjects = value.get("outer_subjects", value.get("subjects"))
    else:
        subjects = None
    if not isinstance(subjects, list) or not subjects or any(type(subject) is not str for subject in subjects):
        raise ValueError("frozen manifest must declare a non-empty subjects list")
    return tuple(subjects)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", "--loso-manifest", dest="manifest", required=True, type=Path, help="frozen LOSO manifest JSON")
    parser.add_argument("--dataset-lock", required=True, type=Path, help="dataset lock file")
    parser.add_argument(
        "--baseline-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="immutable baseline/evaluation input; may be repeated",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--runner-command", required=True, help="command that writes run_manifest.json; use {run_dir} placeholders")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument("--train-dataset", action="append", default=[])
    parser.add_argument("--held-out-dataset")
    parser.add_argument("--variant", dest="variants", action="append", choices=tuple(VARIANTS), default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    subjects = tuple(args.subjects) if args.subjects else _manifest_subjects(args.manifest)
    input_paths: dict[str, Path] = {
        "frozen_manifest": args.manifest,
        "dataset_lock": args.dataset_lock,
    }
    for entry in args.baseline_path:
        if "=" not in entry:
            raise ValueError("--baseline-path must use NAME=PATH")
        name, raw_path = entry.split("=", 1)
        if not name.strip() or not raw_path.strip():
            raise ValueError("--baseline-path must use NAME=PATH")
        if name.strip() in input_paths:
            raise ValueError(f"duplicate immutable input name: {name.strip()}")
        input_paths[name.strip()] = Path(raw_path.strip())
    input_hashes = hash_input_paths(input_paths)
    matrix = build_experiment_matrix(
        outer_subjects=subjects,
        seeds=tuple(args.seeds),
        train_datasets=tuple(args.train_dataset),
        held_out_dataset=args.held_out_dataset,
        variants=tuple(args.variants) if args.variants else None,
    )
    result = run_experiment_matrix(
        matrix,
        output_root=args.output_root,
        runner=_subprocess_runner(args.runner_command),
        input_hashes=input_hashes,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "aggregate_matrix_artifacts",
    "aggregate_experiment_matrix",
    "hash_input_paths",
    "run_matrix",
    "run_experiment_matrix",
    "sha256_path",
]
