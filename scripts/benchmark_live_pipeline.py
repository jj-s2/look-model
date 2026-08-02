"""Measure local replay capture latency without inventing model-performance data.

This script deliberately reports an incomplete benchmark when the full vision
inference pipeline is not available.  A capture/decode measurement is useful
for diagnosis, but is never release-eligible on its own.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATENCY_GATE_SECONDS = 2.0


@dataclass(frozen=True)
class BenchmarkReport:
    """A redaction-safe record of what this benchmark actually measured."""

    input_source: str
    input_status: str
    requested_duration_seconds: float
    measured_duration_seconds: float
    frames_or_records: int
    p50_latency_seconds: float | None
    p95_latency_seconds: float | None
    pipeline_scope: str
    release_gate_passed: bool
    release_gate_reasons: tuple[str, ...]
    hardware: dict[str, str]
    python_version: str
    cuda_version: str
    model_versions: dict[str, str]
    dataset: str
    threshold: float | None
    random_seed: int | None
    fall_precision: float | None
    fall_recall: float | None
    fall_f1: float | None
    false_alarms_per_hour: float | None
    per_class_confusion_matrix: dict[str, str]
    measured_at: str

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["schema_version"] = "1.0"
        result["kind"] = "pipeline_benchmark"
        result["latency_measurement"] = "local capture/decode elapsed time"
        return result


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * percentile
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _runtime_metadata() -> tuple[dict[str, str], str, dict[str, str]]:
    hardware = {
        "platform": platform.platform(),
        "machine": platform.machine() or "unknown",
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
    }
    cuda_version = "unavailable"
    try:
        import torch  # type: ignore[import-not-found]

        cuda_version = str(torch.version.cuda or "unavailable")
        if torch.cuda.is_available():
            hardware["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return hardware, cuda_version, {
        "opencv-python": _package_version("opencv-python"),
        "torch": _package_version("torch"),
        "mmpose": _package_version("mmpose"),
        "mmaction2": _package_version("mmaction2"),
    }


def _base_report(input_path: Path, duration_seconds: float, *, status: str, elapsed: float = 0.0,
                 count: int = 0, latencies: list[float] | None = None) -> BenchmarkReport:
    hardware, cuda_version, model_versions = _runtime_metadata()
    p50 = _percentile(latencies or [], 0.50)
    p95 = _percentile(latencies or [], 0.95)
    reasons: list[str] = []
    if status != "measured":
        reasons.append(f"input is {status}")
    if p95 is None:
        reasons.append("P95 latency was not measured")
    elif p95 > LATENCY_GATE_SECONDS:
        reasons.append(f"P95 latency {p95:.3f}s exceeds {LATENCY_GATE_SECONDS:.1f}s")
    reasons.append("capture/decode-only benchmark is not a full inference release evaluation")
    return BenchmarkReport(
        input_source=str(input_path), input_status=status,
        requested_duration_seconds=duration_seconds, measured_duration_seconds=elapsed,
        frames_or_records=count, p50_latency_seconds=p50, p95_latency_seconds=p95,
        pipeline_scope="capture_decode_only", release_gate_passed=False,
        release_gate_reasons=tuple(reasons), hardware=hardware,
        python_version=sys.version.split()[0], cuda_version=cuda_version,
        model_versions=model_versions,
        dataset=input_path.name if status == "measured" else "unavailable",
        threshold=None, random_seed=None, fall_precision=None, fall_recall=None, fall_f1=None,
        false_alarms_per_hour=None,
        per_class_confusion_matrix={"fall": "unavailable", "adl": "unavailable"},
        measured_at=datetime.now(timezone.utc).isoformat(),
    )


def _benchmark_jsonl(path: Path, duration_seconds: float) -> tuple[str, float, int, list[float]]:
    latencies: list[float] = []
    count = 0
    started = time.perf_counter()
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if time.perf_counter() - started >= duration_seconds:
                    break
                before = time.perf_counter()
                if line.strip():
                    json.loads(line)
                    count += 1
                latencies.append(time.perf_counter() - before)
    except (OSError, json.JSONDecodeError):
        return "unavailable", time.perf_counter() - started, count, latencies
    return ("measured" if count else "unavailable"), time.perf_counter() - started, count, latencies


def _benchmark_video(path: Path, duration_seconds: float) -> tuple[str, float, int, list[float]]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return "dependency_unavailable", 0.0, 0, []
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return "unavailable", 0.0, 0, []
    latencies: list[float] = []
    count = 0
    started = time.perf_counter()
    try:
        while time.perf_counter() - started < duration_seconds:
            before = time.perf_counter()
            ok, _frame = capture.read()
            elapsed = time.perf_counter() - before
            if not ok:
                break
            latencies.append(elapsed)
            count += 1
    finally:
        capture.release()
    elapsed = time.perf_counter() - started
    return ("measured" if count else "unavailable"), elapsed, count, latencies


def benchmark_live_pipeline(input_source: str | Path, duration_seconds: float) -> BenchmarkReport:
    """Measure a local replay source and return its evidence-bearing report.

    No network source is opened by this helper.  It never converts unavailable
    input, missing models, or a replay-only measurement into a passing result.
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    input_path = Path(input_source)
    if not input_path.is_file():
        return _base_report(input_path, duration_seconds, status="unavailable")
    if input_path.suffix.lower() == ".jsonl":
        status, elapsed, count, latencies = _benchmark_jsonl(input_path, duration_seconds)
    else:
        status, elapsed, count, latencies = _benchmark_video(input_path, duration_seconds)
    return _base_report(input_path, duration_seconds, status=status, elapsed=elapsed, count=count, latencies=latencies)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="local MP4 or JSONL replay; network URLs are unsupported")
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = benchmark_live_pipeline(args.input, args.duration_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"benchmark status={report.input_status}; report={args.output}")
    for reason in report.release_gate_reasons:
        print(f"release gate: FAIL - {reason}", file=sys.stderr)
    return 0 if report.release_gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
