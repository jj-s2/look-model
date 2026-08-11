"""Create traceable RG-PCNet evaluation figures from machine-readable JSON.

The plotting surface is deliberately small and strict.  Four source files are
read as bytes, parsed with duplicate-key and non-finite-value rejection, and
then rendered into four fixed figures.  Every figure records the exact SHA-256
of the source JSON which supplied its values; no metric is inferred or filled
with a default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_FIGURE_NAMES = ("model_comparison", "subject_f1", "risk_coverage", "continuous_events")
_MODEL_ALIASES = {
    "baseline_tcn": "Baseline TCN",
    "tcn": "Baseline TCN",
    "baseline": "Baseline TCN",
    "phase_only": "Phase-only",
    "phase": "Phase-only",
    "reliability_only": "Reliability-only",
    "reliability": "Reliability-only",
    "full_rg_pcnet": "Full RG-PCNet",
    "rg_pcnet": "Full RG-PCNet",
    "full": "Full RG-PCNet",
    "distillation": "Distillation",
    "distilled": "Distillation",
}
_REQUIRED_MODELS = ("Baseline TCN", "Phase-only", "Reliability-only", "Full RG-PCNet")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, name: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise ValueError(f"{name} must be a regular JSON file: {path}")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(("duplicate JSON key", "non-finite JSON")):
            raise
        raise ValueError(f"{name} must contain valid UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{name} root must be a JSON object")
    return value, raw


def _number(value: Any, name: str, *, lower: float | None = None, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if lower is not None and number < lower:
        raise ValueError(f"{name} must be in [{lower:g}, {upper:g}]")
    if upper is not None and number > upper:
        raise ValueError(f"{name} must be in [{lower:g}, {upper:g}]")
    return 0.0 if number == 0.0 else number


def _required(mapping: Mapping[str, Any], key: str, name: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{name} is missing required field: {key}")
    return mapping[key]


def _model_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _model_value(value: Any, name: str) -> float:
    if isinstance(value, Mapping):
        value = _required(value, "f1", f"model {name}")
    return _number(value, f"model {name} f1", lower=0.0, upper=1.0)


def _models(payload: Mapping[str, Any], name: str) -> dict[str, float]:
    candidates = payload.get("models", payload.get("model_comparison", payload.get("ablation", payload.get("variants"))))
    if candidates is None:
        direct = {key: payload[key] for key in payload if isinstance(key, str) and _model_key(key) in _MODEL_ALIASES}
        candidates = direct
    if isinstance(candidates, list):
        normalized: dict[str, Any] = {}
        for index, item in enumerate(candidates):
            if not isinstance(item, Mapping):
                raise ValueError(f"{name} models[{index}] must be an object")
            raw_name = item.get("name", item.get("model"))
            if type(raw_name) is not str or not raw_name.strip():
                raise ValueError(f"{name} models[{index}] is missing a model name")
            normalized[raw_name] = _required(item, "f1", f"{name} models[{index}]")
        candidates = normalized
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError(f"{name} must contain a non-empty models object")
    parsed: dict[str, float] = {}
    for raw_name, raw_value in candidates.items():
        if type(raw_name) is not str:
            raise ValueError(f"{name} model names must be strings")
        label = _MODEL_ALIASES.get(_model_key(raw_name), raw_name.strip())
        if not label:
            raise ValueError(f"{name} model names must be non-empty")
        parsed[label] = _model_value(raw_value, label)
    missing = [label for label in _REQUIRED_MODELS if label not in parsed]
    if missing:
        raise ValueError(f"{name} is missing required models: {', '.join(missing)}")
    ordered = {label: parsed[label] for label in _REQUIRED_MODELS}
    ordered.update({label: parsed[label] for label in sorted(parsed) if label not in ordered})
    return ordered


def _subject_f1(payload: Mapping[str, Any]) -> tuple[dict[str, float], int]:
    values = payload.get("subject_f1", payload.get("per_subject_f1", payload.get("subject_metrics", payload.get("per_subject", payload.get("subjects")))))
    parsed: dict[str, float] = {}
    if isinstance(values, Mapping):
        for subject, raw_value in values.items():
            if type(subject) is not str or not subject.strip():
                raise ValueError("aggregate_loso subject IDs must be non-empty strings")
            parsed[subject] = _model_value(raw_value, f"subject {subject}")
    elif isinstance(values, list):
        for index, item in enumerate(values):
            if not isinstance(item, Mapping):
                raise ValueError(f"aggregate_loso subjects[{index}] must be an object")
            subject = _required(item, "subject_id", f"aggregate_loso subjects[{index}]")
            if type(subject) is not str or not subject.strip():
                raise ValueError("aggregate_loso subject_id must be a non-empty string")
            if subject in parsed:
                raise ValueError(f"aggregate_loso contains duplicate subject: {subject}")
            parsed[subject] = _model_value(_required(item, "f1", f"aggregate_loso subjects[{index}]"), f"subject {subject}")
    else:
        raise ValueError("aggregate_loso must contain subject_f1, per_subject_f1, or subjects")
    if not parsed:
        raise ValueError("aggregate_loso subject F1 cannot be empty")
    raw_count = payload.get("subject_count", len(parsed))
    if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count != len(parsed):
        raise ValueError("aggregate_loso subject_count must equal the number of subjects")
    return dict(sorted(parsed.items())), raw_count


def _curve(value: Any, name: str) -> tuple[list[float], list[float]]:
    if isinstance(value, Mapping) and "risk_coverage_points" in value:
        value = value["risk_coverage_points"]
    if isinstance(value, Mapping):
        coverage = value.get("coverage")
        risk = value.get("risk", value.get("error"))
        if not isinstance(coverage, list) or not isinstance(risk, list):
            raise ValueError(f"{name} must contain coverage and risk arrays")
        if len(coverage) != len(risk):
            raise ValueError(f"{name} coverage and risk arrays must have equal length")
        points = list(zip(coverage, risk))
    elif isinstance(value, list):
        points = value
    else:
        raise ValueError(f"{name} must be an object or point array")
    if len(points) < 2:
        raise ValueError(f"{name} must contain at least two points")
    coverage_values: list[float] = []
    risk_values: list[float] = []
    previous = -math.inf
    for index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{name}[{index}] must be a [coverage, risk] pair")
        coverage_value = _number(point[0], f"{name}[{index}] coverage", lower=0.0, upper=1.0)
        risk_value = _number(point[1], f"{name}[{index}] risk", lower=0.0, upper=1.0)
        if coverage_value < previous:
            raise ValueError(f"{name} coverage must be nondecreasing")
        previous = coverage_value
        coverage_values.append(coverage_value)
        risk_values.append(risk_value)
    return coverage_values, risk_values


def _calibration_curves(payload: Mapping[str, Any]) -> tuple[tuple[list[float], list[float]], tuple[list[float], list[float]]]:
    source = payload.get("risk_coverage", payload)
    if not isinstance(source, Mapping):
        raise ValueError("calibration must contain a risk_coverage object")
    clean = source.get("clean")
    corrupted = source.get("corrupted", source.get("corrupt"))
    if clean is None or corrupted is None:
        raise ValueError("calibration must contain clean and corrupted risk-coverage curves")
    return _curve(clean, "calibration clean"), _curve(corrupted, "calibration corrupted")


def _continuous_metrics(payload: Mapping[str, Any]) -> tuple[dict[str, float], int | None]:
    values = payload.get("metrics", payload)
    if not isinstance(values, Mapping):
        raise ValueError("continuous_metrics must contain a metrics object")
    required = ("event_recall", "false_alerts_per_hour", "median_delay_seconds", "coverage")
    parsed = {
        key: _number(_required(values, key, "continuous_metrics"), f"continuous_metrics {key}", lower=0.0, upper=1.0 if key in {"event_recall", "coverage"} else None)
        for key in required
    }
    raw_count = values.get("subject_count", payload.get("subject_count"))
    if raw_count is not None and (isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 1):
        raise ValueError("continuous_metrics subject_count must be a positive integer")
    return parsed, raw_count


def _source_hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("matplotlib is required to generate RG-PCNet plots") from exc
    return plt


def _save_figure(fig: Any, output_dir: Path, name: str) -> tuple[Path, Path]:
    png = output_dir / f"{name}.png"
    svg = output_dir / f"{name}.svg"
    fig.savefig(png, format="png", dpi=200, metadata={"Software": "RG-PCNet evaluation"})
    fig.savefig(svg, format="svg", metadata={"Date": None})
    fig.clf()
    return png, svg


def plot_rg_pcnet_results(
    aggregate_loso: Path | str,
    continuous_metrics: Path | str,
    calibration: Path | str,
    ablation: Path | str,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Render four fixed plots and write a source-hash manifest.

    All arguments are paths to UTF-8 JSON files.  Inputs are parsed before
    matplotlib is imported, making schema failures deterministic even when the
    optional plotting dependency is unavailable.
    """
    aggregate_payload, aggregate_raw = _read_json(Path(aggregate_loso), "aggregate_loso")
    continuous_payload, continuous_raw = _read_json(Path(continuous_metrics), "continuous_metrics")
    calibration_payload, calibration_raw = _read_json(Path(calibration), "calibration")
    ablation_payload, ablation_raw = _read_json(Path(ablation), "ablation")
    subjects, subject_count = _subject_f1(aggregate_payload)
    continuous, continuous_subject_count = _continuous_metrics(continuous_payload)
    clean_curve, corrupted_curve = _calibration_curves(calibration_payload)
    models = _models(ablation_payload, "ablation")
    if "subject_count" in ablation_payload and ablation_payload["subject_count"] != subject_count:
        raise ValueError("ablation subject_count must match aggregate_loso")
    figure_subject_count = continuous_subject_count or subject_count

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt = _matplotlib()
    paths: dict[str, Path] = {}
    manifest_figures: dict[str, dict[str, Any]] = {}

    fig, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    labels = list(models)
    palette = ["#718096", "#4299e1", "#ed8936", "#38a169", "#805ad5"]
    axis.bar(labels, [models[label] for label in labels], color=[palette[index % len(palette)] for index in range(len(labels))])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("F1")
    axis.set_title("LOSO model comparison")
    axis.tick_params(axis="x", rotation=18)
    png, svg = _save_figure(fig, output, "model_comparison")
    paths.update({"model_comparison_png": png, "model_comparison_svg": svg})
    manifest_figures["model_comparison"] = {"png": str(png.resolve()), "svg": str(svg.resolve()), "source_sha256": _source_hash(ablation_raw), "caption": f"LOSO model comparison; subject count: {subject_count}; intervals: descriptive."}

    fig, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    subject_labels = list(subjects)
    axis.bar(subject_labels, [subjects[label] for label in subject_labels], color="#4299e1")
    axis.axhline(sum(subjects.values()) / len(subjects), color="#c53030", linestyle="--", label="subject macro")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("F1")
    axis.set_title("LOSO per-subject F1")
    axis.tick_params(axis="x", rotation=25)
    axis.legend()
    png, svg = _save_figure(fig, output, "subject_f1")
    paths.update({"subject_f1_png": png, "subject_f1_svg": svg})
    manifest_figures["subject_f1"] = {"png": str(png.resolve()), "svg": str(svg.resolve()), "source_sha256": _source_hash(aggregate_raw), "caption": f"LOSO subject F1 with macro line; subject count: {subject_count}; intervals: descriptive."}

    fig, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    axis.plot(clean_curve[0], clean_curve[1], marker="o", label="clean")
    axis.plot(corrupted_curve[0], corrupted_curve[1], marker="o", label="corrupted")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Risk")
    axis.set_title("LOSO risk-coverage")
    axis.legend()
    png, svg = _save_figure(fig, output, "risk_coverage")
    paths.update({"risk_coverage_png": png, "risk_coverage_svg": svg})
    manifest_figures["risk_coverage"] = {"png": str(png.resolve()), "svg": str(svg.resolve()), "source_sha256": _source_hash(calibration_raw), "caption": f"LOSO risk-coverage curves; subject count: {subject_count}; intervals: descriptive."}

    fig, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    metric_labels = ("event recall", "false alerts/hour", "median delay (s)", "coverage")
    metric_values = tuple(continuous[key] for key in ("event_recall", "false_alerts_per_hour", "median_delay_seconds", "coverage"))
    axis.bar(metric_labels, metric_values, color=["#38a169", "#c53030", "#805ad5", "#4299e1"])
    axis.set_ylabel("Value")
    axis.set_title("continuous replay event metrics")
    axis.tick_params(axis="x", rotation=18)
    for index, value in enumerate(metric_values):
        axis.text(index, value, f"{value:.3g}", ha="center", va="bottom")
    png, svg = _save_figure(fig, output, "continuous_events")
    paths.update({"continuous_events_png": png, "continuous_events_svg": svg})
    manifest_figures["continuous_events"] = {"png": str(png.resolve()), "svg": str(svg.resolve()), "source_sha256": _source_hash(continuous_raw), "caption": f"continuous replay event metrics; subject count: {figure_subject_count}; intervals: descriptive."}

    manifest = {"schema_version": "rgpc.plots.v1", "figures": manifest_figures}
    manifest_path = output / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    paths["plot_manifest"] = manifest_path
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-loso", required=True, type=Path)
    parser.add_argument("--continuous-metrics", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--ablation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plot_rg_pcnet_results(args.aggregate_loso, args.continuous_metrics, args.calibration, args.ablation, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
