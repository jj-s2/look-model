"""Generate reproducible held-out evaluation plots for the PA-DTSF checkpoint.

The CLI evaluates only the frozen ``test`` partition from the release
manifest. It uses the same normalized 64-frame long-pose input and
``short_quality=0`` contract as the live runner, then writes prediction records,
metrics, a short evidence report, and presentation-ready PNGs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PHASE_NAMES = ("normal_adl", "prefall_abnormal", "descending", "impact", "fallen", "recovering")
DEFAULT_RELEASE = PROJECT_ROOT / "outputs" / "releases" / "padtfs-gmdcsa24-gpu-norm"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot held-out PA-DTSF phase-model results without fitting or threshold tuning.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RELEASE / "checkpoint.pt")
    parser.add_argument("--dataset-lock", type=Path, default=DEFAULT_RELEASE / "dataset_lock.json")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_RELEASE / "split_manifest.json")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "datasets" / "raw" / "training" / "GMDCSA24")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "docs" / "figures" / "padtfs-gmdcsa24-gpu-norm")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5, help="fixed reporting threshold; no threshold fitting is performed")
    parser.add_argument("--calibration", type=Path, default=None, help="path to calibration.json; defaults to checkpoint parent / calibration.json")
    return parser


def _load_calibration(calibration_path: Path | None, checkpoint: Path) -> tuple[float, float]:
    """Return (temperature, threshold) from a calibration file or defaults."""
    path = calibration_path
    if path is None:
        path = checkpoint.parent / "calibration.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        temperature = float(data.get("temperature", 1.0))
        threshold = float(data.get("threshold", 0.5))
        if math.isfinite(temperature) and temperature > 0.0 and math.isfinite(threshold) and 0.0 <= threshold <= 1.0:
            return temperature, threshold
    return 1.0, 0.5


def _apply_temperature(scores: Sequence[float], temperature: float) -> list[float]:
    """Convert raw probabilities with a positive temperature."""
    from risk.phase_model.calibration import apply_temperature as _apply

    logits = [math.log(max(score, 1e-9) / max(1.0 - score, 1e-9)) for score in scores]
    return _apply(logits, temperature)


def _records_arrays(records: Sequence[Mapping[str, object]]):
    import numpy as np

    labels: list[int] = []
    scores: list[float] = []
    for record in records:
        label = int(record["label"])
        score = float(record["score"])
        if label not in (0, 1) or not 0.0 <= score <= 1.0:
            raise ValueError("each record must contain label 0/1 and score in [0, 1]")
        labels.append(label)
        scores.append(score)
    if not labels or not any(labels) or all(labels):
        raise ValueError("held-out plot data must contain both positive and negative labels")
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def _threshold_table(labels, scores) -> list[dict[str, float]]:
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    rows = []
    for threshold in np.linspace(0.0, 1.0, 101):
        predictions = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        negatives = tn + fp
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "false_positive_rate": float(fp / negatives if negatives else 0.0),
            "tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn),
        })
    return rows


def _calibration_bins(labels, scores, *, bins: int = 10) -> list[dict[str, float]]:
    rows = []
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        selected = [i for i, score in enumerate(scores) if low <= score < high or (index == bins - 1 and low <= score <= high)]
        if not selected:
            continue
        rows.append({
            "bin_lower": low,
            "bin_upper": high,
            "mean_predicted": sum(float(scores[i]) for i in selected) / len(selected),
            "fraction_positive": sum(int(labels[i]) for i in selected) / len(selected),
            "count": float(len(selected)),
        })
    return rows


def summarize_predictions(records: Sequence[Mapping[str, object]], *, threshold: float = 0.5) -> dict[str, object]:
    """Compute fixed-threshold and threshold-free test metrics."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    labels, scores = _records_arrays(records)
    from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    rows = _threshold_table(labels, scores)
    best = max(rows, key=lambda row: (row["f1"], row["threshold"]))
    calibration = _calibration_bins(labels, scores)
    ece = sum(row["count"] * abs(row["mean_predicted"] - row["fraction_positive"]) for row in calibration) / len(labels)
    return {
        "samples": int(len(labels)),
        "positive_samples": int(labels.sum()),
        "negative_samples": int(len(labels) - labels.sum()),
        "threshold": float(threshold),
        "precision_at_threshold": float(precision_score(labels, predictions, zero_division=0)),
        "recall_at_threshold": float(recall_score(labels, predictions, zero_division=0)),
        "f1_at_threshold": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "expected_calibration_error": float(ece),
        "best_f1_threshold": float(best["threshold"]),
        "best_f1": float(best["f1"]),
        "threshold_table": rows,
        "calibration_bins": calibration,
    }


def _save_figure(figure, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(figure)
    return destination


def _plot_roc_pr(labels, scores, summary: Mapping[str, object], destination: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="#1565c0", linewidth=2, label=f"ROC-AUC = {summary['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
    axes[0].set(title="Held-out ROC curve", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    axes[0].legend(loc="lower right")
    axes[1].plot(recall, precision, color="#2e7d32", linewidth=2, label=f"AP = {summary['average_precision']:.3f}")
    axes[1].axhline(float(labels.mean()), linestyle="--", color="#999999", linewidth=1, label=f"Prevalence = {labels.mean():.2f}")
    axes[1].set(title="Held-out precision–recall curve", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].legend(loc="lower left")
    figure.suptitle("PA-DTSF fall-event discrimination (test split)")
    figure.tight_layout()
    return _save_figure(figure, destination)


def _plot_confusion(labels, scores, summary: Mapping[str, object], destination: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.asarray(summary["confusion_matrix"], dtype=int)
    figure, axis = plt.subplots(figsize=(5.3, 4.7))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal / ADL", "Fall"], yticklabels=["Normal / ADL", "Fall"], xlabel="Predicted label", ylabel="True label", title=f"Confusion matrix (threshold {summary['threshold']:.2f})")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > matrix.max() / 2 else "black", fontsize=16, fontweight="bold")
    figure.tight_layout()
    return _save_figure(figure, destination)


def _plot_thresholds(summary: Mapping[str, object], destination: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = summary["threshold_table"]
    thresholds = [row["threshold"] for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for key, color in (("precision", "#ef6c00"), ("recall", "#1565c0"), ("f1", "#2e7d32"), ("false_positive_rate", "#9e9e9e")):
        axis.plot(thresholds, [row[key] for row in rows], label=key.replace("_", " ").title(), color=color, linewidth=2 if key == "f1" else 1.6)
    axis.axvline(summary["threshold"], color="#212121", linestyle="--", linewidth=1, label=f"fixed = {summary['threshold']:.2f}")
    axis.axvline(summary["best_f1_threshold"], color="#7b1fa2", linestyle=":", linewidth=1.4, label=f"best F1 = {summary['best_f1_threshold']:.2f}")
    axis.set(title="Threshold trade-off on held-out test split", xlabel="Fall probability threshold", ylabel="Metric", xlim=(0, 1), ylim=(0, 1.02))
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    return _save_figure(figure, destination)


def _plot_calibration(summary: Mapping[str, object], destination: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    bins = summary["calibration_bins"]
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    axis.plot([0, 1], [0, 1], "--", color="#9e9e9e", label="perfect calibration")
    if bins:
        axis.scatter([row["mean_predicted"] for row in bins], [row["fraction_positive"] for row in bins], s=[40 + 12 * row["count"] for row in bins], color="#6a1b9a", edgecolor="white", linewidth=0.8, label="test bins")
    axis.set(title=f"Reliability diagram (ECE = {summary['expected_calibration_error']:.3f})", xlabel="Mean predicted probability", ylabel="Observed fall fraction", xlim=(0, 1), ylim=(0, 1))
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left")
    figure.tight_layout()
    return _save_figure(figure, destination)


def _plot_distribution(labels, scores, destination: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    groups = [scores[labels == 0], scores[labels == 1]]
    figure, axis = plt.subplots(figsize=(6.2, 4.7))
    try:
        axis.boxplot(groups, tick_labels=["Normal / ADL", "Fall"], patch_artist=True, boxprops={"facecolor": "#bbdefb"}, medianprops={"color": "#b71c1c", "linewidth": 2})
    except TypeError:  # Matplotlib < 3.9 used ``labels``.
        axis.boxplot(groups, labels=["Normal / ADL", "Fall"], patch_artist=True, boxprops={"facecolor": "#bbdefb"}, medianprops={"color": "#b71c1c", "linewidth": 2})
    rng = np.random.default_rng(42)
    for index, values in enumerate(groups, start=1):
        axis.scatter(index + rng.uniform(-0.08, 0.08, len(values)), values, s=24, alpha=0.8, color="#263238", zorder=3)
    axis.set(title="Fall-score separation on held-out clips", ylabel="Predicted fall probability", ylim=(0, 1))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return _save_figure(figure, destination)


def _write_report(summary: Mapping[str, object], metadata: Mapping[str, object], destination: Path) -> Path:
    def pct(value: object) -> str:
        return f"{float(value) * 100:.1f}%"

    text = f"""# PA-DTSF 留出测试集效果图

本报告由 `scripts/plot_phase_results.py` 根据冻结的 GMDCSA24 v2.1 受试者划分自动生成。所有图表均来自未参与训练的 test split，固定阈值为 `{float(summary['threshold']):.2f}`，没有在测试集上调阈值。

## 核心结果

| 指标 | 结果 |
| --- | ---: |
| 测试片段 | {summary['samples']} |
| 阳性（跌倒） | {summary['positive_samples']} |
| Precision | {pct(summary['precision_at_threshold'])} |
| Recall | {pct(summary['recall_at_threshold'])} |
| F1 | {pct(summary['f1_at_threshold'])} |
| ROC-AUC | {float(summary['roc_auc']):.3f} |
| Average Precision | {float(summary['average_precision']):.3f} |
| Brier score | {float(summary['brier_score']):.3f} |

## 证据边界

- 数据集：GMDCSA24 v2.1；按 subject 分组，当前 test split 为留出的受试者集合。
- 推理路径：归一化 64 帧骨架、长时 PA-DTSF 分支、`short_quality=0`，与实时入口一致。
- 这些结果证明该冻结测试划分上的算法区分能力；不能直接外推为真实老人群体、临床诊断或萤石设备现场效果。
- 本 release 没有经过心理健康标签验证，因此不生成心理健康“准确率”图，避免把筛查功能误报成诊断模型。

## 生成信息

- release_id：`{metadata.get('release_id', 'unknown')}`
- checkpoint_sha256：`{metadata.get('checkpoint_sha256', 'unknown')}`
- device：`{metadata.get('device', 'unknown')}`

图文件：ROC/PR、混淆矩阵、阈值权衡、校准图和得分分布图。
"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def generate_artifacts(records: Sequence[Mapping[str, object]], output_dir: Path, *, metadata: Mapping[str, object] | None = None, threshold: float = 0.5) -> dict[str, Path]:
    """Write plots and auditable records from already materialized predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata or {})
    summary = summarize_predictions(records, threshold=threshold)
    summary["metadata"] = metadata
    predictions_path = output_dir / "predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    labels, scores = _records_arrays(records)
    paths = {
        "metrics": metrics_path,
        "predictions": predictions_path,
        "report": _write_report(summary, metadata, output_dir / "report.md"),
        "roc_pr": _plot_roc_pr(labels, scores, summary, output_dir / "roc_pr_curves.png"),
        "confusion_matrix": _plot_confusion(labels, scores, summary, output_dir / "confusion_matrix.png"),
        "thresholds": _plot_thresholds(summary, output_dir / "threshold_curves.png"),
        "calibration": _plot_calibration(summary, output_dir / "calibration.png"),
        "score_distribution": _plot_distribution(labels, scores, output_dir / "score_distribution.png"),
    }
    return paths


def _resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    return torch.device(requested)


def _resolve_feature_path(root: Path, clip: Mapping[str, object]) -> Path:
    media = Path(str(clip["media_path"]))
    candidates = [root / media]
    dataset = str(clip.get("dataset") or "")
    if dataset:
        candidates.append(root / dataset / media)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"pose feature is missing for clip {clip.get('clip_id')}")


def collect_checkpoint_predictions(checkpoint: Path, dataset_lock: Path, split_manifest: Path, data_root: Path, *, device: str = "auto") -> tuple[list[dict[str, object]], str]:
    """Run the frozen checkpoint over the manifest's held-out test clips."""
    import numpy as np
    import torch
    from risk.phase_model.model import PhaseAwareFusionModel
    from risk.phase_model.normalization import normalize_pose_array

    lock = json.loads(Path(dataset_lock).read_text(encoding="utf-8"))
    split = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    clips = {str(item["clip_id"]): item for item in lock.get("clips", []) if isinstance(item, Mapping) and "clip_id" in item}
    test_ids = split.get("partitions", {}).get("test", [])
    if not test_ids:
        raise ValueError("split manifest does not contain a test partition")
    target_device = _resolve_device(torch, device)
    payload = torch.load(Path(checkpoint), map_location=target_device, weights_only=False)
    raw_state = payload.get("model") if isinstance(payload, Mapping) else None
    if not isinstance(raw_state, Mapping):
        raise ValueError("checkpoint is missing a model state dict")
    model = PhaseAwareFusionModel(short_dim=512, joints=17, hidden_dim=128).to(target_device)
    modules = {
        "long_branch": model.long_branch.network, "short_projection": model.short_projection,
        "long_projection": model.long_projection, "phase_head": model.phase_head,
        "fall_head": model.fall_head, "prefall_head": model.prefall_head,
        "recovery_head": model.recovery_head, "abstain_head": model.abstain_head,
    }
    for name, module in modules.items():
        module.load_state_dict(raw_state[name])
    model.eval()
    records: list[dict[str, object]] = []
    with torch.no_grad():
        for clip_id in test_ids:
            clip = clips.get(str(clip_id))
            if clip is None:
                raise ValueError(f"test clip {clip_id} is absent from dataset lock")
            feature_path = _resolve_feature_path(Path(data_root), clip)
            with np.load(feature_path) as data:
                long_pose = normalize_pose_array(np.asarray(data["long_pose"], dtype=np.float32))
            pose = torch.as_tensor(long_pose, dtype=torch.float32, device=target_device).unsqueeze(0)
            short = torch.zeros((1, 512), dtype=torch.float32, device=target_device)
            short_quality = torch.zeros((1,), dtype=torch.float32, device=target_device)
            long_quality = torch.ones((1,), dtype=torch.float32, device=target_device)
            output = model(short, pose, short_quality, long_quality)
            phase_probs = torch.softmax(output.phase_logits, dim=-1)[0].detach().cpu().tolist()
            phase_index = max(range(len(phase_probs)), key=phase_probs.__getitem__)
            records.append({
                "clip_id": str(clip["clip_id"]),
                "subject_id": str(clip.get("subject_id", "unknown")),
                "dataset": str(clip.get("dataset", "unknown")),
                "event_group_id": str(clip.get("event_group_id", "unknown")),
                "label": int(clip.get("coarse_event") == "fall"),
                "score": float(torch.sigmoid(output.fall_event_logit)[0].detach().cpu()),
                "prefall_score": float(torch.sigmoid(output.prefall_logit)[0].detach().cpu()),
                "recovery_score": float(torch.sigmoid(output.recovery_logit)[0].detach().cpu()),
                "phase_true": str(clip.get("phase") or "unknown"),
                "phase_pred": PHASE_NAMES[phase_index],
                "split": "test",
            })
    return records, str(target_device)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    temperature, threshold = _load_calibration(args.calibration, args.checkpoint)
    metadata = {
        "release_id": args.checkpoint.parent.name,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "device_requested": args.device,
        "mode": "live_long_only",
        "split": "test",
        "subject_level_split": True,
        "temperature": temperature,
        "threshold": threshold,
    }
    records, resolved_device = collect_checkpoint_predictions(args.checkpoint, args.dataset_lock, args.split_manifest, args.data_root, device=args.device)
    metadata["device"] = resolved_device
    if temperature != 1.0:
        scores = [record["score"] for record in records]
        calibrated = _apply_temperature(scores, temperature)
        for record, score in zip(records, calibrated):
            record["score"] = score
            record["calibrated"] = True
    paths = generate_artifacts(records, args.output_dir, metadata=metadata, threshold=threshold)
    summary = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    print(json.dumps({key: summary[key] for key in ("samples", "positive_samples", "precision_at_threshold", "recall_at_threshold", "f1_at_threshold", "roc_auc", "average_precision", "brier_score", "best_f1_threshold")}, ensure_ascii=False, sort_keys=True))
    print(f"artifacts: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
