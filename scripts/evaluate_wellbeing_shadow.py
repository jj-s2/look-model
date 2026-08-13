"""Evaluate research-only wellbeing shadow artifacts with auditable slices."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score

from scripts.train_wellbeing_shadow import _read_records


def _ece(labels: list[int], probabilities: list[float], bins: int = 10) -> float:
    if not labels:
        return 0.0
    result = 0.0
    total = len(labels)
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [i for i, probability in enumerate(probabilities) if lower <= probability < upper or (index == bins - 1 and probability <= upper)]
        if not selected:
            continue
        accuracy = sum(labels[i] == (probabilities[i] >= 0.5) for i in selected) / len(selected)
        confidence = sum(probabilities[i] for i in selected) / len(selected)
        result += len(selected) / total * abs(accuracy - confidence)
    return float(result)


def evaluate_shadow_model(artifact_path: str | Path, input_path: str | Path, output_path: str | Path | None = None) -> dict:
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != "pace-wb.shadow.v1":
        raise ValueError("unsupported shadow artifact schema")
    if artifact.get("research_only") is not True or artifact.get("promoted") is not False:
        raise ValueError("shadow artifact must remain research-only and unpromoted")
    rows = _read_records(input_path)
    train_subjects = set(artifact.get("training_subjects", ()))
    rows = [row for row in rows if row.get("split") == "test"] or rows
    if train_subjects & {row["subject_id"] for row in rows}:
        raise ValueError("evaluation subjects overlap training subjects")
    vectorizer = artifact.get("vectorizer")
    classifier = artifact.get("classifier")
    if vectorizer is None or classifier is None:
        raise ValueError("shadow artifact is missing model components")
    labels = [row["label"] for row in rows]
    probabilities = [float(probability) for probability in classifier.predict_proba(vectorizer.transform([row["text"] for row in rows]))[:, 1]]
    predictions = [probability >= 0.5 for probability in probabilities]
    metrics = {
        "sample_count": len(rows),
        "positive_rate": float(sum(labels) / len(labels)) if labels else 0.0,
        "average_precision": float(average_precision_score(labels, probabilities)) if len(set(labels)) > 1 else None,
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) > 1 else None,
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": _ece(labels, probabilities),
        "coverage": 1.0,
        "risk_at_coverage": float(sum(label != prediction for label, prediction in zip(labels, predictions)) / len(labels)) if labels else 0.0,
    }
    missingness: dict[str, dict] = {}
    for value in (False, True):
        subset = [i for i, row in enumerate(rows) if bool(row.get("audio_present", False)) is value]
        if subset:
            missingness[str(value)] = {
                "count": len(subset),
                "brier": float(brier_score_loss([labels[i] for i in subset], [probabilities[i] for i in subset])),
            }
    result = {
        "schema_version": "pace-wb.shadow-evaluation.v1",
        "model_version": artifact.get("model_version"),
        "model_mode": "research_shadow",
        "research_only": True,
        "promoted": False,
        "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "artifact_sha256": hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest(),
        "metrics": metrics,
        "missingness_slices": missingness,
        "limitations": ["research dataset only", "no elderly external validation", "text baseline is not MacBERT weights"],
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_shadow_model(args.artifact, args.input, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
