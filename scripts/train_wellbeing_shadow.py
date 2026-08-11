"""Train a reproducible text baseline for the voluntary wellbeing shadow lane.

This is a research-only TF-IDF + logistic-regression baseline.  It is a safe
fallback for the MacBERT adapter contract, not a clinical model and not a
replacement for elderly external validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def _read_records(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError("each record must be an object")
        if row.get("demo") is True:
            raise ValueError("demo records cannot train the shadow model")
        subject = row.get("subject_id")
        text = row.get("text")
        label = row.get("label")
        if not isinstance(subject, str) or not subject.strip():
            raise ValueError("subject_id must be a non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if type(label) is not int or label not in (0, 1):
            raise ValueError("label must be integer 0 or 1")
        split = row.get("split")
        if split is not None and split not in {"train", "test"}:
            raise ValueError("split must be train or test")
        rows.append({**row, "subject_id": subject.strip(), "text": text.strip(), "label": label})
    if not rows:
        raise ValueError("dataset is empty")
    return rows


def _assign_splits(rows: list[dict]) -> None:
    for row in rows:
        if "split" not in row or row["split"] is None:
            digest = hashlib.sha256(row["subject_id"].encode("utf-8")).digest()[0]
            row["split"] = "test" if digest % 5 == 0 else "train"
    train_subjects = {row["subject_id"] for row in rows if row["split"] == "train"}
    test_subjects = {row["subject_id"] for row in rows if row["split"] == "test"}
    overlap = train_subjects & test_subjects
    if overlap:
        raise ValueError(f"subject appears in both train and test: {sorted(overlap)!r}")
    if not train_subjects:
        raise ValueError("training split is empty")


def train_shadow_model(input_path: str | Path, output_dir: str | Path, *, seed: int = 13, model_version: str = "research_shadow_v1") -> Path:
    rows = _read_records(input_path)
    _assign_splits(rows)
    train = [row for row in rows if row["split"] == "train"]
    labels = [row["label"] for row in train]
    if len(set(labels)) < 2:
        raise ValueError("training split must contain both classes")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    features = vectorizer.fit_transform([row["text"] for row in train])
    classifier = LogisticRegression(random_state=seed, solver="liblinear", max_iter=500, class_weight="balanced")
    classifier.fit(features, labels)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    artifact_path = destination / "shadow_model.joblib"
    artifact = {
        "schema_version": "pace-wb.shadow.v1",
        "model_version": model_version,
        "text_encoder": "tfidf_char_fallback",
        "borrowed_architecture": "MacBERT-compatible text adapter; TF-IDF baseline until MacBERT weights are approved",
        "research_only": True,
        "promoted": False,
        "training_subjects": sorted({row["subject_id"] for row in train}),
        "vectorizer": vectorizer,
        "classifier": classifier,
    }
    joblib.dump(artifact, artifact_path, compress=3)
    input_bytes = Path(input_path).read_bytes()
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "pace-wb.shadow-manifest.v1",
        "model_version": model_version,
        "research_only": True,
        "promoted": False,
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "artifact_sha256": artifact_hash,
        "training_subjects": artifact["training_subjects"],
        "test_subjects": sorted({row["subject_id"] for row in rows if row["split"] == "test"}),
        "record_count": len(rows),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return artifact_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    print(train_shadow_model(args.input, args.output_dir, seed=args.seed))


if __name__ == "__main__":
    main()
