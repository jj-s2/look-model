"""Train a reproducible subject-level EATD audio-text baseline.

The public EATD archive is kept outside the repository.  This script only
reads it and writes model artifacts to the requested output directory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from scipy.fft import rfft, rfftfreq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def _audio_features(path: Path) -> np.ndarray:
    x, sr = sf.read(path, always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.size == 0:
        return np.zeros(11, dtype=np.float32)
    x = x - float(x.mean())
    scale = float(np.max(np.abs(x))) or 1.0
    x = x / scale
    spec = np.abs(rfft(x))
    freqs = rfftfreq(x.size, 1.0 / sr)
    power = spec * spec
    denom = float(power.sum()) + 1e-8
    centroid = float((freqs * power).sum() / denom)
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * power).sum() / denom))
    cdf = np.cumsum(power) / denom
    rolloff = float(freqs[min(len(freqs) - 1, int(np.searchsorted(cdf, 0.85)))])
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x)))))
    rms = float(np.sqrt(np.mean(x * x)))
    q = np.quantile(np.abs(x), [0.1, 0.25, 0.5, 0.75, 0.9]).astype(np.float32)
    return np.asarray([rms, zcr, centroid / max(sr, 1), bandwidth / max(sr, 1),
                       rolloff / max(sr, 1), math.log1p(x.size / max(sr, 1)), *q], dtype=np.float32)


def _subject(root: Path, split: str, number: str) -> tuple[str, int, str, np.ndarray]:
    folder = root / split / number
    label_path = folder / "new_label.txt"
    label = float(label_path.read_text(encoding="utf-8").strip())
    # SDS >= 53 is the convention used by the supplied EATD classification code.
    target = int(label >= 53.0)
    texts: list[str] = []
    feats: list[np.ndarray] = []
    for emotion in ("positive", "negative", "neutral"):
        txt = folder / f"{emotion}.txt"
        if txt.exists():
            texts.append(txt.read_text(encoding="utf-8", errors="ignore"))
        wav = folder / f"{emotion}_out.wav"
        if not wav.exists():
            wav = folder / f"{emotion}.wav"
        if wav.exists():
            feats.append(_audio_features(wav))
    audio = np.mean(np.stack(feats), axis=0) if feats else np.zeros(11, dtype=np.float32)
    return number, target, " ".join(texts), audio


def load_split(root: Path, split: str):
    rows = [_subject(root, split, p.name) for p in sorted((root / split).iterdir()) if p.is_dir()]
    if not rows:
        raise RuntimeError(f"no EATD subjects found under {root / split}")
    return rows


def select_f1_threshold(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, float]]:
    """Select a deterministic research threshold from training-only OOF scores."""
    scores = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 1 or target.ndim != 1 or len(scores) != len(target) or not len(scores):
        raise ValueError("probabilities and labels must be equally sized non-empty vectors")
    if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if not np.isin(target, (0, 1)).all() or len(np.unique(target)) != 2:
        raise ValueError("labels must contain both binary classes")
    best: tuple[float, float, float, float] | None = None
    for threshold in sorted(float(value) for value in np.unique(scores)):
        predicted = scores >= threshold
        f1 = float(f1_score(target, predicted, zero_division=0))
        recall = float(recall_score(target, predicted, zero_division=0))
        precision = float(precision_score(target, predicted, zero_division=0))
        candidate = (f1, recall, threshold, precision)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    f1, recall, threshold, precision = best
    return threshold, {"f1": f1, "precision": precision, "recall": recall}


def train_text_baseline(
    texts: list[str], labels: np.ndarray, *, c_candidates: tuple[float, ...] = (.1, 1.0, 10.0), cv_splits: int = 5,
) -> dict[str, object]:
    """Fit an EATD research baseline without exposing validation text to fitting."""
    target = np.asarray(labels, dtype=np.int64)
    if len(texts) != len(target) or not texts:
        raise ValueError("texts and labels must be equally sized and non-empty")
    if not np.isin(target, (0, 1)).all() or len(np.unique(target)) != 2:
        raise ValueError("labels must contain both binary classes")
    if cv_splits < 2 or min(int((target == value).sum()) for value in (0, 1)) < cv_splits:
        raise ValueError("each class must contain at least cv_splits examples")
    candidates = tuple(float(value) for value in c_candidates)
    if not candidates or any(not math.isfinite(value) or value <= 0.0 for value in candidates):
        raise ValueError("c_candidates must contain positive finite values")
    folds = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    selection: tuple[float, float, float, dict[str, float], np.ndarray] | None = None
    for c_value in candidates:
        vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 5), min_df=1, max_features=30000, sublinear_tf=True,
        )
        classifier = LogisticRegression(
            C=c_value, max_iter=5000, class_weight="balanced", random_state=42,
        )
        oof = cross_val_predict(
            Pipeline([( "vectorizer", vectorizer), ("classifier", classifier)]),
            texts, target, cv=folds, method="predict_proba",
        )[:, 1]
        threshold, metrics = select_f1_threshold(oof, target)
        candidate = (metrics["f1"], metrics["recall"], c_value, metrics, oof)
        if selection is None or candidate[:3] > selection[:3]:
            selection = candidate
    assert selection is not None
    _, _, c_value, oof_metrics, oof_scores = selection
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 5), min_df=1, max_features=30000, sublinear_tf=True,
    )
    classifier = LogisticRegression(
        C=c_value, max_iter=5000, class_weight="balanced", random_state=42,
    )
    features = vectorizer.fit_transform(texts)
    classifier.fit(features, target)
    threshold, oof_metrics = select_f1_threshold(oof_scores, target)
    return {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "threshold": threshold,
        "candidate_c": c_value,
        "oof_metrics": oof_metrics,
        "threshold_selection_scope": "stratified_train_oof_only",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    train = load_split(args.data, "train")
    valid = load_split(args.data, "validation")
    train_texts = [row[2] for row in train]
    valid_texts = [row[2] for row in valid]
    train_labels = np.asarray([row[1] for row in train], dtype=np.int64)
    valid_labels = np.asarray([row[1] for row in valid], dtype=np.int64)
    result = train_text_baseline(train_texts, train_labels)
    vectorizer = result["vectorizer"]
    clf = result["classifier"]
    threshold = float(result["threshold"])
    probabilities = clf.predict_proba(vectorizer.transform(valid_texts))[:, 1]
    pred = (probabilities >= threshold).astype(np.int64)
    metrics = {
        "train_subjects": len(train), "validation_subjects": len(valid),
        "train_positive": int(train_labels.sum()), "validation_positive": int(valid_labels.sum()),
        "accuracy": float(accuracy_score(valid_labels, pred)),
        "precision": float(precision_score(valid_labels, pred, zero_division=0)),
        "recall": float(recall_score(valid_labels, pred, zero_division=0)),
        "f1": float(f1_score(valid_labels, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(valid_labels, probabilities)) if len(np.unique(valid_labels)) == 2 else None,
        "label_rule": "new_label >= 53 => elevated depression-risk class",
        "model": "train-only char TF-IDF + balanced logistic regression",
        "feature_fit_scope": "official_train_partition_only",
        "threshold_selection_scope": result["threshold_selection_scope"],
        "threshold": threshold,
        "selected_c": result["candidate_c"],
        "train_oof": result["oof_metrics"],
        "promoted": False,
        "claim_boundary": "research_screening_only_not_clinical_diagnosis",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": clf, "vectorizer": vectorizer, "threshold": threshold, "metrics": metrics}, args.out / "eatd_text_baseline.joblib")
    (args.out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "validation_predictions.jsonl").write_text(
        "".join(json.dumps({"subject_id": r[0], "label": int(r[1]), "probability": float(p), "prediction": int(q), "threshold": threshold}) + "\n"
                for r, p, q in zip(valid, probabilities, pred)), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
