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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    train = load_split(args.data, "train")
    valid = load_split(args.data, "validation")
    all_rows = train + valid
    texts = [r[2] for r in all_rows]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=12000)
    text_matrix = vectorizer.fit_transform(texts)
    audio_matrix = np.vstack([r[3] for r in all_rows])
    scaler = StandardScaler()
    audio_matrix = scaler.fit_transform(audio_matrix)
    from scipy.sparse import csr_matrix, hstack
    X = hstack([text_matrix, csr_matrix(audio_matrix)], format="csr")
    y = np.asarray([r[1] for r in all_rows], dtype=np.int64)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    clf.fit(X[: len(train)], y[: len(train)])
    pred = clf.predict(X[len(train):])
    prob = clf.predict_proba(X[len(train):])[:, 1]
    metrics = {
        "train_subjects": len(train), "validation_subjects": len(valid),
        "train_positive": int(y[: len(train)].sum()), "validation_positive": int(y[len(train):].sum()),
        "accuracy": float(accuracy_score(y[len(train):], pred)),
        "precision": float(precision_score(y[len(train):], pred, zero_division=0)),
        "recall": float(recall_score(y[len(train):], pred, zero_division=0)),
        "f1": float(f1_score(y[len(train):], pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y[len(train):], prob)) if len(np.unique(y[len(train):])) == 2 else None,
        "label_rule": "new_label >= 53 => elevated depression-risk class",
        "model": "char TF-IDF + 11-dim audio statistics + balanced logistic regression",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": clf, "vectorizer": vectorizer, "audio_scaler": scaler}, args.out / "eatd_audio_text_baseline.joblib")
    (args.out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "validation_predictions.jsonl").write_text(
        "".join(json.dumps({"subject_id": r[0], "label": int(r[1]), "probability": float(p), "prediction": int(q)}) + "\n"
                for r, p, q in zip(valid, prob, pred)), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
