from __future__ import annotations

import numpy as np

from scripts.train_eatd_baseline import select_f1_threshold, train_text_baseline


def test_select_f1_threshold_uses_only_supplied_oof_scores() -> None:
    threshold, metrics = select_f1_threshold(
        np.asarray([.1, .2, .7, .9]), np.asarray([0, 1, 1, 0])
    )

    assert threshold == .2
    assert metrics["f1"] == .8
    assert metrics["recall"] == 1.0


def test_text_baseline_fits_vocabulary_and_threshold_inside_train_partition() -> None:
    train_texts = [
        "calm routine normal", "calm routine normal", "calm routine normal",
        "withdrawn hopeless sleep", "withdrawn hopeless sleep", "withdrawn hopeless sleep",
    ]
    train_labels = np.asarray([0, 0, 0, 1, 1, 1])
    validation_texts = ["validationonlytoken calm", "validationonlytoken withdrawn"]

    result = train_text_baseline(
        train_texts, train_labels, c_candidates=(.1,), cv_splits=3,
    )

    vocabulary = result["vectorizer"].vocabulary_
    assert "valid" not in vocabulary
    assert result["threshold_selection_scope"] == "stratified_train_oof_only"
    assert 0.0 <= result["threshold"] <= 1.0
    probabilities = result["classifier"].predict_proba(
        result["vectorizer"].transform(validation_texts)
    )[:, 1]
    assert probabilities.shape == (2,)
