from risk.prefall_evaluation import make_subject_folds, should_promote


SUBJECT_IDS = ["a", "a", "b", "b", "c", "c"]


def test_subjects_never_cross_train_and_validation_folds():
    folds = make_subject_folds(SUBJECT_IDS)

    for train_idx, valid_idx in folds:
        train_subjects = {SUBJECT_IDS[index] for index in train_idx}
        valid_subjects = {SUBJECT_IDS[index] for index in valid_idx}
        assert train_subjects.isdisjoint(valid_subjects)


def test_candidate_is_not_promoted_when_recall_regresses():
    assert should_promote(
        candidate={"f1": 0.92, "recall": 0.84},
        baseline={"f1": 0.90, "recall": 0.88},
    ) is False


def test_promoted_candidate_must_meet_baseline_and_global_gates():
    assert should_promote(
        candidate={"f1": 0.91, "recall": 0.89},
        baseline={"f1": 0.90, "recall": 0.88},
    ) is True
    assert should_promote(
        candidate={"f1": 0.89, "recall": 0.90},
        baseline={"f1": 0.80, "recall": 0.80},
    ) is False
