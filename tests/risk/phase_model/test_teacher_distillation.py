import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from risk.phase_model.teacher_distillation import load_teacher_logits, masked_binary_distillation


def _record(clip_id="train-a", outer_fold="s4", fall_logit=2.0, checkpoint="a" * 64):
    return {"clip_id": clip_id, "outer_fold": outer_fold, "fall_logit": fall_logit, "checkpoint_sha256": checkpoint}


def _manifest(tmp_path, *records):
    path = tmp_path / "teacher.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_teacher_manifest_rejects_outer_test_clip(tmp_path):
    """Break caught: held-out validation logits silently enter the training teacher."""
    path = _manifest(tmp_path, _record("test-a"))
    with pytest.raises(ValueError, match="outer test clip"):
        load_teacher_logits(path, train_clip_ids={"train-a"}, outer_test_clip_ids={"test-a"}, outer_fold="s4")


def test_manifest_orders_immutable_records_and_snapshots_hash(tmp_path):
    """Break caught: mutable or unordered teacher lookup can mismatch shuffled batches."""
    path = _manifest(tmp_path, _record("z", fall_logit=-1.5), _record("a", fall_logit=3.0))
    teacher = load_teacher_logits(path, train_clip_ids={"a", "z"}, outer_test_clip_ids=set(), outer_fold="s4")
    assert tuple(teacher) == ("a", "z")
    assert teacher["a"] == pytest.approx(3.0)
    assert teacher.checkpoint_sha256 == "a" * 64
    assert teacher.manifest_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(TypeError):
        teacher["a"] = 0.0


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ((_record(), _record()), "duplicate"),
        ((_record(outer_fold="other"),), "outer_fold"),
        ((_record("unknown"),), "not present in train"),
        ((_record(fall_logit=float("nan")),), "finite"),
        ((_record(checkpoint="X"),), "SHA-256"),
        ((_record(checkpoint="a" * 64), _record("other", checkpoint="b" * 64)), "one checkpoint"),
    ],
)
def test_manifest_rejects_unsafe_or_invalid_records(tmp_path, records, message):
    """Break caught: invalid, cross-fold, or unknown teacher records are accepted."""
    path = _manifest(tmp_path, *records)
    with pytest.raises(ValueError, match=message):
        load_teacher_logits(path, train_clip_ids={"train-a", "other"}, outer_test_clip_ids=set(), outer_fold="s4")


def test_manifest_rejects_empty_and_non_object_lines(tmp_path):
    """Break caught: malformed manifests turn into a silent no-teacher run."""
    path = tmp_path / "teacher.jsonl"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_teacher_logits(path, train_clip_ids={"train-a"}, outer_test_clip_ids=set(), outer_fold="s4")
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_teacher_logits(path, train_clip_ids={"train-a"}, outer_test_clip_ids=set(), outer_fold="s4")


def test_matching_teacher_and_student_logits_have_near_zero_distillation():
    """Break caught: the binary KL pair or temperature scaling is implemented incorrectly."""
    student = torch.tensor([2.0, -1.0])
    teacher = torch.tensor([2.0, -1.0])
    loss = masked_binary_distillation(student, teacher, torch.tensor([True, True]), temperature=2.0)
    assert loss.item() < 1e-7


def test_empty_distillation_mask_is_differentiable_zero():
    """Break caught: absent teacher rows detach the training loss graph."""
    student = torch.tensor([2.0], requires_grad=True)
    loss = masked_binary_distillation(student, torch.tensor([1.0]), torch.tensor([False]))
    loss.backward()
    assert loss.item() == 0.0
    assert student.grad.item() == 0.0


@pytest.mark.parametrize("temperature", (0.0, -1.0, float("nan"), float("inf")))
def test_distillation_rejects_invalid_temperature(temperature):
    """Break caught: invalid temperature produces undefined softmax probabilities."""
    with pytest.raises(ValueError, match="temperature"):
        masked_binary_distillation(torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([True]), temperature=temperature)
