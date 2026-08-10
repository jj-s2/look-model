import random
import pickle
import json
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from risk.phase_model.rg_training import EarlyStopping, _load_provenance, _worker_init, seed_everything


def test_seed_everything_repeats_python_numpy_and_torch_values():
    seed_everything(23)
    first = (random.random(), np.random.rand(3), torch.rand(4))
    seed_everything(23)
    second = (random.random(), np.random.rand(3), torch.rand(4))
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2])


def test_early_stopping_restores_a_deep_copy_of_the_best_epoch_state():
    state = EarlyStopping(patience=2, mode="max")
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    assert state.update(0.4, epoch=0, model=model)
    with torch.no_grad():
        model.weight.fill_(2.0)
    assert state.update(0.6, epoch=1, model=model)
    with torch.no_grad():
        model.weight.fill_(99.0)
    assert not state.update(0.5, epoch=2, model=model)
    assert not state.update(0.4, epoch=3, model=model)
    assert state.should_stop
    assert state.best_epoch == 1
    state.restore(model)
    assert model.weight.item() == pytest.approx(2.0)


def test_worker_initializer_is_pickleable_for_windows_spawn():
    assert callable(pickle.loads(pickle.dumps(_worker_init)))


def test_default_rgpc_config_constants_are_exact():
    from configs.skeleton import rg_pcnet_v1 as config
    assert {name: getattr(config, name) for name in ("SEED", "INPUT_DIM", "HIDDEN_DIM", "DROPOUT", "BATCH_SIZE", "EPOCHS", "PATIENCE", "LEARNING_RATE", "WEIGHT_DECAY", "GRAD_CLIP_NORM", "NUM_WORKERS", "CORRUPTION_PROBABILITY")} == {"SEED": 42, "INPUT_DIM": 112, "HIDDEN_DIM": 128, "DROPOUT": 0.20, "BATCH_SIZE": 8, "EPOCHS": 40, "PATIENCE": 6, "LEARNING_RATE": 3e-4, "WEIGHT_DECAY": 1e-4, "GRAD_CLIP_NORM": 1.0, "NUM_WORKERS": 0, "CORRUPTION_PROBABILITY": 0.50}


def _provenance():
    lock = {"schema_version": "1.0", "clips": [{"clip_id": "a", "subject_id": "s1", "media_path": "a.npz"}, {"clip_id": "b", "subject_id": "s2", "media_path": "b.npz"}]}
    split = {"schema_version": "1.0", "release_id": "r", "partitions": {"train": ["a"], "validation": ["b"]}}
    return lock, split


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda lock, split: lock.update(schema_version="2.0"), "dataset lock schema_version"),
        (lambda lock, split: lock.update(clips={}), "clips must be a list"),
        (lambda lock, split: lock.update(clips=["bad"]), "contain mappings"),
        (lambda lock, split: lock["clips"][0].update(clip_id=""), "clip_id must be"),
        (lambda lock, split: lock["clips"][0].update(subject_id=1), "subject_id must be"),
        (lambda lock, split: lock["clips"][0].update(media_path=""), "media_path must be"),
        (lambda lock, split: lock["clips"].append(dict(lock["clips"][0])), "duplicate clip_id"),
        (lambda lock, split: split.update(partitions=[]), "partitions must be"),
        (lambda lock, split: split["partitions"].update(train="a"), "partition train"),
        (lambda lock, split: split["partitions"].update(validation=[]), "partition validation"),
        (lambda lock, split: split["partitions"].update(train=["",]), "invalid or duplicate"),
        (lambda lock, split: split["partitions"].update(train=["a", "a"]), "invalid or duplicate"),
        (lambda lock, split: split["partitions"].update(train=["missing"]), "missing from"),
    ],
)
def test_provenance_schema_rejects_each_invalid_contract_independently(mutate, message):
    lock, split = _provenance()
    mutate(lock, split)
    with pytest.raises(ValueError, match=message):
        _load_provenance(json.dumps(lock).encode(), json.dumps(split).encode(), "r")


@pytest.mark.parametrize(
    ("lock", "split", "expected"),
    [
        ([], _provenance()[1], "dataset lock schema_version must be '1.0'"),
        (_provenance()[0], [], "split manifest schema_version must be '1.0'"),
        ({**_provenance()[0], "schema_version": "2.0"}, _provenance()[1], "dataset lock schema_version must be '1.0'"),
        (_provenance()[0], {**_provenance()[1], "schema_version": "2.0"}, "split manifest schema_version must be '1.0'"),
        ({**_provenance()[0], "clips": None}, _provenance()[1], "dataset lock clips must be a list"),
        (_provenance()[0], {**_provenance()[1], "partitions": None}, "split manifest partitions must be a mapping"),
        (_provenance()[0], {**_provenance()[1], "release_id": "other"}, "split_manifest release_id does not match"),
    ],
)
def test_provenance_top_level_errors_have_exact_messages(lock, split, expected):
    with pytest.raises(ValueError, match="^" + re.escape(expected) + "$"):
        _load_provenance(json.dumps(lock).encode(), json.dumps(split).encode(), "r")
