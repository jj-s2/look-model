import random
import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from risk.phase_model.rg_training import EarlyStopping, _worker_init, seed_everything


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
