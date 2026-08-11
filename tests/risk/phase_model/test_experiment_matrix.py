from __future__ import annotations

from risk.phase_model.experiment_matrix import VARIANTS, build_experiment_matrix


def test_matrix_contains_fixed_ablations_seeds_and_outer_subjects():
    matrix = build_experiment_matrix(
        outer_subjects=("s1", "s2", "s3", "s4"), seeds=(41, 42, 43)
    )
    assert {run.variant for run in matrix} == set(VARIANTS)
    assert {run.seed for run in matrix} == {41, 42, 43}
    assert {run.outer_subject for run in matrix} == {"s1", "s2", "s3", "s4"}
    assert len(matrix) == 7 * 3 * 4
    assert all(run.run_id for run in matrix)
    assert len({run.run_id for run in matrix}) == len(matrix)


def test_cross_domain_runs_never_train_on_held_out_dataset():
    matrix = build_experiment_matrix(
        outer_subjects=("s1",),
        seeds=(42,),
        train_datasets=("GMDCSA24", "URFD"),
        held_out_dataset="OmniFall",
    )
    assert all("OmniFall" not in run.train_datasets for run in matrix)
    assert all(run.held_out_dataset == "OmniFall" for run in matrix)


def test_run_specification_is_immutable_and_has_canonical_configuration():
    run = build_experiment_matrix(outer_subjects=("s1",), seeds=(42,))[0]
    assert run.config_overrides["phase_weight"] == 0.0
    assert run.to_dict()["run_id"] == run.run_id
    try:
        run.seed = 99
    except Exception as exc:  # frozen dataclass contract
        assert type(exc).__name__ in {"FrozenInstanceError", "AttributeError"}
    else:  # pragma: no cover - makes the contract explicit
        raise AssertionError("ExperimentRun must be immutable")
