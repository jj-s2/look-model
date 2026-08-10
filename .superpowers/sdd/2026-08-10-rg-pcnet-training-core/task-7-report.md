# Task 7: Fold-safe optional PoseC3D distillation

## RED / GREEN

- RED: `test_teacher_distillation.py` initially failed during collection with
  `ModuleNotFoundError: risk.phase_model.teacher_distillation`.
- GREEN: the teacher contract, loss, batch collation, training, and integration
  suite passed after the minimal implementation.

## Provenance and parity

- Teacher manifests are read once into bytes; their SHA-256 is retained as
  `teacher_manifest_sha256` in `run_manifest.json`.
- Every manifest enforces exactly one normalized lower-case checkpoint SHA-256,
  recorded as `teacher_checkpoint_sha256`.
- The integration test verifies explicit `teacher_manifest=None` and omission
  produce identical deterministic checkpoint SHA-256 values. The same suite
  checks a teacher snapshot hash generated from its exact JSONL bytes and
  checkpoint SHA `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`.

## Validation evidence

- `python -m pytest tests/risk/phase_model tests/integration/test_train_phase_model.py tests/integration/test_train_rg_pcnet.py -q`
  - `203 passed, 1 skipped`
- CPU and CUDA one-epoch smoke training both ran from the same fixture. Their
  metrics and run-manifest schemas matched; checkpoint values can differ by
  device. CUDA-specific loss checks are also included in the phase-model suite
  (the one skip was unrelated to the optional path).

## Self-review

- Teacher records are immutable, clip-id sorted, train-only, fold checked, and
  reject outer-test rows before generic train-membership errors.
- KL uses the specified binary `[logit, 0]` distributions, boolean mask,
  finite positive temperature, and differentiable empty-mask zero.
- The original six loss components and checkpoint state remain unchanged when
  no teacher is supplied; only the optional `distill` component is appended.
- Validation batches are never teacher-injected.

## GitNexus

`detect_changes(scope="all")` saw unrelated dirty workspace changes and reported
critical aggregate risk. The Task 7-relevant affected flows were
`Compute_rgpc_loss → _validate_tensor` and `Main → Set_epoch` / training;
their targeted tests pass. No model weights or output artifacts are staged.

## Concern

The shared worktree contains unrelated uncommitted changes. This task stages
only its own files.

## Review fix round 1

- `TeacherLogits.__post_init__` now defensively copies, validates and sorts any
  public-constructor mapping before wrapping it in `MappingProxyType`.
- A present split `outer_fold` is now strict: only an absent key permits
  single-validation-subject derivation.
- A snapshot-race integration test mutates the JSONL during the first forward
  and proves the initial logits and both initial hashes remain in use.
- True pre/post parity was measured in Python 3.12 / CPU using the Task 6 tiny
  fixture. Baseline worktree `05e11a0d187efca28bcd7f394cc42772bc50976b` and
  current omission/explicit-None all produced checkpoint SHA
  `cc61283411aca3cbb45592468ee1572bcb977c073287bc53b1a5c9814ce860ec` and
  sorted-state digest `c0c06807edd754c8bf097d0c8fec30a866852907994c3b92fca0a6c27cd21ae7`.
