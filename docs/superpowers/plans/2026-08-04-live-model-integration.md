# Live PA-DTSF Inference Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load the published PA-DTSF checkpoint and connect it to the existing C6c-compatible video, pose, temporal-window, decision, alert, and dashboard pipeline.

**Architecture:** Add a dependency-isolated `TorchPhasePredictor` that converts `DualWindow` observations into normalized COCO-17 tensors and maps model logits into the existing `PhaseModelOutput` contract. Add an Ultralytics pose-frame adapter and a lightweight live runner that composes the existing `EzvizStreamAdapter`, `VisionPhaseSource`, `PhaseRiskService`, `LiveMonitoringService`, `DecisionEngine`, and `AlertDispatcher` without importing OpenMMLab.

**Tech Stack:** Python 3.12, PyTorch 2.4.1+cu121, Ultralytics YOLO11n-pose, NumPy, OpenCV, existing phase-model contracts, Gradio dashboard, JSONL alert persistence, pytest.

## Global Constraints

- Keep short-branch quality at `0.0` because the published checkpoint contains no real short embedding.
- Use `normalize_pose_array` and COCO-17 indices; never feed raw pixel coordinates to the trained model.
- A missing/low-quality stream or model error must degrade/abstain, never fabricate a confirmed fall.
- No device credentials, playback URLs, access tokens, or verification codes may enter logs, fixtures, or committed files.
- Do not import OpenMMLab in the new live path; the current validated runtime is PyTorch + Ultralytics.
- Do not change the mental-health screening semantics or present any result as a diagnosis.
- Keep the existing `PhaseRiskService`, `DecisionEngine`, and `AlertDispatcher` contracts backward compatible.
- Every production behavior change must have a failing pytest first, then a minimal implementation.

---

### Task 1: Implement the checkpoint-backed phase predictor

**Files:**
- Create: `risk/phase_model/torch_predictor.py`
- Test: `tests/risk/phase_model/test_torch_predictor.py`
- Reference: `risk/phase_model/model.py`, `risk/phase_model/schema.py`, `risk/phase_model/normalization.py`, `configs/skeleton/padtfs_v1.py`

**Interfaces:**
- Consumes: `DualWindow`, released checkpoint dictionary with `release_id` and `model` state dict.
- Produces: `TorchPhasePredictor.predict(window) -> PhaseModelOutput`.

- [ ] **Step 1: Write failing tests**

  Add tests for: a deterministic tiny checkpoint loads on CPU; a valid output contains six phase probabilities summing to one and all four risk probabilities in `[0,1]`; missing `model` keys raise `ValueError`; explicit `cuda:0` on a CPU-only runtime raises a readable `RuntimeError`; long observations are normalized and short quality is zero.

- [ ] **Step 2: Run the predictor tests and verify the expected failure**

  Run:

  ```powershell
  python -m pytest tests/risk/phase_model/test_torch_predictor.py -q
  ```

  Expected: collection or import failure because `TorchPhasePredictor` does not exist yet.

- [ ] **Step 3: Implement the minimal predictor**

  Implement `TorchPhasePredictor` with:

  ```python
  class TorchPhasePredictor:
      def __init__(self, checkpoint: Path, *, device: str = "auto", model_version: str | None = None,
                   embedding_version: str = "short_embedding_unavailable") -> None: ...
      def predict(self, window: DualWindow) -> PhaseModelOutput: ...
  ```

  Load the configured dimensions (`SHORT_DIM=512`, `JOINTS=17`, `HIDDEN_DIM=128`), validate required state-dict keys, call `model.eval()` and `torch.no_grad()`, convert each long observation to `(17,3)` using keypoints and scores, call `normalize_pose_array`, use a zero `(1,512)` short embedding with `short_quality=0` and `long_quality=1`, apply sigmoid/softmax, and map the argmax to `Phase`.

- [ ] **Step 4: Run tests and refactor only after green**

  Run the targeted tests again, then:

  ```powershell
  python -m pytest tests/risk/phase_model/test_torch_predictor.py tests/risk/phase_model/test_service.py -q
  ```

- [ ] **Step 5: Commit**

  ```powershell
  git add risk/phase_model/torch_predictor.py tests/risk/phase_model/test_torch_predictor.py
  git commit -m "feat: add checkpoint-backed phase predictor"
  ```

### Task 2: Add the Ultralytics live pose adapter

**Files:**
- Create: `vision/ultralytics_pose.py`
- Test: `tests/vision/test_ultralytics_pose.py`
- Reference: `vision/pose_pipeline.py`, `pipeline/vision_phase_source.py`

**Interfaces:**
- Consumes: one BGR NumPy frame and a loaded YOLO pose model.
- Produces: `PoseFrameResult`-compatible values with bounding boxes, 17-point keypoints, confidence scores, and a stable single-person tracker ID.

- [ ] **Step 1: Write failing tests**

  Use a small injected fake model result to verify that a person box, keypoints, confidence scores, and `person-0` tracking ID are emitted; verify an empty result produces no observations; verify malformed model output is treated as unavailable rather than raising from the frame loop.

- [ ] **Step 2: Run the adapter tests and verify failure**

  ```powershell
  python -m pytest tests/vision/test_ultralytics_pose.py -q
  ```

  Expected: import failure because the adapter module does not exist.

- [ ] **Step 3: Implement the adapter**

  Add a lazy `YOLO` loader and a `UltralyticsPosePipeline.process(frame, timestamp)` implementation. Use `model.predict(source=frame, device=..., verbose=False, imgsz=512)`, select person boxes, convert tensors through `.cpu().numpy()`, and return the existing `PoseFrameResult` shape. Keep the model import out of module import time so tests remain CPU/dependency-light.

- [ ] **Step 4: Run adapter and existing vision tests**

  ```powershell
  python -m pytest tests/vision/test_ultralytics_pose.py tests/vision/test_input_adapter.py tests/pipeline/test_vision_phase_source.py -q
  ```

- [ ] **Step 5: Commit**

  ```powershell
  git add vision/ultralytics_pose.py tests/vision/test_ultralytics_pose.py
  git commit -m "feat: add lazy Ultralytics live pose adapter"
  ```

### Task 3: Wire the real-time runner

**Files:**
- Create: `scripts/run_live_monitor.py`
- Test: `tests/integration/test_run_live_monitor.py`
- Modify: `pipeline/vision_phase_source.py` only if a backward-compatible adapter hook is required.

**Interfaces:**
- Consumes: `--checkpoint`, `--device`, `--input`, `--smoke-seconds`, `--no-browser`, optional `--model`.
- Produces: bounded local monitoring iterations, `ServiceSnapshot`, JSONL alerts, and optional Gradio UI.

- [ ] **Step 1: Write failing runner tests**

  Verify parser defaults and explicit arguments; verify an injected fake input and fake pose model can run a bounded one-step service without opening a real camera or network URL; verify the runner rejects a missing checkpoint before opening the input.

- [ ] **Step 2: Run tests and verify failure**

  ```powershell
  python -m pytest tests/integration/test_run_live_monitor.py -q
  ```

- [ ] **Step 3: Implement composition and bounded loop**

  Compose `create_input_adapter`, `UltralyticsPosePipeline`, a stable single-person tracker, `DualTimescaleBuffer`, `TorchPhasePredictor`, `PhaseRiskService`, `VisionPhaseSource`, `LiveMonitoringService`, and `AlertDispatcher`. For local video, stop at EOF; for `--smoke-seconds`, stop at the deadline; for `--no-browser`, print a redaction-safe snapshot. Never print the input URL or credentials.

- [ ] **Step 4: Run offline integration tests**

  ```powershell
  python -m pytest tests/integration/test_run_live_monitor.py tests/pipeline/test_live_service.py tests/alerts/test_dispatcher.py -q
  ```

- [ ] **Step 5: Commit**

  ```powershell
  git add scripts/run_live_monitor.py tests/integration/test_run_live_monitor.py
  git commit -m "feat: wire checkpoint into live monitoring runner"
  ```

### Task 4: Add offline smoke verification and usage documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/env_compatibility_check.md`
- Create: `tests/integration/fixtures/` only if a tiny synthetic frame fixture is needed; do not add user video or credentials.

**Interfaces:**
- Consumes: the runner from Task 3 and the published checkpoint.
- Produces: a documented CPU smoke command and a documented GPU/C6c command.

- [ ] **Step 1: Write failing documentation smoke test**

  Add a test that runs `python scripts/run_live_monitor.py --help` by path and asserts the new flags are present without importing OpenMMLab.

- [ ] **Step 2: Run the documentation smoke test and verify failure**

  ```powershell
  python -m pytest tests/integration/test_run_live_monitor.py::test_live_runner_help -q
  ```

- [ ] **Step 3: Document commands and limitations**

  Document CPU offline smoke, GPU local-video run, and the C6c playback-address run. State explicitly that the current checkpoint is long-branch-only, SDNL1 remains field-mapping gated, and live device validation must be reported separately from public-dataset metrics.

- [ ] **Step 4: Run help and offline smoke**

  ```powershell
  python scripts/run_live_monitor.py --help
  python scripts/run_live_monitor.py --checkpoint outputs/releases/padtfs-gmdcsa24-gpu-norm/checkpoint.pt --input <offline-video> --device cpu --smoke-seconds 5 --no-browser
  ```

- [ ] **Step 5: Commit**

  ```powershell
  git add README.md docs/env_compatibility_check.md tests/integration/test_run_live_monitor.py
  git commit -m "docs: document live PA-DTSF monitoring runner"
  ```

### Task 5: Full verification and publish

**Files:**
- No new production files; inspect all changed files and generated local outputs.

- [ ] **Step 1: Run targeted and full test suites**

  ```powershell
  python -m pytest tests/risk/phase_model tests/vision tests/pipeline tests/alerts tests/integration -q
  python -m pytest -q
  ```

- [ ] **Step 2: Run static safety checks**

  Confirm no credentials, live URLs, tokens, or verification codes are in the diff; confirm generated videos, raw datasets, and caches remain ignored.

- [ ] **Step 3: Run GitNexus change-impact review**

  Run `gitnexus analyze` if the index is stale, then inspect changed execution flows for `TorchPhasePredictor`, `VisionPhaseSource`, and `LiveMonitoringService`.

- [ ] **Step 4: Commit any final test-only or documentation corrections**

  ```powershell
  git status --short
  git diff --check
  ```

- [ ] **Step 5: Push the branch**

  ```powershell
  git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 push
  ```

