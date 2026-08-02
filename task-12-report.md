# Task 12 report — privacy-aware monitoring demo

## Delivered

- Added `LiveMonitoringService.step()` for bounded, local one-step orchestration. Source and alert failures are isolated so another component can continue.
- Added `CircularClipBuffer` with explicit `recording_opt_in`; without consent it keeps no frames and writes no files. Opted-in captures contain the 10 seconds before an event plus 20 seconds after it.
- Added `RetentionPolicy` with a seven-day default that deletes only `event_*.npz` clips.
- Added a Gradio-ready local dashboard and testable view model for device quality, fall events/trend, wellbeing change/trend, evidence, alert history, demo watermark, and user-initiated GDS-15 screening. The UI states that screening does not constitute a diagnosis.
- Added `app.py` with strictly opt-in `--demo-fixtures`; default operation has no automatic demo fallback and no real network source.

## Verification

- `python -m pytest tests/pipeline tests/storage tests/ui -q` — 6 passed.
- `python app.py --demo-fixtures --no-browser --smoke-seconds 5` — processed marked `demo=true` fixture events for five seconds and exited cleanly.
- `python -m pytest -q` — 150 passed, 3 skipped, 1 existing unrelated failure in `tests/test_scripts.py::TestManifest::test_gmdcsa24_has_official_source`: `datasets/manifest.json` supplies `expected_size_bytes: null`.
- `git diff --check` — no whitespace errors.

## Environment note

The requested `.conda\\python.exe` interpreter and Gradio are not installed in this worktree. The app's browser mode presents a clear Gradio dependency error; no-browser smoke mode remains usable without it.
