# Deployment and verification

This repository is a local research/competition prototype. It is not a
clinical device and does not diagnose mental-health conditions. The wellbeing
module can only surface screening results, changes, and a recommendation for
human attention.

## Minimal local path

```powershell
Copy-Item .env.example .env
# Fill only the local settings needed for the selected source; never commit .env.
python scripts/probe_ezviz_devices.py --offline-fixture --write-report docs/device-capability-report.md
python -m pytest tests/integration/test_monitoring_flow.py -q
python scripts/run_pipeline.py --input <local-video.mp4>
```

The offline probe is the default safe verification route. It reads only
`tests/fixtures/ezviz_offline_unavailable.json`, makes no HTTP request, and
must report `unavailable`. It is not a device validation.

## Released-model live runner

The released normalized checkpoint can be exercised without the OpenMMLab
stack. The runner uses the lazy Ultralytics pose adapter and the existing
quality-gated `LiveMonitoringService`:

```powershell
python scripts/run_live_monitor.py `
  --checkpoint outputs/releases/padtfs-gmdcsa24-gpu-norm/checkpoint.pt `
  --input <local-video.mp4> --device auto `
  --smoke-seconds 10 --no-browser
```

Use `--device cpu` for an offline compatibility check. Use `--device auto` on
the verified RTX 4060 environment to select CUDA when it is available. A
browser dashboard is optional; omit `--no-browser` to launch the local Gradio
view. The runner prints only camera/radar health, decision counts, and error
counts. Alerts are appended to `outputs/live/local_alerts.jsonl` and no video
is written by default.

The input may be a local replay, webcam index, or a playback address obtained
through the authorized EZVIZ interface. Do not paste access tokens, device
verification codes, or signed playback URLs into reports or shell history.
The runner validates the checkpoint before opening the stream and always closes
the stream and worker service on exit.

The current release is long-branch-only in live inference: the short branch is
passed a zero embedding with `short_quality=0`. This is an explicit quality
boundary, not a claim that a short-term image embedding has been trained.

## Live-device route (only after a device is connected)

1. Copy `.env.example` to `.env`, then set `EZVIZ_APP_KEY` and
   `EZVIZ_APP_SECRET` locally. Set a stream URL only when the platform has
   provided one. Do not paste secrets, tokens, verification codes, or full
   serials into logs or reports.
2. Run `python scripts/probe_ezviz_devices.py --write-report docs/device-capability-report.md`.
   This inventories the platform response; it does **not** prove that live
   streaming or intercom works. Mark any absent field `unavailable`.
3. Verify C6c streaming, reconnect behavior, and intercom separately with the
   real device/API. Record a continuous 30-minute observation: disconnect
   count, recovery time, average FPS, and end-to-end P95 alert latency.
4. Verify SDNL1 only through fields actually returned by its authorized
   interface. If the endpoint or field is absent, keep the UI, events, and
   reports at `unavailable`; do not substitute demo physiology.

No C6c, live stream, intercom, or SDNL1 field has been validated by the
offline fixture. A device test belongs behind an explicit `--run-device-tests`
opt-in once such tests are added; the normal test suite must stay offline.

## Release evidence

```powershell
python scripts/benchmark_live_pipeline.py --input <controlled-local-replay.mp4> --duration-seconds 300 --output outputs/benchmark.json
python scripts/generate_evaluation_report.py --metrics experiments/outputs/losocv/summary.json outputs/benchmark.json --output docs/evaluation-report.md
```

Both scripts write an auditable file even when evidence is missing. They return
a non-zero exit status when a gate fails. A capture/decode-only result is
diagnostic, not a full-inference latency claim and cannot pass release.

The required gates are fall F1 ≥ 0.90, fall recall ≥ 0.88, P95 end-to-end
latency ≤ 2 seconds, and false alarms ≤ 1/hour. Evaluation data must use
subject-level splits. Do not claim a pass until all four values come from the
same documented release evaluation.

## Privacy and safe operation

- Continuous video and audio are not retained by default. Event video requires
  explicit opt-in and is limited to 10 seconds before plus 20 seconds after a
  confirmed event, with a seven-day retention policy.
- Demo fixtures are visibly marked as demo. Do not represent a replay as a
  real device stream.
- Never ask an older adult to fall. Use safeguarded healthy-adult simulations
  or approved public fall/ADL data.
- A fall can trigger one immediate “do you need help?” query; decline, silence,
  or a later response ends the interaction. Wellbeing prompts are not diagnoses
  and should recommend human follow-up where appropriate.
