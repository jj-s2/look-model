"""Run the released phase model through the local live-monitoring service.

The command is intentionally small and dependency-lazy: parsing ``--help`` does
not load Torch, Ultralytics, Gradio, or any device SDK.  Real inputs are opened
only after the checkpoint has been validated and the service is ready to close
them in a ``finally`` block.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

# ``python scripts/run_live_monitor.py`` sets ``sys.path[0]`` to ``scripts``;
# add the repository root so the same entry point works outside an installed
# package as well as under pytest.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from alerts.dispatcher import AlertDispatcher
from pipeline.live_service import LiveMonitoringService, ServiceSnapshot
from pipeline.vision_phase_source import VisionPhaseSource
from risk.phase_model.service import PhaseRiskService
from risk.phase_model.torch_predictor import TorchPhasePredictor
from risk.phase_model.windows import DualTimescaleBuffer
from vision.input_adapter import create_input_adapter
from vision.ultralytics_pose import SinglePersonTracker, UltralyticsPosePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the released PA-DTSF phase model on a local video, webcam, or EZVIZ stream."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="released phase-model checkpoint (.pt)")
    parser.add_argument("--input", dest="input_source", required=True, help="video path, webcam index, or stream address")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="Ultralytics pose checkpoint")
    parser.add_argument("--device", default="auto", help="inference device: auto, cpu, cuda, or cuda:0")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/live"), help="local alert and runtime output directory")
    parser.add_argument("--smoke-seconds", type=float, default=0.0, help="bounded smoke-run duration; zero means no duration limit")
    parser.add_argument("--no-browser", action="store_true", help="do not launch the optional local Gradio dashboard")
    return parser


def run_monitor(
    *,
    checkpoint: Path,
    stream: Any,
    pose_pipeline: Any,
    predictor: Any | None = None,
    output_dir: Path = Path("outputs/live"),
    device: str = "auto",
    steps: int | None = None,
    smoke_seconds: float = 0.0,
    poll_timeout_seconds: float = 2.0,
    launch_browser: bool = False,
    clock: Callable[[], datetime] | None = None,
) -> ServiceSnapshot:
    """Run bounded live monitoring and return the last redaction-safe snapshot.

    ``stream``, ``pose_pipeline`` and ``predictor`` are injectable so the same
    orchestration can be tested without opening a camera or loading a GPU model.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"phase-model checkpoint not found: {checkpoint}")
    if steps is not None and steps <= 0:
        raise ValueError("steps must be positive when provided")
    if smoke_seconds < 0:
        raise ValueError("smoke_seconds must be non-negative")
    if poll_timeout_seconds <= 0:
        raise ValueError("poll_timeout_seconds must be positive")

    now = clock or (lambda: datetime.now(timezone.utc))
    model_predictor = predictor or TorchPhasePredictor(checkpoint, device=device)
    buffer = DualTimescaleBuffer(short_frames=48, short_fps=10.0, long_frames=64, long_fps=2.0)
    phase_service = PhaseRiskService(model_predictor, buffer, clock=now)
    output_dir = Path(output_dir)
    dispatcher = AlertDispatcher(output_dir / "local_alerts.jsonl")
    dispatcher.path.parent.mkdir(parents=True, exist_ok=True)
    dispatcher.path.touch(exist_ok=True)
    source = VisionPhaseSource(stream, pose_pipeline, SinglePersonTracker(), phase_service, clock=now)
    service = LiveMonitoringService(
        (source,), dispatcher=dispatcher, clock=now,
        poll_timeout_seconds=poll_timeout_seconds,
        component_timeout_seconds=poll_timeout_seconds,
    )
    last_snapshot: ServiceSnapshot | None = None
    opened = False
    try:
        stream.open()
        opened = True
        if launch_browser:
            from ui.dashboard import build_dashboard

            build_dashboard(service).launch(inbrowser=True)
            last_snapshot = service.last_snapshot
        else:
            if steps is None and smoke_seconds <= 0:
                steps = 1
            deadline = time.monotonic() + smoke_seconds if smoke_seconds > 0 else None
            iteration = 0
            while True:
                if steps is not None and iteration >= steps:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break
                last_snapshot = service.step()
                iteration += 1
                if deadline is not None:
                    time.sleep(0.01)
        if last_snapshot is None:
            last_snapshot = service.last_snapshot
        if last_snapshot is None:
            raise RuntimeError("live monitor did not produce a snapshot")
        _print_snapshot(last_snapshot)
        return last_snapshot
    finally:
        service.close()
        if opened:
            try:
                stream.close()
            except Exception:
                pass


def _print_snapshot(snapshot: ServiceSnapshot) -> None:
    """Print only health and counts; never print source addresses or credentials."""
    payload = {
        "camera_health": snapshot.camera_health,
        "radar_health": snapshot.radar_health,
        "decision_count": len(snapshot.decisions),
        "alert_count": len(snapshot.alert_history),
        "source_error_count": len(snapshot.source_errors),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stream = create_input_adapter(args.input_source)
    pose_pipeline = UltralyticsPosePipeline(model_path=args.model, device=args.device)
    run_monitor(
        checkpoint=args.checkpoint,
        stream=stream,
        pose_pipeline=pose_pipeline,
        output_dir=args.output_dir,
        device=args.device,
        smoke_seconds=args.smoke_seconds,
        launch_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
