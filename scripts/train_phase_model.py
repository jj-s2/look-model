"""Train PA-DTSF from frozen provenance and an explicit pose-feature cache.

The ``--demo`` path is a smoke test only and is always marked non-promotable.
Real training refuses raw videos without an extracted pose cache, preventing a
random or silently different preprocessing path from becoming a release.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_config(path: Path | None):
    if path is None:
        from configs.skeleton import padtfs_v1 as config
        return config
    spec = importlib.util.spec_from_file_location("padtfs_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_provenance(dataset_lock: Path, split_manifest: Path, release_id: str, *, demo: bool) -> tuple[dict, dict]:
    if not dataset_lock.exists():
        raise FileNotFoundError(f"dataset_lock not found: {dataset_lock}")
    if not split_manifest.exists():
        raise FileNotFoundError(f"split_manifest not found: {split_manifest}")
    lock = json.loads(dataset_lock.read_text(encoding="utf-8"))
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or not isinstance(split, dict):
        raise ValueError("frozen provenance files must contain JSON objects")
    if not demo and bool(lock.get("demo")):
        raise ValueError("demo dataset lock cannot be used for real training")
    if split.get("release_id") not in (None, release_id):
        raise ValueError("split_manifest release_id does not match")
    return lock, split


def _demo_batches(torch, device, *, count: int = 16, batch_size: int = 4):
    generator = torch.Generator(device="cpu").manual_seed(42)
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        yield (
            torch.randn(size, 512, generator=generator).to(device),
            torch.randn(size, 64, 17, 3, generator=generator).to(device),
            torch.rand(size, generator=generator).to(device), torch.rand(size, generator=generator).to(device),
            torch.randint(0, 6, (size,), generator=generator).to(device),
            torch.randint(0, 2, (size,), generator=generator).float().to(device),
            torch.randint(0, 2, (size,), generator=generator).float().to(device),
            torch.randint(0, 2, (size,), generator=generator).float().to(device),
            {"phase", "fall_event", "prefall", "recovery"},
        )


def _resolve_feature_path(root: Path, clip: Mapping[str, object]) -> Path:
    """Resolve a relative cache path whether root is dataset or collection level."""
    media_path = Path(str(clip["media_path"]))
    candidates = [root / media_path]
    dataset = str(clip.get("dataset") or "")
    if dataset:
        candidates.append(root / dataset / media_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _real_batches(torch, device, lock: Mapping[str, object], split: Mapping[str, object], root: Path):
    from risk.phase_model.normalization import normalize_pose_array

    clips = {str(item["clip_id"]): item for item in lock.get("clips", []) if isinstance(item, Mapping) and "clip_id" in item}
    train_ids = split.get("partitions", {}).get("train", [])
    samples = []
    for clip_id in train_ids:
        clip = clips.get(str(clip_id))
        if not clip:
            continue
        path = _resolve_feature_path(root, clip)
        if path.suffix.lower() not in {".npz", ".npy"}:
            raise RuntimeError("real training requires extracted .npz/.npy pose features; raw video was not silently converted")
        if path.suffix.lower() == ".npz":
            data = __import__("numpy").load(path)
            samples.append((data["short_embedding"], normalize_pose_array(data["long_pose"]), clip))
        else:
            data = __import__("numpy").load(path, allow_pickle=False)
            samples.append((data, normalize_pose_array(data), clip))
    if not samples:
        raise RuntimeError("no trainable pose-feature samples found in frozen dataset lock")
    phase_names = ["normal_adl", "prefall_abnormal", "descending", "impact", "fallen", "recovering"]
    for short, long_pose, clip in samples:
        supervision = {str(item) for item in clip.get("supervision_mask", []) if str(item) != "none"}
        phase_name = clip.get("phase")
        phase = phase_names.index(phase_name) if phase_name in phase_names else -1
        # A coarse fall label does not justify inventing impact/fallen phase targets.
        prefall = 0.0 if "prefall" in supervision else -1.0
        recovery = 0.0 if "recovery" in supervision else -1.0
        yield (
            torch.as_tensor(short, dtype=torch.float32, device=device).reshape(1, -1),
            torch.as_tensor(long_pose, dtype=torch.float32, device=device).reshape(1, 64, 17, 3),
            torch.zeros(1, device=device), torch.ones(1, device=device),
            torch.tensor([phase], device=device),
            torch.tensor([float(clip.get("coarse_event") == "fall")], device=device),
            torch.tensor([prefall], device=device),
            torch.tensor([recovery], device=device),
            supervision or {"fall_event"},
        )


def train_phase_model(*, dataset_lock: Path, split_manifest: Path, output_dir: Path, release_id: str, config_path: Path | None = None, demo: bool = False, epochs: int | None = None, data_root: Path | None = None) -> dict[str, object]:
    lock, split = _validate_provenance(dataset_lock, split_manifest, release_id, demo=demo)
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError("PyTorch is required for training") from error
    from risk.phase_model.losses import LossTargets, compute_multitask_loss
    from risk.phase_model.model import PhaseAwareFusionModel
    config = _load_config(config_path)
    seed = int(getattr(config, "SEED", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhaseAwareFusionModel(short_dim=int(getattr(config, "SHORT_DIM", 512)), joints=int(getattr(config, "JOINTS", 17)), hidden_dim=int(getattr(config, "HIDDEN_DIM", 128))).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=float(getattr(config, "LEARNING_RATE", 1e-3)), weight_decay=float(getattr(config, "WEIGHT_DECAY", 1e-4)))
    total_epochs = int(epochs if epochs is not None else getattr(config, "EPOCHS", 5))
    root = data_root or PROJECT_ROOT
    losses = []
    for _epoch in range(total_epochs):
        model.train()
        for batch in (_demo_batches(torch, device) if demo else _real_batches(torch, device, lock, split, root)):
            short, long_pose, short_q, long_q, phase, fall, prefall, recovery, supervision = batch
            optimizer.zero_grad(set_to_none=True)
            output = model(short, long_pose, short_q, long_q)
            loss = compute_multitask_loss(output, LossTargets(phase, fall, prefall, recovery), supervision)
            loss.total.backward()
            optimizer.step()
            losses.append(float(loss.total.detach().cpu()))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"release_id": release_id, "device": str(device), "model": model.state_dict()}, output_dir / "checkpoint.pt")
    (output_dir / "dataset_lock.json").write_text(dataset_lock.read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "split_manifest.json").write_text(split_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    checkpoint_sha = hashlib.sha256((output_dir / "checkpoint.pt").read_bytes()).hexdigest()
    metrics = {"release_id": release_id, "demo": demo, "promoted": False, "device": str(device), "epochs": total_epochs, "train_loss_last": losses[-1] if losses else None, "checkpoint_sha256": checkpoint_sha, "reason": "demo training" if demo else "training checkpoint requires held-out evaluation before promotion"}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-lock", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    metrics = train_phase_model(dataset_lock=args.dataset_lock, split_manifest=args.split_manifest, output_dir=args.output, release_id=args.release_id, config_path=args.config, demo=args.demo, epochs=args.epochs, data_root=args.data_root)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
