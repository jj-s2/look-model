"""Shared adapter protocol and clip construction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol

from ..labels import LabelMapping, map_source_label
from ..schema import UnifiedClip, stable_clip_id


class DatasetAdapter(Protocol):
    name: str
    version: str

    def scan(self, root: Path) -> tuple[UnifiedClip, ...]:
        ...


def make_clip(
    *,
    dataset: str,
    version: str,
    root: Path,
    media_path: str,
    subject_id: str,
    camera_id: str,
    event_group_id: str,
    start_sec: float,
    end_sec: float,
    source_label: str,
    source_file: str,
    metadata: Mapping[str, object] | None = None,
    metadata_only: bool = False,
) -> UnifiedClip:
    mapping: LabelMapping = map_source_label(dataset, source_label, metadata or {})
    provenance = {
        "source_file": source_file.replace("\\", "/"),
        "mapping_reason": mapping.mapping_reason,
        "adapter_version": "1",
    }
    if metadata_only:
        provenance["metadata_only"] = "true"
        phase = None
        coarse_event = None
        supervision_mask: tuple[str, ...] = ()
    else:
        phase = mapping.phase
        coarse_event = mapping.coarse_event
        supervision_mask = mapping.supervision_mask
    relative_media = media_path.replace("\\", "/")
    clip_id = stable_clip_id(
        dataset,
        version,
        subject_id,
        camera_id,
        relative_media,
        start_sec,
        end_sec,
    )
    return UnifiedClip(
        clip_id=clip_id,
        dataset=dataset,
        dataset_version=version,
        subject_id=subject_id,
        camera_id=camera_id,
        event_group_id=event_group_id,
        media_path=relative_media,
        start_sec=float(start_sec),
        end_sec=float(end_sec),
        phase=phase,
        coarse_event=coarse_event,
        hard_negative=mapping.hard_negative,
        supervision_mask=supervision_mask,
        source_label=source_label,
        provenance=provenance,
    )


def ensure_root(root: Path) -> Path:
    resolved = Path(root).resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"dataset root must be an existing directory: {root}")
    return resolved
