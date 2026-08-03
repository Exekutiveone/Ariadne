"""Kritisch-Meldungen: fehlerhafte Frames und markierte Fehlerbereiche.

Gedacht fuer Aufnahmen, in denen das Modell grob falsch liegt. Eine Meldung
speichert den betroffenen Frame, einen Schweregrad von 1 bis 5 und optional
eine per Pinsel markierte Fehlerregion. Wie Ground Truth und Refinements sind
Meldungen Handarbeit und werden unter data/missions/<mission_id>/critical_flags
versioniert.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .labeling import probe_labeling_video
from .models import CriticalFlagInput, MissionRecord

CRITICAL_FLAG_SCHEMA_VERSION = "1.0"
CRITICAL_FLAG_KIND = "no_path_false_detection"


def _flag_root(mission_dir: Path):
    return mission_dir / "critical_flags"


def _flag_path(mission_dir: Path, video_id: str, frame_index: int):
    return _flag_root(mission_dir) / video_id / f"{frame_index:09d}.json"


def load_critical_flag_records(mission_dir: Path, video_id: str | None = None):
    """Rohliste aller Meldungen, ohne Missionsvalidierung - fuer Training und Zaehlung."""
    records = []
    pattern = f"{video_id}/*.json" if video_id else "*/*.json"
    for path in _flag_root(mission_dir).glob(pattern):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(records, key=lambda item: (item.get("video_id", ""), item.get("frame_index", 0)))


def count_critical_flags(mission_dir: Path):
    return sum(1 for _ in _flag_root(mission_dir).glob("*/*.json"))


def save_critical_flag(
    mission: MissionRecord,
    mission_dir: Path,
    video_id: str,
    frame_index: int,
    payload: CriticalFlagInput,
):
    metadata = probe_labeling_video(mission, mission_dir, video_id)
    if frame_index < 0 or frame_index >= metadata["total_frames"]:
        raise LookupError("Videoframe nicht gefunden")
    annotation_path = mission_dir / "ground_truth" / video_id / f"{frame_index:09d}.json"
    if annotation_path.is_file():
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            annotation = {}
        if annotation.get("status") == "confirmed" and annotation.get("polygons"):
            raise ValueError(
                "Dieser Frame hat eine bestaetigte Ground Truth mit Wegflaeche; "
                "entferne zuerst das Label oder waehle einen anderen Frame"
            )
    record = {
        "schema_version": CRITICAL_FLAG_SCHEMA_VERSION,
        "kind": CRITICAL_FLAG_KIND,
        "meaning": "Aufnahme mit grobem Modellfehler; gemeldete Bereiche sollen im naechsten Training besonders beachtet werden",
        "mission_id": mission.id,
        "video_id": video_id,
        "frame_index": frame_index,
        "timestamp_ms": round(frame_index / metadata["fps"] * 1000),
        "severity": int(payload.severity),
        "note": payload.note,
        "annotator": payload.annotator,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if payload.brush_mask is not None:
        record["brush_mask"] = payload.brush_mask.model_dump()
    target = _flag_path(mission_dir, video_id, frame_index)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return record


def delete_critical_flag(mission_dir: Path, video_id: str, frame_index: int):
    path = _flag_path(mission_dir, video_id, frame_index)
    if not path.is_file():
        return False
    path.unlink()
    return True


def list_critical_flags(mission: MissionRecord, mission_dir: Path, video_id: str | None = None):
    if video_id is not None and not any(video.id == video_id for video in mission.videos):
        raise LookupError("Video nicht gefunden")
    items = load_critical_flag_records(mission_dir, video_id)
    return {
        "schema_version": CRITICAL_FLAG_SCHEMA_VERSION,
        "mission_id": mission.id,
        "kind": CRITICAL_FLAG_KIND,
        "counts": {"total": len(items)},
        "items": items,
    }
