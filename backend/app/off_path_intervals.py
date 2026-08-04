"""Off-Path-Intervalle: Zeitspannen ganz ohne befahrbare Flaeche.

Waehrend des Anschauens eines Videos markiert, nicht durch Einzelbild-Labeling:
ein Start- und ein Endzeitpunkt, zwischen denen KEIN einziger Frame einen
befahrbaren Bereich zeigt — das Fahrzeug ist vom Weg abgekommen, im Graben,
zwischen dichtem Gebuesch. Anders als eine normale bestaetigte Ground Truth ist
das nicht "kein Weg markiert", sondern ausdruecklich "es gibt hier keinen Weg".

`synthetic_off_path_records` erzeugt daraus vollstaendig negative Frame-Records
im selben Format wie `confirmed_annotations` in `path_dataset.py`. Ein Record
ohne Polygone ergibt ueber `polygon_mask` eine durchgehend negative Maske;
`read_frames(..., allow_unlabelled=True)` nimmt solche Frames als
Trainingsbeispiele auf. Das Training bestraft das Modell dann dafuer, hier
trotzdem Wegflaeche zu erfinden (invented_path).

Intervalle sind Handarbeit wie Ground Truth und liegen unter
data/missions/<mission_id>/off_path_intervals/<video_id>.json als Liste.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .labeling import probe_labeling_video
from .models import MissionRecord, OffPathIntervalInput

OFF_PATH_INTERVAL_SCHEMA_VERSION = "1.0"
# Jeder n-te Frame innerhalb eines Intervalls wird als Trainingsbeispiel
# gezogen. Kleiner als beim Terrainmodell, weil Intervalle typischerweise kurz
# sind und sonst zu wenige Frames uebrig blieben.
DEFAULT_FRAME_STRIDE = 10


def _path(mission_dir: Path, video_id: str) -> Path:
    return mission_dir / "off_path_intervals" / f"{video_id}.json"


def list_off_path_intervals(mission_dir: Path, video_id: str):
    path = _path(mission_dir, video_id)
    if not path.is_file():
        return []
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return sorted(items, key=lambda item: item["start_ms"])


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _write(mission_dir: Path, video_id: str, items: list):
    path = _path(mission_dir, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def save_off_path_interval(mission: MissionRecord, mission_dir: Path, video_id: str, payload: OffPathIntervalInput):
    if not any(video.id == video_id for video in mission.videos):
        raise LookupError("Video nicht gefunden")
    existing = list_off_path_intervals(mission_dir, video_id)
    if any(_overlaps(payload.start_ms, payload.end_ms, item["start_ms"], item["end_ms"]) for item in existing):
        raise ValueError("Das Intervall überschneidet sich mit einem bereits gespeicherten Intervall")
    record = {
        "id": f"off-{uuid4().hex[:12]}",
        "schema_version": OFF_PATH_INTERVAL_SCHEMA_VERSION,
        "mission_id": mission.id,
        "video_id": video_id,
        "start_ms": payload.start_ms,
        "end_ms": payload.end_ms,
        "note": payload.note,
        "annotator": payload.annotator,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write(mission_dir, video_id, [*existing, record])
    return record


def delete_off_path_interval(mission_dir: Path, video_id: str, interval_id: str) -> bool:
    existing = list_off_path_intervals(mission_dir, video_id)
    remaining = [item for item in existing if item["id"] != interval_id]
    if len(remaining) == len(existing):
        return False
    _write(mission_dir, video_id, remaining)
    return True


def synthetic_off_path_records(mission: MissionRecord, mission_dir: Path, frame_stride: int = DEFAULT_FRAME_STRIDE):
    """Vollstaendig negative Frame-Records aus allen gespeicherten Intervallen
    aller Videos der Mission — fuer `read_frames(..., allow_unlabelled=True)`.
    """
    records = []
    for video in mission.videos:
        intervals = list_off_path_intervals(mission_dir, video.id)
        if not intervals:
            continue
        try:
            metadata = probe_labeling_video(mission, mission_dir, video.id)
        except (OSError, ValueError):
            continue
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        for interval in intervals:
            start_frame = max(0, round(interval["start_ms"] / 1000 * fps))
            end_frame = min(total_frames - 1, round(interval["end_ms"] / 1000 * fps))
            for frame_index in range(start_frame, end_frame + 1, max(1, frame_stride)):
                records.append(
                    {
                        "video_id": video.id,
                        "frame_index": frame_index,
                        "timestamp_ms": round(frame_index / fps * 1000),
                        "mission_id": mission.id,
                        "mission_name": mission.name,
                        "status": "confirmed",
                        "polygons": [],
                        "off_path_interval_id": interval["id"],
                    }
                )
    return records
