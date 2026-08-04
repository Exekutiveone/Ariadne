"""Von Hand geplante Trajektorien je Frame.

Der Korridorcheck schlaegt eine Trajektorie vor; hier wird festgehalten, was
der Mensch daraus gemacht hat. Das ist Handarbeit wie die Ground-Truth-Polygone
und liegt deshalb im selben Missionsordner unter `trajectories/`, nicht in
`derived/` — es wird nie neu berechnet.

Die Punkte sind wie die Ground-Truth-Polygone auf das Originalbild normiert
(0..1), nie in Pixeln: das Modellraster ist 160 px breit, das Video nicht, und
eine Aenderung an einem Frame darf keinen anderen Frame veraendern.

Mehrere Trajektorien je Frame sind gleichrangig moeglich — eine Engstelle kann
mehrere sinnvolle Linien haben, keine ist die einzig gueltige. Dateien vor
Schema 2.0 enthalten ein einzelnes Objekt statt einer Liste; `get_trajectories`
erkennt das an der fehlenden Liste und liefert es unveraendert als
Einzelelement-Liste zurueck, ohne die Datei anzufassen — die Migration auf das
Listenformat passiert erst beim naechsten tatsaechlichen Schreiben.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

TRAJECTORY_SCHEMA_VERSION = "2.0"


def _directory(mission_dir: Path, video_id: str):
    return mission_dir / "trajectories" / video_id


def _path(mission_dir: Path, video_id: str, frame_index: int):
    return _directory(mission_dir, video_id) / f"{frame_index:09d}.json"


def get_trajectories(mission_dir: Path, video_id: str, frame_index: int):
    path = _path(mission_dir, video_id, frame_index)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if isinstance(data, list):
        return data
    # Schema vor 2.0: eine einzelne Trajektorie ohne Liste und ohne id.
    return [{**data, "id": data.get("id", "legacy")}]


def _write(mission_dir: Path, video_id: str, frame_index: int, items: list):
    directory = _directory(mission_dir, video_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = _path(mission_dir, video_id, frame_index)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.replace(temporary, path)
    except PermissionError:
        # OneDrive kann das Umbenennen kurz sperren; direkt schreiben statt scheitern.
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.unlink(missing_ok=True)


def _record_from_payload(payload, *, existing: dict | None) -> dict:
    return {
        "timestamp_ms": payload.timestamp_ms,
        "points": [[round(float(x), 6), round(float(y), 6)] for x, y in payload.points],
        "corridor": payload.corridor,
        "origin": payload.origin,
        "note": payload.note,
        "annotator": payload.annotator,
        "coordinate_space": "normalized_to_original_frame",
        "revision": (existing.get("revision", 0) + 1) if existing else 1,
    }


def create_trajectory(mission, mission_dir: Path, video_id: str, frame_index: int, payload):
    if not any(video.id == video_id for video in mission.videos):
        raise LookupError("Video nicht gefunden")
    if frame_index < 0:
        raise ValueError("Frameindex muss null oder groesser sein")
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "id": f"traj-{uuid4().hex[:12]}",
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "mission_id": mission.id,
        "video_id": video_id,
        "frame_index": frame_index,
        **_record_from_payload(payload, existing=None),
        "created_at": now,
        "updated_at": now,
    }
    existing_items = get_trajectories(mission_dir, video_id, frame_index)
    _write(mission_dir, video_id, frame_index, [*existing_items, record])
    return record


def update_trajectory(mission, mission_dir: Path, video_id: str, frame_index: int, trajectory_id: str, payload):
    if not any(video.id == video_id for video in mission.videos):
        raise LookupError("Video nicht gefunden")
    existing_items = get_trajectories(mission_dir, video_id, frame_index)
    current = next((item for item in existing_items if item["id"] == trajectory_id), None)
    if not current:
        raise LookupError("Trajektorie nicht gefunden")
    updated = {
        **current,
        **_record_from_payload(payload, existing=current),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write(
        mission_dir,
        video_id,
        frame_index,
        [updated if item["id"] == trajectory_id else item for item in existing_items],
    )
    return updated


def delete_trajectory(mission_dir: Path, video_id: str, frame_index: int, trajectory_id: str) -> bool:
    existing_items = get_trajectories(mission_dir, video_id, frame_index)
    remaining = [item for item in existing_items if item["id"] != trajectory_id]
    if len(remaining) == len(existing_items):
        return False
    if remaining:
        _write(mission_dir, video_id, frame_index, remaining)
    else:
        _path(mission_dir, video_id, frame_index).unlink(missing_ok=True)
    return True


def list_trajectories(mission, mission_dir: Path, video_id: str | None = None):
    if video_id and not any(video.id == video_id for video in mission.videos):
        raise LookupError("Video nicht gefunden")
    pattern = f"{video_id}/*.json" if video_id else "*/*.json"
    items = []
    for path in sorted((mission_dir / "trajectories").glob(pattern)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        items.extend(data if isinstance(data, list) else [{**data, "id": data.get("id", "legacy")}])
    items.sort(key=lambda item: (item["video_id"], item["frame_index"]))
    return {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "mission_id": mission.id,
        "counts": {
            "total": len(items),
            "manual_edit": sum(item.get("origin") == "manual_edit" for item in items),
            "manual": sum(item.get("origin") == "manual" for item in items),
            "model_proposal": sum(item.get("origin") == "model_proposal" for item in items),
        },
        "items": items,
    }
