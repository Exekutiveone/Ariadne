"""Auswertungsbereich je Video — ein Profil statt Handarbeit pro Frame.

Steht die Kamera fest, ist derselbe Bildbereich in jedem Frame irrelevant: oben
Himmel und Ferne, unten oft die Motorhaube oder der eigene Schatten. Ein Profil
je Video haelt das einmal fest, statt es hunderte Male neu zu zeichnen.

Das Profil **ersetzt keine Frame-Labels**: es ist ein Vorschlag, der beim
Labeln uebernommen werden kann. Was tatsaechlich gilt, steht im jeweiligen
Frame. Sonst waere spaeter nicht mehr unterscheidbar, was bewusst markiert und
was nur aus einer Voreinstellung gefallen ist.

Liegt neben der Ground Truth im Missionsordner, weil es Handarbeit ist und nicht
neu berechnet werden kann.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROI_PROFILE_SCHEMA_VERSION = "1.0"
# Erfahrungswerte als Startpunkt der Oberflaeche, keine Vorgabe: die Kamerahoehe
# entscheidet, und die ist pro Aufbau anders.
SUGGESTED_TOP_IGNORE = 0.2
SUGGESTED_BOTTOM_IGNORE = 0.1


def _path(mission_dir: Path, video_id: str):
    return mission_dir / "roi_profiles" / f"{video_id}.json"


def band_polygon(kind: str, fraction: float):
    """Waagerechtes Band am oberen oder unteren Bildrand, auf 0..1 normiert."""
    if not 0 < fraction < 1:
        raise ValueError("Der Anteil muss zwischen 0 und 1 liegen")
    top, bottom = (0.0, fraction) if kind == "top" else (1.0 - fraction, 1.0)
    return {
        "id": f"roi-{kind}",
        "class_id": "roi_ignore",
        "points": [[0.0, top], [1.0, top], [1.0, bottom], [0.0, bottom]],
        "certainty": "certain",
        "origin": "manual",
        "hard_negative": False,
        "note": f"{'Oberer' if kind == 'top' else 'Unterer'} Bildrand, {round(fraction * 100)} % ignoriert",
    }


def get_roi_profile(mission_dir: Path, video_id: str):
    path = _path(mission_dir, video_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_roi_profile(mission, mission_dir: Path, video_id: str, payload):
    if not any(video.id == video_id for video in mission.videos):
        raise LookupError("Video nicht gefunden")
    existing = get_roi_profile(mission_dir, video_id)
    record = {
        "schema_version": ROI_PROFILE_SCHEMA_VERSION,
        "mission_id": mission.id,
        "video_id": video_id,
        "top_ignore_fraction": payload.top_ignore_fraction,
        "bottom_ignore_fraction": payload.bottom_ignore_fraction,
        "roi": [polygon.model_dump() for polygon in payload.roi],
        "note": payload.note,
        "annotator": payload.annotator,
        "coordinate_space": "normalized_to_original_frame",
        "revision": (existing.get("revision", 0) + 1) if existing else 1,
        "created_at": existing.get("created_at") if existing else datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "applies_as": "suggestion_only_frame_labels_decide",
    }
    path = _path(mission_dir, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.replace(temporary, path)
    except PermissionError:
        # OneDrive kann das Umbenennen kurz sperren; direkt schreiben statt scheitern.
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.unlink(missing_ok=True)
    return record


def delete_roi_profile(mission_dir: Path, video_id: str):
    path = _path(mission_dir, video_id)
    if not path.is_file():
        return False
    path.unlink()
    return True


def resolved_roi(mission_dir: Path, video_id: str):
    """Profil samt abgeleiteter Baender — das, was die Oberflaeche anbietet."""
    profile = get_roi_profile(mission_dir, video_id)
    if profile:
        return profile
    return {
        "schema_version": ROI_PROFILE_SCHEMA_VERSION,
        "video_id": video_id,
        "top_ignore_fraction": None,
        "bottom_ignore_fraction": None,
        "roi": [],
        "note": "",
        "revision": 0,
        "suggested": {"top_ignore_fraction": SUGGESTED_TOP_IGNORE, "bottom_ignore_fraction": SUGGESTED_BOTTOM_IGNORE},
        "applies_as": "suggestion_only_frame_labels_decide",
    }
