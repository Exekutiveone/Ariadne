"""Spuren: dieselbe markierte Stelle ueber mehrere Frames hinweg.

Die zeitliche Information steht am Polygon (`tracking_id`, `carried_from_frame`,
`edit`) und wird hier nur noch zusammengelesen. Es gibt bewusst kein zweites
Verzeichnis fuer Aenderungen: eine Loeschung ist keine gespeicherte Tatsache,
sondern ergibt sich daraus, dass eine `tracking_id` im naechsten gelabelten
Frame fehlt. Damit kann sie nicht von den Polygonen abweichen.

Wofuer das gut ist: das Modell soll spaeter lernen, wie sich eine Stelle ueber
die Zeit veraendert — und die Auswertung soll zeigen, wo ein uebernommenes
Polygon von Hand nachgezogen werden musste. Genau dort ist die Aufnahme schwierig.
"""

import json
from pathlib import Path

TRACK_SCHEMA_VERSION = "1.0"

EDIT_LABELS = {
    "new": "Neu gezeichnet",
    "carried_unchanged": "Unverändert übernommen",
    "carried_adjusted": "Übernommen und angepasst",
    "corrected": "Korrigiert",
}


def _annotations_of_video(mission_dir: Path, video_id: str):
    records = []
    for path in sorted((mission_dir / "ground_truth" / video_id).glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(records, key=lambda item: item.get("frame_index", 0))


def video_tracks(mission_dir: Path, video_id: str):
    """Alle Spuren eines Videos, chronologisch.

    Ein Polygon ohne `tracking_id` bekommt keine Spur — es stammt entweder aus
    der Zeit vor der Verkettung oder wurde einzeln gesetzt. Das ist kein Fehler
    und wird als eigene Zahl ausgewiesen, statt still unterzugehen.
    """
    records = _annotations_of_video(mission_dir, video_id)
    tracks: dict[str, dict] = {}
    untracked = 0
    frames_with_labels = []

    for record in records:
        frame_index = record.get("frame_index", 0)
        seen_here = set()
        for polygon in record.get("polygons", []):
            tracking_id = polygon.get("tracking_id")
            if not tracking_id:
                untracked += 1
                continue
            seen_here.add(tracking_id)
            track = tracks.setdefault(
                tracking_id,
                {
                    "tracking_id": tracking_id,
                    "class_id": polygon.get("class_id", "traversable"),
                    "first_frame": frame_index,
                    "last_frame": frame_index,
                    "frames": [],
                    "class_changes": [],
                    "ended_at_frame": None,
                },
            )
            if polygon.get("class_id", "traversable") != track["class_id"]:
                track["class_changes"].append(
                    {"frame_index": frame_index, "from": track["class_id"], "to": polygon.get("class_id")}
                )
                track["class_id"] = polygon.get("class_id", "traversable")
            track["last_frame"] = frame_index
            track["frames"].append(
                {
                    "frame_index": frame_index,
                    "edit": polygon.get("edit", "new"),
                    "carried_from_frame": polygon.get("carried_from_frame"),
                    "certainty": polygon.get("certainty", "certain"),
                    "point_count": len(polygon.get("points", [])),
                }
            )
        if seen_here or record.get("polygons"):
            frames_with_labels.append(frame_index)

    # Loeschung: die Spur taucht in einem spaeteren gelabelten Frame nicht mehr
    # auf. Abgeleitet statt gespeichert.
    for track in tracks.values():
        later = [frame for frame in frames_with_labels if frame > track["last_frame"]]
        track["ended_at_frame"] = later[0] if later else None
        track["frame_count"] = len(track["frames"])
        track["adjusted_count"] = sum(1 for item in track["frames"] if item["edit"] == "carried_adjusted")
        track["corrected_count"] = sum(1 for item in track["frames"] if item["edit"] == "corrected")

    ordered = sorted(tracks.values(), key=lambda item: (item["first_frame"], item["tracking_id"]))
    return {
        "schema_version": TRACK_SCHEMA_VERSION,
        "video_id": video_id,
        "edit_labels": EDIT_LABELS,
        "totals": {
            "tracks": len(ordered),
            "labelled_frames": len(frames_with_labels),
            "untracked_polygons": untracked,
            "ended_tracks": sum(1 for item in ordered if item["ended_at_frame"] is not None),
            "adjusted_polygons": sum(item["adjusted_count"] for item in ordered),
            "corrected_polygons": sum(item["corrected_count"] for item in ordered),
            "class_changes": sum(len(item["class_changes"]) for item in ordered),
        },
        "tracks": ordered,
        "note": (
            "Eine Löschung wird nicht gespeichert: sie ergibt sich daraus, dass eine tracking_id im nächsten "
            "gelabelten Frame fehlt (ended_at_frame)."
        ),
    }
