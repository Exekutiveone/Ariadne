"""Labels und Frames von der Platte lesen.

Der I/O-Teil des Wegmodells: bestaetigte Annotationen einsammeln, nach Videos
in Train und Validierung trennen, Frames dekodieren. Die Numerik liegt in
`path_features`, die Masken in `path_masks`.
"""

import json
import threading
from collections import OrderedDict
from pathlib import Path

import cv2

from .models import MissionRecord
from .path_features import MODEL_WIDTH
from .path_masks import PATH_POSITIVE_CLASSES, apply_refinements, polygon_mask
from .processor import video_path

# Eigener Lock je Modul: der Modell-Cache in path_model hat seinen eigenen.
_CACHE_LOCK = threading.Lock()


def confirmed_annotations(mission_dir: Path):
    records = []
    for path in (mission_dir / "ground_truth").glob("*/*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if record.get("status") != "confirmed":
            continue
        # Ein Frame, auf dem nur Hindernisse oder Problemzonen markiert sind,
        # taugt nicht als Trainingsbeispiel fuer das binaere Wegmodell — er
        # enthaelt keine einzige Wegflaeche.
        if any(
            polygon.get("class_id", "traversable") in PATH_POSITIVE_CLASSES for polygon in record.get("polygons", [])
        ):
            records.append(record)
    return sorted(records, key=lambda item: (item["video_id"], item["frame_index"]))


def frame_split(records):
    train, validation = [], []
    by_video = {}
    for record in records:
        by_video.setdefault(record["video_id"], []).append(record)
    for video_records in by_video.values():
        if len(video_records) < 5:
            train.extend(video_records[:-1] or video_records)
            if len(video_records) > 1:
                validation.append(video_records[-1])
            continue
        for index, record in enumerate(video_records):
            (validation if index % 5 == 4 else train).append(record)
    if not validation and len(train) > 1:
        validation.append(train.pop())
    return train, validation


def read_frames(
    mission: MissionRecord,
    mission_dir: Path,
    records,
    width: int = MODEL_WIDTH,
    *,
    allow_unlabelled: bool = False,
    progress=None,
):
    """Dekodiert Labelframes samt effektiver Ground-Truth-Maske.

    `allow_unlabelled=True` nimmt zusätzlich Frames ganz ohne Wegfläche auf
    (Kritisch-Meldungen: der gesamte Frame ist Negativbeispiel). `progress`
    wird, falls gesetzt, nach jedem verarbeiteten Record ohne Argumente
    aufgerufen — für Fortschrittsanzeigen langer Trainingsläufe.
    """
    by_video = {}
    for record in records:
        by_video.setdefault(record["video_id"], []).append(record)
    decoded = []
    for video_id, video_records in by_video.items():
        capture = cv2.VideoCapture(str(video_path(mission_dir, video_id)))
        try:
            source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            height = max(48, round(width * source_height / max(1, source_width)))
            for record in video_records:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(record["frame_index"]))
                ok, image = capture.read()
                if progress is not None:
                    progress()
                if not ok:
                    continue
                resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                mask = polygon_mask(record, width, height)
                mask = apply_refinements(mask, mission_dir, video_id, int(record["frame_index"]))
                labelled = mask.any() and (~mask.astype(bool)).any()
                if labelled or (allow_unlabelled and not mask.any()):
                    decoded.append({"record": record, "image": resized, "mask": mask})
        finally:
            capture.release()
    return decoded


_CAPTURE_CACHE: OrderedDict = OrderedDict()
_CAPTURE_CACHE_SIZE = 4
# Bis zu dieser Distanz wird vorwaerts weitergelesen statt gesprungen. Ein
# Seek in Long-GOP-Videos springt zum letzten Keyframe zurueck und dekodiert
# von dort (gemessen ~150 ms pro Frame-Zugriff); sequenzielles Weiterlesen
# kostet nur ~8 ms pro Frame und ist frame-exakt.
_SEQUENTIAL_READ_LIMIT = 5


def _cached_capture_entry(source: Path):
    key = str(source)
    with _CACHE_LOCK:
        entry = _CAPTURE_CACHE.get(key)
        if entry is not None:
            _CAPTURE_CACHE.move_to_end(key)
            return entry
        entry = {"capture": cv2.VideoCapture(key), "lock": threading.Lock()}
        _CAPTURE_CACHE[key] = entry
        while len(_CAPTURE_CACHE) > _CAPTURE_CACHE_SIZE:
            _CAPTURE_CACHE.popitem(last=False)
        return entry


def read_original_frame(mission_dir: Path, video_id: str, frame_index: int):
    """Liest einen exakten Originalframe ueber ein gecachtes VideoCapture-Handle."""
    source = video_path(mission_dir, video_id)
    entry = _cached_capture_entry(source)
    with entry["lock"]:
        capture = entry["capture"]
        for _attempt in range(2):
            if not capture.isOpened():
                capture.release()
                entry["capture"] = capture = cv2.VideoCapture(str(source))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_index < 0 or frame_index >= total_frames:
                raise LookupError("Videoframe nicht gefunden")
            position = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            step = frame_index - position
            if 0 <= step <= _SEQUENTIAL_READ_LIMIT:
                ok, image = False, None
                for _ in range(step + 1):
                    ok, image = capture.read()
                    if not ok:
                        break
            else:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, image = capture.read()
            if ok:
                return image, fps, source_width, source_height
            # Veraltetes Handle (z. B. nach Dateiwechsel): einmal neu oeffnen.
            capture.release()
            entry["capture"] = capture = cv2.VideoCapture(str(source))
        raise LookupError("Videoframe konnte nicht dekodiert werden")
