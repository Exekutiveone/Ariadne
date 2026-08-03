import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .labeling import frame_reference, probe_labeling_video
from .models import GroundTruthAnnotationInput, MissionRecord
from .segmentation import current_segmentation_dir

ANNOTATION_SCHEMA_VERSION = "2.0"
GROUND_TRUTH_ONTOLOGY = {
    "unlabelled": {"value": 0, "label": "Nicht markiert / im Training ignorieren", "color": "#00000000"},
    "traversable": {"value": 1, "label": "Befahrbar", "color": "#55d96f"},
    "not_traversable": {"value": 2, "label": "Nicht befahrbar", "color": "#e05b52"},
    "unknown": {"value": 3, "label": "Nicht bewertbar", "color": "#737c78"},
}


def _annotation_root(mission_dir: Path):
    return mission_dir / "ground_truth"


def _annotation_path(mission_dir: Path, video_id: str, frame_index: int):
    return _annotation_root(mission_dir) / video_id / f"{frame_index:09d}.json"


def _frame_manifest(mission_dir: Path):
    run_dir = current_segmentation_dir(mission_dir)
    manifest_path = run_dir / "annotation_frames.json"
    if not manifest_path.is_file():
        result = json.loads((run_dir / "segmentation.json").read_text(encoding="utf-8"))
        manifest = {
            video["video_id"]: {
                str(frame["frame_index"]): {
                    "timestamp_ms": frame["timestamp_ms"],
                    "source_frame_hash": frame.get("terrain", {}).get("source_frame_hash"),
                    "mask_width": frame.get("terrain", {}).get("traversability", {}).get("mask", {}).get("width"),
                    "mask_height": frame.get("terrain", {}).get("traversability", {}).get("mask", {}).get("height"),
                }
                for frame in video["frames"]
                if frame.get("terrain")
            }
            for video in result["videos"]
        }
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, manifest_path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def frame_provenance(mission_dir: Path, video_id: str, frame_index: int):
    manifest = _frame_manifest(mission_dir)
    return manifest.get(video_id, {}).get(str(frame_index))


def _mask_statistics(payload: GroundTruthAnnotationInput):
    counts = {"unlabelled": 0, "traversable": 0, "not_traversable": 0, "unknown": 0}
    by_value = {item["value"]: key for key, item in GROUND_TRUTH_ONTOLOGY.items()}
    if payload.mask is None:
        return None
    for index in range(0, len(payload.mask.rle), 2):
        counts[by_value[payload.mask.rle[index]]] += payload.mask.rle[index + 1]
    total = payload.mask.width * payload.mask.height
    labelled = total - counts["unlabelled"]
    return {
        "pixels": counts,
        "labelled_pixels": labelled,
        "labelled_fraction": round(labelled / max(1, total), 5),
    }


def _polygon_statistics(payload: GroundTruthAnnotationInput):
    return {
        "polygon_count": len(payload.polygons),
        "point_count": sum(len(polygon.points) for polygon in payload.polygons),
        "classes": {"traversable": len(payload.polygons)},
    }


def save_annotation(
    mission: MissionRecord,
    mission_dir: Path,
    video_id: str,
    frame_index: int,
    payload: GroundTruthAnnotationInput,
):
    if not any(video.id == video_id for video in mission.videos):
        raise LookupError("Video nicht gefunden")
    video = next(item for item in mission.videos if item.id == video_id)
    full_frame_mask_only = (
        payload.mask is not None
        and not payload.polygons
        and len(payload.mask.rle) == 2
        and payload.mask.rle[0] == 2
        and payload.mask.rle[1] == payload.mask.width * payload.mask.height
    )
    if payload.polygons or payload.mask is None or full_frame_mask_only:
        metadata = probe_labeling_video(mission, mission_dir, video_id)
        if frame_index < 0 or frame_index >= metadata["total_frames"]:
            raise LookupError("Videoframe nicht gefunden")
        expected_timestamp, expected_hash = frame_reference(video, frame_index, metadata["fps"])
        tolerance_ms = max(2, round(1000 / metadata["fps"]))
        if abs(payload.timestamp_ms - expected_timestamp) > tolerance_ms:
            raise ValueError("Zeitstempel passt nicht zum Originalvideoframe")
        if payload.source_frame_hash and payload.source_frame_hash != expected_hash:
            raise ValueError("Quellframe-Hash passt nicht zum Originalvideoframe")
        if full_frame_mask_only and (payload.mask.width, payload.mask.height) != (
            metadata["width"],
            metadata["height"],
        ):
            raise ValueError("Maskengröße passt nicht zum Originalvideoframe")
        source_frame_hash = expected_hash
    else:
        provenance = frame_provenance(mission_dir, video_id, frame_index)
        if not provenance:
            raise LookupError("Analyseframe nicht gefunden")
        if provenance["timestamp_ms"] != payload.timestamp_ms:
            raise ValueError("Zeitstempel passt nicht zum Analyseframe")
        if provenance["source_frame_hash"] != payload.source_frame_hash:
            raise ValueError("Quellframe-Hash passt nicht zum Analyseframe")
        if (provenance["mask_width"], provenance["mask_height"]) != (payload.mask.width, payload.mask.height):
            raise ValueError("Maskengröße passt nicht zum Analyseframe")
        source_frame_hash = payload.source_frame_hash

    mask_statistics = _mask_statistics(payload)
    polygon_statistics = _polygon_statistics(payload)
    has_label = bool(payload.polygons) or bool(mask_statistics and mask_statistics["labelled_pixels"])
    if payload.status == "confirmed" and not has_label:
        raise ValueError("Eine bestätigte Ground Truth muss ein Polygon oder markierte Pixel enthalten")
    if payload.status == "skipped" and has_label:
        raise ValueError("Ein übersprungener Frame darf keine Ground Truth enthalten")
    statistics = polygon_statistics | ({"mask": mask_statistics} if mask_statistics else {})
    if mask_statistics:
        statistics.update(mask_statistics)

    target = _annotation_path(mission_dir, video_id, frame_index)
    target.parent.mkdir(parents=True, exist_ok=True)
    revision = 1
    if target.is_file():
        try:
            revision = int(json.loads(target.read_text(encoding="utf-8")).get("revision", 0)) + 1
        except (OSError, ValueError, TypeError):
            revision = 1
    record = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "mission_id": mission.id,
        "video_id": video_id,
        "frame_index": frame_index,
        "timestamp_ms": payload.timestamp_ms,
        "source_frame_hash": source_frame_hash,
        "polygons": [polygon.model_dump() for polygon in payload.polygons],
        "status": payload.status,
        "annotator": payload.annotator,
        "notes": payload.notes,
        "revision": revision,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "statistics": statistics,
        "ontology": GROUND_TRUTH_ONTOLOGY,
    }
    if payload.mask is not None:
        record["mask"] = payload.mask.model_dump()
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    return record


def get_annotation(mission_dir: Path, video_id: str, frame_index: int):
    path = _annotation_path(mission_dir, video_id, frame_index)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def delete_annotation(mission_dir: Path, video_id: str, frame_index: int):
    path = _annotation_path(mission_dir, video_id, frame_index)
    if not path.is_file():
        return False
    path.unlink()
    return True


def list_annotations(
    mission: MissionRecord, mission_dir: Path, video_id: str | None = None, include_geometry: bool = False
):
    if video_id is not None and not any(video.id == video_id for video in mission.videos):
        raise LookupError("Video nicht gefunden")
    items = []
    root = _annotation_root(mission_dir)
    paths = (root / video_id).glob("*.json") if video_id else root.glob("*/*.json")
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            item = {
                key: record[key]
                for key in (
                    "video_id",
                    "frame_index",
                    "timestamp_ms",
                    "source_frame_hash",
                    "status",
                    "annotator",
                    "revision",
                    "updated_at",
                    "statistics",
                )
            }
            if include_geometry:
                item["polygons"] = record.get("polygons", [])
                if "mask" in record:
                    item["mask"] = record["mask"]
            items.append(item)
        except (OSError, ValueError, KeyError):
            continue
    items.sort(key=lambda item: (item["video_id"], item["frame_index"]))
    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "mission_id": mission.id,
        "ontology": GROUND_TRUTH_ONTOLOGY,
        "counts": {
            "total": len(items),
            "draft": sum(item["status"] == "draft" for item in items),
            "confirmed": sum(item["status"] == "confirmed" for item in items),
            "skipped": sum(item["status"] == "skipped" for item in items),
        },
        "items": items,
    }
