import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .path_model import (
    MODEL_ID,
    MODEL_SCHEMA_VERSION,
    MODEL_WIDTH,
    RANDOM_FEATURES,
    RANDOM_SEED,
    RIDGE_LAMBDA,
    SAMPLES_PER_CLASS_PER_FRAME,
    _apply_refinements,
    _choose_threshold,
    _clean_prediction,
    _comparison_mask,
    _confirmed_annotations,
    _encode_binary_rle,
    _evaluate,
    _features,
    _fit_kernel_classifier,
    _frame_split,
    _load_refinements,
    _polygon_mask,
    _predict_scores,
    _read_frames,
    _write_evidence,
    confusion_counts,
    symmetric_metrics,
)
from .processor import video_path


def _global_root(missions_root: Path):
    return missions_root.parent / "global_models" / "path_model"


def global_dataset_summary(store):
    missions = []
    for mission in store.list():
        mission_dir = store.root / mission.id
        records = _confirmed_annotations(mission_dir)
        refinements = 0
        for path in (mission_dir / "path_refinements").glob("*/*.json"):
            try:
                refinements += len(json.loads(path.read_text(encoding="utf-8")).get("items", []))
            except (OSError, ValueError):
                continue
        missions.append(
            {
                "mission_id": mission.id,
                "name": mission.name,
                "confirmed_frames": len(records),
                "videos": len({record["video_id"] for record in records}),
                "refinements": refinements,
            }
        )
    return {
        "missions": missions,
        "totals": {
            "missions": sum(item["confirmed_frames"] > 0 for item in missions),
            "confirmed_frames": sum(item["confirmed_frames"] for item in missions),
            "videos": sum(item["videos"] for item in missions),
            "refinements": sum(item["refinements"] for item in missions),
        },
    }


def current_global_model(missions_root: Path):
    root = _global_root(missions_root)
    pointer = json.loads((root / "current.json").read_text(encoding="utf-8"))
    return json.loads((root / "runs" / pointer["run_id"] / "result.json").read_text(encoding="utf-8"))


def current_global_model_dir(missions_root: Path):
    root = _global_root(missions_root)
    pointer = json.loads((root / "current.json").read_text(encoding="utf-8"))
    return root / "runs" / pointer["run_id"]


def train_global_path_model(store):
    started = time.perf_counter()
    train_frames, validation_frames = [], []
    included = []
    all_records = []
    for mission in store.list():
        mission_dir = store.root / mission.id
        records = _confirmed_annotations(mission_dir)
        if not records:
            continue
        records = [{**record, "mission_id": mission.id, "mission_name": mission.name} for record in records]
        train_records, validation_records = _frame_split(records)
        decoded_train = _read_frames(mission, mission_dir, train_records, MODEL_WIDTH)
        decoded_validation = _read_frames(mission, mission_dir, validation_records, MODEL_WIDTH)
        train_frames.extend(decoded_train)
        validation_frames.extend(decoded_validation)
        all_records.extend(records)
        included.append(
            {
                "mission_id": mission.id,
                "name": mission.name,
                "confirmed_frames": len(records),
                "train_frames": len(decoded_train),
                "validation_frames": len(decoded_validation),
            }
        )
    if len(included) < 2:
        raise ValueError("Für ein globales Modell werden Labels aus mindestens zwei Missionen benötigt")
    if len(train_frames) < 8 or len(validation_frames) < 2:
        raise ValueError("Zu wenige dekodierbare Trainings- oder Validierungsframes")

    rng = np.random.default_rng(RANDOM_SEED)
    samples, labels = [], []
    for item in train_frames:
        features = _features(item["image"])
        flat = item["mask"].reshape(-1)
        positive = np.flatnonzero(flat == 1)
        negative = np.flatnonzero(flat == 0)
        count = min(SAMPLES_PER_CLASS_PER_FRAME, len(positive), len(negative))
        if not count:
            continue
        indices = np.concatenate(
            [rng.choice(positive, count, replace=False), rng.choice(negative, count, replace=False)]
        )
        samples.append(features[indices])
        labels.append(np.concatenate([np.ones(count, np.uint8), np.zeros(count, np.uint8)]))
    training_samples = np.vstack(samples)
    training_labels = np.concatenate(labels)
    order = rng.permutation(len(training_labels))
    model = _fit_kernel_classifier(
        training_samples[order], training_labels[order], RANDOM_FEATURES, RIDGE_LAMBDA, RANDOM_SEED
    )
    threshold, threshold_metrics = _choose_threshold(validation_frames, model)
    validation_metrics, evaluated = _evaluate(validation_frames, model, threshold)
    train_metrics, _ = _evaluate(train_frames, model, threshold)

    run_id = f"global-path-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    root = _global_root(store.root)
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=runs))
    evidence_dir = staging / "evidence"
    evidence_dir.mkdir()
    try:
        np.savez_compressed(staging / "model.npz", **model, threshold=np.asarray([threshold], np.float32))
        evidence = _write_evidence(evidence_dir, "global", evaluated, "/api/v1/path-model/global/evidence")
        result = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "scope": "global_cross_mission",
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "id": MODEL_ID,
                "type": "random_fourier_kernel_ridge_pixel_classifier",
                "hardware": "CPU",
                "cloud_used": False,
                "input_width": MODEL_WIDTH,
                "feature_count": int(model["mean"].shape[0]),
                "random_features": RANDOM_FEATURES,
                "threshold": round(threshold, 6),
            },
            "dataset": {
                "missions": included,
                "confirmed_frames": len(all_records),
                "videos": len({record["video_id"] for record in all_records}),
                "refinements_included": global_dataset_summary(store)["totals"]["refinements"],
            },
            "split": {
                "strategy": "per_mission_per_video_frame_split",
                "train_frames": len(train_frames),
                "validation_frames": len(validation_frames),
                "training_pixels_sampled": len(training_labels),
                "same_frame_in_train_and_validation": False,
            },
            "scoring": {
                "rule": "50 percent missed labeled path plus 50 percent invented path",
                "threshold_selection": threshold_metrics,
            },
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "evidence": evidence,
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "limitations": [
                "Cross-mission CPU baseline; more diverse missions improve generalization.",
                "This is not a safety-relevant driving approval.",
            ],
        }
        (staging / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        target = runs / run_id
        try:
            os.replace(staging, target)
        except PermissionError:
            shutil.copytree(staging, target)
            shutil.rmtree(staging, ignore_errors=True)
        temporary = root / "current.tmp"
        temporary.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
        os.replace(temporary, root / "current.json")
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def predict_global_path_frame(store, mission_id: str, video_id: str, frame_index: int):
    mission = store.get(mission_id)
    if not mission:
        raise LookupError("Mission nicht gefunden")
    video = next((item for item in mission.videos if item.id == video_id), None)
    if not video:
        raise LookupError("Video nicht gefunden")
    model_dir = current_global_model_dir(store.root)
    result = json.loads((model_dir / "result.json").read_text(encoding="utf-8"))
    with np.load(model_dir / "model.npz") as stored:
        model = {key: stored[key] for key in ("mean", "scale", "projection", "phase", "weights")}
        threshold = float(stored["threshold"][0])
    mission_dir = store.root / mission_id
    capture = cv2.VideoCapture(str(video_path(mission_dir, video_id)))
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_index < 0 or frame_index >= total_frames:
            raise LookupError("Videoframe nicht gefunden")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok:
        raise LookupError("Videoframe konnte nicht dekodiert werden")
    width = int(result["model"]["input_width"])
    height = max(48, round(width * source_height / max(1, source_width)))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    scores = _predict_scores(_features(resized), model)
    prediction = _clean_prediction(scores, (height, width), threshold)
    response = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "scope": "global_cross_mission",
        "model_run_id": result["run_id"],
        "mission_id": mission_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "timestamp_ms": round(frame_index / fps * 1000),
        "mask": {"width": width, "height": height, "rle": _encode_binary_rle(prediction)},
        "path_fraction": round(float(prediction.mean()), 5),
        "source": "global_cpu_model_inference_on_exact_original_video_frame",
    }
    annotation_path = mission_dir / "ground_truth" / video_id / f"{frame_index:09d}.json"
    if annotation_path.is_file():
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            if annotation.get("polygons"):
                truth = _apply_refinements(_polygon_mask(annotation, width, height), mission_dir, video_id, frame_index)
                comparison = _comparison_mask(truth, prediction)
                response["evaluation"] = {
                    "annotation_status": annotation.get("status"),
                    "metrics": symmetric_metrics(confusion_counts(truth, prediction)),
                    "comparison_mask": {"width": width, "height": height, "rle": _encode_binary_rle(comparison)},
                    "legend": {"1": "correct_path", "2": "missed_label", "3": "invented_path"},
                    "refinement_count": len(_load_refinements(mission_dir, video_id, frame_index)),
                }
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return response
