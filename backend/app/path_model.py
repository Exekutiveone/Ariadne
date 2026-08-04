import json
import os
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .models import MissionRecord
from .path_dataset import confirmed_annotations, frame_split, read_frames, read_original_frame
from .path_features import (
    GRADE_ONTOLOGY,
    MODEL_WIDTH,
    RANDOM_FEATURES,
    RANDOM_SEED,
    RIDGE_LAMBDA,
    SAMPLES_PER_CLASS_PER_FRAME,
    clean_prediction,
    fit_kernel_classifier,
    grade_prediction,
    grading_summary,
    pixel_features,
    predict_scores,
)
from .path_masks import (
    apply_refinements,
    comparison_mask,
    confusion_counts,
    decode_rle,
    encode_binary_rle,
    load_refinements,
    polygon_mask,
    refinement_path,
    symmetric_metrics,
)

MODEL_SCHEMA_VERSION = "1.0"
MODEL_ID = "ariadne-cpu-path-rff"


def evaluate_frames(frames, model, threshold):
    total = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    per_frame = []
    for item in frames:
        features = pixel_features(item["image"])
        scores = predict_scores(features, model)
        prediction = clean_prediction(scores, item["mask"].shape, threshold)
        counts = confusion_counts(item["mask"], prediction)
        for key in total:
            total[key] += counts[key]
        per_frame.append(
            {"item": item, "scores": scores, "prediction": prediction, "metrics": symmetric_metrics(counts)}
        )
    return symmetric_metrics(total), per_frame


def choose_threshold(frames, model):
    score_sets = [predict_scores(pixel_features(item["image"]), model) for item in frames]
    merged = np.concatenate(score_sets)
    candidates = np.unique(np.quantile(merged, np.linspace(0.12, 0.88, 33)))
    best = None
    for threshold in candidates:
        total = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for item, scores in zip(frames, score_sets, strict=True):
            counts = confusion_counts(item["mask"], clean_prediction(scores, item["mask"].shape, float(threshold)))
            for key in total:
                total[key] += counts[key]
        metrics = symmetric_metrics(total)
        if best is None or metrics["symmetric_penalty_points"] < best[1]["symmetric_penalty_points"]:
            best = (float(threshold), metrics)
    return best


def write_evidence(directory: Path, mission_id: str, evaluated, evidence_base_url: str | None = None):
    evidence = []
    ranked = sorted(evaluated, key=lambda item: item["metrics"]["symmetric_penalty_points"])
    chosen = []
    for label, index in (("best", 0), ("median", len(ranked) // 2), ("worst", len(ranked) - 1)):
        if not ranked:
            break
        item = ranked[index]
        if any(
            existing[1]["item"]["record"]["video_id"] == item["item"]["record"]["video_id"]
            and existing[1]["item"]["record"]["frame_index"] == item["item"]["record"]["frame_index"]
            for existing in chosen
        ):
            continue
        chosen.append((label, item))
    for label, evaluated_item in chosen:
        item = evaluated_item["item"]
        image = item["image"]
        truth = item["mask"].astype(bool)
        prediction = evaluated_item["prediction"].astype(bool)
        overlay = image.copy()
        colours = np.zeros_like(image)
        colours[truth & prediction] = (70, 210, 90)
        colours[truth & ~prediction] = (55, 55, 235)
        colours[~truth & prediction] = (40, 210, 240)
        changed = truth | prediction
        overlay[changed] = cv2.addWeighted(image, 0.38, colours, 0.62, 0)[changed]
        combined = np.hstack([image, overlay])
        record = item["record"]
        filename = f"{label}-{record['video_id'][:8]}-{record['frame_index']:07d}.jpg"
        cv2.imwrite(str(directory / filename), combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
        evidence.append(
            {
                "kind": label,
                "video_id": record["video_id"],
                "frame_index": record["frame_index"],
                "timestamp_ms": record["timestamp_ms"],
                "metrics": evaluated_item["metrics"],
                "image_url": f"{evidence_base_url}/{filename}"
                if evidence_base_url
                else f"/api/v1/missions/{mission_id}/path-model/evidence/{filename}",
                "legend": {"green": "correct_path", "red": "missed_label", "yellow": "invented_path"},
            }
        )
    return evidence


def train_path_model(
    mission: MissionRecord,
    mission_dir: Path,
    *,
    width: int = MODEL_WIDTH,
    random_features: int = RANDOM_FEATURES,
    samples_per_class_per_frame: int = SAMPLES_PER_CLASS_PER_FRAME,
    ridge_lambda: float = RIDGE_LAMBDA,
    seed: int = RANDOM_SEED,
):
    started = time.perf_counter()
    records = confirmed_annotations(mission_dir)
    if len(records) < 10:
        raise ValueError("Mindestens 10 bestätigte Polygonframes werden für das CPU-Training benötigt")
    train_records, validation_records = frame_split(records)
    train_frames = read_frames(mission, mission_dir, train_records, width)
    validation_frames = read_frames(mission, mission_dir, validation_records, width)
    if len(train_frames) < 8 or len(validation_frames) < 2:
        raise ValueError("Zu wenige dekodierbare Trainings- oder Validierungsframes")

    rng = np.random.default_rng(seed)
    samples, labels = [], []
    for item in train_frames:
        features = pixel_features(item["image"])
        flat = item["mask"].reshape(-1)
        positive = np.flatnonzero(flat == 1)
        negative = np.flatnonzero(flat == 0)
        count = min(samples_per_class_per_frame, len(positive), len(negative))
        if not count:
            continue
        selected_positive = rng.choice(positive, count, replace=False)
        selected_negative = rng.choice(negative, count, replace=False)
        indices = np.concatenate([selected_positive, selected_negative])
        samples.append(features[indices])
        labels.append(np.concatenate([np.ones(count, np.uint8), np.zeros(count, np.uint8)]))
    training_samples = np.vstack(samples)
    training_labels = np.concatenate(labels)
    order = rng.permutation(len(training_labels))
    model = fit_kernel_classifier(training_samples[order], training_labels[order], random_features, ridge_lambda, seed)

    threshold, threshold_metrics = choose_threshold(validation_frames, model)
    validation_metrics, evaluated = evaluate_frames(validation_frames, model, threshold)
    train_metrics, _ = evaluate_frames(train_frames, model, threshold)

    run_id = f"path-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    derived = mission_dir / "derived"
    runs = derived / "path_model_runs"
    runs.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=runs))
    evidence_dir = staging / "evidence"
    evidence_dir.mkdir()
    try:
        np.savez_compressed(staging / "model.npz", **model, threshold=np.asarray([threshold], np.float32))
        evidence = write_evidence(evidence_dir, mission.id, evaluated)
        result = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "run_id": run_id,
            "mission_id": mission.id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "id": MODEL_ID,
                "type": "random_fourier_kernel_ridge_pixel_classifier",
                "hardware": "CPU",
                "cloud_used": False,
                "input_width": width,
                "feature_count": int(model["mean"].shape[0]),
                "random_features": random_features,
                "threshold": round(threshold, 6),
                "postprocessing": "3x3 opening, 7x7 closing",
            },
            "ground_truth": {
                "positive": "pixels inside confirmed traversable polygons",
                "negative": "pixels outside confirmed polygons in the same labeled frame",
                "confirmed_frames": len(records),
                "videos": len({record["video_id"] for record in records}),
            },
            "split": {
                "strategy": "frame_level_every_fifth_frame_validation_per_video",
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
                "First CPU baseline trained from a limited mission-specific dataset.",
                "Validation frames are separate, but adjacent video frames remain visually correlated.",
                "This is not a safety-relevant driving approval.",
            ],
        }
        (staging / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        report = f"""# ARIADNE CPU-Wegerkennung\n\nMission: {mission.name} ({mission.id})\n\n- Bestätigte Polygonframes: {len(records)}\n- Training: {len(train_frames)} Frames; Validierung: {len(validation_frames)} getrennte Frames\n- Symmetrischer Validierungsscore: {validation_metrics["symmetric_score"]:.2f}/100\n- Punktabzug: {validation_metrics["symmetric_penalty_points"]:.2f}\n- Übersehene gelabelte Fläche: {validation_metrics["missed_label_fraction"]:.2%}\n- Fälschlich erfundene Wegfläche: {validation_metrics["invented_path_fraction"]:.2%}\n- IoU: {validation_metrics["iou"]:.3f}; Dice: {validation_metrics["dice"]:.3f}\n- Laufzeit CPU: {result["runtime_seconds"]:.1f} s\n\nDie Metrik gewichtet übersehene Wegfläche und fälschlich erkannte Wegfläche gleich. Das Modell ist eine missionsspezifische CPU-Baseline und keine Fahrfreigabe.\n"""
        (staging / "evaluation.md").write_text(report, encoding="utf-8")
        target = runs / run_id
        try:
            os.replace(staging, target)
        except PermissionError:
            shutil.copytree(staging, target)
            shutil.rmtree(staging, ignore_errors=True)
        pointer = derived / "path_model_current.tmp"
        pointer.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
        os.replace(pointer, derived / "path_model_current.json")
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def current_path_model_dir(mission_dir: Path):
    pointer = json.loads((mission_dir / "derived" / "path_model_current.json").read_text(encoding="utf-8"))
    return mission_dir / "derived" / "path_model_runs" / pointer["run_id"]


def select_path_model_run(mission_dir: Path, run_id: str):
    target = mission_dir / "derived" / "path_model_runs" / run_id / "result.json"
    if not target.is_file():
        raise LookupError("CPU-Wegmodelllauf nicht gefunden")
    pointer = mission_dir / "derived" / "path_model_current.tmp"
    pointer.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    os.replace(pointer, mission_dir / "derived" / "path_model_current.json")


# Wiederverwendung teurer Ressourcen fuer die interaktiven Einzelframe-Endpunkte.
# Beide Caches sind klein und threadsicher; FastAPI bedient synchrone Endpunkte
# aus einem Threadpool. Modelle werden ueber die mtime von model.npz invalidiert
# (jeder Trainingslauf schreibt ein neues Run-Verzeichnis, der Schluessel aendert
# sich also ohnehin). Verdraengte VideoCapture-Handles werden nicht hart
# geschlossen, weil ein anderer Thread sie noch nutzen kann; der letzte Nutzer
# gibt sie ueber den Destruktor frei, sobald die Referenz faellt.
_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: OrderedDict = OrderedDict()
_MODEL_CACHE_SIZE = 4


def load_model_bundle(model_dir: Path):
    """Laedt model.npz und result.json eines Laufs mit mtime-invalidiertem Cache."""
    npz_path = model_dir / "model.npz"
    key = str(npz_path)
    stamp = npz_path.stat().st_mtime_ns
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None and cached["stamp"] == stamp:
            _MODEL_CACHE.move_to_end(key)
            return cached["model"], cached["threshold"], cached["result"]
    with np.load(npz_path) as stored:
        model = {name: stored[name] for name in ("mean", "scale", "projection", "phase", "weights")}
        threshold = float(stored["threshold"][0])
    result = json.loads((model_dir / "result.json").read_text(encoding="utf-8"))
    with _CACHE_LOCK:
        _MODEL_CACHE[key] = {"stamp": stamp, "model": model, "threshold": threshold, "result": result}
        while len(_MODEL_CACHE) > _MODEL_CACHE_SIZE:
            _MODEL_CACHE.popitem(last=False)
    return model, threshold, result


def predict_path_frame(mission: MissionRecord, mission_dir: Path, video_id: str, frame_index: int):
    video = next((item for item in mission.videos if item.id == video_id), None)
    if not video:
        raise LookupError("Video nicht gefunden")
    model_dir = current_path_model_dir(mission_dir)
    model, threshold, result = load_model_bundle(model_dir)
    image, fps, source_width, source_height = read_original_frame(mission_dir, video_id, frame_index)
    width = int(result["model"]["input_width"])
    height = max(48, round(width * source_height / max(1, source_width)))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    scores = predict_scores(pixel_features(resized), model)
    prediction = clean_prediction(scores, (height, width), threshold)
    grades = grade_prediction(scores, prediction, threshold, (height, width))
    margin = np.abs(scores - threshold)
    response = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_run_id": result["run_id"],
        "video_id": video_id,
        "frame_index": frame_index,
        "timestamp_ms": round(frame_index / fps * 1000),
        "mask": {"width": width, "height": height, "rle": encode_binary_rle(prediction)},
        "grade_mask": {"width": width, "height": height, "rle": encode_binary_rle(grades)},
        "grade_ontology": GRADE_ONTOLOGY,
        "grading": grading_summary(threshold),
        "path_fraction": round(float(prediction.mean()), 5),
        "mean_separation": round(float(margin.mean()), 5),
        "confidence_note": "Uncalibrated distance from the learned decision threshold.",
        "source": "cpu_model_inference_on_exact_original_video_frame",
    }
    annotation_path = mission_dir / "ground_truth" / video_id / f"{frame_index:09d}.json"
    if annotation_path.is_file():
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
            if annotation.get("polygons"):
                truth = polygon_mask(annotation, width, height)
                truth = apply_refinements(truth, mission_dir, video_id, frame_index)
                metrics = symmetric_metrics(confusion_counts(truth, prediction))
                comparison = comparison_mask(truth, prediction)
                response["evaluation"] = {
                    "annotation_status": annotation.get("status"),
                    "metrics": metrics,
                    "comparison_mask": {"width": width, "height": height, "rle": encode_binary_rle(comparison)},
                    "legend": {"1": "correct_path", "2": "missed_label", "3": "invented_path"},
                    "refinement_count": len(load_refinements(mission_dir, video_id, frame_index)),
                }
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return response


def save_path_refinement(mission: MissionRecord, mission_dir: Path, video_id: str, frame_index: int, payload):
    prediction = predict_path_frame(mission, mission_dir, video_id, frame_index)
    evaluation = prediction.get("evaluation")
    if not evaluation:
        raise ValueError("Für diesen Frame ist kein Ground-Truth-Vergleich verfügbar")
    comparison = decode_rle(evaluation["comparison_mask"])
    x = min(comparison.shape[1] - 1, max(0, round(float(payload.x) * (comparison.shape[1] - 1))))
    y = min(comparison.shape[0] - 1, max(0, round(float(payload.y) * (comparison.shape[0] - 1))))
    expected_value = 2 if payload.expected_kind == "missed_label" else 3
    if int(comparison[y, x]) != expected_value:
        raise ValueError("Die angeklickte Fläche entspricht nicht mehr der gewählten Fehlerklasse")
    count, labels = cv2.connectedComponents((comparison == expected_value).astype(np.uint8), connectivity=8)
    component_id = int(labels[y, x])
    if count < 2 or component_id == 0:
        raise ValueError("Keine zusammenhängende Fehlerfläche an dieser Position gefunden")
    region = (labels == component_id).astype(np.uint8)
    item = {
        "id": f"refine-{uuid4().hex[:12]}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model_run_id": prediction["model_run_id"],
        "kind": payload.expected_kind,
        "action": "accept_model",
        "correct_value": 0 if expected_value == 2 else 1,
        "seed": {"x": float(payload.x), "y": float(payload.y)},
        "pixel_count": int(region.sum()),
        "region_mask": {"width": comparison.shape[1], "height": comparison.shape[0], "rle": encode_binary_rle(region)},
    }
    path = refinement_path(mission_dir, video_id, frame_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = load_refinements(mission_dir, video_id, frame_index)
    items.append(item)
    record = {
        "schema_version": "1.0",
        "mission_id": mission.id,
        "video_id": video_id,
        "frame_index": frame_index,
        "items": items,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return {**item, "refinement_count": len(items)}
