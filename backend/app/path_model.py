import json
import math
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
from .processor import video_path

MODEL_SCHEMA_VERSION = "1.0"
MODEL_ID = "ariadne-cpu-path-rff"
MODEL_WIDTH = 160
RANDOM_FEATURES = 64
SAMPLES_PER_CLASS_PER_FRAME = 450
RIDGE_LAMBDA = 0.08
RANDOM_SEED = 42

# Abstufung der binären Wegmaske in sechs Anzeigeklassen (AGENT_ANWEISUNG.md).
# Farb- und Wertetabelle ist mit dem Nutzer abgestimmt und im Frontend exakt
# zu übernehmen; Aufbau wie GROUND_TRUTH_ONTOLOGY in annotations.py.
GRADE_ONTOLOGY = {
    "unrated": {"value": 0, "label": "Nicht bewertet / Umgebung", "color": "#00000000"},
    "safe": {"value": 1, "label": "Sicher befahrbar", "color": "#1e8c46"},
    "good": {"value": 2, "label": "Gut befahrbar", "color": "#55d96f"},
    "marginal": {"value": 3, "label": "Knapp befahrbar", "color": "#a3ecb4"},
    "risky": {"value": 4, "label": "Potenziell befahrbar, mit Risiko", "color": "#f08c3a"},
    "problem": {"value": 5, "label": "Problemzone / Hindernis", "color": "#e05b52"},
}
# Bandgrenzen auf dem normierten Abstand m = (score - threshold) / max(1e-6, 1 - threshold).
# Startwerte laut Arbeitsanweisung; sie stehen in jeder API-Antwort (grading-Block),
# damit Ergebnisse reproduzierbar bleiben.
GRADE_SAFE_MIN_MARGIN = 0.6
GRADE_GOOD_MIN_MARGIN = 0.25
GRADE_RISKY_MIN_MARGIN = -0.2
# Rote Problemzonen: nur zusammenhängende sicher-negative Flächen, die mindestens
# diesen Bildanteil belegen und der Grünfläche nahe kommen (Dilatationsradius in
# Pixeln des Modellrasters, MODEL_WIDTH breit).
GRADE_PROBLEM_MIN_AREA_FRACTION = 0.002
GRADE_PROBLEM_NEIGHBOURHOOD = 9
# Qualifizierte Komponenten färben nur ihre Pixel in diesem Band um den Fahrbereich
# (Kernelgröße der Dilatation, Reichweite ~12 px im Modellraster). Nötig, weil in
# echten Waldframes Wald und Himmel EINE zusammenhängende Negativkomponente bilden,
# die den Weg immer irgendwo berührt — ohne Begrenzung würde der gesamte Hintergrund
# rot. Realdaten-Befund vom 03.08.2026: ~50 % Rotanteil pro Frame statt Hindernissen
# direkt am Weg. Himmel und ferne Umgebung bleiben mit dem Band transparent.
GRADE_PROBLEM_CLIP = 25


def _confirmed_annotations(mission_dir: Path):
    records = []
    for path in (mission_dir / "ground_truth").glob("*/*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if record.get("status") == "confirmed" and record.get("polygons"):
            records.append(record)
    return sorted(records, key=lambda item: (item["video_id"], item["frame_index"]))


def _frame_split(records):
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


def _polygon_mask(record, width: int, height: int):
    mask = np.zeros((height, width), np.uint8)
    for polygon in record.get("polygons", []):
        points = np.asarray(
            [[round(float(x) * (width - 1)), round(float(y) * (height - 1))] for x, y in polygon.get("points", [])],
            np.int32,
        )
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
    return mask


def _decode_rle(mask_record):
    values = np.empty(int(mask_record["width"]) * int(mask_record["height"]), dtype=np.uint8)
    cursor = 0
    for value, length in zip(mask_record["rle"][::2], mask_record["rle"][1::2]):
        values[cursor : cursor + int(length)] = int(value)
        cursor += int(length)
    if cursor != len(values):
        raise ValueError("Ungültige Refinement-Maske")
    return values.reshape(int(mask_record["height"]), int(mask_record["width"]))


def _refinement_path(mission_dir: Path, video_id: str, frame_index: int):
    return mission_dir / "path_refinements" / video_id / f"{frame_index:09d}.json"


def _load_refinements(mission_dir: Path, video_id: str, frame_index: int):
    path = _refinement_path(mission_dir, video_id, frame_index)
    if not path.is_file():
        return []
    record = json.loads(path.read_text(encoding="utf-8"))
    return record.get("items", [])


def _apply_refinements(mask: np.ndarray, mission_dir: Path, video_id: str, frame_index: int):
    refined = mask.copy()
    for item in _load_refinements(mission_dir, video_id, frame_index):
        region = _decode_rle(item["region_mask"])
        if region.shape != refined.shape:
            region = cv2.resize(region, (refined.shape[1], refined.shape[0]), interpolation=cv2.INTER_NEAREST)
        refined[region > 0] = int(item["correct_value"])
    return refined


# Die fuenf reinen Positionskanaele haengen nur von der Rastergroesse ab und
# werden pro Shape einmal berechnet. Werte sind identisch zur Direktberechnung;
# es treten nur wenige Shapes auf (MODEL_WIDTH x videoabhaengige Hoehe).
_POSITION_CACHE: dict = {}


def _position_channels(height: int, width: int):
    cached = _POSITION_CACHE.get((height, width))
    if cached is None:
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :].repeat(height, axis=0)
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None].repeat(width, axis=1)
        cached = (x, y, x**2, y**2, np.abs(x - 0.5))
        _POSITION_CACHE[(height, width)] = cached
    return cached


def _features(image: np.ndarray):
    pixels = image.astype(np.float32) / 255.0
    blue, green, red = cv2.split(pixels)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    hue = hsv[:, :, 0] * (2 * math.pi / 180.0)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.clip(np.sqrt(gradient_x**2 + gradient_y**2), 0, 1)
    local_mean = cv2.blur(gray, (9, 9))
    local_square = cv2.blur(gray**2, (9, 9))
    local_std = np.sqrt(np.maximum(0, local_square - local_mean**2))
    height, width = gray.shape
    x, y, x_squared, y_squared, x_center_distance = _position_channels(height, width)
    excess_green = np.clip(2 * green - red - blue, -1, 1)
    channels = [
        blue,
        green,
        red,
        np.sin(hue),
        np.cos(hue),
        saturation,
        value,
        lab[:, :, 0],
        lab[:, :, 1],
        lab[:, :, 2],
        excess_green,
        gradient,
        local_mean,
        local_std,
        x,
        y,
        x_squared,
        y_squared,
        x_center_distance,
        green * y,
        saturation * y,
        value * y,
    ]
    return np.stack(channels, axis=-1).reshape(-1, len(channels)).astype(np.float32)


def _read_frames(mission: MissionRecord, mission_dir: Path, records, width: int = MODEL_WIDTH):
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
                if not ok:
                    continue
                resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                mask = _polygon_mask(record, width, height)
                mask = _apply_refinements(mask, mission_dir, video_id, int(record["frame_index"]))
                if mask.any() and (~mask.astype(bool)).any():
                    decoded.append({"record": record, "image": resized, "mask": mask})
        finally:
            capture.release()
    return decoded


def _standardize(values, mean, scale):
    return (values - mean) / scale


def _random_projection(values, projection, phase):
    return np.sqrt(2.0 / projection.shape[1]) * np.cos(values @ projection + phase)


def _fit_kernel_classifier(samples, labels, random_features: int, ridge_lambda: float, seed: int):
    mean = samples.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = samples.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-5] = 1
    normalized = _standardize(samples, mean, scale)
    rng = np.random.default_rng(seed)
    projection = rng.normal(0, 1 / math.sqrt(samples.shape[1]), size=(samples.shape[1], random_features)).astype(
        np.float32
    )
    phase = rng.uniform(0, 2 * math.pi, size=random_features).astype(np.float32)
    hidden = _random_projection(normalized, projection, phase)
    design = np.column_stack([hidden, np.ones(len(hidden), np.float32)])
    targets = labels.astype(np.float32) * 2 - 1
    gram = design.T @ design
    gram.flat[:: gram.shape[0] + 1] += ridge_lambda
    weights = np.linalg.solve(gram.astype(np.float64), (design.T @ targets).astype(np.float64)).astype(np.float32)
    return {"mean": mean, "scale": scale, "projection": projection, "phase": phase, "weights": weights}


def _predict_scores(features, model, chunk_size: int = 120_000):
    # Mathematisch identisch zu standardize -> projizieren -> cos -> gewichten,
    # aber Standardisierung und Kosinus-Amplitude sind in Projektionsmatrix und
    # Gewichte gefaltet: ((x - m) / s) @ P == x @ (P / s) - (m / s) @ P. Das
    # spart die grossen Zwischenarrays der Normalisierung und Skalierung; cos
    # laeuft in-place. Training (_fit_kernel_classifier) nutzt weiter den
    # Referenzweg, Abweichungen liegen im float32-Rundungsbereich.
    inverse_scale = (1.0 / model["scale"]).astype(np.float32)
    projection = model["projection"] * inverse_scale[:, None]
    offset = (model["phase"] - (model["mean"] * inverse_scale) @ model["projection"]).astype(np.float32)
    amplitude = np.float32(math.sqrt(2.0 / model["projection"].shape[1]))
    cosine_weights = amplitude * model["weights"][:-1]
    bias = model["weights"][-1]
    output = np.empty(len(features), np.float32)
    for start in range(0, len(features), chunk_size):
        stop = min(len(features), start + chunk_size)
        hidden = features[start:stop] @ projection
        hidden += offset
        np.cos(hidden, out=hidden)
        output[start:stop] = hidden @ cosine_weights + bias
    return output


def _clean_prediction(scores, shape, threshold):
    mask = (scores.reshape(shape) >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return mask


def _grade_prediction(scores, prediction, threshold, shape):
    """Stuft die bereinigte Binärmaske in die sechs Klassen von GRADE_ONTOLOGY ab.

    Rein additiv zur bestehenden Inferenz: Innerhalb von prediction==1 entstehen
    ausschließlich die Grünstufen 1-3, außerhalb ausschließlich 0, 4 oder 5.
    """
    # 3x3-Mittelung der Abstände glättet die Stufengrenzen, damit das Overlay im
    # Video nicht flimmert (leichte Glättung analog _clean_prediction).
    margins = (scores.reshape(shape).astype(np.float32) - threshold) / max(1e-6, 1.0 - threshold)
    margins = cv2.blur(margins, (3, 3))
    inside = prediction.astype(bool)
    grades = np.zeros(shape, np.uint8)
    grades[inside] = GRADE_ONTOLOGY["marginal"]["value"]
    grades[inside & (margins >= GRADE_GOOD_MIN_MARGIN)] = GRADE_ONTOLOGY["good"]["value"]
    grades[inside & (margins >= GRADE_SAFE_MIN_MARGIN)] = GRADE_ONTOLOGY["safe"]["value"]
    grades[~inside & (margins >= GRADE_RISKY_MIN_MARGIN)] = GRADE_ONTOLOGY["risky"]["value"]
    # Rot nur für sicher-negative, zusammenhängende Flächen mit Mindestgröße in
    # unmittelbarer Nachbarschaft des Fahrbereichs; alles Übrige bleibt transparent.
    candidates = (~inside & (margins < GRADE_RISKY_MIN_MARGIN)).astype(np.uint8)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if candidates.any() and inside.any():
        kernel = np.ones((GRADE_PROBLEM_NEIGHBOURHOOD, GRADE_PROBLEM_NEIGHBOURHOOD), np.uint8)
        near_path = cv2.dilate(inside.astype(np.uint8), kernel).astype(bool)
        clip_kernel = np.ones((GRADE_PROBLEM_CLIP, GRADE_PROBLEM_CLIP), np.uint8)
        clip_band = cv2.dilate(inside.astype(np.uint8), clip_kernel).astype(bool)
        min_area = max(1, int(GRADE_PROBLEM_MIN_AREA_FRACTION * shape[0] * shape[1]))
        count, components = cv2.connectedComponents(candidates, connectivity=8)
        for component in range(1, count):
            member = components == component
            if int(member.sum()) >= min_area and bool((member & near_path).any()):
                grades[member & clip_band] = GRADE_ONTOLOGY["problem"]["value"]
    return grades


def _grading_summary(threshold: float):
    return {
        "margin": "m = (score - threshold) / max(1e-6, 1 - threshold)",
        "threshold": round(float(threshold), 6),
        "bands": {
            "safe_min_margin": GRADE_SAFE_MIN_MARGIN,
            "good_min_margin": GRADE_GOOD_MIN_MARGIN,
            "risky_min_margin": GRADE_RISKY_MIN_MARGIN,
        },
        "problem_min_area_fraction": GRADE_PROBLEM_MIN_AREA_FRACTION,
        "problem_neighbourhood_px": GRADE_PROBLEM_NEIGHBOURHOOD,
        "problem_clip_px": GRADE_PROBLEM_CLIP,
        "smoothing": "3x3 mean blur on margins, 3x3 opening on problem candidates",
        "note": "KI-Einschätzung der Befahrbarkeit, keine sicherheitsrelevante Fahrfreigabe.",
    }


def confusion_counts(truth: np.ndarray, prediction: np.ndarray):
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)
    return {
        "tp": int(np.count_nonzero(truth & prediction)),
        "tn": int(np.count_nonzero(~truth & ~prediction)),
        "fp": int(np.count_nonzero(~truth & prediction)),
        "fn": int(np.count_nonzero(truth & ~prediction)),
    }


def symmetric_metrics(counts):
    tp, tn, fp, fn = (counts[key] for key in ("tp", "tn", "fp", "fn"))
    missed = fn / max(1, tp + fn)
    invented = fp / max(1, tn + fp)
    balanced_error = 0.5 * (missed + invented)
    return {
        **counts,
        "missed_label_fraction": round(missed, 5),
        "invented_path_fraction": round(invented, 5),
        "symmetric_penalty_points": round(100 * balanced_error, 2),
        "symmetric_score": round(100 * (1 - balanced_error), 2),
        "iou": round(tp / max(1, tp + fp + fn), 5),
        "dice": round(2 * tp / max(1, 2 * tp + fp + fn), 5),
        "precision": round(tp / max(1, tp + fp), 5),
        "recall": round(tp / max(1, tp + fn), 5),
    }


def _evaluate(frames, model, threshold):
    total = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    per_frame = []
    for item in frames:
        features = _features(item["image"])
        scores = _predict_scores(features, model)
        prediction = _clean_prediction(scores, item["mask"].shape, threshold)
        counts = confusion_counts(item["mask"], prediction)
        for key in total:
            total[key] += counts[key]
        per_frame.append(
            {"item": item, "scores": scores, "prediction": prediction, "metrics": symmetric_metrics(counts)}
        )
    return symmetric_metrics(total), per_frame


def _choose_threshold(frames, model):
    score_sets = [_predict_scores(_features(item["image"]), model) for item in frames]
    merged = np.concatenate(score_sets)
    candidates = np.unique(np.quantile(merged, np.linspace(0.12, 0.88, 33)))
    best = None
    for threshold in candidates:
        total = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        for item, scores in zip(frames, score_sets, strict=True):
            counts = confusion_counts(item["mask"], _clean_prediction(scores, item["mask"].shape, float(threshold)))
            for key in total:
                total[key] += counts[key]
        metrics = symmetric_metrics(total)
        if best is None or metrics["symmetric_penalty_points"] < best[1]["symmetric_penalty_points"]:
            best = (float(threshold), metrics)
    return best


def _write_evidence(directory: Path, mission_id: str, evaluated, evidence_base_url: str | None = None):
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
    records = _confirmed_annotations(mission_dir)
    if len(records) < 10:
        raise ValueError("Mindestens 10 bestätigte Polygonframes werden für das CPU-Training benötigt")
    train_records, validation_records = _frame_split(records)
    train_frames = _read_frames(mission, mission_dir, train_records, width)
    validation_frames = _read_frames(mission, mission_dir, validation_records, width)
    if len(train_frames) < 8 or len(validation_frames) < 2:
        raise ValueError("Zu wenige dekodierbare Trainings- oder Validierungsframes")

    rng = np.random.default_rng(seed)
    samples, labels = [], []
    for item in train_frames:
        features = _features(item["image"])
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
    model = _fit_kernel_classifier(training_samples[order], training_labels[order], random_features, ridge_lambda, seed)

    threshold, threshold_metrics = _choose_threshold(validation_frames, model)
    validation_metrics, evaluated = _evaluate(validation_frames, model, threshold)
    train_metrics, _ = _evaluate(train_frames, model, threshold)

    run_id = f"path-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    derived = mission_dir / "derived"
    runs = derived / "path_model_runs"
    runs.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=runs))
    evidence_dir = staging / "evidence"
    evidence_dir.mkdir()
    try:
        np.savez_compressed(staging / "model.npz", **model, threshold=np.asarray([threshold], np.float32))
        evidence = _write_evidence(evidence_dir, mission.id, evaluated)
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


def _encode_binary_rle(mask: np.ndarray):
    # Vektorisiert (Laufgrenzen ueber np.diff statt Python-Schleife ueber jeden
    # Pixel); Ausgabeformat unveraendert: flache Liste aus (Wert, Lauflaenge).
    values = mask.reshape(-1).astype(np.uint8)
    if not len(values):
        return []
    boundaries = np.flatnonzero(values[1:] != values[:-1])
    starts = np.concatenate([[0], boundaries + 1])
    lengths = np.diff(np.append(starts, len(values)))
    encoded = np.empty(2 * len(starts), dtype=np.int64)
    encoded[0::2] = values[starts]
    encoded[1::2] = lengths
    return encoded.tolist()


def _comparison_mask(truth: np.ndarray, prediction: np.ndarray):
    comparison = np.zeros(truth.shape, dtype=np.uint8)
    comparison[(truth == 1) & (prediction == 1)] = 1
    comparison[(truth == 1) & (prediction == 0)] = 2
    comparison[(truth == 0) & (prediction == 1)] = 3
    return comparison


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
_CAPTURE_CACHE: OrderedDict = OrderedDict()
_CAPTURE_CACHE_SIZE = 4
# Bis zu dieser Distanz wird vorwaerts weitergelesen statt gesprungen. Ein
# Seek in Long-GOP-Videos springt zum letzten Keyframe zurueck und dekodiert
# von dort (gemessen ~150 ms pro Frame-Zugriff); sequenzielles Weiterlesen
# kostet nur ~8 ms pro Frame und ist frame-exakt.
_SEQUENTIAL_READ_LIMIT = 5


def _load_model_bundle(model_dir: Path):
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


def _read_original_frame(mission_dir: Path, video_id: str, frame_index: int):
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


def predict_path_frame(mission: MissionRecord, mission_dir: Path, video_id: str, frame_index: int):
    video = next((item for item in mission.videos if item.id == video_id), None)
    if not video:
        raise LookupError("Video nicht gefunden")
    model_dir = current_path_model_dir(mission_dir)
    model, threshold, result = _load_model_bundle(model_dir)
    image, fps, source_width, source_height = _read_original_frame(mission_dir, video_id, frame_index)
    width = int(result["model"]["input_width"])
    height = max(48, round(width * source_height / max(1, source_width)))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    scores = _predict_scores(_features(resized), model)
    prediction = _clean_prediction(scores, (height, width), threshold)
    grades = _grade_prediction(scores, prediction, threshold, (height, width))
    margin = np.abs(scores - threshold)
    response = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_run_id": result["run_id"],
        "video_id": video_id,
        "frame_index": frame_index,
        "timestamp_ms": round(frame_index / fps * 1000),
        "mask": {"width": width, "height": height, "rle": _encode_binary_rle(prediction)},
        "grade_mask": {"width": width, "height": height, "rle": _encode_binary_rle(grades)},
        "grade_ontology": GRADE_ONTOLOGY,
        "grading": _grading_summary(threshold),
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
                truth = _polygon_mask(annotation, width, height)
                truth = _apply_refinements(truth, mission_dir, video_id, frame_index)
                metrics = symmetric_metrics(confusion_counts(truth, prediction))
                comparison = _comparison_mask(truth, prediction)
                response["evaluation"] = {
                    "annotation_status": annotation.get("status"),
                    "metrics": metrics,
                    "comparison_mask": {"width": width, "height": height, "rle": _encode_binary_rle(comparison)},
                    "legend": {"1": "correct_path", "2": "missed_label", "3": "invented_path"},
                    "refinement_count": len(_load_refinements(mission_dir, video_id, frame_index)),
                }
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return response


def save_path_refinement(mission: MissionRecord, mission_dir: Path, video_id: str, frame_index: int, payload):
    prediction = predict_path_frame(mission, mission_dir, video_id, frame_index)
    evaluation = prediction.get("evaluation")
    if not evaluation:
        raise ValueError("Für diesen Frame ist kein Ground-Truth-Vergleich verfügbar")
    comparison = _decode_rle(evaluation["comparison_mask"])
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
        "region_mask": {"width": comparison.shape[1], "height": comparison.shape[0], "rle": _encode_binary_rle(region)},
    }
    path = _refinement_path(mission_dir, video_id, frame_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = _load_refinements(mission_dir, video_id, frame_index)
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
