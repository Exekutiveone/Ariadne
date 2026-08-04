"""Videobasierte Terrainklassifizierung: Bildausschnitt -> Terrainklasse.

Jedes Video traegt genau eine Terrainkategorie (`VideoMeta.terrain_category`).
Frames haben bewusst kein eigenes Terrainlabel, sondern erben immer die aktuell
gesetzte Kategorie ihres Videos. Wird ein Video umgelabelt, gilt die neue Klasse
damit sofort fuer alle seine Frames -- auch fuer bereits geladene oder frueher
ausgewertete. Die video_id bleibt die eindeutige Referenz.

Trainings-, Validierungs- und Testdaten werden strikt nach video_id getrennt.
Eine zufaellige Aufteilung einzelner Frames waere ein Datenleck, weil benachbarte
Frames desselben Videos nahezu identisch sind.

Jeder Trainings- und jeder Video-Vorhersagelauf schreibt ein eigenes, danach
unveraendertes Verzeichnis unter `data/global_models/terrain_model`. Vorhandene
Laeufe, Ergebnisse und Modelle werden nie ueberschrieben; nur der Zeiger
`current.json` auf den aktiven Trainingslauf wird ersetzt.
"""

import json
import math
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .path_dataset import read_original_frame
from .processor import video_path

TERRAIN_MODEL_SCHEMA_VERSION = "1.0"
TERRAIN_MODEL_ID = "ariadne-cpu-terrain-rff"
TERRAIN_MODEL_WIDTH = 160
# Raster fuer die ortsabhaengige Aggregation: Untergrund liegt im unteren
# Bilddrittel, Himmel und Kronendach im oberen. Ein 3x3-Gitter haelt diesen
# Unterschied fest, ohne die Merkmalszahl aufzublaehen.
TERRAIN_GRID = 3
TERRAIN_DESCRIPTOR_CHANNELS = 13
# Mittelwert und Streuung je Kanal und Gitterzelle.
TERRAIN_FEATURE_COUNT = TERRAIN_DESCRIPTOR_CHANNELS * TERRAIN_GRID * TERRAIN_GRID * 2
TERRAIN_RANDOM_FEATURES = 256
TERRAIN_RIDGE_LAMBDA = 1.0
TERRAIN_RANDOM_SEED = 4711
# Jeder n-te Frame eines kategorisierten Videos geht in den Datensatz. Benachbarte
# Frames sind nahezu identisch; die Schrittweite ist pro Lauf einstellbar und
# wird im Ergebnis mitgeschrieben.
DEFAULT_FRAME_STRIDE = 15
# Unterhalb dieser Konfidenz gilt ein Frame als unsicher und wird in der UI nicht
# verbindlich einer Klasse zugeordnet.
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
# Kandidaten fuer die Temperatur der Softmax-Kalibrierung. Gewaehlt wird der Wert
# mit der kleinsten negativen Log-Likelihood auf den Validierungsframes -- analog
# zur Schwellenwahl des Wegmodells ausschliesslich auf Validierungsdaten.
SOFTMAX_SCALE_CANDIDATES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def _terrain_root(missions_root: Path):
    return missions_root.parent / "global_models" / "terrain_model"


def _descriptor_channels(image: np.ndarray):
    """Dreizehn Kanaele je Pixel: Farbe, Saettigung, Helligkeit, Textur.

    Bewusst ohne Positionskanaele -- die Ortsabhaengigkeit steckt im Gitter von
    `_frame_descriptor`.
    """
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
        local_std,
    ]
    return np.stack(channels, axis=-1).astype(np.float32)


def _frame_descriptor(image: np.ndarray):
    """Ein Merkmalsvektor je Frame statt je Pixel: das Modell klassifiziert
    den ganzen Bildausschnitt, nicht einzelne Pixel."""
    channels = _descriptor_channels(image)
    height, width, count = channels.shape
    parts = []
    for rows in np.array_split(np.arange(height), TERRAIN_GRID):
        for columns in np.array_split(np.arange(width), TERRAIN_GRID):
            cell = channels[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1].reshape(-1, count)
            parts.append(cell.mean(axis=0))
            parts.append(cell.std(axis=0))
    return np.concatenate(parts).astype(np.float32)


def terrain_dataset_summary(store):
    """Welche Videos tragen welche Terrainkategorie -- Grundlage der UI und
    des Trainings."""
    categorized, uncategorized = [], []
    for mission in store.list():
        for video in mission.videos:
            entry = {
                "mission_id": mission.id,
                "mission_name": mission.name,
                "video_id": video.id,
                "original_name": video.original_name,
                "terrain_category": video.terrain_category,
            }
            (categorized if video.terrain_category else uncategorized).append(entry)
    classes: dict[str, list] = {}
    for entry in categorized:
        classes.setdefault(entry["terrain_category"], []).append(entry)
    return {
        "videos": categorized + uncategorized,
        "classes": [
            {
                "terrain_category": label,
                "videos": len(classes[label]),
                "missions": len({entry["mission_id"] for entry in classes[label]}),
            }
            for label in sorted(classes)
        ],
        "totals": {
            "categorized_videos": len(categorized),
            "uncategorized_videos": len(uncategorized),
            "classes": len(classes),
            "missions": len({entry["mission_id"] for entry in categorized}),
        },
        "label_source": "video_terrain_category_inherited_by_all_frames",
    }


def _categorized_videos(store):
    videos = []
    for mission in store.list():
        for video in mission.videos:
            if video.terrain_category:
                videos.append(
                    {
                        "mission_id": mission.id,
                        "mission_name": mission.name,
                        "video_id": video.id,
                        "original_name": video.original_name,
                        "label": video.terrain_category,
                    }
                )
    return videos


def _split_videos(videos):
    """Aufteilung ausschliesslich auf Videoebene, je Klasse getrennt.

    Ab drei Videos einer Klasse entstehen Train, Validierung und Test; bei zwei
    Videos entfaellt der Testteil, bei einem Video auch die Validierung. Kein
    Video landet je in zwei Teilmengen.
    """
    train, validation, test = [], [], []
    by_label: dict[str, list] = {}
    for video in videos:
        by_label.setdefault(video["label"], []).append(video)
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda item: (item["mission_id"], item["video_id"]))
        if len(group) >= 3:
            train.extend(group[:-2])
            validation.append(group[-2])
            test.append(group[-1])
        elif len(group) == 2:
            train.append(group[0])
            validation.append(group[1])
        else:
            train.extend(group)
    return train, validation, test


def _video_descriptors(mission_dir: Path, video_id: str, frame_stride: int):
    """Dekodiert jeden `frame_stride`-ten Frame eines Videos sequenziell.

    `grab()` ohne `retrieve()` ueberspringt die Farbkonvertierung der nicht
    benoetigten Frames; ein `set(CAP_PROP_POS_FRAMES)` je Frame waere um ein
    Vielfaches langsamer.
    """
    capture = cv2.VideoCapture(str(video_path(mission_dir, video_id)))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or TERRAIN_MODEL_WIDTH
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or TERRAIN_MODEL_WIDTH
        height = max(48, round(TERRAIN_MODEL_WIDTH * source_height / max(1, source_width)))
        index = 0
        while True:
            if not capture.grab():
                break
            if index % frame_stride == 0:
                ok, image = capture.retrieve()
                if ok and image is not None:
                    resized = cv2.resize(image, (TERRAIN_MODEL_WIDTH, height), interpolation=cv2.INTER_AREA)
                    yield {
                        "frame_index": index,
                        "timestamp_ms": round(index / fps * 1000),
                        "descriptor": _frame_descriptor(resized),
                    }
            index += 1
    finally:
        capture.release()


def _collect(store, videos, frame_stride: int):
    descriptors, labels, included = [], [], []
    for video in videos:
        mission_dir = store.root / video["mission_id"]
        try:
            frames = list(_video_descriptors(mission_dir, video["video_id"], frame_stride))
        except (FileNotFoundError, cv2.error):
            frames = []
        for frame in frames:
            descriptors.append(frame["descriptor"])
            labels.append(video["label"])
        included.append(
            {
                **{key: video[key] for key in video if key != "label"},
                "terrain_category": video["label"],
                "frames": len(frames),
            }
        )
    stacked = (
        np.vstack(descriptors).astype(np.float32) if descriptors else np.zeros((0, TERRAIN_FEATURE_COUNT), np.float32)
    )
    return stacked, labels, included


def _fit_multiclass(descriptors, indices, class_count: int):
    """Ridge-Regression ueber Random-Fourier-Merkmalen, eine Spalte je Klasse.

    Klassen werden ueber die Zeilengewichte ausbalanciert, damit lange Videos
    kurze nicht ueberstimmen.
    """
    mean = descriptors.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = descriptors.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-5] = 1
    normalized = (descriptors - mean) / scale
    rng = np.random.default_rng(TERRAIN_RANDOM_SEED)
    projection = rng.normal(
        0, 1 / math.sqrt(descriptors.shape[1]), size=(descriptors.shape[1], TERRAIN_RANDOM_FEATURES)
    ).astype(np.float32)
    phase = rng.uniform(0, 2 * math.pi, size=TERRAIN_RANDOM_FEATURES).astype(np.float32)
    hidden = math.sqrt(2.0 / TERRAIN_RANDOM_FEATURES) * np.cos(normalized @ projection + phase)
    design = np.column_stack([hidden, np.ones(len(hidden), np.float32)])
    targets = np.full((len(indices), class_count), -1.0, np.float32)
    targets[np.arange(len(indices)), indices] = 1.0
    counts = np.bincount(indices, minlength=class_count).astype(np.float64)
    counts[counts == 0] = 1
    weights_per_sample = (len(indices) / (class_count * counts))[indices]
    root = np.sqrt(weights_per_sample).astype(np.float32)[:, None]
    weighted_design = design * root
    gram = weighted_design.T @ weighted_design
    gram.flat[:: gram.shape[0] + 1] += TERRAIN_RIDGE_LAMBDA
    weights = np.linalg.solve(
        gram.astype(np.float64), (weighted_design.T @ (targets * root)).astype(np.float64)
    ).astype(np.float32)
    return {"mean": mean, "scale": scale, "projection": projection, "phase": phase, "weights": weights}


def _class_scores(descriptors, model):
    if not len(descriptors):
        return np.zeros((0, model["weights"].shape[1]), np.float32)
    normalized = (descriptors - model["mean"]) / model["scale"]
    hidden = math.sqrt(2.0 / model["projection"].shape[1]) * np.cos(normalized @ model["projection"] + model["phase"])
    design = np.column_stack([hidden, np.ones(len(hidden), np.float32)])
    return (design @ model["weights"]).astype(np.float32)


def _probabilities(scores, softmax_scale: float):
    if not len(scores):
        return np.zeros_like(scores)
    shifted = (scores - scores.max(axis=1, keepdims=True)) * softmax_scale
    exponent = np.exp(shifted)
    return (exponent / exponent.sum(axis=1, keepdims=True)).astype(np.float32)


def _choose_softmax_scale(scores, indices):
    """Temperatur der Konfidenz ausschliesslich auf Validierungsframes waehlen."""
    best_scale, best_loss = SOFTMAX_SCALE_CANDIDATES[0], math.inf
    for scale in SOFTMAX_SCALE_CANDIDATES:
        probabilities = _probabilities(scores, scale)
        truth = np.clip(probabilities[np.arange(len(indices)), indices], 1e-9, 1.0)
        loss = float(-np.log(truth).mean())
        if loss < best_loss:
            best_scale, best_loss = scale, loss
    return best_scale, round(best_loss, 6)


def _metrics(indices, predicted, confidence, classes, confidence_threshold: float):
    class_count = len(classes)
    confusion = np.zeros((class_count, class_count), np.int64)
    for truth, guess in zip(indices, predicted):
        confusion[truth, guess] += 1
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    per_class = []
    recalls = []
    for index, label in enumerate(classes):
        support = int(confusion[index].sum())
        predicted_count = int(confusion[:, index].sum())
        hits = int(confusion[index, index])
        precision = hits / predicted_count if predicted_count else 0.0
        recall = hits / support if support else 0.0
        if support:
            recalls.append(recall)
        per_class.append(
            {
                "terrain_category": label,
                "support": support,
                "precision": round(precision, 5),
                "recall": round(recall, 5),
                "f1": round(2 * precision * recall / (precision + recall), 5) if precision + recall else 0.0,
            }
        )
    confident = np.asarray(confidence) >= confidence_threshold
    confident_count = int(confident.sum())
    confident_correct = int(sum(1 for ok, truth, guess in zip(confident, indices, predicted) if ok and truth == guess))
    return {
        "frames": total,
        "accuracy": round(correct / total, 5) if total else 0.0,
        "balanced_accuracy": round(float(np.mean(recalls)), 5) if recalls else 0.0,
        "mean_confidence": round(float(np.mean(confidence)), 5) if len(confidence) else 0.0,
        "uncertain_frames": total - confident_count,
        "uncertain_fraction": round((total - confident_count) / total, 5) if total else 0.0,
        "accuracy_on_confident": round(confident_correct / confident_count, 5) if confident_count else 0.0,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def _part_summary(included, classes):
    return {
        "videos": len(included),
        "frames": sum(item["frames"] for item in included),
        "classes": sorted({item["terrain_category"] for item in included}),
        "video_ids": [item["video_id"] for item in included],
        "all_classes": classes,
    }


def _commit_run(runs: Path, run_id: str, payload: dict, model: dict | None = None):
    runs.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}-", dir=runs))
    try:
        if model is not None:
            np.savez_compressed(staging / "model.npz", **model)
        (staging / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        target = runs / run_id
        try:
            os.replace(staging, target)
        except PermissionError:
            # OneDrive kann das Umbenennen kurzzeitig sperren; kopieren statt scheitern.
            shutil.copytree(staging, target)
            shutil.rmtree(staging, ignore_errors=True)
        return target
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def train_terrain_model(
    store,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
):
    started = time.perf_counter()
    videos = _categorized_videos(store)
    classes = sorted({video["label"] for video in videos})
    if len(classes) < 2:
        raise ValueError(
            "Für ein Terrainmodell werden Videos aus mindestens zwei verschiedenen Terrainkategorien benötigt"
        )
    train_videos, validation_videos, test_videos = _split_videos(videos)
    if len({video["label"] for video in train_videos}) < 2:
        raise ValueError("Der Trainingsteil enthält nur eine Terrainkategorie — kategorisiere weitere Videos")
    if len({video["label"] for video in validation_videos}) < 2:
        raise ValueError(
            "Der Validierungsteil enthält weniger als zwei Terrainkategorien — pro Kategorie werden mindestens "
            "zwei Videos benötigt, damit Training und Validierung nach Video getrennt werden können"
        )

    notes = []
    if len({video["label"] for video in test_videos}) < 2:
        notes.append(
            "Kein Testteil gebildet: dafür werden mindestens drei Videos je Terrainkategorie benötigt. "
            "Die berichteten Werte stammen ausschließlich aus der Validierung."
        )
        test_videos = []

    train_descriptors, train_labels, train_included = _collect(store, train_videos, frame_stride)
    validation_descriptors, validation_labels, validation_included = _collect(store, validation_videos, frame_stride)
    test_descriptors, test_labels, test_included = _collect(store, test_videos, frame_stride)
    if len(train_descriptors) < 2 * len(classes) or len(validation_descriptors) < len(classes):
        raise ValueError("Zu wenige dekodierbare Frames für Training und Validierung")

    index_of = {label: index for index, label in enumerate(classes)}
    train_indices = np.asarray([index_of[label] for label in train_labels])
    validation_indices = np.asarray([index_of[label] for label in validation_labels])
    test_indices = np.asarray([index_of[label] for label in test_labels])

    model = _fit_multiclass(train_descriptors, train_indices, len(classes))
    validation_scores = _class_scores(validation_descriptors, model)
    softmax_scale, calibration_loss = _choose_softmax_scale(validation_scores, validation_indices)

    def evaluate(descriptors, indices):
        scores = _class_scores(descriptors, model)
        probabilities = _probabilities(scores, softmax_scale)
        if not len(probabilities):
            return _metrics(indices, np.zeros(0, int), np.zeros(0), classes, confidence_threshold)
        predicted = probabilities.argmax(axis=1)
        return _metrics(indices, predicted, probabilities.max(axis=1), classes, confidence_threshold)

    run_id = f"terrain-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    root = _terrain_root(store.root)
    result = {
        "schema_version": TERRAIN_MODEL_SCHEMA_VERSION,
        "scope": "video_terrain_classification",
        "kind": "training",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": TERRAIN_MODEL_ID,
            "type": "random_fourier_kernel_ridge_frame_classifier",
            "hardware": "CPU",
            "cloud_used": False,
            "input_width": TERRAIN_MODEL_WIDTH,
            "grid": TERRAIN_GRID,
            "feature_count": TERRAIN_FEATURE_COUNT,
            "random_features": TERRAIN_RANDOM_FEATURES,
            "softmax_scale": softmax_scale,
            "confidence_threshold": round(float(confidence_threshold), 5),
        },
        "classes": classes,
        "dataset": {
            "frame_stride": frame_stride,
            "label_source": "video_terrain_category_inherited_by_all_frames",
            "categorized_videos": len(videos),
            "uncategorized_videos": terrain_dataset_summary(store)["totals"]["uncategorized_videos"],
            "videos": train_included + validation_included + test_included,
            "frames": len(train_descriptors) + len(validation_descriptors) + len(test_descriptors),
        },
        "split": {
            "strategy": "grouped_by_video_id",
            "random_frame_split_used": False,
            "same_video_in_multiple_parts": False,
            "train": _part_summary(train_included, classes),
            "validation": _part_summary(validation_included, classes),
            "test": _part_summary(test_included, classes) if test_included else None,
            "notes": notes,
        },
        "calibration": {
            "softmax_scale": softmax_scale,
            "selected_on": "validation_frames_only",
            "negative_log_likelihood": calibration_loss,
        },
        "train_metrics": evaluate(train_descriptors, train_indices),
        "validation_metrics": evaluate(validation_descriptors, validation_indices),
        "test_metrics": evaluate(test_descriptors, test_indices) if test_included else None,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "limitations": [
            "Die Klasse eines Frames stammt aus dem Videolabel, nicht aus einer Einzelbildprüfung.",
            "Die Konfidenz ist ein auf der Validierung kalibrierter Score, keine belastbare Wahrscheinlichkeit.",
            "Dies ist eine KI-gestützte Einschätzung und keine sicherheitsrelevante Fahrfreigabe.",
        ],
    }
    _commit_run(root / "runs", run_id, result, {**model, "classes": np.asarray(classes)})
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "current.tmp"
    temporary.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    os.replace(temporary, root / "current.json")
    return result


def current_terrain_model_dir(missions_root: Path):
    root = _terrain_root(missions_root)
    pointer = json.loads((root / "current.json").read_text(encoding="utf-8"))
    return root / "runs" / pointer["run_id"]


def current_terrain_model(missions_root: Path):
    return json.loads((current_terrain_model_dir(missions_root) / "result.json").read_text(encoding="utf-8"))


def _load_terrain_bundle(missions_root: Path):
    directory = current_terrain_model_dir(missions_root)
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    with np.load(directory / "model.npz", allow_pickle=False) as bundle:
        model = {key: bundle[key] for key in ("mean", "scale", "projection", "phase", "weights")}
    return model, result


def _run_summaries(directory: Path):
    runs = []
    if not directory.is_dir():
        return runs
    for path in sorted(directory.glob("*/result.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        runs.append(record)
    return sorted(runs, key=lambda item: item.get("created_at", ""), reverse=True)


def list_terrain_runs(missions_root: Path):
    root = _terrain_root(missions_root)
    try:
        active = json.loads((root / "current.json").read_text(encoding="utf-8"))["run_id"]
    except (OSError, ValueError, KeyError):
        active = None
    training = [
        {
            "run_id": record["run_id"],
            "kind": record.get("kind", "training"),
            "created_at": record["created_at"],
            "active": record["run_id"] == active,
            "classes": record.get("classes", []),
            "frame_stride": record.get("dataset", {}).get("frame_stride"),
            "confidence_threshold": record.get("model", {}).get("confidence_threshold"),
            "validation_accuracy": record.get("validation_metrics", {}).get("accuracy"),
            "test_accuracy": (record.get("test_metrics") or {}).get("accuracy"),
            "runtime_seconds": record.get("runtime_seconds"),
        }
        for record in _run_summaries(root / "runs")
    ]
    predictions = [
        {
            "run_id": record["run_id"],
            "kind": record.get("kind", "prediction"),
            "created_at": record["created_at"],
            "model_run_id": record.get("model_run_id"),
            "mission_id": record.get("mission_id"),
            "video_id": record.get("video_id"),
            "original_name": record.get("original_name"),
            "frame_stride": record.get("frame_stride"),
            "confidence_threshold": record.get("confidence_threshold"),
            "predicted_frames": record.get("summary", {}).get("frames"),
            "uncertain_frames": record.get("summary", {}).get("uncertain_frames"),
            "dominant_category": record.get("summary", {}).get("dominant_category"),
        }
        for record in _run_summaries(root / "predictions")
    ]
    return {"training_runs": training, "prediction_runs": predictions, "active_run_id": active}


def _resolve_video(store, mission_id: str, video_id: str):
    mission = store.get(mission_id)
    if not mission:
        raise LookupError("Mission nicht gefunden")
    video = next((item for item in mission.videos if item.id == video_id), None)
    if not video:
        raise LookupError("Video nicht gefunden")
    return mission, video


def _frame_verdict(probabilities, classes, confidence_threshold: float):
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    certain = confidence >= confidence_threshold
    return {
        "predicted_category": classes[index] if certain else None,
        "top_category": classes[index],
        "confidence": round(confidence, 5),
        "uncertain": not certain,
        "scores": {label: round(float(value), 5) for label, value in zip(classes, probabilities)},
    }


def predict_terrain_frame(
    store, mission_id: str, video_id: str, frame_index: int, confidence_threshold: float | None = None
):
    """Einzelframe-Vorschau. Sie legt bewusst keinen Modelllauf an; persistiert
    werden die Videolaeufe aus `predict_terrain_video`."""
    _, video = _resolve_video(store, mission_id, video_id)
    model, result = _load_terrain_bundle(store.root)
    classes = result["classes"]
    threshold = result["model"]["confidence_threshold"] if confidence_threshold is None else float(confidence_threshold)
    image, fps, source_width, source_height = read_original_frame(store.root / mission_id, video_id, frame_index)
    height = max(48, round(TERRAIN_MODEL_WIDTH * source_height / max(1, source_width)))
    resized = cv2.resize(image, (TERRAIN_MODEL_WIDTH, height), interpolation=cv2.INTER_AREA)
    scores = _class_scores(_frame_descriptor(resized)[None, :], model)
    probabilities = _probabilities(scores, result["model"]["softmax_scale"])[0]
    return {
        "schema_version": TERRAIN_MODEL_SCHEMA_VERSION,
        "scope": "video_terrain_classification",
        "model_run_id": result["run_id"],
        "mission_id": mission_id,
        "video_id": video_id,
        "frame_index": frame_index,
        "timestamp_ms": round(frame_index / fps * 1000),
        "confidence_threshold": round(threshold, 5),
        "video_terrain_category": video.terrain_category,
        **_frame_verdict(probabilities, classes, threshold),
    }


def predict_terrain_video(
    store,
    mission_id: str,
    video_id: str,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    confidence_threshold: float | None = None,
):
    """Klassifiziert jeden `frame_stride`-ten Frame eines Videos und legt das
    Ergebnis als eigenen, unveraenderlichen Vorhersagelauf ab."""
    started = time.perf_counter()
    _, video = _resolve_video(store, mission_id, video_id)
    model, result = _load_terrain_bundle(store.root)
    classes = result["classes"]
    threshold = result["model"]["confidence_threshold"] if confidence_threshold is None else float(confidence_threshold)

    frames, descriptors = [], []
    for frame in _video_descriptors(store.root / mission_id, video_id, frame_stride):
        frames.append({"frame_index": frame["frame_index"], "timestamp_ms": frame["timestamp_ms"]})
        descriptors.append(frame["descriptor"])
    if not frames:
        raise ValueError("Das Video konnte nicht dekodiert werden")

    probabilities = _probabilities(_class_scores(np.vstack(descriptors), model), result["model"]["softmax_scale"])
    predicted = [{**frame, **_frame_verdict(row, classes, threshold)} for frame, row in zip(frames, probabilities)]
    counts: dict[str, int] = {}
    for item in predicted:
        counts[item["top_category"]] = counts.get(item["top_category"], 0) + 1
    uncertain = sum(1 for item in predicted if item["uncertain"])

    run_id = f"terrain-predict-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    payload = {
        "schema_version": TERRAIN_MODEL_SCHEMA_VERSION,
        "scope": "video_terrain_classification",
        "kind": "prediction",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_run_id": result["run_id"],
        "mission_id": mission_id,
        "video_id": video_id,
        "original_name": video.original_name,
        "video_terrain_category": video.terrain_category,
        "frame_stride": frame_stride,
        "confidence_threshold": round(threshold, 5),
        "classes": classes,
        "summary": {
            "frames": len(predicted),
            "uncertain_frames": uncertain,
            "uncertain_fraction": round(uncertain / len(predicted), 5),
            "mean_confidence": round(float(np.mean([item["confidence"] for item in predicted])), 5),
            "dominant_category": max(counts, key=lambda label: counts[label]),
            "counts": counts,
            "matches_video_category": (
                None
                if not video.terrain_category
                else round(
                    sum(1 for item in predicted if item["top_category"] == video.terrain_category) / len(predicted), 5
                )
            ),
        },
        "frames": predicted,
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "limitations": [
            "Die Konfidenz ist ein auf der Validierung kalibrierter Score, keine belastbare Wahrscheinlichkeit.",
            "Frames unterhalb des Schwellenwerts bleiben ohne verbindliche Klassenzuweisung.",
            "Dies ist eine KI-gestützte Einschätzung und keine sicherheitsrelevante Fahrfreigabe.",
        ],
    }
    _commit_run(_terrain_root(store.root) / "predictions", run_id, payload)
    return payload


def terrain_prediction_run(missions_root: Path, run_id: str):
    path = _terrain_root(missions_root) / "predictions" / run_id / "result.json"
    if not path.is_file():
        raise LookupError("Vorhersagelauf nicht gefunden")
    return json.loads(path.read_text(encoding="utf-8"))
