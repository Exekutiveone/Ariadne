"""Masken des Wegmodells: Darstellung, Korrektur und Vergleich.

Kennt weder Modell noch Video — nur Raster. Das macht die Maskenlogik ohne
trainiertes Modell testbar und gibt den frueher privaten Funktionen einen
Vertrag: bis 04.08.2026 importierten drei Module bis zu 20 Namen mit
fuehrendem Unterstrich aus `path_model`, also eine Schnittstelle ohne Zusage.
"""

import json
from pathlib import Path

import cv2
import numpy as np


def polygon_mask(record, width: int, height: int):
    mask = np.zeros((height, width), np.uint8)
    for polygon in record.get("polygons", []):
        points = np.asarray(
            [[round(float(x) * (width - 1)), round(float(y) * (height - 1))] for x, y in polygon.get("points", [])],
            np.int32,
        )
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
    return mask


def decode_rle(mask_record):
    values = np.empty(int(mask_record["width"]) * int(mask_record["height"]), dtype=np.uint8)
    cursor = 0
    for value, length in zip(mask_record["rle"][::2], mask_record["rle"][1::2]):
        values[cursor : cursor + int(length)] = int(value)
        cursor += int(length)
    if cursor != len(values):
        raise ValueError("Ungültige Refinement-Maske")
    return values.reshape(int(mask_record["height"]), int(mask_record["width"]))


def refinement_path(mission_dir: Path, video_id: str, frame_index: int):
    return mission_dir / "path_refinements" / video_id / f"{frame_index:09d}.json"


def load_refinements(mission_dir: Path, video_id: str, frame_index: int):
    path = refinement_path(mission_dir, video_id, frame_index)
    if not path.is_file():
        return []
    record = json.loads(path.read_text(encoding="utf-8"))
    return record.get("items", [])


def apply_refinements(mask: np.ndarray, mission_dir: Path, video_id: str, frame_index: int):
    refined = mask.copy()
    for item in load_refinements(mission_dir, video_id, frame_index):
        region = decode_rle(item["region_mask"])
        if region.shape != refined.shape:
            region = cv2.resize(region, (refined.shape[1], refined.shape[0]), interpolation=cv2.INTER_NEAREST)
        refined[region > 0] = int(item["correct_value"])
    return refined


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


def encode_binary_rle(mask: np.ndarray):
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


def comparison_mask(truth: np.ndarray, prediction: np.ndarray):
    comparison = np.zeros(truth.shape, dtype=np.uint8)
    comparison[(truth == 1) & (prediction == 1)] = 1
    comparison[(truth == 1) & (prediction == 0)] = 2
    comparison[(truth == 0) & (prediction == 1)] = 3
    return comparison
