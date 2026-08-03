import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .models import MissionRecord
from .processor import video_path
from .terrain import (
    TRAVERSABILITY_ONTOLOGY,
    TerrainAnalyzer,
    VehicleConfiguration,
    render_evidence,
)

MODEL_ID = "opencv-forest-instance-cv"
MODEL_VERSION = "3.0.0"
TERRAIN_MODEL_ID = "opencv-ground-traversability-cv"
TERRAIN_MODEL_VERSION = "1.0.0"
INPUT_WIDTH = 640
ANALYSIS_HZ = 4
ONTOLOGY = {
    "tree": {"label": "Einzelbaum", "color": "#55c878", "countable": True, "default_enabled": True},
    "shrub": {"label": "Einzelstrauch", "color": "#d5e75c", "countable": True, "default_enabled": True},
    "vegetation_cluster": {
        "label": "Vegetationsgruppe (nicht zählbar)",
        "color": "#368e59",
        "countable": False,
        "default_enabled": False,
    },
    "unknown_obstacle": {
        "label": "Mögliches Hindernis",
        "color": "#e16c55",
        "countable": False,
        "default_enabled": True,
    },
}


def _environment_float(name, default, minimum, maximum):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return float(default), False
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} muss eine Zahl sein") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} muss zwischen {minimum} und {maximum} liegen")
    return value, True


def terrain_configuration_from_environment():
    width_m, width_configured = _environment_float("ARIADNE_ARGUS_WIDTH_M", 0.35, 0.15, 3.0)
    margin_m, margin_configured = _environment_float("ARIADNE_ARGUS_SAFETY_MARGIN_M", 0.20, 0.0, 1.5)
    near_field_width_m, _ = _environment_float("ARIADNE_TERRAIN_NEAR_FIELD_WIDTH_M", 3.2, 0.8, 12.0)
    calibration = os.getenv("ARIADNE_TERRAIN_METRIC_CALIBRATION", "perspective_estimate").strip().lower()
    if calibration not in {"calibrated", "perspective_estimate"}:
        raise ValueError("ARIADNE_TERRAIN_METRIC_CALIBRATION muss 'calibrated' oder 'perspective_estimate' sein")
    configured = width_configured or margin_configured
    vehicle = VehicleConfiguration(
        width_m=width_m,
        safety_margin_per_side_m=margin_m,
        source="environment" if configured else "documented_default_assumption",
    )
    return vehicle, near_field_width_m, calibration


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / max(1e-9, union)


def _center_distance(a, b):
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return float(np.hypot(ax - bx, ay - by))


def _resize(image, width=INPUT_WIDTH):
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _normalise_box(box, width, height):
    x1, y1, x2, y2 = box
    return [round(x1 / width, 5), round(y1 / height, 5), round(x2 / width, 5), round(y2 / height, 5)]


def _mask_geometry(mask):
    height, width = mask.shape
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    epsilon = max(1.0, 0.012 * cv2.arcLength(contour, True))
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(polygon) < 3:
        polygon = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    return {
        "bbox": _normalise_box([x, y, x + w, y + h], width, height),
        "polygon": [[round(float(px / width), 5), round(float(py / height), 5)] for px, py in polygon],
        "area_ratio": round(float(cv2.countNonZero(mask) / (width * height)), 5),
    }


def _quality_flags(box):
    flags = []
    if box[0] <= 0.006 or box[1] <= 0.006 or box[2] >= 0.994 or box[3] >= 0.994:
        flags.append("border_truncated")
    return flags


def _estimate_motion(previous, current):
    """Estimate previous->current camera motion for track prediction."""
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    if previous is None or current is None or previous.shape != current.shape:
        return identity, 0
    points = cv2.goodFeaturesToTrack(previous, maxCorners=350, qualityLevel=0.012, minDistance=7, blockSize=7)
    if points is None or len(points) < 12:
        return identity, 0
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, winSize=(25, 25), maxLevel=3)
    if moved is None or status is None:
        return identity, 0
    valid = status.reshape(-1).astype(bool)
    source, target = points.reshape(-1, 2)[valid], moved.reshape(-1, 2)[valid]
    if len(source) < 10:
        return identity, 0
    matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None:
        return identity, 0
    scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    if (
        not 0.86 <= scale <= 1.16
        or abs(matrix[0, 2]) > current.shape[1] * 0.28
        or abs(matrix[1, 2]) > current.shape[0] * 0.28
    ):
        return identity, 0
    return matrix.astype(np.float32), int(np.count_nonzero(inliers)) if inliers is not None else 0


class MultiFrameTracker:
    def __init__(self, video_id, max_age=2, confirmation_hits=3):
        self.video_id = video_id
        self.max_age = max_age
        self.confirmation_hits = confirmation_hits
        self.next_number = {"tree": 1, "shrub": 1, "vegetation_cluster": 1, "unknown_obstacle": 1}
        self.tracks = {}

    def _prefix(self, class_id):
        return {"tree": "T", "shrub": "S", "vegetation_cluster": "V", "unknown_obstacle": "O"}[class_id]

    def _predict(self, motion, shape):
        height, width = shape
        for track in self.tracks.values():
            box = track["bbox"]
            corners = np.array(
                [[[box[0] * width, box[1] * height], [box[2] * width, box[3] * height]]], dtype=np.float32
            )
            warped = cv2.transform(corners, motion)[0]
            x1, y1 = np.minimum(warped[0], warped[1])
            x2, y2 = np.maximum(warped[0], warped[1])
            track["bbox"] = [
                float(np.clip(x1 / width, 0, 1)),
                float(np.clip(y1 / height, 0, 1)),
                float(np.clip(x2 / width, 0, 1)),
                float(np.clip(y2 / height, 0, 1)),
            ]

    def update(self, detections, frame_number, timestamp_ms, motion, shape):
        self._predict(motion, shape)
        candidate_pairs = []
        for detection_index, detection in enumerate(detections):
            for track_id, track in self.tracks.items():
                age = frame_number - track["last_frame"]
                if age > self.max_age or track["class_id"] != detection["class_id"]:
                    continue
                overlap, distance = (
                    _iou(detection["bbox"], track["bbox"]),
                    _center_distance(detection["bbox"], track["bbox"]),
                )
                if overlap < 0.015 and distance > 0.16:
                    continue
                score = 0.65 * overlap + 0.35 * max(0.0, 1 - distance / 0.22) - 0.025 * max(0, age - 1)
                if score >= 0.20:
                    candidate_pairs.append((score, detection_index, track_id, age))
        assigned_detections, assigned_tracks = set(), set()
        for _, detection_index, track_id, age in sorted(candidate_pairs, reverse=True):
            if detection_index in assigned_detections or track_id in assigned_tracks:
                continue
            self._attach(
                detections[detection_index],
                track_id,
                frame_number,
                timestamp_ms,
                "continued" if age == 1 else "reacquired",
            )
            assigned_detections.add(detection_index)
            assigned_tracks.add(track_id)
        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detections:
                continue
            class_id = detection["class_id"]
            number = self.next_number[class_id]
            self.next_number[class_id] += 1
            track_id = f"{self.video_id[:8]}-{self._prefix(class_id)}-{number:03d}"
            self.tracks[track_id] = {
                "track_id": track_id,
                "instance_label": f"{self._prefix(class_id)}-{number:02d}",
                "class_id": class_id,
                "first_timestamp_ms": timestamp_ms,
                "last_timestamp_ms": timestamp_ms,
                "last_frame": frame_number,
                "observation_count": 0,
                "max_confidence": 0.0,
                "representative_detection_id": detection["detection_id"],
                "bbox": detection["bbox"],
            }
            self._attach(detection, track_id, frame_number, timestamp_ms, "new")

    def _attach(self, detection, track_id, frame_number, timestamp_ms, status):
        track = self.tracks[track_id]
        track["last_frame"] = frame_number
        track["last_timestamp_ms"] = timestamp_ms
        track["observation_count"] += 1
        track["bbox"] = detection["bbox"]
        if detection["confidence"] >= track["max_confidence"]:
            track["max_confidence"] = detection["confidence"]
            track["representative_detection_id"] = detection["detection_id"]
        detection.update(
            {
                "track_id": track_id,
                "instance_id": track_id,
                "instance_label": track["instance_label"],
                "tracking_status": status,
                "observation_id": f"{detection['class_id']}-{track_id}",
            }
        )

    def finalise(self, frames):
        summaries = []
        status_by_track = {}
        for track in self.tracks.values():
            individual = track["class_id"] in {"tree", "shrub"}
            confirmed = individual and track["observation_count"] >= self.confirmation_hits
            status = (
                "confirmed"
                if confirmed
                else "tentative"
                if individual
                else "cluster"
                if track["class_id"] == "vegetation_cluster"
                else "uncertain"
            )
            status_by_track[track["track_id"]] = (status, confirmed)
            summaries.append(
                {key: value for key, value in track.items() if key not in {"bbox", "last_frame"}}
                | {
                    "instance_status": status,
                    "countable": confirmed,
                    "max_confidence": round(track["max_confidence"], 3),
                }
            )
        for frame in frames:
            for detection in frame["detections"]:
                status, countable = status_by_track[detection["track_id"]]
                detection["instance_status"] = status
                detection["countable"] = countable
        return sorted(summaries, key=lambda item: item["track_id"])


class ForestInstanceAdapter:
    def __init__(self, min_area=0.004, input_width=INPUT_WIDTH):
        self.min_area = min_area
        self.input_width = input_width

    def prepare(self, image):
        small = _resize(image, self.input_width)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        lightness, green_red, blue_yellow = cv2.split(lab)
        lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
        balanced = cv2.cvtColor(cv2.merge([lightness, green_red, blue_yellow]), cv2.COLOR_LAB2BGR)
        return small, balanced

    def _vegetation_mask(self, balanced):
        hsv = cv2.cvtColor(balanced, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
        blue, green, red = cv2.split(balanced.astype(np.int16))
        excess_green = 2 * green - red - blue
        hsv_green = cv2.inRange(hsv, (19, 24, 18), (105, 255, 255))
        spectral_green = np.where((excess_green > 10) & (lab[:, :, 1] < 133), 255, 0).astype(np.uint8)
        mask = cv2.bitwise_or(hsv_green, spectral_green)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask

    def _tree_instances(self, balanced, vegetation):
        height, width = vegetation.shape
        hsv = cv2.cvtColor(balanced, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY)
        trunk_colour = cv2.bitwise_or(
            cv2.inRange(hsv, (2, 14, 28), (36, 180, 220)), cv2.inRange(hsv, (0, 0, 38), (179, 62, 215))
        )
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 42, 125)
        raw_lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(18, round(height * 0.075)),
            minLineLength=round(height * 0.12),
            maxLineGap=round(height * 0.045),
        )
        vertical_lines = []
        if raw_lines is not None:
            for x1, y1, x2, y2 in np.asarray(raw_lines).reshape(-1, 4):
                dx, dy = abs(int(x2) - int(x1)), abs(int(y2) - int(y1))
                if dy < height * 0.12 or dx > max(5, dy * 0.30) or max(y1, y2) < height * 0.38:
                    continue
                vertical_lines.append((int(x1), min(int(y1), int(y2)), int(x2), max(int(y1), int(y2))))
        vertical = trunk_colour.copy()
        vertical[: round(height * 0.10)] = 0
        vertical[round(height * 0.92) :] = 0
        vertical = cv2.morphologyEx(
            vertical, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(17, round(height * 0.10))))
        )
        vertical = cv2.morphologyEx(
            vertical, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, max(9, round(height * 0.035))))
        )
        components = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        results = []
        road = self._road_exclusion((height, width))
        for contour in components:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            fill = cv2.contourArea(contour) / max(1, box_width * box_height)
            if box_height < height * 0.12 or box_width > width * 0.10 or box_height / max(1, box_width) < 2.2:
                continue
            if (
                y > height * 0.58
                or y + box_height < height * 0.42
                or fill < 0.20
                or x < width * 0.012
                or x + box_width > width * 0.988
            ):
                continue
            centre_x, centre_y = x + box_width / 2, y + box_height / 2
            if road[min(height - 1, int(centre_y)), min(width - 1, int(centre_x))] and centre_y > height * 0.66:
                continue
            matching_lines = 0
            for line_x1, line_y1, line_x2, line_y2 in vertical_lines:
                line_x = (line_x1 + line_x2) / 2
                overlap = min(y + box_height, line_y2) - max(y, line_y1)
                if abs(line_x - centre_x) <= max(width * 0.025, box_width) and overlap >= height * 0.035:
                    matching_lines += 1
            if matching_lines == 0 and not (box_height >= height * 0.23 and fill >= 0.48):
                continue
            context = np.zeros_like(gray)
            cv2.rectangle(
                context,
                (max(0, x - round(width * 0.035)), max(0, y - round(height * 0.04))),
                (
                    min(width - 1, x + box_width + round(width * 0.035)),
                    min(height - 1, y + box_height + round(height * 0.04)),
                ),
                255,
                -1,
            )
            context_ratio = cv2.countNonZero(cv2.bitwise_and(vegetation, context)) / max(1, cv2.countNonZero(context))
            if context_ratio < 0.10:
                continue
            visible = np.zeros_like(gray)
            cv2.drawContours(visible, [contour], -1, 255, -1)
            geometry = _mask_geometry(visible)
            if not geometry:
                continue
            length_score = min(1.0, box_height / (height * 0.34))
            confidence = round(
                float(
                    np.clip(
                        0.34
                        + 0.22 * length_score
                        + 0.14 * min(1, fill)
                        + 0.08 * min(1, matching_lines / 2)
                        + 0.08 * min(1, context_ratio * 2),
                        0.38,
                        0.83,
                    )
                ),
                3,
            )
            box = geometry["bbox"]
            extent_width = max(0.06, min(0.22, (box[3] - box[1]) * 0.45))
            centre = (box[0] + box[2]) / 2
            results.append(
                geometry
                | {
                    "class_id": "tree",
                    "class_label": ONTOLOGY["tree"]["label"],
                    "confidence": confidence,
                    "geometry_basis": "visible_trunk",
                    "estimated_extent": [
                        round(max(0, centre - extent_width), 5),
                        round(max(0, box[1] - 0.18), 5),
                        round(min(1, centre + extent_width), 5),
                        box[3],
                    ],
                    "scores": {
                        "objectness": confidence,
                        "classification": round(min(0.82, 0.55 + matching_lines * 0.06), 3),
                        "boundary": round(min(0.9, 0.48 + fill * 0.4), 3),
                        "temporal": 0.0,
                        "combined": confidence,
                    },
                    "quality_flags": _quality_flags(box),
                    "observed": True,
                }
            )
        return self._suppress_near_duplicates(results)

    def _suppress_near_duplicates(self, detections):
        kept = []
        for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
            centre = (detection["bbox"][0] + detection["bbox"][2]) / 2
            if any(
                abs(centre - (item["bbox"][0] + item["bbox"][2]) / 2) < 0.026
                and _iou(detection["bbox"], item["bbox"]) > 0.04
                for item in kept
            ):
                continue
            kept.append(detection)
        return kept

    def _road_exclusion(self, shape):
        height, width = shape
        mask = np.zeros((height, width), np.uint8)
        bottom_left, bottom_right = (0.07, 0.93) if height > width else (0.20, 0.80)
        points = np.array(
            [
                [width * 0.44, height * 0.46],
                [width * 0.56, height * 0.46],
                [width * bottom_right, height],
                [width * bottom_left, height],
            ],
            np.int32,
        )
        cv2.fillConvexPoly(mask, points, 255)
        return mask

    def _split_region(self, component, limit=8):
        height, width = component.shape
        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        if float(distance.max()) < 3:
            return [component]
        kernel_size = max(21, round(width * 0.085) | 1)
        dilated = cv2.dilate(distance, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
        maxima = np.where(
            (distance >= dilated - 1e-4) & (distance >= max(3.0, float(distance.max()) * 0.22)), 255, 0
        ).astype(np.uint8)
        count, labels, _, _ = cv2.connectedComponentsWithStats(maxima)
        seeds = []
        for label in range(1, count):
            ys, xs = np.where(labels == label)
            if len(xs):
                index = int(np.argmax(distance[ys, xs]))
                seeds.append((int(xs[index]), int(ys[index]), float(distance[ys[index], xs[index]])))
        seeds = sorted(seeds, key=lambda item: item[2], reverse=True)[:limit]
        if len(seeds) <= 1:
            return [component]
        ys, xs = np.where(component > 0)
        costs = np.stack(
            [((xs - sx) / max(1, width)) ** 2 + 1.35 * ((ys - sy) / max(1, height)) ** 2 for sx, sy, _ in seeds], axis=1
        )
        assignments = np.argmin(costs, axis=1)
        instances = []
        for index in range(len(seeds)):
            instance = np.zeros_like(component)
            selected = assignments == index
            instance[ys[selected], xs[selected]] = 255
            instance = cv2.morphologyEx(instance, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            instances.append(instance)
        return instances

    def _shrub_instances(self, vegetation):
        height, width = vegetation.shape
        shrub_mask = vegetation.copy()
        shrub_mask[: round(height * 0.38)] = 0
        shrub_mask[self._road_exclusion((height, width)) > 0] = 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(shrub_mask)
        frame_area = width * height
        results, clusters = [], []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < frame_area * self.min_area:
                continue
            component = np.where(labels == label, 255, 0).astype(np.uint8)
            split_limit = min(8, max(1, round(area / (frame_area * 0.022))))
            instances = self._split_region(component, split_limit) if area > frame_area * 0.018 else [component]
            for instance in instances:
                geometry = _mask_geometry(instance)
                if not geometry or geometry["area_ratio"] < self.min_area * 0.72:
                    continue
                box = geometry["bbox"]
                box_width, box_height = box[2] - box[0], box[3] - box[1]
                centre_y = (box[1] + box[3]) / 2
                compactness = min(1.0, geometry["area_ratio"] / max(1e-6, box_width * box_height))
                is_separable = (
                    box_width <= 0.34
                    and box_height <= 0.40
                    and centre_y >= 0.58
                    and box[3] >= 0.69
                    and geometry["area_ratio"] <= 0.095
                )
                class_id = "shrub" if is_separable else "vegetation_cluster"
                confidence = round(
                    float(np.clip(0.34 + 0.24 * compactness + (0.10 if len(instances) > 1 else 0), 0.34, 0.76)), 3
                )
                target = results if class_id == "shrub" else clusters
                target.append(
                    geometry
                    | {
                        "class_id": class_id,
                        "class_label": ONTOLOGY[class_id]["label"],
                        "confidence": confidence,
                        "geometry_basis": "watershed_region" if len(instances) > 1 else "connected_region",
                        "scores": {
                            "objectness": confidence,
                            "classification": 0.64 if class_id == "shrub" else 0.42,
                            "boundary": round(0.45 + 0.35 * compactness, 3),
                            "temporal": 0.0,
                            "combined": confidence,
                        },
                        "quality_flags": _quality_flags(box),
                        "observed": True,
                    }
                )
        return (
            sorted(results, key=lambda item: item["confidence"], reverse=True)[:20]
            + sorted(clusters, key=lambda item: item["area_ratio"], reverse=True)[:3]
        )

    def infer_prepared(self, balanced, vegetation):
        return self._tree_instances(balanced, vegetation) + self._shrub_instances(vegetation)

    def infer(self, image):
        _, balanced = self.prepare(image)
        vegetation = self._vegetation_mask(balanced)
        return self.infer_prepared(balanced, vegetation)


# Backward-compatible import name used by existing callers and tests.
ForestCvAdapter = ForestInstanceAdapter


def _process_video(
    path,
    video,
    adapter,
    terrain_analyzer,
    evidence_dir,
    mission_id,
    run_id,
    interval=1 / ANALYSIS_HZ,
):
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise ValueError(f"Video kann nicht dekodiert werden: {video.original_name}")
    duration = frame_count / fps
    step = max(1, round(fps * interval))
    tracker = MultiFrameTracker(video.id)
    frames, previous_gray, decoded_index, analyzed_index = [], None, 0, 0
    motion_inliers = []
    terrain_mask_hashes, terrain_source_hashes = [], []
    evidence_candidates = {}
    started = time.perf_counter()

    def consider_evidence(key, score, reason, source_image, terrain, class_map):
        current = evidence_candidates.get(key)
        if current is None or score > current["score"]:
            evidence_candidates[key] = {
                "score": float(score),
                "reason": reason,
                "image": source_image.copy(),
                "terrain": terrain,
                "class_map": class_map.copy(),
            }

    while True:
        ok, image = cap.read()
        if not ok:
            break
        if decoded_index % step:
            decoded_index += 1
            continue
        timestamp_ms = round(decoded_index / fps * 1000)
        tracking_image, balanced = adapter.prepare(image)
        current_gray = cv2.cvtColor(tracking_image, cv2.COLOR_BGR2GRAY)
        motion, inliers = _estimate_motion(previous_gray, current_gray)
        motion_inliers.append(inliers)
        vegetation = adapter._vegetation_mask(balanced)
        detections = adapter.infer_prepared(balanced, vegetation)
        for index, detection in enumerate(detections):
            detection.update(
                {
                    "detection_id": f"{video.id[:8]}-{timestamp_ms:07d}-{index:02d}",
                    "video_id": video.id,
                    "video_name": video.original_name,
                    "frame_index": decoded_index,
                    "timestamp_ms": timestamp_ms,
                    "model_id": MODEL_ID,
                    "model_version": MODEL_VERSION,
                }
            )
        tracker.update(detections, analyzed_index, timestamp_ms, motion, current_gray.shape)
        terrain, terrain_maps = terrain_analyzer.analyze(
            balanced,
            vegetation,
            detections,
            motion,
            inliers,
            timestamp_ms,
            source_image=tracking_image,
        )
        terrain["source_video_id"] = video.id
        terrain["source_frame_index"] = decoded_index
        class_map = terrain_maps["class_map"]
        terrain_mask_hashes.append(hashlib.sha256(class_map.tobytes()).hexdigest())
        terrain_source_hashes.append(terrain["source_frame_hash"])
        overall_class = terrain["traversability"]["overall_class"]
        overall_confidence = terrain["traversability"]["overall_confidence"]
        coverage = terrain["traversability"]["class_coverage"]
        consider_evidence(
            f"overall-{overall_class}",
            overall_confidence + coverage.get(overall_class, 0),
            f"representative_{overall_class}",
            tracking_image,
            terrain,
            class_map,
        )
        if terrain["corridor"]["stable_frames"] >= 3:
            consider_evidence(
                "stable-corridor",
                terrain["corridor"]["confidence"] + min(0.3, terrain["corridor"]["stable_frames"] / 20),
                "temporally_stable_corridor",
                tracking_image,
                terrain,
                class_map,
            )
        if coverage.get("not_traversable", 0) >= 0.015:
            consider_evidence(
                "visible-obstacle",
                coverage["not_traversable"] + overall_confidence,
                "visible_obstacle_evidence",
                tracking_image,
                terrain,
                class_map,
            )
        if coverage.get("unknown", 0) >= 0.35:
            consider_evidence(
                "low-visibility",
                coverage["unknown"] + (1 - terrain["factors"]["visibility_score"]),
                "substantial_non_assessable_area",
                tracking_image,
                terrain,
                class_map,
            )
        frames.append(
            {
                "video_id": video.id,
                "video_name": video.original_name,
                "frame_index": decoded_index,
                "timestamp_ms": timestamp_ms,
                "quality": {
                    "sharpness": round(float(cv2.Laplacian(current_gray, cv2.CV_64F).var()), 2),
                    "motion_inliers": inliers,
                },
                "detections": detections,
                "terrain": terrain,
            }
        )
        previous_gray = current_gray
        decoded_index += 1
        analyzed_index += 1
    cap.release()

    evidence_dir.mkdir(parents=True, exist_ok=True)
    deduplicated = {}
    for candidate in sorted(evidence_candidates.values(), key=lambda item: item["score"], reverse=True):
        timestamp_ms = candidate["terrain"]["source_frame_timestamp_ms"]
        if timestamp_ms in deduplicated:
            deduplicated[timestamp_ms]["reasons"].append(candidate["reason"])
            continue
        deduplicated[timestamp_ms] = candidate | {"reasons": [candidate["reason"]]}
    selected_evidence = sorted(deduplicated.values(), key=lambda item: item["score"], reverse=True)[:6]
    evidence_base_url = f"/api/v1/missions/{mission_id}/segmentation/evidence"
    for index, candidate in enumerate(
        sorted(selected_evidence, key=lambda item: item["terrain"]["source_frame_timestamp_ms"])
    ):
        timestamp_ms = candidate["terrain"]["source_frame_timestamp_ms"]
        stem = f"{run_id[:8]}-{video.id[:8]}-{timestamp_ms:07d}-{index:02d}"
        original_name, overlay_name = f"{stem}-original.jpg", f"{stem}-overlay.jpg"
        overlay = render_evidence(candidate["image"], candidate["terrain"], candidate["class_map"])
        original_written = cv2.imwrite(
            str(evidence_dir / original_name), candidate["image"], [cv2.IMWRITE_JPEG_QUALITY, 91]
        )
        overlay_written = cv2.imwrite(str(evidence_dir / overlay_name), overlay, [cv2.IMWRITE_JPEG_QUALITY, 91])
        if not original_written or not overlay_written:
            raise OSError(f"Terrain-Evidenz konnte nicht geschrieben werden: {stem}")
        candidate["terrain"]["evidence"] = {
            "representative": True,
            "reasons": sorted(set(candidate["reasons"])),
            "image_url": f"{evidence_base_url}/{original_name}",
            "overlay_url": f"{evidence_base_url}/{overlay_name}",
        }

    tracks = tracker.finalise(frames)
    confirmed_trees = sum(1 for item in tracks if item["class_id"] == "tree" and item["countable"])
    confirmed_shrubs = sum(1 for item in tracks if item["class_id"] == "shrub" and item["countable"])
    latest_individuals = sum(
        1
        for item in (frames[-1]["detections"] if frames else [])
        if item["countable"] and item["class_id"] in {"tree", "shrub"}
    )
    track_lengths = [item["observation_count"] for item in tracks]
    total_detections = sum(len(frame["detections"]) for frame in frames)
    terrain_frames = [frame["terrain"] for frame in frames]
    corridor_statuses = [item["corridor"]["status"] for item in terrain_frames]
    overall_classes = [item["traversability"]["overall_class"] for item in terrain_frames]
    mask_transitions = sum(before != after for before, after in zip(terrain_mask_hashes, terrain_mask_hashes[1:]))
    return {
        "video_id": video.id,
        "video_name": video.original_name,
        "duration_seconds": round(duration, 3),
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "analysis_interval_ms": round(interval * 1000),
        "frames": frames,
        "tracks": tracks,
        "counts": {
            "visible_individuals_latest_frame": latest_individuals,
            "confirmed_unique_per_video": {"tree": confirmed_trees, "shrub": confirmed_shrubs},
        },
        "metrics": {
            "analyzed_frames": len(frames),
            "raw_detections": total_detections,
            "tracks": len(tracks),
            "confirmed_tree_instances": confirmed_trees,
            "confirmed_shrub_instances": confirmed_shrubs,
            "average_track_length_frames": round(float(np.mean(track_lengths)) if track_lengths else 0, 2),
            "short_track_fraction": round(
                sum(1 for value in track_lengths if value <= 2) / max(1, len(track_lengths)), 3
            ),
            "empty_frame_fraction": round(
                sum(1 for frame in frames if not frame["detections"]) / max(1, len(frames)), 3
            ),
            "median_motion_inliers": round(float(np.median(motion_inliers)) if motion_inliers else 0, 1),
            "terrain_frames": len(terrain_frames),
            "terrain_unique_source_hashes": len(set(terrain_source_hashes)),
            "terrain_source_hash_unique_fraction": round(
                len(set(terrain_source_hashes)) / max(1, len(terrain_source_hashes)), 4
            ),
            "terrain_unique_masks": len(set(terrain_mask_hashes)),
            "terrain_mask_unique_fraction": round(len(set(terrain_mask_hashes)) / max(1, len(terrain_mask_hashes)), 4),
            "terrain_mask_transition_fraction": round(mask_transitions / max(1, len(terrain_mask_hashes) - 1), 4),
            "terrain_masks_vary": len(set(terrain_mask_hashes)) > 1,
            "corridor_available_fraction": round(
                corridor_statuses.count("available") / max(1, len(corridor_statuses)), 4
            ),
            "corridor_uncertain_fraction": round(
                corridor_statuses.count("uncertain") / max(1, len(corridor_statuses)), 4
            ),
            "median_corridor_stability_px": round(
                float(np.median([item["corridor"]["stability_px"] for item in terrain_frames]))
                if terrain_frames
                else 0,
                2,
            ),
            "terrain_overall_class_frames": {
                class_id: overall_classes.count(class_id) for class_id in TRAVERSABILITY_ONTOLOGY
            },
            "representative_evidence_frames": len(selected_evidence),
            "inference_seconds": round(time.perf_counter() - started, 2),
        },
    }


def process_segmentation(mission: MissionRecord, mission_dir: Path, min_area=0.004):
    started = time.perf_counter()
    run_id = str(uuid4())
    derived = mission_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".segmentation-", dir=derived))
    run_dir = derived / "segmentation_runs" / run_id
    try:
        vehicle, near_field_width_m, metric_calibration = terrain_configuration_from_environment()
        adapter = ForestInstanceAdapter(min_area=min_area)
        videos = []
        for video in mission.videos:
            terrain_analyzer = TerrainAnalyzer(
                vehicle=vehicle,
                near_field_width_m=near_field_width_m,
                metric_calibration=metric_calibration,
            )
            videos.append(
                _process_video(
                    video_path(mission_dir, video.id),
                    video,
                    adapter,
                    terrain_analyzer,
                    staging / "evidence",
                    mission.id,
                    run_id,
                )
            )

        tree_count = sum(video["metrics"]["confirmed_tree_instances"] for video in videos)
        shrub_count = sum(video["metrics"]["confirmed_shrub_instances"] for video in videos)
        terrain_frames = sum(video["metrics"]["terrain_frames"] for video in videos)
        unique_source_hashes = sum(video["metrics"]["terrain_unique_source_hashes"] for video in videos)
        unique_masks = sum(video["metrics"]["terrain_unique_masks"] for video in videos)
        evidence_frames = sum(video["metrics"]["representative_evidence_frames"] for video in videos)
        class_frame_counts = {
            class_id: sum(video["metrics"]["terrain_overall_class_frames"][class_id] for video in videos)
            for class_id in TRAVERSABILITY_ONTOLOGY
        }
        result = {
            "schema_version": "3.0",
            "run_id": run_id,
            "mission_id": mission.id,
            "model": {
                "adapter": "ForestInstanceAdapter",
                "model_id": MODEL_ID,
                "version": MODEL_VERSION,
                "hardware": "CPU",
                "weights": "none; deterministic classical CV instance baseline",
                "license": "project source",
            },
            "terrain_model": {
                "adapter": "TerrainAnalyzer",
                "model_id": TERRAIN_MODEL_ID,
                "version": TERRAIN_MODEL_VERSION,
                "hardware": "CPU",
                "weights": "none; deterministic image-derived terrain baseline",
                "license": "project source",
            },
            "configuration": {
                "analysis_hz": ANALYSIS_HZ,
                "input_width": INPUT_WIDTH,
                "min_area_ratio": min_area,
                "confirmation_hits": 3,
                "max_track_gap_frames": 2,
                "random_seed": 0,
                "confidence_meaning": "algorithmic evidence proxy, not calibrated accuracy",
            },
            "vehicle_configuration": {
                "width_m": round(vehicle.width_m, 3),
                "safety_margin_per_side_m": round(vehicle.safety_margin_per_side_m, 3),
                "required_width_m": round(vehicle.required_width_m, 3),
                "source": vehicle.source,
            },
            "terrain_configuration": {
                "near_field_width_m": round(near_field_width_m, 3),
                "metric_calibration": metric_calibration,
                "source_frames": "decoded_video_frames_resized_without_content_substitution",
                "temporal_motion_minimum_inliers": 30,
            },
            "ontology": ONTOLOGY,
            "terrain_ontology": TRAVERSABILITY_ONTOLOGY,
            "videos": videos,
            "counts": {
                "confirmed_unique_per_video_sum": {"tree": tree_count, "shrub": shrub_count},
                "mission_unique": None,
                "mission_unique_reason": "Dieselben Pflanzen aus vier Begehungen sind ohne 3D-Objektlokalisierung nicht belastbar zusammenführbar.",
            },
            "metrics": {
                "runtime_seconds": round(time.perf_counter() - started, 2),
                "analyzed_frames": sum(video["metrics"]["analyzed_frames"] for video in videos),
                "raw_detections": sum(video["metrics"]["raw_detections"] for video in videos),
                "tracks": sum(video["metrics"]["tracks"] for video in videos),
                "confirmed_tree_instances": tree_count,
                "confirmed_shrub_instances": shrub_count,
                "empty_frame_fraction": round(
                    float(np.mean([video["metrics"]["empty_frame_fraction"] for video in videos])), 3
                ),
                "average_track_length_frames": round(
                    float(np.mean([video["metrics"]["average_track_length_frames"] for video in videos])), 2
                ),
                "terrain_frames": terrain_frames,
                "terrain_unique_source_hashes": unique_source_hashes,
                "terrain_source_hash_unique_fraction": round(unique_source_hashes / max(1, terrain_frames), 4),
                "terrain_unique_masks": unique_masks,
                "terrain_mask_unique_fraction": round(unique_masks / max(1, terrain_frames), 4),
                "terrain_mask_transition_fraction": round(
                    float(np.mean([video["metrics"]["terrain_mask_transition_fraction"] for video in videos])), 4
                ),
                "terrain_masks_vary": any(video["metrics"]["terrain_masks_vary"] for video in videos),
                "corridor_available_fraction": round(
                    sum(
                        video["metrics"]["corridor_available_fraction"] * video["metrics"]["terrain_frames"]
                        for video in videos
                    )
                    / max(1, terrain_frames),
                    4,
                ),
                "corridor_uncertain_fraction": round(
                    sum(
                        video["metrics"]["corridor_uncertain_fraction"] * video["metrics"]["terrain_frames"]
                        for video in videos
                    )
                    / max(1, terrain_frames),
                    4,
                ),
                "terrain_overall_class_frames": class_frame_counts,
                "representative_evidence_frames": evidence_frames,
            },
            "truth_rules": {
                "ground_truth_available": False,
                "species_inference": False,
                "navigation_grade": False,
                "individual_definition": "Baum nur mit sichtbarem Stammanker; Strauch nur als separierbare bodennahe Vegetationsregion; mindestens drei Beobachtungen für bestätigte Zählung.",
                "terrain_inference": "Every terrain mask is calculated from its timestamped source video frame; no manual or static result mask is used.",
                "terrain_uncertainty": "Hidden, clipped, disconnected or insufficiently supported areas remain unknown or limited and are never promoted by temporal smoothing alone.",
                "metric_scale": "Configured perspective estimate unless explicitly marked calibrated; reported widths are estimates.",
                "safety_disclaimer": "KI-gestützte Einschätzung; keine sicherheitsrelevante Fahrfreigabe.",
                "cross_video_fusion": False,
                "overlays": "stored image-derived polygons and RLE masks rendered client-side; originals unchanged",
            },
        }
        (staging / "segmentation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        annotation_frames = {
            video["video_id"]: {
                str(frame["frame_index"]): {
                    "timestamp_ms": frame["timestamp_ms"],
                    "source_frame_hash": frame["terrain"]["source_frame_hash"],
                    "mask_width": frame["terrain"]["traversability"]["mask"]["width"],
                    "mask_height": frame["terrain"]["traversability"]["mask"]["height"],
                }
                for frame in video["frames"]
            }
            for video in videos
        }
        (staging / "annotation_frames.json").write_text(
            json.dumps(annotation_frames, ensure_ascii=False), encoding="utf-8"
        )
        report = f"""# ARIADNE Goal 4 – Vegetations-, Boden- und Befahrbarkeitsbericht

Mission: {mission.name} ({mission.id})

## Technischer Nachweis
- {len(videos)} reale Originalvideos unverändert mit {ANALYSIS_HZ} Hz analysiert.
- {result["metrics"]["analyzed_frames"]} zeitgestempelte Frames besitzen Vegetations- und bildberechnete Terrainergebnisse.
- {tree_count} bestätigte Einzelbaum-Tracks und {shrub_count} bestätigte Einzelstrauch-Tracks; Summe pro Video, keine missionsweite Pflanzenzahl.
- Bodenmasken entstehen aus Farbe, Textur, Kanten, Sichtbarkeit und Verbindung zum aktuellen Bildunterrand.
- Befahrbarkeit unterscheidet Grün, Gelb, Rot und Grau und berücksichtigt freie Breite, Hindernisse, Zusammenhang, Rauheit, Engstellen und Sichtbarkeit.
- Der Fahrkorridor wird zwischen Frames bewegungskompensiert stabilisiert; aktuelle rote oder graue Evidenz hat Vorrang.
- {evidence_frames} repräsentative Frames wurden als echtes Quell-JPEG und berechnetes Overlay gespeichert.

## Proxy-Metriken ohne Ground Truth
- Eindeutige Quellbild-Hashes: {unique_source_hashes}/{terrain_frames}
- Eindeutige Terrainmasken: {unique_masks}/{terrain_frames}
- Anteil aufeinanderfolgender Maskenänderungen: {result["metrics"]["terrain_mask_transition_fraction"]}
- Korridor verfügbar: {result["metrics"]["corridor_available_fraction"]}
- Korridor eingeschränkt/unsicher: {result["metrics"]["corridor_uncertain_fraction"]}
- Fahrzeugbreite: {vehicle.width_m:.2f} m; Sicherheitsrand je Seite: {vehicle.safety_margin_per_side_m:.2f} m; erforderliche Breite: {vehicle.required_width_m:.2f} m (`{vehicle.source}`).
- Metrische Skalierung: `{metric_calibration}`; Gesamtlaufzeit: {result["metrics"]["runtime_seconds"]} s.

## Grenzen
Die Offline-CPU-Pipeline ist eine deterministische klassische Computer-Vision-Baseline ohne Ground Truth. Unebenheit, Stufen, Wasser und metrische Breite sind aus monokularem Video nur eingeschränkt schätzbar. Unsichere oder verdeckte Bereiche werden nicht automatisch als befahrbar behandelt. Die Ausgabe ist ausschließlich eine KI-gestützte Einschätzung und keine sicherheitsrelevante Fahrfreigabe.
"""
        (staging / "evaluation.md").write_text(report, encoding="utf-8")
        runs = derived / "segmentation_runs"
        runs.mkdir(exist_ok=True)
        try:
            os.replace(staging, run_dir)
        except PermissionError:
            shutil.copytree(staging, run_dir)
            shutil.rmtree(staging, ignore_errors=True)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    pointer = derived / "segmentation_current.tmp"
    pointer.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    os.replace(pointer, derived / "segmentation_current.json")
    return result


def current_segmentation_dir(mission_dir: Path):
    run_id = json.loads((mission_dir / "derived" / "segmentation_current.json").read_text(encoding="utf-8"))["run_id"]
    return mission_dir / "derived" / "segmentation_runs" / run_id
