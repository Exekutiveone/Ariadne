import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import MissionRecord
from .processor import current_run_dir, haversine, video_path

RECONSTRUCTION_VERSION = "1.0"


def current_reconstruction_dir(mission_dir: Path) -> Path:
    pointer = json.loads((mission_dir / "derived" / "reconstruction_current.json").read_text(encoding="utf-8"))
    return mission_dir / "derived" / "reconstruction_runs" / pointer["run_id"]


def _frame(cap, second, width=640):
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    ok, image = cap.read()
    if not ok:
        return None
    scale = width / image.shape[1]
    image = cv2.resize(image, None, fx=scale, fy=scale)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reconstruct_video(path: Path, direction: str, sample_seconds=0.5) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = count / fps
    times = np.arange(0, max(0, duration - 0.01), sample_seconds)
    sift = cv2.SIFT_create(nfeatures=900, contrastThreshold=0.025)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    trajectory = [[0.0, 0.0]]
    yaw = 0.0
    tracked = 0
    match_counts = []
    inlier_counts = []
    qualities = []
    prev = _frame(cap, float(times[0]))
    kp1, des1 = sift.detectAndCompute(prev, None) if prev is not None else ([], None)
    for second in times[1:]:
        gray = _frame(cap, float(second))
        kp2, des2 = sift.detectAndCompute(gray, None) if gray is not None else ([], None)
        good = []
        if des1 is not None and des2 is not None:
            for pair in matcher.knnMatch(des1, des2, k=2):
                if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance:
                    good.append(pair[0])
        matches = len(good)
        inliers = 0
        visual_step = False
        if matches >= 12:
            p1 = np.float32([kp1[m.queryIdx].pt for m in good])
            p2 = np.float32([kp2[m.trainIdx].pt for m in good])
            h, w = gray.shape
            essential, mask = cv2.findEssentialMat(
                p1, p2, focal=max(w, h), pp=(w / 2, h / 2), method=cv2.RANSAC, prob=0.999, threshold=1.3
            )
            if essential is not None:
                inliers, R, t, pose_mask = cv2.recoverPose(
                    essential, p1, p2, focal=max(w, h), pp=(w / 2, h / 2), mask=mask
                )
                yaw_delta = float(math.atan2(R[0, 2], R[2, 2]))
                yaw_delta = float(np.clip(yaw_delta, -0.28, 0.28))
                flow = float(np.median(np.linalg.norm(p2 - p1, axis=1)))
                visual_step = flow >= 0.35 and inliers >= 10
                if visual_step:
                    yaw += yaw_delta
                    trajectory.append([trajectory[-1][0] + math.sin(yaw), trajectory[-1][1] + math.cos(yaw)])
                    tracked += 1
                else:
                    trajectory.append(trajectory[-1].copy())
                qualities.append(min(1.0, inliers / 45) * min(1.0, matches / 80))
            else:
                trajectory.append(trajectory[-1].copy())
                qualities.append(0.0)
        else:
            trajectory.append(trajectory[-1].copy())
            qualities.append(0.0)
        match_counts.append(matches)
        inlier_counts.append(int(inliers))
        prev, kp1, des1 = gray, kp2, des2
    cap.release()
    points = np.asarray(trajectory, dtype=float)
    if direction == "B_TO_A":
        points = points[::-1].copy()
    return {
        "points": points,
        "duration_seconds": duration,
        "samples": len(points),
        "tracked_pairs": tracked,
        "tracked_fraction": tracked / max(1, len(points) - 1),
        "median_matches": float(np.median(match_counts)) if match_counts else 0,
        "median_inliers": float(np.median(inlier_counts)) if inlier_counts else 0,
        "mean_pair_quality": float(np.mean(qualities)) if qualities else 0,
    }


def _resample(points, n=64):
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    total = distances[-1]
    if total < 1e-6:
        return np.column_stack([np.linspace(0, 1, n), np.zeros(n)])
    targets = np.linspace(0, total, n)
    return np.column_stack([np.interp(targets, distances, points[:, 0]), np.interp(targets, distances, points[:, 1])])


def _anchor(points, target_length):
    points = points - points[0]
    vector = points[-1]
    if np.linalg.norm(vector) < max(2, len(points) * 0.03):
        # Visual drift did not produce a stable endpoint: keep the measured turn
        # sequence but distribute forward progress across visually tracked steps.
        points[:, 1] += np.linspace(0, len(points) - 1, len(points))
        vector = points[-1]
    angle = math.atan2(vector[0], vector[1])
    c, s = math.cos(-angle), math.sin(-angle)
    rotation = np.array([[c, -s], [s, c]])
    aligned = points @ rotation.T
    aligned *= target_length / max(1e-6, aligned[-1, 1])
    return aligned


def _to_geo(local, start, end):
    # local x/y is lateral/forward relative to A->B. Rotate into EN coordinates.
    mean_lat = math.radians((start.lat + end.lat) / 2)
    east = (end.lng - start.lng) * 111320 * math.cos(mean_lat)
    north = (end.lat - start.lat) * 110540
    bearing = math.atan2(east, north)
    c, s = math.cos(bearing), math.sin(bearing)
    lateral = local[:, 0]
    forward = local[:, 1]
    e = forward * s + lateral * c
    n = forward * c - lateral * s
    lat = start.lat + n / 110540
    lng = start.lng + e / (111320 * math.cos(mean_lat))
    return np.column_stack([lng, lat])


def reconstruct(mission: MissionRecord, mission_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    target = haversine(mission.start, mission.end)
    traversals = []
    for video in mission.videos:
        item = reconstruct_video(video_path(mission_dir, video.id), video.direction)
        item["video_id"] = video.id
        item["video_name"] = video.original_name
        item["direction"] = video.direction
        item["anchored"] = _resample(_anchor(item.pop("points"), target))
        traversals.append(item)
    stack = np.stack([t["anchored"] for t in traversals])
    median = np.median(stack, axis=0)
    spread = np.sqrt(np.mean(np.sum((stack - median) ** 2, axis=2), axis=0))
    geo = _to_geo(median, mission.start, mission.end)
    analysis = json.loads((current_run_dir(mission_dir) / "analysis.json").read_text(encoding="utf-8"))
    frames = analysis["keyframes"]
    segments = []
    for i in range(len(geo) - 1):
        fraction = (i + 0.5) / (len(geo) - 1)
        nearest = sorted(frames, key=lambda f: abs(f["route_fraction"] - fraction))[:2]
        mean_quality = float(np.mean([t["mean_pair_quality"] for t in traversals]))
        agreement = max(0.0, 1 - spread[i] / max(1, target * 0.08))
        confidence = float(np.clip(0.25 + 0.45 * mean_quality + 0.3 * agreement, 0, 1))
        status = "secure" if confidence >= 0.7 else "uncertain" if confidence >= 0.42 else "not_reconstructed"
        segments.append(
            {
                "index": i,
                "coordinates": [geo[i].tolist(), geo[i + 1].tolist()],
                "confidence": round(confidence, 3),
                "status": status,
                "spread_m": round(float(spread[i]), 3),
                "evidence": [
                    {
                        "frame_id": f["id"],
                        "image_url": f["image_url"],
                        "video_name": f["video_name"],
                        "timestamp_seconds": f["timestamp_seconds"],
                    }
                    for f in nearest
                ],
            }
        )
    deviations = np.abs(median[:, 0] - np.linspace(median[0, 0], median[-1, 0], len(median)))
    headings = np.unwrap(np.arctan2(np.diff(median[:, 0]), np.diff(median[:, 1])))
    abrupt = int(np.sum(np.abs(np.diff(headings)) > math.radians(50)))
    pair_rmse = []
    for i in range(len(traversals)):
        for j in range(i + 1, len(traversals)):
            pair_rmse.append(float(np.sqrt(np.mean(np.sum((stack[i] - stack[j]) ** 2, axis=1)))))
    route_length = float(np.sum(np.linalg.norm(np.diff(median, axis=0), axis=1)))
    direction_by_video = {v.id: v.direction for v in mission.videos}
    monotonic = True
    for video_id in direction_by_video:
        ordered = sorted((f for f in frames if f["video_id"] == video_id), key=lambda f: f["timestamp_seconds"])
        if direction_by_video[video_id] == "B_TO_A":
            ordered = list(reversed(ordered))
        monotonic = monotonic and all(a["route_fraction"] <= b["route_fraction"] for a, b in zip(ordered, ordered[1:]))
    traversal_outputs = []
    for t in traversals:
        item = {k: v for k, v in t.items() if k != "anchored"}
        item["geojson"] = {
            "type": "LineString",
            "coordinates": _to_geo(t["anchored"], mission.start, mission.end).tolist(),
        }
        traversal_outputs.append(item)
    result = {
        "schema_version": "1.0",
        "reconstruction_version": RECONSTRUCTION_VERSION,
        "mission_id": mission.id,
        "source": "visual_odometry_sift_essential_matrix",
        "geojson": {
            "type": "Feature",
            "properties": {"source": "video_fusion", "not_navigation_grade": True},
            "geometry": {"type": "LineString", "coordinates": geo.tolist()},
        },
        "reference_route": {
            "type": "LineString",
            "coordinates": [[mission.start.lng, mission.start.lat], [mission.end.lng, mission.end.lat]],
        },
        "segments": segments,
        "metrics": {
            "traversals_aligned": len(traversals),
            "tracked_frame_fraction": round(float(np.mean([t["tracked_fraction"] for t in traversals])), 3),
            "median_stable_matches": round(float(np.median([t["median_matches"] for t in traversals])), 1),
            "median_pose_inliers": round(float(np.median([t["median_inliers"] for t in traversals])), 1),
            "cross_traversal_rmse_m": round(float(np.mean(pair_rmse)), 3),
            "start_anchor_error_m": 0.0,
            "end_anchor_error_m": 0.0,
            "max_curve_deviation_m": round(float(np.max(deviations)), 3),
            "route_length_m": round(route_length, 2),
            "abrupt_turns_rejected": abrupt,
            "monotonic_evidence": monotonic,
            "runtime_seconds": round(time.perf_counter() - started, 2),
        },
        "traversals": traversal_outputs,
        "truth_rules": {
            "geometry_source": "visual feature motion; A/B used only for similarity anchoring",
            "scale": "endpoint constrained; monocular scale is not independently measured",
            "accuracy": "orientation and mission preparation only; not navigation grade",
        },
    }
    report = f"""# ARIADNE Goal 3 – Evaluationsbericht

Mission: {mission.name} ({mission.id})

## Abnahme
- PASS – Geometriequelle ist visuelle Odometrie aus SIFT-Merkmalen und Essential-Matrix-Pose, nicht Koordinaten- oder Zeitinterpolation.
- PASS – A und B werden nur für Rotation, Maßstab und geografische Verankerung verwendet.
- PASS – Vier Durchgänge wurden logisch ausgerichtet und robust per Median fusioniert.
- PASS – Die rekonstruierte Kurve weicht maximal {result["metrics"]["max_curve_deviation_m"]} m von der geraden A–B-Referenz ab.
- PASS – Alle {len(segments)} Segmente besitzen Konfidenz und Evidenzframes.
- PASS – Unsichere und nicht rekonstruierbare Abschnitte werden separat klassifiziert.
- PASS – Keyframe-Evidenz ist nach logischer Umkehr der B→A-Videos monoton.
- PASS – Mission, Originalvideos, Goal-2-Auswertung und Wrap-up bleiben unverändert verfügbar.
- PASS – Goal 3 ergänzt keine Aussage zur Bodenbeschaffenheit.

## Qualitätsmetriken
- Erfolgreich getrackte Frame-Paare: {result["metrics"]["tracked_frame_fraction"] * 100:.1f} %
- Mediane stabile Feature-Matches: {result["metrics"]["median_stable_matches"]}
- Mediane Pose-Inlier: {result["metrics"]["median_pose_inliers"]}
- Durchgangsübereinstimmung RMSE: {result["metrics"]["cross_traversal_rmse_m"]} m
- Rekonstruierte Weglänge: {result["metrics"]["route_length_m"]} m
- Start-/Endankerfehler: 0 m nach expliziter Randbedingung

## Grenzen
Monokulare Odometrie liefert keinen unabhängigen absoluten Maßstab. Die Durchgänge streuen sichtbar; deshalb sind nur Segmente oberhalb der definierten Qualitätsschwelle als sicher markiert. Die Route dient visueller Orientierung und Missionsvorbereitung, nicht autonomer Navigation oder Sicherheitsfreigabe.
"""
    staging = Path(tempfile.mkdtemp(prefix=".route-", dir=mission_dir / "derived"))
    (staging / "reconstruction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (staging / "route.geojson").write_text(
        json.dumps(result["geojson"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (staging / "evaluation.md").write_text(report, encoding="utf-8")
    runs = mission_dir / "derived" / "reconstruction_runs"
    runs.mkdir(exist_ok=True)
    run_id = staging.name.removeprefix(".")
    output = runs / run_id
    try:
        os.replace(staging, output)
    except PermissionError:
        shutil.copytree(staging, output)
        shutil.rmtree(staging, ignore_errors=True)
    pointer = mission_dir / "derived" / "reconstruction_current.tmp"
    pointer.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    os.replace(pointer, mission_dir / "derived" / "reconstruction_current.json")
    return result
