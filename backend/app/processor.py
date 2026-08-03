import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import MissionRecord

PROCESSOR_VERSION = "2.0"


def current_run_dir(mission_dir: Path) -> Path:
    pointer = json.loads((mission_dir / "derived" / "current.json").read_text(encoding="utf-8"))
    return mission_dir / "derived" / "runs" / pointer["run_id"]


def haversine(a, b):
    radius = 6371000
    p1 = math.radians(a.lat)
    p2 = math.radians(b.lat)
    dp = math.radians(b.lat - a.lat)
    dl = math.radians(b.lng - a.lng)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


def route_length(route):
    return sum(haversine(a, b) for a, b in zip(route, route[1:]))


def interpolate(route, fraction):
    if len(route) == 2:
        return {
            "lat": route[0].lat + (route[1].lat - route[0].lat) * fraction,
            "lng": route[0].lng + (route[1].lng - route[0].lng) * fraction,
        }
    lengths = [haversine(a, b) for a, b in zip(route, route[1:])]
    target = sum(lengths) * fraction
    walked = 0
    for i, length in enumerate(lengths):
        if walked + length >= target:
            f = (target - walked) / length if length else 0
            a, b = route[i], route[i + 1]
            return {"lat": a.lat + (b.lat - a.lat) * f, "lng": a.lng + (b.lng - a.lng) * f}
        walked += length
    return route[-1].model_dump()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def video_path(mission_dir, video_id):
    matches = list((mission_dir / "videos").glob(f"{video_id}.*"))
    if not matches:
        raise FileNotFoundError(video_id)
    return matches[0]


def sharpness(frame):
    return float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def visual_features(frame):
    small = cv2.resize(frame, (320, 180))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    vegetation = float(cv2.inRange(hsv, (25, 35, 25), (95, 255, 255)).mean() / 255)
    lower = hsv[90:]
    ground = float(cv2.inRange(lower, (5, 20, 20), (30, 230, 220)).mean() / 255)
    edges = cv2.Canny(cv2.cvtColor(small[72:], cv2.COLOR_BGR2GRAY), 80, 180)
    return {
        "vegetation_ratio": round(vegetation, 4),
        "ground_ratio_lower": round(ground, 4),
        "edge_ratio_lower": round(float(edges.mean() / 255), 4),
        "brightness": round(float(hsv[:, :, 2].mean() / 255), 4),
    }


def read_candidate(capture, times):
    best = None
    for second in times:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0, second) * 1000)
        ok, frame = capture.read()
        if ok:
            score = sharpness(frame)
            if best is None or score > best[0]:
                best = (score, second, frame)
    return best


def process(
    mission: MissionRecord, mission_dir: Path, interval=6.0, samples=3, include_endpoints=True
) -> dict[str, Any]:
    started = time.perf_counter()
    derived = mission_dir / "derived"
    derived.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".processing-", dir=derived))
    (staging / "frames").mkdir()
    errors = []
    integrity = []
    videos = []
    keyframes = []
    observations = []
    unique_length = route_length(mission.route)
    for video in mission.videos:
        path = video_path(mission_dir, video.id)
        actual = sha256(path)
        valid = actual == video.sha256
        integrity.append(
            {"video_id": video.id, "valid": valid, "expected_sha256": video.sha256, "actual_sha256": actual}
        )
        if not valid:
            errors.append(f"Checksum mismatch: {video.original_name}")
            continue
        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frames / fps if fps else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or duration <= 0:
            errors.append(f"Unreadable video: {video.original_name}")
            cap.release()
            continue
        videos.append(
            {
                "id": video.id,
                "name": video.original_name,
                "duration_seconds": round(duration, 3),
                "fps": round(fps, 3),
                "frames": frames,
                "width": width,
                "height": height,
                "direction": video.direction,
                "orientation": video.orientation,
            }
        )
        times = list(np.arange(0 if include_endpoints else interval / 2, duration, interval))
        if include_endpoints and (not times or duration - 1 - times[-1] > interval / 2):
            times.append(max(0, duration - 1))
        for seq, target in enumerate(times):
            offsets = np.linspace(-0.6, 0.6, samples) if samples > 1 else [0]
            candidate = read_candidate(cap, [target + float(o) for o in offsets])
            if not candidate:
                continue
            score, second, frame = candidate
            progress = min(1, max(0, second / duration))
            route_fraction = progress if video.direction == "A_TO_B" else 1 - progress
            point = interpolate(mission.route, route_fraction)
            segment = min(max(0, len(mission.route) - 2), int(route_fraction * max(1, len(mission.route) - 1)))
            frame_id = f"{video.id[:8]}-{seq:03d}"
            output = staging / "frames" / f"{frame_id}.jpg"
            scale = min(1, 1280 / max(frame.shape[:2]))
            saved = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
            cv2.imwrite(str(output), saved, [cv2.IMWRITE_JPEG_QUALITY, 86])
            features = visual_features(frame)
            item = {
                "id": frame_id,
                "video_id": video.id,
                "video_name": video.original_name,
                "timestamp_seconds": round(second, 3),
                "route_fraction": round(route_fraction, 5),
                "segment_index": segment,
                "position": point,
                "sharpness": round(score, 2),
                "image_url": f"/api/v1/missions/{mission.id}/analysis/frames/{frame_id}.jpg",
                "features": features,
            }
            keyframes.append(item)
            if features["vegetation_ratio"] >= 0.12:
                observations.append(
                    {
                        "category": "vegetation",
                        "label": "deutliche Vegetationspräsenz",
                        "confidence": round(min(0.88, 0.42 + features["vegetation_ratio"]), 2),
                        "value_status": "calculated",
                        "frame_id": frame_id,
                        "video_id": video.id,
                        "timestamp_seconds": item["timestamp_seconds"],
                        "segment_index": segment,
                        "position": point,
                        "evidence_url": item["image_url"],
                    }
                )
                observations.append(
                    {
                        "category": "tree_species",
                        "label": "Baumart unklar",
                        "confidence": 0.1,
                        "value_status": "estimated",
                        "frame_id": frame_id,
                        "video_id": video.id,
                        "timestamp_seconds": item["timestamp_seconds"],
                        "segment_index": segment,
                        "position": point,
                        "evidence_url": item["image_url"],
                    }
                )
            if features["ground_ratio_lower"] >= 0.12:
                observations.append(
                    {
                        "category": "ground",
                        "label": "möglicher unbefestigter Untergrund",
                        "confidence": round(min(0.76, 0.35 + features["ground_ratio_lower"]), 2),
                        "value_status": "estimated",
                        "frame_id": frame_id,
                        "video_id": video.id,
                        "timestamp_seconds": item["timestamp_seconds"],
                        "segment_index": segment,
                        "position": point,
                        "evidence_url": item["image_url"],
                    }
                )
            if features["edge_ratio_lower"] >= 0.16:
                observations.append(
                    {
                        "category": "obstacle_candidate",
                        "label": "visuell komplexes Wegsegment – manuell prüfen",
                        "confidence": 0.3,
                        "value_status": "estimated",
                        "frame_id": frame_id,
                        "video_id": video.id,
                        "timestamp_seconds": item["timestamp_seconds"],
                        "segment_index": segment,
                        "position": point,
                        "evidence_url": item["image_url"],
                    }
                )
        cap.release()
    if errors:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("; ".join(errors))
    grouped = defaultdict(list)
    for obs in observations:
        grouped[(obs["category"], obs["segment_index"])].append(obs)
    merged = []
    for (category, segment), items in grouped.items():
        representative = max(items, key=lambda x: x["confidence"])
        merged.append(
            {
                **representative,
                "id": f"{category}-{segment}",
                "raw_detection_count": len(items),
                "source_video_ids": sorted(set(x["video_id"] for x in items)),
                "evidence_urls": [x["evidence_url"] for x in items[:4]],
                "object_status": "estimated_group" if category in {"vegetation", "tree_species"} else "raw_detection",
            }
        )
    video_seconds = sum(v["duration_seconds"] for v in videos)
    traversals = len(videos)
    total_walked = unique_length * traversals
    categories = {item["category"] for item in merged}
    wrap_up = {
        "vegetation": {
            "status": "observed" if "vegetation" in categories else "not_observed",
            "basis": "calculated color ratio with evidence",
        },
        "tree_species": {"status": "unclear", "basis": "no domain model or labeled ground truth"},
        "shrub_groups": {
            "status": "not_reliably_assessed",
            "basis": "cannot be separated from general vegetation by current method",
        },
        "regeneration": {
            "status": "not_reliably_assessed",
            "basis": "cannot be separated from general vegetation by current method",
        },
        "ground": {
            "status": "candidate_observed" if "ground" in categories else "not_observed",
            "basis": "estimated lower-frame color ratio",
        },
        "obstacles": {
            "status": "manual_review_candidates" if "obstacle_candidate" in categories else "none_detected",
            "basis": "low-confidence visual complexity heuristic",
        },
        "drivability": {
            "status": "manual_review_required",
            "confidence": 0.25,
            "basis": "no safety-grade terrain or clearance model",
        },
    }
    result = {
        "schema_version": "1.0",
        "processor_version": PROCESSOR_VERSION,
        "mission_id": mission.id,
        "generated_at": time.time(),
        "source_status": mission.status,
        "integrity": integrity,
        "videos": videos,
        "route": {
            "coordinates": [p.model_dump() for p in mission.route],
            "unique_length_m": {"value": round(unique_length, 2), "status": "calculated"},
            "walked_total_m": {
                "value": round(total_walked, 2),
                "status": "calculated",
                "assumption": "one complete traversal per video",
            },
        },
        "metrics": {
            "media_duration_seconds": {"value": round(video_seconds, 2), "status": "measured"},
            "movement_time_seconds": {
                "value": round(video_seconds, 2),
                "status": "estimated",
                "reason": "no manual movement timestamps; media duration used",
            },
            "mean_speed_mps": {
                "value": round(total_walked / video_seconds, 3) if video_seconds else None,
                "status": "estimated",
            },
            "traversals": {"value": traversals, "status": "measured"},
            "keyframes": {"value": len(keyframes), "status": "measured"},
            "raw_observations": {"value": len(observations), "status": "calculated"},
            "merged_observations": {"value": len(merged), "status": "calculated"},
        },
        "wrap_up": wrap_up,
        "keyframes": keyframes,
        "observations": merged,
        "truth_rules": {
            "ground_truth_available": False,
            "frame_position_is_object_position": False,
            "coverage": "visible route corridor only",
            "species_statement": "No species model or ground truth; species remains unknown.",
        },
        "technical": {
            "runtime_seconds": round(time.perf_counter() - started, 2),
            "failed_files": 0,
            "unassigned_keyframes": sum(1 for f in keyframes if "position" not in f),
            "map_objects_without_evidence": sum(1 for o in merged if not o.get("evidence_urls")),
            "keyframe_interval_seconds": interval,
            "sharpness_samples": samples,
            "mean_sharpness": round(sum(f["sharpness"] for f in keyframes) / len(keyframes), 2) if keyframes else 0,
        },
    }
    (staging / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    runs = derived / "runs"
    runs.mkdir(exist_ok=True)
    run_id = staging.name.removeprefix(".")
    final = runs / run_id
    try:
        os.replace(staging, final)
    except PermissionError:
        shutil.copytree(staging, final)
        shutil.rmtree(staging, ignore_errors=True)
    pointer_tmp = derived / "current.tmp"
    pointer_tmp.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    os.replace(pointer_tmp, derived / "current.json")
    return result


def autonomous_loop(mission, mission_dir):
    configs = [("Baseline", 10.0, 1, False), ("Iteration 1", 6.0, 3, False), ("Iteration 2", 6.0, 3, True)]
    runs = []
    best = None
    for name, interval, samples, endpoints in configs:
        result = process(mission, mission_dir, interval, samples, endpoints)
        metric = {
            "name": name,
            "config": {"interval": interval, "samples": samples, "endpoints": endpoints},
            "keyframes": result["metrics"]["keyframes"]["value"],
            "mean_sharpness": result["technical"]["mean_sharpness"],
            "unassigned": result["technical"]["unassigned_keyframes"],
            "without_evidence": result["technical"]["map_objects_without_evidence"],
            "runtime_seconds": result["technical"]["runtime_seconds"],
        }
        accepted = (
            best is None or metric["keyframes"] > best["keyframes"] or metric["mean_sharpness"] > best["mean_sharpness"]
        )
        metric["decision"] = "accepted" if accepted else "rejected"
        runs.append(metric)
        if accepted:
            best = metric
    report = [
        "# ARIADNE Goal 2 – Evaluationsbericht",
        "",
        f"Mission: {mission.name} ({mission.id})",
        "",
        "## Iterationen",
    ]
    for run in runs:
        report += [
            f"### {run['name']}",
            f"Konfiguration: {run['config']}",
            f"Keyframes: {run['keyframes']}; mittlere Schärfe: {run['mean_sharpness']}; nicht zugeordnet: {run['unassigned']}; Kartenobjekte ohne Evidenz: {run['without_evidence']}; Laufzeit: {run['runtime_seconds']} s.",
            f"Entscheidung: {run['decision']}.",
            "",
        ]
    report += [
        "## Grenzen",
        "Keine gelabelte Ground Truth. Baumarten bleiben unklar. Objektpositionen entsprechen nur dem zeitbasiert zugeordneten Routensegment. Bewegungszeit nutzt mangels manueller Zeitmarken die Videodauer. Hindernisse sind niedrig-konfidente Prüfkandidaten, keine Sicherheitsfreigabe.",
        "",
        "## Nächste sinnvolle Schritte",
        "Manuelle Bestätigung ausgewählter Evidenzframes; echte Bewegungszeitmarken; domänenspezifisch gelabelter Walddatensatz; zusätzliche GPS-/IMU-Synchronisation.",
    ]
    current = current_run_dir(mission_dir)
    (current / "evaluation.md").write_text("\n".join(report), encoding="utf-8")
    (current / "iterations.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    return result
