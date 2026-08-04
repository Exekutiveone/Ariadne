import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from uuid import uuid4

import cv2
import numpy as np

from .global_path_model import current_global_model_dir
from .path_features import (
    GRADE_ONTOLOGY,
    clean_prediction,
    grade_prediction,
    grading_summary,
    pixel_features,
    predict_scores,
)
from .path_masks import (
    apply_refinements,
    comparison_mask,
    confusion_counts,
    encode_binary_rle,
    load_refinements,
    polygon_mask,
    symmetric_metrics,
)
from .processor import video_path
from .runtime_paths import runtime_root
from .storage import MissionStore

# 1.1 fuegt die Abstufung (grade_mask) je Frame hinzu, vorher enthielt der
# Batch-Lauf nur die binaere Maske; die Abstufung entstand ausschliesslich
# live im Einzelframe-Endpunkt. Deshalb blieb die Anzeige waehrend der
# Wiedergabe auf die einfarbige Maske beschraenkt und die abgestufte Ansicht
# erschien nur im Pausenzustand, wo der Player den Einzelframe live nachholte.
# Alte Ergebnisse und Checkpoints ohne diese Version werden verworfen statt
# wiederverwendet, damit kein Lauf halb mit und halb ohne Abstufung entsteht.
GLOBAL_VIDEO_ANALYSIS_SCHEMA_VERSION = "1.1"


def _runtime_root():
    return runtime_root()


def _analysis_dir(missions_root: Path, model_run_id: str, mission_id: str, video_id: str):
    # Derived video inference is a reproducible runtime cache. Keeping it out
    # of OneDrive/network workspaces prevents sync-provider file locks.
    return _runtime_root() / "global_video_analyses" / model_run_id / mission_id / video_id


def _portable_analysis_dir(missions_root: Path, model_run_id: str, mission_id: str, video_id: str):
    return missions_root.parent / "runtime_cache" / "global_video_analyses" / model_run_id / mission_id / video_id


def _write_json(path: Path, payload: dict, *, attempts: int = 12, direct_fallback: bool = True):
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    temporary = None
    preparation_error = None
    for preparation_attempt in range(attempts):
        candidate = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text(serialized, encoding="utf-8")
            temporary = candidate
            break
        except (FileNotFoundError, PermissionError) as exc:
            preparation_error = exc
            if preparation_attempt < attempts - 1:
                time.sleep(min(1.0, 0.04 * (2**preparation_attempt)))
    if temporary is None:
        raise preparation_error or OSError("Temporäre JSON-Datei konnte nicht erstellt werden")
    try:
        for attempt in range(attempts):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == attempts - 1:
                    if direct_fallback:
                        for direct_attempt in range(attempts):
                            try:
                                path.write_text(serialized, encoding="utf-8")
                                return
                            except PermissionError:
                                if direct_attempt == attempts - 1:
                                    raise
                                time.sleep(min(1.0, 0.04 * (2**direct_attempt)))
                    raise
                time.sleep(min(1.0, 0.04 * (2**attempt)))
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _write_status(path: Path, payload: dict):
    try:
        _write_json(path, payload, direct_fallback=False)
        return True
    except OSError:
        # Status is advisory. A temporary sync-provider lock must never stop
        # model inference; the next periodic update will try again.
        return False


def _chunk_dir(directory: Path):
    return directory / "frame_chunks"


def _load_checkpoint(directory: Path):
    frames = []
    for path in sorted(_chunk_dir(directory).glob("*.json")):
        try:
            chunk = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            break
        if chunk.get("start_frame") != len(frames):
            break
        chunk_frames = chunk.get("frames", [])
        if chunk_frames and "grade_mask" not in chunk_frames[0]:
            # Checkpoint stammt von vor GLOBAL_VIDEO_ANALYSIS_SCHEMA_VERSION 1.1.
            # Nicht fortsetzen, sonst blieben die fruehen Frames dauerhaft ohne
            # Abstufung, waehrend die spaeteren sie haetten.
            return []
        frames.extend(chunk_frames)
    return frames


def _completed_result_is_current_schema(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return payload.get("schema_version") == GLOBAL_VIDEO_ANALYSIS_SCHEMA_VERSION


def _write_chunk(directory: Path, frames: list[dict], start: int):
    if start >= len(frames):
        return
    chunk = frames[start:]
    path = _chunk_dir(directory) / f"{start:09d}-{len(frames) - 1:09d}.json"
    _write_json(path, {"start_frame": start, "end_frame": len(frames) - 1, "frames": chunk})


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def global_video_analysis_status(store, mission_id: str, video_id: str):
    model_dir = current_global_model_dir(store.root)
    directory = _analysis_dir(store.root, model_dir.name, mission_id, video_id)
    path = directory / "status.json"
    if not path.is_file():
        portable = _portable_analysis_dir(store.root, model_dir.name, mission_id, video_id) / "status.json"
        if not portable.is_file():
            return None
        return json.loads(portable.read_text(encoding="utf-8"))
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("status") in {"queued", "running"} and state.get("pid") and not _alive(state["pid"]):
        state.update(
            {
                "status": "interrupted",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "message": "Videoanalyse wurde unterbrochen",
            }
        )
        _write_status(path, state)
    return state


def start_global_video_analysis(store, mission_id: str, video_id: str):
    mission = store.get(mission_id)
    if not mission:
        raise LookupError("Mission nicht gefunden")
    if not any(video.id == video_id for video in mission.videos):
        raise LookupError("Video nicht gefunden")
    model_dir = current_global_model_dir(store.root)
    current = global_video_analysis_status(store, mission_id, video_id)
    if current and current.get("status") in {"queued", "running"}:
        return current
    completed_result = _analysis_dir(store.root, model_dir.name, mission_id, video_id) / "result.json"
    portable_result = _portable_analysis_dir(store.root, model_dir.name, mission_id, video_id) / "result.json"
    existing_result = (
        completed_result if completed_result.is_file() else (portable_result if portable_result.is_file() else None)
    )
    if (
        current
        and current.get("status") == "completed"
        and existing_result
        and _completed_result_is_current_schema(existing_result)
    ):
        return current
    directory = _analysis_dir(store.root, model_dir.name, mission_id, video_id)
    directory.mkdir(parents=True, exist_ok=True)
    job_id = f"global-video-{uuid4().hex[:10]}"
    command = [
        sys.executable,
        "-m",
        "backend.app.global_video_analysis",
        str(store.root),
        mission_id,
        video_id,
        model_dir.name,
        job_id,
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    stdout = (directory / "worker.out.log").open("ab")
    stderr = (directory / "worker.err.log").open("ab")
    try:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
        )
    finally:
        stdout.close()
        stderr.close()
    state = {
        "job_id": job_id,
        "status": "running",
        "model_run_id": model_dir.name,
        "mission_id": mission_id,
        "video_id": video_id,
        "pid": process.pid,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "processed_frames": 0,
        "total_frames": 0,
        "progress": 0,
        "elapsed_seconds": 0,
        "eta_seconds": None,
        "message": "Globale Videoanalyse gestartet",
    }
    _write_status(directory / "status.json", state)
    return state


def global_video_analysis_result(store, mission_id: str, video_id: str):
    model_dir = current_global_model_dir(store.root)
    path = _analysis_dir(store.root, model_dir.name, mission_id, video_id) / "result.json"
    if not path.is_file():
        path = _portable_analysis_dir(store.root, model_dir.name, mission_id, video_id) / "result.json"
    if not path.is_file():
        raise LookupError("Noch keine abgeschlossene globale Videoanalyse vorhanden")
    return json.loads(path.read_text(encoding="utf-8"))


def run_worker(missions_root: Path, mission_id: str, video_id: str, model_run_id: str, job_id: str):
    store = MissionStore(missions_root)
    mission = store.get(mission_id)
    if not mission:
        raise LookupError("Mission nicht gefunden")
    model_dir = current_global_model_dir(store.root)
    if model_dir.name != model_run_id:
        raise ValueError("Das globale Modell wurde vor dem Analysestart gewechselt")
    directory = _analysis_dir(store.root, model_run_id, mission_id, video_id)
    status_path = directory / "status.json"
    with np.load(model_dir / "model.npz") as stored:
        model = {key: stored[key] for key in ("mean", "scale", "projection", "phase", "weights")}
        threshold = float(stored["threshold"][0])
    model_result = json.loads((model_dir / "result.json").read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(video_path(store.root / mission_id, video_id)))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(model_result["model"]["input_width"])
    height = max(48, round(width * source_height / max(1, source_width)))
    frames = _load_checkpoint(directory)
    start_frame = len(frames)
    if start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    started = time.perf_counter()
    original_started_at = datetime.now(timezone.utc).isoformat()
    state = {
        "job_id": job_id,
        "status": "running",
        "model_run_id": model_run_id,
        "mission_id": mission_id,
        "video_id": video_id,
        "pid": os.getpid(),
        "started_at": original_started_at,
        "finished_at": None,
        "processed_frames": start_frame,
        "total_frames": total,
        "progress": round(start_frame / max(1, total), 5),
        "elapsed_seconds": 0,
        "eta_seconds": None,
        "message": f"Analyse ab Frame {start_frame + 1} gestartet" if start_frame else "Videoanalyse gestartet",
    }
    _write_status(status_path, state)
    # Decodieren und Inferenz ueberlappen: ein Prefetch-Thread liest Frames,
    # waehrend der Hauptthread rechnet. Das Videodecoding (~7 ms/Frame bei 1080p)
    # verschwindet dadurch fast vollstaendig aus der Gesamtzeit. Nur der
    # Hauptthread beruehrt capture nach dem Threadstart nicht mehr; released
    # wird erst nach dem Join im finally.
    stop_decoding = Event()
    frame_queue: Queue = Queue(maxsize=8)

    def _decode_frames():
        try:
            for decode_index in range(start_frame, total):
                if stop_decoding.is_set():
                    return
                ok, decoded = capture.read()
                if not ok:
                    return
                while not stop_decoding.is_set():
                    try:
                        frame_queue.put((decode_index, decoded), timeout=0.2)
                        break
                    except Full:
                        continue
        finally:
            # Endsignal; bricht ab, wenn der Konsument bereits aufgegeben hat.
            while not stop_decoding.is_set():
                try:
                    frame_queue.put(None, timeout=0.2)
                    return
                except Full:
                    continue

    decoder = Thread(target=_decode_frames, daemon=True)
    decoder.start()
    try:
        chunk_start = start_frame
        while True:
            item = frame_queue.get()
            if item is None:
                break
            frame_index, image = item
            resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            scores = predict_scores(pixel_features(resized), model)
            prediction = clean_prediction(scores, (height, width), threshold)
            grades = grade_prediction(scores, prediction, threshold, (height, width))
            frame = {
                "frame_index": frame_index,
                "timestamp_ms": round(frame_index / fps * 1000),
                "mask": {"width": width, "height": height, "rle": encode_binary_rle(prediction)},
                "grade_mask": {"width": width, "height": height, "rle": encode_binary_rle(grades)},
                "path_fraction": round(float(prediction.mean()), 5),
            }
            annotation_path = store.root / mission_id / "ground_truth" / video_id / f"{frame_index:09d}.json"
            if annotation_path.is_file():
                try:
                    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
                    if annotation.get("polygons"):
                        truth = apply_refinements(
                            polygon_mask(annotation, width, height), store.root / mission_id, video_id, frame_index
                        )
                        comparison = comparison_mask(truth, prediction)
                        frame["evaluation"] = {
                            "metrics": symmetric_metrics(confusion_counts(truth, prediction)),
                            "comparison_mask": {
                                "width": width,
                                "height": height,
                                "rle": encode_binary_rle(comparison),
                            },
                            "refinement_count": len(load_refinements(store.root / mission_id, video_id, frame_index)),
                        }
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            frames.append(frame)
            processed = frame_index + 1
            if processed % 100 == 0:
                _write_chunk(directory, frames, chunk_start)
                chunk_start = processed
            if processed == 1 or processed % 10 == 0 or processed == total:
                elapsed = time.perf_counter() - started
                effective_processed = processed - start_frame
                rate = effective_processed / max(0.001, elapsed)
                state = {
                    "job_id": job_id,
                    "status": "running",
                    "model_run_id": model_run_id,
                    "mission_id": mission_id,
                    "video_id": video_id,
                    "pid": os.getpid(),
                    "started_at": original_started_at,
                    "finished_at": None,
                    "processed_frames": processed,
                    "total_frames": total,
                    "progress": round(processed / max(1, total), 5),
                    "elapsed_seconds": round(elapsed, 1),
                    "eta_seconds": round((total - processed) / max(0.001, rate), 1),
                    "message": f"Frame {processed} von {total} analysiert",
                }
                _write_status(status_path, state)
        _write_chunk(directory, frames, chunk_start)
        elapsed = time.perf_counter() - started
        result = {
            "schema_version": GLOBAL_VIDEO_ANALYSIS_SCHEMA_VERSION,
            "model_run_id": model_run_id,
            "mission_id": mission_id,
            "video_id": video_id,
            "fps": fps,
            "total_frames": total,
            "width": source_width,
            "height": source_height,
            "analyzed_frames": len(frames),
            "runtime_seconds": round(elapsed, 2),
            # Threshold-abhaengig, aber gleich fuer alle Frames dieses Laufs -
            # einmal hier statt in jedem Frame gespeichert.
            "grade_ontology": GRADE_ONTOLOGY,
            "grading": grading_summary(threshold),
            "frames": frames,
        }
        _write_json(directory / "result.json", result)
        state.update(
            {
                "status": "completed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "processed_frames": len(frames),
                "progress": 1,
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": 0,
                "message": f"Videoanalyse abgeschlossen: {len(frames)} Frames",
            }
        )
        _write_status(status_path, state)
        portable = _portable_analysis_dir(store.root, model_run_id, mission_id, video_id)
        _write_json(portable / "result.json", result)
        _write_json(portable / "status.json", {**state, "pid": 0, "portable_cache": True})
    finally:
        stop_decoding.set()
        while decoder.is_alive():
            try:
                frame_queue.get_nowait()
            except Empty:
                decoder.join(timeout=0.1)
        capture.release()


if __name__ == "__main__":
    root, mission_arg, video_arg, model_arg, job_arg = (
        Path(sys.argv[1]).resolve(),
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
    )
    try:
        run_worker(root, mission_arg, video_arg, model_arg, job_arg)
    except Exception as exc:
        directory = _analysis_dir(root, model_arg, mission_arg, video_arg)
        state_path = directory / "status.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {"job_id": job_arg, "model_run_id": model_arg, "mission_id": mission_arg, "video_id": video_arg}
        state.update(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
        )
        _write_status(state_path, state)
        raise
