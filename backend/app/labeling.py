import hashlib
from pathlib import Path

import cv2

from .models import MissionRecord, StoredVideo
from .processor import video_path


def _stored_video(mission: MissionRecord, video_id: str) -> StoredVideo:
    video = next((item for item in mission.videos if item.id == video_id), None)
    if not video:
        raise LookupError("Video nicht gefunden")
    return video


def probe_labeling_video(mission: MissionRecord, mission_dir: Path, video_id: str):
    video = _stored_video(mission, video_id)
    path = video_path(mission_dir, video_id)
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or total_frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Videometadaten von {video.original_name} konnten nicht gelesen werden")
    return {
        "video_id": video.id,
        "original_name": video.original_name,
        "fps": round(fps, 6),
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": round(total_frames / fps, 6),
        "terrain_category": video.terrain_category,
    }


def list_labeling_videos(mission: MissionRecord, mission_dir: Path):
    return {
        "mission_id": mission.id,
        "source": "original_video_metadata_only",
        "automatic_processing_started": False,
        "videos": [probe_labeling_video(mission, mission_dir, video.id) for video in mission.videos],
    }


def frame_reference(video: StoredVideo, frame_index: int, fps: float):
    timestamp_ms = round(frame_index / fps * 1000)
    value = f"{video.sha256}:{frame_index}:{timestamp_ms}".encode("utf-8")
    return timestamp_ms, hashlib.sha256(value).hexdigest()
