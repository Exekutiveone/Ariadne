import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from .models import MissionRecord, StoredVideo, SurveyPayload, VideoMeta

CHUNK = 1024 * 1024


class MissionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[MissionRecord]:
        records = []
        for path in self.root.glob("*/mission.json"):
            try:
                records.append(MissionRecord.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def get(self, mission_id: str) -> MissionRecord | None:
        path = self.root / mission_id / "mission.json"
        if not path.is_file():
            return None
        return MissionRecord.model_validate_json(path.read_text(encoding="utf-8"))

    async def create(
        self, payload: SurveyPayload, uploads: list[UploadFile], metadata: list[VideoMeta]
    ) -> MissionRecord:
        mission_id = str(uuid4())
        staging = Path(tempfile.mkdtemp(prefix=f".{mission_id}-", dir=self.root))
        final = self.root / mission_id
        try:
            videos = []
            video_dir = staging / "videos"
            video_dir.mkdir()
            for upload, meta in zip(uploads, metadata, strict=True):
                video_id = str(uuid4())
                suffix = Path(upload.filename or "video").suffix.lower()[:10]
                target = video_dir / f"{video_id}{suffix}"
                digest = hashlib.sha256()
                size = 0
                with target.open("xb") as stream:
                    while chunk := await upload.read(CHUNK):
                        stream.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                videos.append(
                    StoredVideo(
                        id=video_id,
                        original_name=upload.filename or "video",
                        content_type=upload.content_type or "application/octet-stream",
                        size_bytes=size,
                        sha256=digest.hexdigest(),
                        **meta.model_dump(),
                    )
                )
            record = MissionRecord(
                id=mission_id,
                status="READY_FOR_GOAL_2",
                created_at=datetime.now(timezone.utc),
                videos=videos,
                **payload.model_dump(),
            )
            manifest = staging / "mission.json"
            manifest.write_text(record.model_dump_json(indent=2), encoding="utf-8")
            with manifest.open("rb+") as stream:
                os.fsync(stream.fileno())
            try:
                os.replace(staging, final)
            except PermissionError:
                # OneDrive on Windows can reject an atomic directory rename even
                # on the same volume. Commit files individually and publish the
                # manifest last; readers ignore directories without mission.json.
                final.mkdir()
                final_videos = final / "videos"
                final_videos.mkdir()
                for source in (staging / "videos").iterdir():
                    os.replace(source, final_videos / source.name)
                os.replace(manifest, final / "mission.json")
                shutil.rmtree(staging, ignore_errors=True)
            return record
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            if final.exists() and not (final / "mission.json").exists():
                shutil.rmtree(final, ignore_errors=True)
            raise

    def video_path(self, mission_id: str, video_id: str) -> tuple[Path, StoredVideo] | None:
        record = self.get(mission_id)
        if not record:
            return None
        video = next((v for v in record.videos if v.id == video_id), None)
        if not video:
            return None
        matches = list((self.root / mission_id / "videos").glob(f"{video_id}.*"))
        return (matches[0], video) if matches else None
