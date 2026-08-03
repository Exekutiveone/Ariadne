from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Coordinate(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)


class Pause(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    note: str = ""

    @model_validator(mode="after")
    def chronological(self):
        if self.end_seconds < self.start_seconds:
            raise ValueError("Pausenende muss nach dem Pausenbeginn liegen")
        return self


class SurveyPayload(BaseModel):
    name: str = Field(min_length=3, max_length=120)
    start: Coordinate
    end: Coordinate
    route: list[Coordinate] = Field(min_length=2)
    movement_start: str | None = None
    movement_end: str | None = None
    pauses: list[Pause] = []
    notes: str = Field(default="", max_length=5000)


class VideoMeta(BaseModel):
    direction: Literal["A_TO_B", "B_TO_A"]
    orientation: Literal["PORTRAIT", "LANDSCAPE"]


class StoredVideo(VideoMeta):
    id: str
    original_name: str
    content_type: str
    size_bytes: int
    sha256: str


class MissionRecord(SurveyPayload):
    id: str
    status: Literal["READY_FOR_GOAL_2"]
    created_at: datetime
    videos: list[StoredVideo]
    schema_version: Literal["1.0"] = "1.0"


class GroundTruthMask(BaseModel):
    width: int = Field(ge=8, le=1024)
    height: int = Field(ge=8, le=2048)
    rle: list[int] = Field(min_length=2, max_length=1_000_000)

    @model_validator(mode="after")
    def valid_rle(self):
        if len(self.rle) % 2:
            raise ValueError("RLE muss aus Wert-/Längenpaaren bestehen")
        total = 0
        for index in range(0, len(self.rle), 2):
            value, length = self.rle[index], self.rle[index + 1]
            if value not in {0, 1, 2, 3}:
                raise ValueError("Ground-Truth-Wert muss 0, 1, 2 oder 3 sein")
            if length <= 0:
                raise ValueError("RLE-Längen müssen positiv sein")
            total += length
        if total != self.width * self.height:
            raise ValueError("RLE-Länge passt nicht zur Maskengröße")
        return self


class GroundTruthPolygon(BaseModel):
    id: str = Field(default="path-1", min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    class_id: Literal["traversable"] = "traversable"
    points: list[tuple[float, float]] = Field(min_length=3, max_length=500)

    @model_validator(mode="after")
    def normalized_and_distinct(self):
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in self.points):
            raise ValueError("Polygonpunkte müssen auf das Videobild normiert sein")
        if len({(round(x, 8), round(y, 8)) for x, y in self.points}) < 3:
            raise ValueError("Ein Polygon benötigt mindestens drei unterschiedliche Punkte")
        return self


class GroundTruthAnnotationInput(BaseModel):
    timestamp_ms: int = Field(ge=0)
    source_frame_hash: str | None = Field(default=None, min_length=12, max_length=128, pattern=r"^[a-fA-F0-9]+$")
    mask: GroundTruthMask | None = None
    polygons: list[GroundTruthPolygon] = Field(default_factory=list, max_length=20)
    status: Literal["draft", "confirmed", "skipped"] = "draft"
    annotator: str = Field(default="human", min_length=1, max_length=80)
    notes: str = Field(default="", max_length=1000)


class PathRefinementInput(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    expected_kind: Literal["missed_label", "invented_path"]
    action: Literal["accept_model"] = "accept_model"
