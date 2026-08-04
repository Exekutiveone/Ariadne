import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app import main
from backend.app.models import OffPathIntervalInput
from backend.app.off_path_intervals import synthetic_off_path_records
from backend.app.path_dataset import synthetic_fully_not_traversable_records, synthetic_hard_negative_records
from backend.app.path_features import sample_training_pixels
from backend.app.storage import MissionStore

WIDTH, HEIGHT, FPS, FRAMES = 32, 24, 10, 40
SQUARE = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


def _write_video(path, frames: int = FRAMES, fps: int = FPS):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (WIDTH, HEIGHT))
    for index in range(frames):
        writer.write(np.full((HEIGHT, WIDTH, 3), index * 4 % 255, dtype=np.uint8))
    writer.release()


def _video_entry(video_id: str, fully_not_traversable: bool):
    return {
        "direction": "A_TO_B",
        "orientation": "LANDSCAPE",
        "terrain_category": None,
        "fully_not_traversable": fully_not_traversable,
        "id": video_id,
        "original_name": f"{video_id}.avi",
        "content_type": "video/x-msvideo",
        "size_bytes": 10,
        "sha256": "0" * 64,
    }


def _mission(root, mission_id: str, name: str, videos: list[tuple[str, bool]]):
    mission_dir = root / mission_id
    (mission_dir / "videos").mkdir(parents=True)
    entries = []
    for video_id, fully_not_traversable in videos:
        _write_video(mission_dir / "videos" / f"{video_id}.avi")
        entries.append(_video_entry(video_id, fully_not_traversable))
    (mission_dir / "mission.json").write_text(
        json.dumps(
            {
                "name": name,
                "start": {"lat": 48.73, "lng": 9.28},
                "end": {"lat": 48.74, "lng": 9.27},
                "route": [{"lat": 48.73, "lng": 9.28}, {"lat": 48.74, "lng": 9.27}],
                "movement_start": None,
                "movement_end": None,
                "pauses": [],
                "notes": "",
                "id": mission_id,
                "status": "READY_FOR_GOAL_2",
                "created_at": "2026-08-04T08:00:00Z",
                "videos": entries,
                "schema_version": "1.0",
            }
        ),
        encoding="utf-8",
    )


def _confirmed_frame(mission_dir, video_id: str, frame_index: int):
    target = mission_dir / "ground_truth" / video_id
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{frame_index:09d}.json").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "frame_index": frame_index,
                "timestamp_ms": round(frame_index / FPS * 1000),
                "status": "confirmed",
                "polygons": [{"id": "path-1", "class_id": "traversable", "points": SQUARE}],
            }
        ),
        encoding="utf-8",
    )


def _client(tmp_path, monkeypatch, missions):
    root = tmp_path / "missions"
    root.mkdir()
    for mission_id, name, videos in missions:
        _mission(root, mission_id, name, videos)
    monkeypatch.setattr(main, "store", MissionStore(root))
    return TestClient(main.app), root


# --- Off-Path-Intervalle: CRUD -------------------------------------------


def test_intervals_are_created_listed_and_deleted(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, [("m1", "Mission 1", [("v1", False)])])

    created = client.post(
        "/api/v1/missions/m1/off-path-intervals/v1",
        json={"start_ms": 500, "end_ms": 1500, "note": "im Graben", "annotator": "Simon"},
    )
    assert created.status_code == 201
    interval_id = created.json()["id"]
    assert created.json()["note"] == "im Graben"

    listed = client.get("/api/v1/missions/m1/off-path-intervals/v1").json()
    assert [item["id"] for item in listed] == [interval_id]

    deleted = client.delete(f"/api/v1/missions/m1/off-path-intervals/v1/{interval_id}")
    assert deleted.status_code == 204
    assert client.get("/api/v1/missions/m1/off-path-intervals/v1").json() == []


def test_overlapping_intervals_are_rejected_but_adjacent_ones_are_fine(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, [("m1", "Mission 1", [("v1", False)])])
    client.post("/api/v1/missions/m1/off-path-intervals/v1", json={"start_ms": 1000, "end_ms": 3000})

    overlapping = client.post("/api/v1/missions/m1/off-path-intervals/v1", json={"start_ms": 2000, "end_ms": 4000})
    assert overlapping.status_code == 409
    assert "überschneidet" in overlapping.json()["detail"]

    adjacent = client.post("/api/v1/missions/m1/off-path-intervals/v1", json={"start_ms": 3000, "end_ms": 4000})
    assert adjacent.status_code == 201


def test_deleting_an_unknown_interval_reports_not_found(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, [("m1", "Mission 1", [("v1", False)])])
    assert client.delete("/api/v1/missions/m1/off-path-intervals/v1/off-doesnotexist").status_code == 404


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"start_ms": 1000, "end_ms": 1000}, "nach dem Anfang"),
        ({"start_ms": 1000, "end_ms": 1100}, "mindestens 200 ms"),
    ],
)
def test_interval_bounds_are_validated(payload, message):
    with pytest.raises(ValidationError, match=message):
        OffPathIntervalInput(**payload)


# --- Off-Path-Intervalle: synthetische Frames -----------------------------


def test_synthetic_off_path_records_cover_the_interval_without_any_polygon(tmp_path, monkeypatch):
    client, root = _client(tmp_path, monkeypatch, [("m1", "Mission 1", [("v1", False)])])
    client.post("/api/v1/missions/m1/off-path-intervals/v1", json={"start_ms": 500, "end_ms": 1500})

    mission = MissionStore(root).get("m1")
    records = synthetic_off_path_records(mission, root / "m1", frame_stride=2)

    assert records
    assert all(record["polygons"] == [] for record in records)
    assert all(record["status"] == "confirmed" for record in records)
    # 500 ms bei 10 fps -> Frame 5, 1500 ms -> Frame 15.
    assert min(item["frame_index"] for item in records) == 5
    assert max(item["frame_index"] for item in records) <= 15


def test_a_video_without_intervals_contributes_nothing(tmp_path, monkeypatch):
    client, root = _client(tmp_path, monkeypatch, [("m1", "Mission 1", [("v1", False)])])
    mission = MissionStore(root).get("m1")
    assert synthetic_off_path_records(mission, root / "m1") == []


# --- Video-Komplettlabel ---------------------------------------------------


def test_a_video_defaults_to_not_fully_not_traversable(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, [("m1", "Mission 1", [("v1", False)])])
    mission = client.get("/api/v1/missions/m1").json()
    assert mission["videos"][0]["fully_not_traversable"] is False


def test_toggling_fully_not_traversable_does_not_touch_the_terrain_category(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, [("m1", "Mission 1", [("v1", False)])])
    client.patch("/api/v1/missions/m1/videos/v1", json={"terrain_category": "schotterweg"})

    response = client.patch("/api/v1/missions/m1/videos/v1", json={"fully_not_traversable": True})
    assert response.status_code == 200
    assert response.json()["fully_not_traversable"] is True
    assert response.json()["terrain_category"] == "schotterweg"

    # Und umgekehrt: eine reine Kategorie-Aenderung darf das Komplettlabel
    # nicht stillschweigend zuruecksetzen.
    response = client.patch("/api/v1/missions/m1/videos/v1", json={"terrain_category": "walduntergrund"})
    assert response.json()["fully_not_traversable"] is True


def test_synthetic_fully_not_traversable_records_sample_the_whole_video(tmp_path, monkeypatch):
    client, root = _client(tmp_path, monkeypatch, [("m1", "Mission 1", [("v1", True)])])
    mission = MissionStore(root).get("m1")

    records = synthetic_fully_not_traversable_records(mission, root / "m1", frame_stride=5)

    assert records
    assert all(record["polygons"] == [] for record in records)
    assert max(item["frame_index"] for item in records) < FRAMES
    assert len(records) == len(range(0, FRAMES, 5))


def test_a_video_without_the_flag_contributes_nothing(tmp_path, monkeypatch):
    client, root = _client(tmp_path, monkeypatch, [("m1", "Mission 1", [("v1", False)])])
    mission = MissionStore(root).get("m1")
    assert synthetic_fully_not_traversable_records(mission, root / "m1") == []


def test_hard_negative_records_combine_intervals_and_whole_videos(tmp_path, monkeypatch):
    client, root = _client(tmp_path, monkeypatch, [("m1", "Mission 1", [("v1", False), ("v2", True)])])
    client.post("/api/v1/missions/m1/off-path-intervals/v1", json={"start_ms": 0, "end_ms": 1000})

    mission = MissionStore(root).get("m1")
    records = synthetic_hard_negative_records(mission, root / "m1")

    assert {item["video_id"] for item in records} == {"v1", "v2"}


# --- Sampling: reine Negativframes werden nicht verworfen -----------------


def test_a_fully_negative_frame_still_contributes_negative_samples():
    """Regression: ohne den Sonderfall in sample_training_pixels waere
    min(count, len(positive)==0, len(negative)) immer 0 -- ein Off-Path- oder
    Komplettlabel-Frame haette dann NIE die gelernten Gewichte beeinflusst,
    egal wie deutlich er markiert wurde."""
    rng = np.random.default_rng(0)
    frame = {"image": np.zeros((HEIGHT, WIDTH, 3), np.uint8), "mask": np.zeros((HEIGHT, WIDTH), np.uint8)}

    samples, labels = sample_training_pixels([frame], samples_per_class_per_frame=50, rng=rng)

    assert samples
    assert np.concatenate(labels).sum() == 0


def test_a_mixed_frame_stays_balanced():
    rng = np.random.default_rng(0)
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    mask[:, : WIDTH // 2] = 1
    frame = {"image": np.zeros((HEIGHT, WIDTH, 3), np.uint8), "mask": mask}

    samples, labels = sample_training_pixels([frame], samples_per_class_per_frame=5, rng=rng)

    flat = np.concatenate(labels)
    assert flat.sum() == len(flat) - flat.sum()


def test_an_empty_frame_list_yields_no_samples():
    samples, labels = sample_training_pixels([], samples_per_class_per_frame=5, rng=np.random.default_rng(0))
    assert samples == []
    assert labels == []


# --- Training: eine Mission ganz ohne Handpolygon darf mittrainieren ------


def test_train_global_path_model_includes_a_mission_with_only_hard_negatives(tmp_path, monkeypatch):
    client, root = _client(
        tmp_path,
        monkeypatch,
        [("m1", "Mission 1", [("v1", False)]), ("m2", "Mission 2", [("v2", True)])],
    )
    mission_dir = root / "m1"
    for frame_index in range(10):
        _confirmed_frame(mission_dir, "v1", frame_index)

    response = client.post("/api/v1/path-model/global/train")
    assert response.status_code == 200
    result = response.json()

    included = {item["mission_id"]: item for item in result["dataset"]["missions"]}
    assert included["m1"]["confirmed_frames"] == 10
    assert included["m1"]["hard_negative_records"] == 0
    # Mission 2 hat kein einziges Hand-Polygon und traegt trotzdem Frames bei
    # -- genau der Zweck des Video-Komplettlabels.
    assert included["m2"]["confirmed_frames"] == 0
    assert included["m2"]["train_frames"] > 0
    assert included["m2"]["hard_negative_records"] > 0
    assert result["dataset"]["hard_negative_records_included"] == included["m2"]["hard_negative_records"]


# --- label_mode -------------------------------------------------------------


def test_ground_truth_defaults_to_linear_label_mode(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, [("m1", "Mission 1", [("v1", False)])])

    saved = client.put(
        "/api/v1/missions/m1/ground-truth/v1/0",
        json={
            "timestamp_ms": 0,
            "polygons": [{"id": "path-1", "class_id": "traversable", "points": SQUARE}],
            "status": "confirmed",
        },
    )
    assert saved.status_code == 200
    assert saved.json()["label_mode"] == "linear"


def test_shuffle_label_mode_is_persisted_and_shows_in_the_summary(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch, [("m1", "Mission 1", [("v1", False)])])

    saved = client.put(
        "/api/v1/missions/m1/ground-truth/v1/0",
        json={
            "timestamp_ms": 0,
            "polygons": [{"id": "path-1", "class_id": "traversable", "points": SQUARE}],
            "status": "confirmed",
            "label_mode": "shuffle",
        },
    )
    assert saved.json()["label_mode"] == "shuffle"

    summary = client.get("/api/v1/missions/m1/ground-truth").json()
    assert summary["items"][0]["label_mode"] == "shuffle"


def test_legacy_records_without_label_mode_count_as_linear(tmp_path, monkeypatch):
    client, root = _client(tmp_path, monkeypatch, [("m1", "Mission 1", [("v1", False)])])
    target = root / "m1" / "ground_truth" / "v1"
    target.mkdir(parents=True)
    (target / "000000000.json").write_text(
        json.dumps(
            {
                "video_id": "v1",
                "frame_index": 0,
                "timestamp_ms": 0,
                "source_frame_hash": "a" * 64,
                "status": "confirmed",
                "annotator": "Simon",
                "revision": 1,
                "updated_at": "2026-08-04T08:00:00Z",
                "statistics": {},
                "polygons": [],
            }
        ),
        encoding="utf-8",
    )

    summary = client.get("/api/v1/missions/m1/ground-truth").json()
    assert summary["items"][0]["label_mode"] == "linear"
