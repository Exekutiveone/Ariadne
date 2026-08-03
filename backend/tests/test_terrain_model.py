import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import main, terrain_model
from backend.app.storage import MissionStore

WIDTH, HEIGHT, FRAMES = 32, 24, 30


# Zwei optisch klar verschiedene Untergruende, damit die Tests die Trennung und
# nicht die Bildqualitaet pruefen, plus eine Mischung genau dazwischen fuer die
# Unsicherheitsmarkierung.
BASE_COLORS = {
    "schotterweg": (120, 122, 118),
    "walduntergrund": (40, 150, 55),
    "mischform": (80, 136, 86),
}


def _write_video(path, category: str, seed: int):
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (WIDTH, HEIGHT))
    base = np.array(BASE_COLORS[category], np.float32)
    for _ in range(FRAMES):
        noise = rng.normal(0, 12, size=(HEIGHT, WIDTH, 3)).astype(np.float32)
        writer.write(np.clip(base + noise, 0, 255).astype(np.uint8))
    writer.release()


def _mission(root, mission_id: str, name: str, categories: list[str], seed: int):
    mission_dir = root / mission_id
    (mission_dir / "videos").mkdir(parents=True)
    videos = []
    for index, category in enumerate(categories):
        video_id = f"{mission_id}-video-{index}"
        _write_video(mission_dir / "videos" / f"{video_id}.avi", category, seed + index)
        videos.append(
            {
                "direction": "A_TO_B",
                "orientation": "LANDSCAPE",
                "terrain_category": category,
                "id": video_id,
                "original_name": f"{video_id}.avi",
                "content_type": "video/x-msvideo",
                "size_bytes": 10,
                "sha256": "0" * 64,
            }
        )
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
                "created_at": "2026-08-03T20:00:00Z",
                "videos": videos,
                "schema_version": "1.0",
            }
        ),
        encoding="utf-8",
    )
    return videos


def _add_uncategorized_video(root, mission_id: str, video_id: str, appearance: str, seed: int):
    """Ein Video ohne Terrainkategorie: es geht nicht ins Training ein, kann aber
    vorhergesagt werden."""
    mission_dir = root / mission_id
    _write_video(mission_dir / "videos" / f"{video_id}.avi", appearance, seed)
    manifest = json.loads((mission_dir / "mission.json").read_text(encoding="utf-8"))
    manifest["videos"].append(
        {
            "direction": "A_TO_B",
            "orientation": "LANDSCAPE",
            "terrain_category": None,
            "id": video_id,
            "original_name": f"{video_id}.avi",
            "content_type": "video/x-msvideo",
            "size_bytes": 10,
            "sha256": "0" * 64,
        }
    )
    (mission_dir / "mission.json").write_text(json.dumps(manifest), encoding="utf-8")


def _client(tmp_path, monkeypatch, per_class: int = 3):
    root = tmp_path / "missions"
    root.mkdir()
    _mission(root, "mission-schotter", "Schotterlauf", ["schotterweg"] * per_class, seed=1)
    _mission(root, "mission-wald", "Waldlauf", ["walduntergrund"] * per_class, seed=50)
    monkeypatch.setattr(main, "store", MissionStore(root))
    return TestClient(main.app)


def test_feature_count_is_pinned():
    # Aendert sich die Merkmalszahl, werden alle gespeicherten Terrainmodelle
    # ungueltig — dann muss TERRAIN_MODEL_SCHEMA_VERSION angehoben werden.
    assert terrain_model.TERRAIN_FEATURE_COUNT == 234
    image = np.zeros((24, terrain_model.TERRAIN_MODEL_WIDTH, 3), np.uint8)
    assert terrain_model._frame_descriptor(image).shape == (terrain_model.TERRAIN_FEATURE_COUNT,)


def test_dashboard_lists_categorized_and_uncategorized_videos(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, per_class=2)
    dataset = client.get("/api/v1/terrain-model/dashboard").json()["dataset"]
    assert dataset["totals"] == {
        "categorized_videos": 4,
        "uncategorized_videos": 0,
        "classes": 2,
        "missions": 2,
    }
    assert [item["terrain_category"] for item in dataset["classes"]] == ["schotterweg", "walduntergrund"]
    assert dataset["label_source"] == "video_terrain_category_inherited_by_all_frames"


def test_training_splits_by_video_and_never_reuses_a_video(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, per_class=3)
    response = client.post("/api/v1/terrain-model/train", json={"frame_stride": 5, "confidence_threshold": 0.6})
    assert response.status_code == 201
    result = response.json()

    assert result["classes"] == ["schotterweg", "walduntergrund"]
    split = result["split"]
    assert split["strategy"] == "grouped_by_video_id"
    assert split["random_frame_split_used"] is False
    parts = [split["train"]["video_ids"], split["validation"]["video_ids"], split["test"]["video_ids"]]
    assert sum(len(part) for part in parts) == 6
    assert len({video_id for part in parts for video_id in part}) == 6
    assert split["test"]["classes"] == ["schotterweg", "walduntergrund"]
    assert result["validation_metrics"]["accuracy"] == 1.0
    assert result["test_metrics"]["accuracy"] == 1.0
    assert result["dataset"]["frame_stride"] == 5


def test_test_part_is_dropped_when_a_class_has_too_few_videos(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, per_class=2)
    result = client.post("/api/v1/terrain-model/train", json={"frame_stride": 5}).json()
    assert result["split"]["test"] is None
    assert result["test_metrics"] is None
    assert any("Testteil" in note for note in result["split"]["notes"])
    assert result["validation_metrics"]["accuracy"] == 1.0


def test_training_rejects_a_single_terrain_category(tmp_path, monkeypatch):
    root = tmp_path / "missions"
    root.mkdir()
    _mission(root, "mission-schotter", "Schotterlauf", ["schotterweg"] * 3, seed=1)
    monkeypatch.setattr(main, "store", MissionStore(root))
    client = TestClient(main.app)
    response = client.post("/api/v1/terrain-model/train", json={"frame_stride": 5})
    assert response.status_code == 409
    assert "mindestens zwei" in response.json()["detail"]


def test_every_training_run_is_kept_and_only_the_pointer_moves(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, per_class=2)
    first = client.post("/api/v1/terrain-model/train", json={"frame_stride": 10}).json()
    second = client.post("/api/v1/terrain-model/train", json={"frame_stride": 5}).json()
    assert first["run_id"] != second["run_id"]

    runs = tmp_path / "global_models" / "terrain_model" / "runs"
    assert {path.name for path in runs.iterdir()} == {first["run_id"], second["run_id"]}
    assert (
        json.loads((runs / first["run_id"] / "result.json").read_text(encoding="utf-8"))["dataset"]["frame_stride"]
        == 10
    )

    listing = client.get("/api/v1/terrain-model/dashboard").json()
    assert listing["model"]["run_id"] == second["run_id"]
    assert [item["run_id"] for item in listing["runs"]["training_runs"]] == [second["run_id"], first["run_id"]]
    assert [item["active"] for item in listing["runs"]["training_runs"]] == [True, False]


def test_frame_prediction_returns_class_and_confidence(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, per_class=2)
    client.post("/api/v1/terrain-model/train", json={"frame_stride": 5, "confidence_threshold": 0.6})
    response = client.get("/api/v1/terrain-model/predict/mission-wald/mission-wald-video-1/4")
    assert response.status_code == 200
    frame = response.json()
    assert frame["top_category"] == "walduntergrund"
    assert frame["predicted_category"] == "walduntergrund"
    assert frame["uncertain"] is False
    assert 0 <= frame["confidence"] <= 1
    assert pytest.approx(sum(frame["scores"].values()), abs=1e-3) == 1.0
    assert frame["video_terrain_category"] == "walduntergrund"


def test_prediction_without_a_trained_model_is_a_clear_conflict(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, per_class=2)
    response = client.get("/api/v1/terrain-model/predict/mission-wald/mission-wald-video-0/2")
    assert response.status_code == 409
    assert response.json()["detail"] == "Es wurde noch kein Terrainmodell trainiert"


def test_every_video_prediction_is_stored_as_its_own_run(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, per_class=2)
    client.post("/api/v1/terrain-model/train", json={"frame_stride": 5})

    confident = client.post(
        "/api/v1/terrain-model/predict-video/mission-schotter/mission-schotter-video-0",
        json={"frame_stride": 5, "confidence_threshold": 0.6},
    )
    assert confident.status_code == 201
    run = confident.json()
    assert run["summary"]["dominant_category"] == "schotterweg"
    assert run["summary"]["frames"] == len(run["frames"]) > 0
    assert all(frame["uncertain"] is False for frame in run["frames"])
    assert all(frame["predicted_category"] == "schotterweg" for frame in run["frames"])

    second = client.post(
        "/api/v1/terrain-model/predict-video/mission-schotter/mission-schotter-video-0",
        json={"frame_stride": 5},
    ).json()
    assert second["run_id"] != run["run_id"]

    predictions = tmp_path / "global_models" / "terrain_model" / "predictions"
    assert {path.name for path in predictions.iterdir()} == {run["run_id"], second["run_id"]}
    stored = client.get(f"/api/v1/terrain-model/predictions/{run['run_id']}")
    assert stored.status_code == 200
    assert stored.json()["summary"] == run["summary"]


def test_frames_below_the_confidence_threshold_stay_unassigned(tmp_path, monkeypatch):
    root = tmp_path / "missions"
    root.mkdir()
    _mission(root, "mission-schotter", "Schotterlauf", ["schotterweg"] * 2, seed=1)
    _mission(root, "mission-wald", "Waldlauf", ["walduntergrund"] * 2, seed=50)
    _add_uncategorized_video(root, "mission-wald", "mission-wald-mix", "mischform", seed=900)
    monkeypatch.setattr(main, "store", MissionStore(root))
    client = TestClient(main.app)
    client.post("/api/v1/terrain-model/train", json={"frame_stride": 5})

    strict = client.post(
        "/api/v1/terrain-model/predict-video/mission-wald/mission-wald-mix",
        json={"frame_stride": 5, "confidence_threshold": 0.99},
    ).json()
    uncertain = [frame for frame in strict["frames"] if frame["uncertain"]]
    assert uncertain, "Ein Untergrund zwischen beiden Klassen muss unsichere Frames erzeugen"
    # Unsicher heisst: keine verbindliche Zuweisung, aber die beste Vermutung
    # samt Konfidenz bleibt sichtbar.
    assert all(frame["predicted_category"] is None for frame in uncertain)
    assert all(frame["top_category"] in strict["classes"] for frame in uncertain)
    assert all(frame["confidence"] < 0.99 for frame in uncertain)
    assert strict["summary"]["uncertain_frames"] == len(uncertain)
    assert strict["video_terrain_category"] is None


def test_relabeling_a_video_relabels_all_its_frames(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, per_class=3)
    before = client.post("/api/v1/terrain-model/train", json={"frame_stride": 10}).json()
    schotter_frames = next(
        item["frames"] for item in before["dataset"]["videos"] if item["video_id"] == "mission-schotter-video-0"
    )

    patch = client.patch(
        "/api/v1/missions/mission-schotter/videos/mission-schotter-video-0",
        json={"terrain_category": "wiese_hoch"},
    )
    assert patch.status_code == 200

    after = client.post("/api/v1/terrain-model/train", json={"frame_stride": 10}).json()
    relabeled = next(item for item in after["dataset"]["videos"] if item["video_id"] == "mission-schotter-video-0")
    # Dasselbe Video, dieselben Frames, nur die geerbte Klasse ist eine andere:
    # es gibt kein Frame, das der alten Kategorie zugeordnet bleibt.
    assert relabeled["terrain_category"] == "wiese_hoch"
    assert relabeled["frames"] == schotter_frames
    assert after["classes"] == ["schotterweg", "walduntergrund", "wiese_hoch"]
    assert all(
        item["terrain_category"] != "schotterweg"
        for item in after["dataset"]["videos"]
        if item["video_id"] == "mission-schotter-video-0"
    )
