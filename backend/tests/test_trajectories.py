import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.models import TrajectoryInput
from backend.app.storage import MissionStore
from backend.app.trajectories import (
    create_trajectory,
    delete_trajectory,
    get_trajectories,
    list_trajectories,
    update_trajectory,
)

POINTS = [(0.2, 0.9), (0.3, 0.5), (0.4, 0.1)]


def _mission(mission_id="mission-1", video_id="video-1"):
    return SimpleNamespace(id=mission_id, name="Mission 1", videos=[SimpleNamespace(id=video_id)])


def _payload(**overrides):
    values = {
        "timestamp_ms": 0,
        "points": POINTS,
        "corridor": None,
        "origin": "manual",
        "note": "",
        "annotator": "Simon",
    }
    values.update(overrides)
    return TrajectoryInput(**values)


def test_a_frame_without_any_saved_trajectory_returns_an_empty_list(tmp_path):
    assert get_trajectories(tmp_path, "video-1", 0) == []


def test_create_appends_instead_of_overwriting_so_several_lines_coexist(tmp_path):
    mission = _mission()

    first = create_trajectory(mission, tmp_path, "video-1", 0, _payload(note="erste Linie"))
    second = create_trajectory(mission, tmp_path, "video-1", 0, _payload(note="zweite Linie"))

    assert first["id"] != second["id"]
    items = get_trajectories(tmp_path, "video-1", 0)
    assert {item["id"] for item in items} == {first["id"], second["id"]}
    assert {item["note"] for item in items} == {"erste Linie", "zweite Linie"}
    assert all(item["revision"] == 1 for item in items)


def test_update_touches_only_the_targeted_trajectory(tmp_path):
    mission = _mission()
    first = create_trajectory(mission, tmp_path, "video-1", 0, _payload(note="unveraendert"))
    second = create_trajectory(mission, tmp_path, "video-1", 0, _payload(note="wird geaendert"))

    updated = update_trajectory(
        mission, tmp_path, "video-1", 0, second["id"], _payload(note="geaendert", origin="manual_edit")
    )

    assert updated["note"] == "geaendert"
    assert updated["revision"] == 2
    items = {item["id"]: item for item in get_trajectories(tmp_path, "video-1", 0)}
    assert items[first["id"]]["note"] == "unveraendert"
    assert items[first["id"]]["revision"] == 1
    assert items[second["id"]]["note"] == "geaendert"


def test_update_of_an_unknown_id_raises_lookup_error(tmp_path):
    mission = _mission()
    create_trajectory(mission, tmp_path, "video-1", 0, _payload())

    with pytest.raises(LookupError):
        update_trajectory(mission, tmp_path, "video-1", 0, "does-not-exist", _payload())


def test_delete_removes_only_the_matching_trajectory_and_keeps_the_others(tmp_path):
    mission = _mission()
    first = create_trajectory(mission, tmp_path, "video-1", 0, _payload())
    second = create_trajectory(mission, tmp_path, "video-1", 0, _payload())

    assert delete_trajectory(tmp_path, "video-1", 0, first["id"]) is True
    remaining = get_trajectories(tmp_path, "video-1", 0)
    assert [item["id"] for item in remaining] == [second["id"]]


def test_deleting_the_last_trajectory_removes_the_file_instead_of_an_empty_list(tmp_path):
    mission = _mission()
    only = create_trajectory(mission, tmp_path, "video-1", 0, _payload())

    delete_trajectory(tmp_path, "video-1", 0, only["id"])

    path = tmp_path / "trajectories" / "video-1" / "000000000.json"
    assert not path.is_file()
    assert get_trajectories(tmp_path, "video-1", 0) == []


def test_deleting_an_unknown_id_reports_false_without_touching_the_file(tmp_path):
    mission = _mission()
    create_trajectory(mission, tmp_path, "video-1", 0, _payload())

    assert delete_trajectory(tmp_path, "video-1", 0, "does-not-exist") is False
    assert len(get_trajectories(tmp_path, "video-1", 0)) == 1


def test_a_pre_schema_2_0_file_is_read_as_a_single_item_list_without_being_rewritten(tmp_path):
    # Format vor der Mehrfach-Trajektorie: ein einzelnes Objekt statt einer Liste,
    # genau wie es echte, bereits committete Trajektoriendateien enthalten.
    directory = tmp_path / "trajectories" / "video-1"
    directory.mkdir(parents=True)
    path = directory / "000000000.json"
    legacy_record = {
        "schema_version": "1.0",
        "mission_id": "mission-1",
        "video_id": "video-1",
        "frame_index": 0,
        "timestamp_ms": 0,
        "points": [[0.1, 0.9], [0.2, 0.1]],
        "corridor": None,
        "origin": "manual",
        "note": "",
        "annotator": "Simon",
        "coordinate_space": "normalized_to_original_frame",
        "revision": 1,
        "created_at": "2026-08-01T00:00:00Z",
        "updated_at": "2026-08-01T00:00:00Z",
    }
    path.write_text(json.dumps(legacy_record), encoding="utf-8")

    items = get_trajectories(tmp_path, "video-1", 0)

    assert len(items) == 1
    assert items[0]["id"] == "legacy"
    assert items[0]["points"] == [[0.1, 0.9], [0.2, 0.1]]
    # Datei bleibt im Altformat, bis tatsaechlich geschrieben wird.
    assert json.loads(path.read_text(encoding="utf-8")) == legacy_record


def test_updating_a_legacy_trajectory_migrates_the_file_to_the_list_format(tmp_path):
    directory = tmp_path / "trajectories" / "video-1"
    directory.mkdir(parents=True)
    path = directory / "000000000.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "mission_id": "mission-1",
                "video_id": "video-1",
                "frame_index": 0,
                "timestamp_ms": 0,
                "points": [[0.1, 0.9], [0.2, 0.1]],
                "corridor": None,
                "origin": "manual",
                "note": "alt",
                "annotator": "Simon",
                "coordinate_space": "normalized_to_original_frame",
                "revision": 1,
                "created_at": "2026-08-01T00:00:00Z",
                "updated_at": "2026-08-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    mission = _mission()

    update_trajectory(mission, tmp_path, "video-1", 0, "legacy", _payload(note="neu"))

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(on_disk, list)
    assert on_disk[0]["note"] == "neu"
    assert on_disk[0]["revision"] == 2


def test_list_trajectories_counts_across_legacy_and_current_files(tmp_path):
    mission = _mission()
    create_trajectory(mission, tmp_path, "video-1", 0, _payload(origin="manual"))
    create_trajectory(mission, tmp_path, "video-1", 0, _payload(origin="model_proposal"))
    legacy_dir = tmp_path / "trajectories" / "video-1"
    (legacy_dir / "000000005.json").write_text(
        json.dumps(
            {
                "video_id": "video-1",
                "frame_index": 5,
                "points": [[0.1, 0.9], [0.2, 0.1]],
                "origin": "manual_edit",
                "note": "",
                "annotator": "Simon",
                "revision": 1,
            }
        ),
        encoding="utf-8",
    )

    summary = list_trajectories(mission, tmp_path)

    assert summary["counts"]["total"] == 3
    assert summary["counts"]["manual"] == 1
    assert summary["counts"]["model_proposal"] == 1
    assert summary["counts"]["manual_edit"] == 1


def test_create_rejects_an_unknown_video(tmp_path):
    with pytest.raises(LookupError):
        create_trajectory(_mission(), tmp_path, "no-such-video", 0, _payload())


def _client(tmp_path, monkeypatch):
    root = tmp_path / "missions"
    mission_dir = root / "mission-1"
    (mission_dir / "videos").mkdir(parents=True)
    (mission_dir / "mission.json").write_text(
        json.dumps(
            {
                "name": "Mission 1",
                "start": {"lat": 48.73, "lng": 9.28},
                "end": {"lat": 48.74, "lng": 9.27},
                "route": [{"lat": 48.73, "lng": 9.28}, {"lat": 48.74, "lng": 9.27}],
                "movement_start": None,
                "movement_end": None,
                "pauses": [],
                "notes": "",
                "id": "mission-1",
                "status": "READY_FOR_GOAL_2",
                "created_at": "2026-08-04T08:00:00Z",
                "videos": [
                    {
                        "direction": "A_TO_B",
                        "orientation": "LANDSCAPE",
                        "terrain_category": None,
                        "fully_not_traversable": False,
                        "id": "video-1",
                        "original_name": "video-1.avi",
                        "content_type": "video/x-msvideo",
                        "size_bytes": 10,
                        "sha256": "0" * 64,
                    }
                ],
                "schema_version": "1.0",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "store", MissionStore(root))
    return TestClient(main.app)


def test_endpoint_round_trip_supports_several_trajectories_on_the_same_frame(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    body = {
        "timestamp_ms": 0,
        "points": [[0.2, 0.9], [0.3, 0.1]],
        "origin": "manual",
        "note": "A",
        "annotator": "Simon",
    }

    created_a = client.post("/api/v1/missions/mission-1/trajectories/video-1/0", json=body)
    created_b = client.post("/api/v1/missions/mission-1/trajectories/video-1/0", json={**body, "note": "B"})
    assert created_a.status_code == 201
    assert created_b.status_code == 201
    assert created_a.json()["id"] != created_b.json()["id"]

    listed = client.get("/api/v1/missions/mission-1/trajectories/video-1/0")
    assert listed.status_code == 200
    assert {item["note"] for item in listed.json()} == {"A", "B"}

    trajectory_id = created_a.json()["id"]
    updated = client.put(
        f"/api/v1/missions/mission-1/trajectories/video-1/0/{trajectory_id}", json={**body, "note": "A2"}
    )
    assert updated.status_code == 200
    assert updated.json()["note"] == "A2"
    assert updated.json()["revision"] == 2

    deleted = client.delete(f"/api/v1/missions/mission-1/trajectories/video-1/0/{trajectory_id}")
    assert deleted.status_code == 204
    remaining = client.get("/api/v1/missions/mission-1/trajectories/video-1/0").json()
    assert [item["note"] for item in remaining] == ["B"]


def test_endpoint_returns_404_deleting_an_id_that_does_not_exist(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)

    response = client.delete("/api/v1/missions/mission-1/trajectories/video-1/0/does-not-exist")

    assert response.status_code == 404
