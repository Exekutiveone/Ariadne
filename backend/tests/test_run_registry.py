import json

from fastapi.testclient import TestClient

from backend.app import main
from backend.app.run_registry import registry_path
from backend.app.storage import MissionStore


def _mission(root, mission_id: str, name: str, videos: list[tuple[str, str | None]]):
    mission_dir = root / mission_id
    (mission_dir / "videos").mkdir(parents=True)
    entries = []
    for video_id, category in videos:
        (mission_dir / "videos" / f"{video_id}.mov").write_bytes(b"video-bytes")
        entries.append(
            {
                "direction": "A_TO_B",
                "orientation": "LANDSCAPE",
                "terrain_category": category,
                "id": video_id,
                "original_name": f"{video_id}.MOV",
                "content_type": "video/quicktime",
                "size_bytes": 11,
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
                "created_at": "2026-08-04T08:00:00Z",
                "videos": entries,
                "schema_version": "1.0",
            }
        ),
        encoding="utf-8",
    )


def _client(tmp_path, monkeypatch):
    root = tmp_path / "missions"
    root.mkdir()
    _mission(root, "mission-a", "Waldlauf", [("video-a1", "walduntergrund"), ("video-a2", None)])
    monkeypatch.setattr(main, "store", MissionStore(root))
    return TestClient(main.app), root


def _run(listing, video_id: str):
    return next(item for item in listing["runs"] if item["video_id"] == video_id)


def test_scan_creates_a_run_per_video_and_starts_unlabeled(tmp_path, monkeypatch):
    client, root = _client(tmp_path, monkeypatch)
    listing = client.get("/api/v1/registry/runs").json()

    assert listing["totals"]["runs"] == 2
    assert listing["scan"]["added"] == 2
    assert all(item["status"] == "unlabeled" for item in listing["runs"])
    assert _run(listing, "video-a1")["terrain_category"] == "walduntergrund"
    assert _run(listing, "video-a2")["terrain_category"] is None
    assert _run(listing, "video-a1")["run_id"] == "mission-a/video-a1"
    assert _run(listing, "video-a1")["video_available"] is True
    assert registry_path(root).is_file()


def test_a_new_recording_is_picked_up_automatically(tmp_path, monkeypatch):
    client, root = _client(tmp_path, monkeypatch)
    assert client.get("/api/v1/registry/runs").json()["totals"]["runs"] == 2

    _mission(root, "mission-b", "Schotterlauf", [("video-b1", "schotterweg")])
    listing = client.get("/api/v1/registry/runs").json()
    assert listing["scan"]["added"] == 1
    assert listing["totals"]["runs"] == 3
    assert _run(listing, "video-b1")["mission_name"] == "Schotterlauf"


def test_handwork_survives_the_next_scan(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    client.get("/api/v1/registry/runs")

    updated = client.patch(
        "/api/v1/registry/runs/mission-a/video-a2",
        json={"status": "queued_for_labeling", "note": "Kamera verwackelt ab Minute 3"},
    )
    assert updated.status_code == 200
    assert updated.json()["status_label"] == "Zum Labeln vorgemerkt"

    # Erneuter Scan frischt nur abgeleitete Felder auf.
    listing = client.get("/api/v1/registry/runs").json()
    run = _run(listing, "video-a2")
    assert run["status"] == "queued_for_labeling"
    assert run["note"] == "Kamera verwackelt ab Minute 3"
    assert listing["counts"] == {
        "unlabeled": 1,
        "queued_for_labeling": 1,
        "labeled": 0,
        "training_ready": 0,
    }


def test_terrain_category_is_written_through_to_the_mission_manifest(tmp_path, monkeypatch):
    client, root = _client(tmp_path, monkeypatch)
    client.get("/api/v1/registry/runs")

    response = client.patch(
        "/api/v1/registry/runs/mission-a/video-a2",
        json={"terrain_category": "schotterweg", "status": "labeled"},
    )
    assert response.status_code == 200
    assert response.json()["terrain_category"] == "schotterweg"

    # Keine zweite Wahrheit: mission.json traegt den neuen Wert, und damit auch
    # das Labeling und das Terrainmodell, die von dort lesen.
    manifest = MissionStore(root).get("mission-a")
    assert next(item for item in manifest.videos if item.id == "video-a2").terrain_category == "schotterweg"

    # Und der Scan spiegelt ihn unveraendert zurueck statt ihn zu ueberschreiben.
    assert _run(client.get("/api/v1/registry/runs").json(), "video-a2")["terrain_category"] == "schotterweg"


def test_a_category_changed_in_the_manifest_wins_over_the_registry_copy(tmp_path, monkeypatch):
    client, root = _client(tmp_path, monkeypatch)
    client.get("/api/v1/registry/runs")

    store = MissionStore(root)
    mission = store.get("mission-a")
    store.save(
        mission.model_copy(
            update={
                "videos": [
                    item.model_copy(update={"terrain_category": "wiese_hoch"}) if item.id == "video-a1" else item
                    for item in mission.videos
                ]
            }
        )
    )

    assert _run(client.get("/api/v1/registry/runs").json(), "video-a1")["terrain_category"] == "wiese_hoch"


def test_a_removed_mission_drops_out_of_the_registry(tmp_path, monkeypatch):
    client, root = _client(tmp_path, monkeypatch)
    client.get("/api/v1/registry/runs")

    (root / "mission-a" / "mission.json").unlink()
    listing = client.get("/api/v1/registry/runs").json()
    assert listing["totals"]["runs"] == 0
    assert listing["scan"]["removed"] == ["mission-a/video-a1", "mission-a/video-a2"]


def test_unknown_status_and_unknown_run_are_rejected(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    client.get("/api/v1/registry/runs")

    assert client.patch("/api/v1/registry/runs/mission-a/video-a1", json={"status": "erledigt"}).status_code == 422
    missing = client.patch("/api/v1/registry/runs/mission-a/video-nope", json={"status": "labeled"})
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Run nicht gefunden"


def test_listing_reports_the_available_statuses_and_category_counts(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    listing = client.get("/api/v1/registry/runs").json()

    assert [item["value"] for item in listing["statuses"]] == [
        "unlabeled",
        "queued_for_labeling",
        "labeled",
        "training_ready",
    ]
    assert listing["terrain_categories"] == [{"terrain_category": "walduntergrund", "runs": 1}]
    assert listing["totals"]["with_terrain_category"] == 1


def test_the_database_is_not_left_open_between_calls(tmp_path, monkeypatch):
    client, root = _client(tmp_path, monkeypatch)
    client.get("/api/v1/registry/runs")
    client.patch("/api/v1/registry/runs/mission-a/video-a1", json={"status": "labeled"})

    # Unter OneDrive waere eine offene Verbindung eine Sync-Sperre. Ein offenes
    # Journal wuerde als Datei danebenliegen; unter Windows liesse sich die
    # Datenbank ausserdem nicht umbenennen.
    database = registry_path(root)
    assert not list(database.parent.glob("registry.sqlite-*"))
    moved = database.with_suffix(".sqlite.moved")
    database.rename(moved)
    moved.rename(database)
