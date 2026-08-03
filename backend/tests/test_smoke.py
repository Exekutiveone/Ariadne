import json
import os

from fastapi.testclient import TestClient

from backend.app import main
from backend.app.storage import MissionStore


def test_upload_survives_store_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", MissionStore(tmp_path))
    client = TestClient(main.app)
    survey = {
        "name": "Smoke Waldbegehung",
        "start": {"lat": 48.5, "lng": 8.5},
        "end": {"lat": 48.51, "lng": 8.52},
        "route": [{"lat": 48.5, "lng": 8.5}, {"lat": 48.505, "lng": 8.51}, {"lat": 48.51, "lng": 8.52}],
        "notes": "Test",
    }
    metadata = [
        {"direction": "A_TO_B", "orientation": "LANDSCAPE", "terrain_category": "walduntergrund"},
        {"direction": "B_TO_A", "orientation": "PORTRAIT", "terrain_category": "wiese_flach"},
    ]
    response = client.post(
        "/api/v1/missions",
        data={"survey": json.dumps(survey), "video_metadata": json.dumps(metadata)},
        files=[
            ("videos", ("hin.mp4", b"video-one", "video/mp4")),
            ("videos", ("zurueck.mp4", b"video-two", "video/mp4")),
        ],
    )
    assert response.status_code == 201
    saved = response.json()
    assert saved["status"] == "READY_FOR_GOAL_2"
    assert len(saved["videos"]) == 2
    assert saved["videos"][0]["terrain_category"] == "walduntergrund"
    restarted = MissionStore(tmp_path)
    assert restarted.get(saved["id"]).videos[1].original_name == "zurueck.mp4"
    assert restarted.get(saved["id"]).videos[1].terrain_category == "wiese_flach"
    assert client.get(f"/api/v1/missions/{saved['id']}").status_code == 200


def test_can_relabel_existing_video_and_persist_it(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", MissionStore(tmp_path))
    client = TestClient(main.app)
    survey = {
        "name": "Video Inventar",
        "start": {"lat": 48.5, "lng": 8.5},
        "end": {"lat": 48.51, "lng": 8.52},
        "route": [{"lat": 48.5, "lng": 8.5}, {"lat": 48.51, "lng": 8.52}],
        "notes": "Test",
    }
    response = client.post(
        "/api/v1/missions",
        data={
            "survey": json.dumps(survey),
            "video_metadata": json.dumps([{"direction": "A_TO_B", "orientation": "LANDSCAPE"}]),
        },
        files=[("videos", ("bestand.mp4", b"video-one", "video/mp4"))],
    )
    assert response.status_code == 201
    saved = response.json()
    mission_id = saved["id"]
    video_id = saved["videos"][0]["id"]

    patch = client.patch(
        f"/api/v1/missions/{mission_id}/videos/{video_id}",
        json={"terrain_category": "walduntergrund"},
    )
    assert patch.status_code == 200
    assert patch.json()["terrain_category"] == "walduntergrund"

    restarted = MissionStore(tmp_path)
    assert restarted.get(mission_id).videos[0].terrain_category == "walduntergrund"


def test_rejects_missing_video(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", MissionStore(tmp_path))
    client = TestClient(main.app)
    response = client.post("/api/v1/missions", data={"survey": "{}", "video_metadata": "[]"})
    assert response.status_code == 422


def test_accepts_video_extension_when_browser_omits_mime(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", MissionStore(tmp_path))
    client = TestClient(main.app)
    survey = {
        "name": "MIME Fallback",
        "start": {"lat": 48.5, "lng": 8.5},
        "end": {"lat": 48.51, "lng": 8.52},
        "route": [{"lat": 48.5, "lng": 8.5}, {"lat": 48.51, "lng": 8.52}],
    }
    response = client.post(
        "/api/v1/missions",
        data={
            "survey": json.dumps(survey),
            "video_metadata": json.dumps(
                [{"direction": "A_TO_B", "orientation": "LANDSCAPE", "terrain_category": "schotterweg"}]
            ),
        },
        files=[("videos", ("aufnahme.MP4", b"video", "application/octet-stream"))],
    )
    assert response.status_code == 201


def test_onedrive_directory_rename_fallback_commits_manifest_last(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", MissionStore(tmp_path))
    client = TestClient(main.app)
    original_replace = os.replace

    def one_drive_replace(source, target):
        if str(source).startswith(str(tmp_path / ".")) and os.path.isdir(source):
            raise PermissionError(5, "OneDrive directory rename denied")
        return original_replace(source, target)

    monkeypatch.setattr("backend.app.storage.os.replace", one_drive_replace)
    survey = {
        "name": "OneDrive Test",
        "start": {"lat": 48.5, "lng": 8.5},
        "end": {"lat": 48.51, "lng": 8.52},
        "route": [{"lat": 48.5, "lng": 8.5}, {"lat": 48.51, "lng": 8.52}],
    }
    response = client.post(
        "/api/v1/missions",
        data={
            "survey": json.dumps(survey),
            "video_metadata": json.dumps(
                [{"direction": "A_TO_B", "orientation": "LANDSCAPE", "terrain_category": "klar_definierter_weg"}]
            ),
        },
        files=[("videos", ("aufnahme.mov", b"video", "video/quicktime"))],
    )
    assert response.status_code == 201
    mission_id = response.json()["id"]
    assert (tmp_path / mission_id / "mission.json").is_file()
    assert len(list((tmp_path / mission_id / "videos").iterdir())) == 1
