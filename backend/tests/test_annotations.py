import json

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.storage import MissionStore

MISSION_ID = "mission-ground-truth"
VIDEO_ID = "video-ground-truth"
FRAME_HASH = "a1b2c3d4e5f67890abcd"


def _client(tmp_path, monkeypatch):
    mission_dir = tmp_path / MISSION_ID
    mission_dir.mkdir()
    video_dir = mission_dir / "videos"
    video_dir.mkdir()
    writer = cv2.VideoWriter(str(video_dir / f"{VIDEO_ID}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 12))
    for index in range(20):
        writer.write(np.full((12, 16, 3), index * 8, dtype=np.uint8))
    writer.release()
    (mission_dir / "mission.json").write_text(
        json.dumps(
            {
                "name": "Ground Truth Test",
                "start": {"lat": 48.73, "lng": 9.28},
                "end": {"lat": 48.74, "lng": 9.27},
                "route": [{"lat": 48.73, "lng": 9.28}, {"lat": 48.74, "lng": 9.27}],
                "movement_start": None,
                "movement_end": None,
                "pauses": [],
                "notes": "",
                "id": MISSION_ID,
                "status": "READY_FOR_GOAL_2",
                "created_at": "2026-08-02T20:00:00Z",
                "videos": [
                    {
                        "direction": "A_TO_B",
                        "orientation": "LANDSCAPE",
                        "terrain_category": "schotterweg",
                        "id": VIDEO_ID,
                        "original_name": "test.mp4",
                        "content_type": "video/mp4",
                        "size_bytes": 10,
                        "sha256": "0" * 64,
                    }
                ],
                "schema_version": "1.0",
            }
        ),
        encoding="utf-8",
    )
    run_dir = mission_dir / "derived" / "segmentation_runs" / "run-1"
    run_dir.mkdir(parents=True)
    (mission_dir / "derived" / "segmentation_current.json").write_text(
        json.dumps({"run_id": "run-1"}), encoding="utf-8"
    )
    (run_dir / "annotation_frames.json").write_text(
        json.dumps(
            {
                VIDEO_ID: {
                    "12": {
                        "timestamp_ms": 400,
                        "source_frame_hash": FRAME_HASH,
                        "mask_width": 8,
                        "mask_height": 8,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "store", MissionStore(tmp_path))
    return TestClient(main.app)


def _payload(**changes):
    payload = {
        "timestamp_ms": 400,
        "source_frame_hash": FRAME_HASH,
        "mask": {"width": 8, "height": 8, "rle": [1, 32, 2, 16, 3, 8, 0, 8]},
        "status": "confirmed",
        "annotator": "Simon",
        "notes": "manuell geprüft",
    }
    payload.update(changes)
    return payload


def test_ground_truth_roundtrip_and_listing_are_persistent(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    url = f"/api/v1/missions/{MISSION_ID}/ground-truth/{VIDEO_ID}/12"

    saved = client.put(url, json=_payload())

    assert saved.status_code == 200
    assert saved.json()["status"] == "confirmed"
    assert saved.json()["statistics"]["labelled_pixels"] == 56
    assert client.get(url).json()["mask"]["rle"] == [1, 32, 2, 16, 3, 8, 0, 8]
    listing = client.get(f"/api/v1/missions/{MISSION_ID}/ground-truth?video_id={VIDEO_ID}").json()
    assert listing["counts"] == {"total": 1, "draft": 0, "confirmed": 1, "skipped": 0}
    assert MissionStore(tmp_path).get(MISSION_ID) is not None
    assert (tmp_path / MISSION_ID / "ground_truth" / VIDEO_ID / "000000012.json").is_file()


def test_ground_truth_rejects_wrong_source_and_empty_confirmation(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    url = f"/api/v1/missions/{MISSION_ID}/ground-truth/{VIDEO_ID}/12"

    wrong_hash = client.put(url, json=_payload(source_frame_hash="f" * 20))
    empty = client.put(url, json=_payload(mask={"width": 8, "height": 8, "rle": [0, 64]}))

    assert wrong_hash.status_code == 409
    assert "Hash" in wrong_hash.json()["detail"]
    assert empty.status_code == 409
    assert "markierte Pixel" in empty.json()["detail"]


def test_ground_truth_validates_rle_shape(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    url = f"/api/v1/missions/{MISSION_ID}/ground-truth/{VIDEO_ID}/12"

    response = client.put(url, json=_payload(mask={"width": 8, "height": 8, "rle": [1, 63]}))

    assert response.status_code == 422


def test_original_video_frames_accept_editable_polygons_and_skips(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    manifest = client.get(f"/api/v1/missions/{MISSION_ID}/labeling/videos")

    assert manifest.status_code == 200
    assert manifest.json()["automatic_processing_started"] is False
    assert manifest.json()["videos"][0]["total_frames"] == 20
    assert manifest.json()["videos"][0]["fps"] == 10
    assert manifest.json()["videos"][0]["terrain_category"] == "schotterweg"

    polygon = {"id": "path-1", "class_id": "traversable", "points": [[0.1, 0.8], [0.5, 0.35], [0.9, 0.8]]}
    url = f"/api/v1/missions/{MISSION_ID}/ground-truth/{VIDEO_ID}/5"
    saved = client.put(
        url,
        json={
            "timestamp_ms": 500,
            "polygons": [polygon],
            "status": "confirmed",
            "annotator": "Simon",
            "notes": "Originalframe",
        },
    )

    assert saved.status_code == 200
    assert saved.json()["schema_version"] == "3.0"
    # Ohne Angabe bleibt ein Polygon, was es vor der Mehrklassen-Ontologie war:
    # eine befahrbare, von Hand gesetzte, sichere Fläche.
    assert saved.json()["polygons"][0]["class_id"] == "traversable"
    assert saved.json()["polygons"][0]["certainty"] == "certain"
    assert saved.json()["polygons"][0]["origin"] == "manual"
    assert saved.json()["polygons"][0]["hard_negative"] is False
    assert saved.json()["polygons"][0]["points"] == polygon["points"]
    assert len(saved.json()["source_frame_hash"]) == 64
    assert saved.json()["statistics"]["polygon_count"] == 1
    geometry_listing = client.get(
        f"/api/v1/missions/{MISSION_ID}/ground-truth?video_id={VIDEO_ID}&include_geometry=true"
    ).json()
    assert geometry_listing["items"][0]["polygons"][0]["points"] == polygon["points"]

    skipped = client.put(
        f"/api/v1/missions/{MISSION_ID}/ground-truth/{VIDEO_ID}/6",
        json={
            "timestamp_ms": 600,
            "polygons": [],
            "status": "skipped",
            "annotator": "Simon",
            "notes": "unscharf",
        },
    )
    assert skipped.status_code == 200
    assert skipped.json()["status"] == "skipped"
    assert (
        client.get(f"/api/v1/missions/{MISSION_ID}/ground-truth?video_id={VIDEO_ID}").json()["counts"]["skipped"] == 1
    )


def test_polygon_edit_is_frame_local_and_annotation_can_be_deleted(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    first = {"id": "path-1", "class_id": "traversable", "points": [[0.1, 0.9], [0.5, 0.4], [0.9, 0.9]]}
    changed = {"id": "path-1", "class_id": "traversable", "points": [[0.2, 0.9], [0.5, 0.3], [0.8, 0.9], [0.5, 0.75]]}
    base = {"status": "confirmed", "annotator": "Simon", "notes": ""}

    client.put(
        f"/api/v1/missions/{MISSION_ID}/ground-truth/{VIDEO_ID}/5",
        json=base | {"timestamp_ms": 500, "polygons": [first]},
    )
    client.put(
        f"/api/v1/missions/{MISSION_ID}/ground-truth/{VIDEO_ID}/6",
        json=base | {"timestamp_ms": 600, "polygons": [changed]},
    )

    assert (
        client.get(f"/api/v1/missions/{MISSION_ID}/ground-truth/{VIDEO_ID}/5").json()["polygons"][0]["points"]
        == first["points"]
    )
    assert (
        client.get(f"/api/v1/missions/{MISSION_ID}/ground-truth/{VIDEO_ID}/6").json()["polygons"][0]["points"]
        == changed["points"]
    )
    assert client.delete(f"/api/v1/missions/{MISSION_ID}/ground-truth/{VIDEO_ID}/6").status_code == 204
    assert client.get(f"/api/v1/missions/{MISSION_ID}/ground-truth/{VIDEO_ID}/6").status_code == 404
