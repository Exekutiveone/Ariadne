import json

import pytest
from pydantic import ValidationError

from backend.app.label_tracks import video_tracks
from backend.app.models import GroundTruthPolygon, RoiProfileInput
from backend.app.roi_profiles import band_polygon, get_roi_profile, resolved_roi, save_roi_profile

SQUARE = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]


def _frame(directory, video_id, frame_index, polygons):
    target = directory / "ground_truth" / video_id
    target.mkdir(parents=True, exist_ok=True)
    (target / f"{frame_index:09d}.json").write_text(
        json.dumps({"frame_index": frame_index, "video_id": video_id, "status": "confirmed", "polygons": polygons}),
        encoding="utf-8",
    )


def test_a_track_follows_the_same_place_across_frames(tmp_path):
    for index, edit in enumerate(["new", "carried_unchanged", "carried_adjusted"]):
        _frame(
            tmp_path,
            "video-1",
            index,
            [
                {
                    "class_id": "traversable",
                    "points": SQUARE,
                    "tracking_id": "weg-a",
                    "edit": edit,
                    "carried_from_frame": index - 1 if index else None,
                }
            ],
        )
    result = video_tracks(tmp_path, "video-1")

    assert result["totals"]["tracks"] == 1
    track = result["tracks"][0]
    assert track["tracking_id"] == "weg-a"
    assert (track["first_frame"], track["last_frame"]) == (0, 2)
    assert track["frame_count"] == 3
    assert track["adjusted_count"] == 1
    assert [item["edit"] for item in track["frames"]] == ["new", "carried_unchanged", "carried_adjusted"]


def test_a_deletion_is_derived_from_absence_not_stored(tmp_path):
    """Kein zweites Verzeichnis fuer Loeschungen: verschwindet die tracking_id
    im naechsten gelabelten Frame, ist das die Loeschung."""
    _frame(tmp_path, "video-1", 0, [{"class_id": "tree", "points": SQUARE, "tracking_id": "baum-a"}])
    _frame(tmp_path, "video-1", 1, [{"class_id": "tree", "points": SQUARE, "tracking_id": "baum-a"}])
    _frame(tmp_path, "video-1", 2, [{"class_id": "traversable", "points": SQUARE, "tracking_id": "weg-a"}])

    result = video_tracks(tmp_path, "video-1")
    tracks = {item["tracking_id"]: item for item in result["tracks"]}
    assert tracks["baum-a"]["ended_at_frame"] == 2
    # Die zuletzt gesehene Spur endet nicht — sie laeuft nur noch.
    assert tracks["weg-a"]["ended_at_frame"] is None
    assert result["totals"]["ended_tracks"] == 1


def test_a_class_change_within_a_track_is_recorded(tmp_path):
    _frame(tmp_path, "video-1", 0, [{"class_id": "traversable", "points": SQUARE, "tracking_id": "stelle-a"}])
    _frame(tmp_path, "video-1", 1, [{"class_id": "restricted", "points": SQUARE, "tracking_id": "stelle-a"}])

    track = video_tracks(tmp_path, "video-1")["tracks"][0]
    assert track["class_changes"] == [{"frame_index": 1, "from": "traversable", "to": "restricted"}]


def test_polygons_without_a_tracking_id_are_counted_not_swallowed(tmp_path):
    # Die 276 Bestandsdateien haben keine tracking_id. Sie sollen sichtbar
    # bleiben, statt aus der Statistik zu fallen.
    _frame(tmp_path, "video-1", 0, [{"class_id": "traversable", "points": SQUARE}])
    result = video_tracks(tmp_path, "video-1")
    assert result["totals"]["tracks"] == 0
    assert result["totals"]["untracked_polygons"] == 1


def test_a_polygon_without_temporal_fields_keeps_the_old_meaning():
    polygon = GroundTruthPolygon(points=SQUARE)
    assert polygon.tracking_id is None
    assert polygon.carried_from_frame is None
    assert polygon.edit == "new"


def test_the_ignore_band_covers_the_requested_share_of_the_image():
    top = band_polygon("top", 0.2)
    assert top["class_id"] == "roi_ignore"
    assert [point[1] for point in top["points"]] == [0.0, 0.0, 0.2, 0.2]

    bottom = band_polygon("bottom", 0.1)
    assert [round(point[1], 5) for point in bottom["points"]] == [0.9, 0.9, 1.0, 1.0]

    with pytest.raises(ValueError, match="zwischen 0 und 1"):
        band_polygon("top", 1.5)


def test_bands_that_would_swallow_the_whole_image_are_rejected():
    with pytest.raises(ValidationError, match="ganze Bild ausschließen"):
        RoiProfileInput(top_ignore_fraction=0.6, bottom_ignore_fraction=0.5)


def test_a_profile_without_a_file_offers_a_starting_point(tmp_path):
    profile = resolved_roi(tmp_path, "video-1")
    assert profile["revision"] == 0
    assert profile["roi"] == []
    # Startwerte als Vorschlag, nicht als Vorgabe — die Kamerahoehe entscheidet.
    assert profile["suggested"] == {"top_ignore_fraction": 0.2, "bottom_ignore_fraction": 0.1}


def test_saving_a_profile_keeps_it_a_suggestion(tmp_path):
    class Mission:
        id = "mission-1"
        videos = [type("Video", (), {"id": "video-1"})()]

    payload = RoiProfileInput(
        top_ignore_fraction=0.25,
        roi=[GroundTruthPolygon(**band_polygon("top", 0.25))],
        note="Kamera hoch montiert",
    )
    saved = save_roi_profile(Mission(), tmp_path, "video-1", payload)

    assert saved["revision"] == 1
    assert saved["top_ignore_fraction"] == 0.25
    # Das Profil entscheidet nichts: was gilt, steht am jeweiligen Frame.
    assert saved["applies_as"] == "suggestion_only_frame_labels_decide"
    assert get_roi_profile(tmp_path, "video-1")["note"] == "Kamera hoch montiert"

    again = save_roi_profile(Mission(), tmp_path, "video-1", payload)
    assert again["revision"] == 2
    assert again["created_at"] == saved["created_at"]


def test_one_place_carried_forward_stays_one_track(tmp_path):
    """Regression: die Spur-ID muss ueber den Framewechsel dieselbe bleiben.

    Im Labeler erzeugte `currentShape()` die ID beim Speichern und das
    Weitertragen noch einmal eine zweite — dieselbe Stelle bekam damit zwei
    verschiedene IDs. Ergebnis waren zwei Spuren statt einer, und die erste
    galt faelschlich als beendet, obwohl sie fortlief.
    """
    _frame(
        tmp_path, "video-1", 0, [{"class_id": "traversable", "points": SQUARE, "tracking_id": "weg-a", "edit": "new"}]
    )
    _frame(
        tmp_path,
        "video-1",
        10,
        [
            {
                "class_id": "traversable",
                "points": SQUARE,
                "tracking_id": "weg-a",
                "edit": "carried_unchanged",
                "carried_from_frame": 0,
            }
        ],
    )

    result = video_tracks(tmp_path, "video-1")
    assert result["totals"]["tracks"] == 1
    assert result["totals"]["ended_tracks"] == 0
    track = result["tracks"][0]
    assert (track["first_frame"], track["last_frame"]) == (0, 10)
    assert track["ended_at_frame"] is None
    assert track["frames"][1]["carried_from_frame"] == 0
