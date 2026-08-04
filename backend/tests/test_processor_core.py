import hashlib

import numpy as np
import pytest

from backend.app.models import Coordinate
from backend.app.processor import (
    haversine,
    interpolate,
    route_length,
    sha256,
    sharpness,
    video_path,
    visual_features,
)

A = Coordinate(lat=48.73, lng=9.28)
B = Coordinate(lat=48.74, lng=9.28)


def test_interpolation_walks_the_real_route_not_the_straight_line():
    """Bei mehr als zwei Wegpunkten zaehlt die tatsaechlich gelaufene Strecke.

    Ein Umweg nach Osten und zurueck darf die Haelfte der Route nicht dorthin
    legen, wo die Luftlinie sie haette.
    """
    detour = [A, Coordinate(lat=48.735, lng=9.30), B]
    midpoint = interpolate(detour, 0.5)
    straight = interpolate([A, B], 0.5)

    assert midpoint["lng"] > straight["lng"] + 0.005
    assert route_length(detour) > haversine(A, B) * 1.2


def test_interpolation_hits_both_ends_exactly():
    route = [A, Coordinate(lat=48.735, lng=9.29), B]
    start, end = interpolate(route, 0.0), interpolate(route, 1.0)
    assert (start["lat"], start["lng"]) == pytest.approx((A.lat, A.lng))
    assert (end["lat"], end["lng"]) == pytest.approx((B.lat, B.lng), abs=1e-9)


def test_a_route_that_stands_still_does_not_divide_by_zero():
    # Zwei identische Punkte kommen vor, wenn die Aufnahme an einer Stelle pausiert.
    stationary = [A, A, A]
    assert route_length(stationary) == 0
    assert interpolate(stationary, 0.5)["lat"] == pytest.approx(A.lat)


def test_haversine_is_symmetric_and_matches_a_known_distance():
    # Ein Zehntel Breitengrad sind rund 11,1 km.
    assert haversine(A, B) == pytest.approx(haversine(B, A))
    assert haversine(A, Coordinate(lat=48.83, lng=9.28)) == pytest.approx(11119, rel=0.01)


def test_sha256_matches_the_reference_implementation(tmp_path):
    path = tmp_path / "video.mov"
    payload = b"x" * (3 * 1024 * 1024 + 17)  # ueber die 1-MB-Blockgrenze hinaus
    path.write_bytes(payload)
    assert sha256(path) == hashlib.sha256(payload).hexdigest()


def test_sharpness_separates_a_blurred_frame_from_a_crisp_one():
    rng = np.random.default_rng(7)
    crisp = rng.integers(0, 255, size=(60, 80, 3), dtype=np.uint8)
    flat = np.full((60, 80, 3), 128, np.uint8)
    assert sharpness(crisp) > sharpness(flat)
    assert sharpness(flat) == pytest.approx(0, abs=1e-6)


def test_visual_features_are_stable_for_the_same_frame():
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    assert visual_features(frame) == visual_features(frame)


def test_video_path_reports_the_missing_id_instead_of_returning_nothing(tmp_path):
    (tmp_path / "videos").mkdir()
    with pytest.raises(FileNotFoundError, match="fehlt-hier"):
        video_path(tmp_path, "fehlt-hier")


def test_video_path_finds_the_file_whatever_the_extension(tmp_path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "abc.MOV").write_bytes(b"x")
    assert video_path(tmp_path, "abc").name == "abc.MOV"
