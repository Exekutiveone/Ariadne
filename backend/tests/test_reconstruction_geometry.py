import math

import numpy as np
import pytest

from backend.app.models import Coordinate
from backend.app.reconstruction import _anchor, _resample, _to_geo

A = Coordinate(lat=48.73, lng=9.28)
NORTH = Coordinate(lat=48.74, lng=9.28)
EAST = Coordinate(lat=48.73, lng=9.30)


def test_forward_motion_follows_the_route_bearing():
    """Der lokale Verlauf ist lateral/vorwaerts relativ zu A->B.

    Laeuft A->B nach Norden, muss reines Vorwaerts auch nach Norden zeigen;
    laeuft die Route nach Osten, derselbe Verlauf nach Osten. Ohne die Drehung
    landete jede Rekonstruktion im gleichen Himmelsrichtung.
    """
    forward = np.array([[0.0, 0.0], [0.0, 100.0]])

    north = _to_geo(forward, A, NORTH)
    assert north[1][1] > north[0][1] + 0.0005  # Breite waechst
    assert north[1][0] == pytest.approx(north[0][0], abs=1e-6)  # Laenge bleibt

    east = _to_geo(forward, A, EAST)
    assert east[1][0] > east[0][0] + 0.0005  # Laenge waechst
    assert east[1][1] == pytest.approx(east[0][1], abs=1e-6)


def test_lateral_offset_goes_sideways_not_forward():
    lateral = np.array([[0.0, 0.0], [50.0, 0.0]])
    points = _to_geo(lateral, A, NORTH)
    # Bei einer Nordroute liegt "seitlich" auf der Laenge, nicht auf der Breite.
    assert abs(points[1][0] - points[0][0]) > 0.0005
    assert points[1][1] == pytest.approx(points[0][1], abs=1e-6)


def test_geo_conversion_starts_at_the_route_start():
    points = _to_geo(np.array([[0.0, 0.0]]), A, NORTH)
    assert (points[0][1], points[0][0]) == pytest.approx((A.lat, A.lng))


def test_resampling_keeps_the_ends_and_hits_the_requested_count():
    curve = np.array([[float(i), float(i * i) / 10] for i in range(17)])
    resampled = _resample(curve, 40)
    assert len(resampled) == 40
    assert resampled[0] == pytest.approx(curve[0])
    assert resampled[-1] == pytest.approx(curve[-1])


def test_anchoring_scales_to_the_known_route_length():
    # Der Bildverlauf hat keine Massstabsinformation; erst die bekannte
    # Routenlaenge macht daraus Meter.
    visual = np.array([[0.0, 0.0], [3.0, 4.0]])
    anchored = _anchor(visual, 250)
    assert np.linalg.norm(anchored[0]) == pytest.approx(0, abs=1e-9)
    assert np.linalg.norm(anchored[-1]) == pytest.approx(250, rel=1e-6)


def test_anchoring_keeps_the_shape_instead_of_flattening_it():
    """Ein Bogen muss ein Bogen bleiben — sonst waere die Rekonstruktion nur
    eine teuer berechnete Gerade."""
    angles = np.linspace(0, math.pi, 24)
    arc = np.column_stack([np.sin(angles) * 40, 1 - np.cos(angles)])
    anchored = _anchor(arc, 100)
    straight = np.linalg.norm(anchored[-1] - anchored[0])
    travelled = float(np.sum(np.linalg.norm(np.diff(anchored, axis=0), axis=1)))
    assert travelled > straight * 1.3


def test_anchoring_puts_the_endpoint_straight_ahead_not_off_to_the_side():
    """Regression: das Drehvorzeichen in _anchor war gedreht.

    Der Endpunkt muss seitlich auf null landen — danach skaliert _anchor die
    Vorwaertskomponente auf die bekannte Routenlaenge. Lag der Endpunkt nicht
    ohnehin geradeaus, blieb vorher ein riesiger seitlicher Versatz stehen: bei
    250 m Route 857 m zur Seite. Der alte Test traf nur den Fall, in dem die
    Drehung gar nichts tut.
    """
    for visual in ([[0.0, 0.0], [3.0, 4.0]], [[0.0, 0.0], [-8.0, 2.0]], [[0.0, 0.0], [1.0, -6.0]]):
        anchored = _anchor(np.array(visual), 250)
        assert anchored[-1][0] == pytest.approx(0, abs=1e-6), f"seitlicher Versatz bei {visual}"
        assert anchored[-1][1] == pytest.approx(250)
