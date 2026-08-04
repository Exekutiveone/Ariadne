import numpy as np
import pytest

from backend.app.segmentation import (
    _center_distance,
    _iou,
    _normalise_box,
    _resize,
    terrain_configuration_from_environment,
)


def test_defaults_declare_themselves_as_assumptions(monkeypatch):
    """Ohne gesetzte Umgebung sind Breite und Zuschlag Annahmen, keine Messung.

    Die Quelle steht deshalb im Ergebnis — sonst laesst sich spaeter nicht mehr
    sagen, ob 0,35 m gemessen oder geraten war.
    """
    for name in ("ARIADNE_ARGUS_WIDTH_M", "ARIADNE_ARGUS_SAFETY_MARGIN_M", "ARIADNE_TERRAIN_METRIC_CALIBRATION"):
        monkeypatch.delenv(name, raising=False)
    vehicle, near_field, calibration = terrain_configuration_from_environment()
    assert vehicle.source == "documented_default_assumption"
    assert (vehicle.width_m, vehicle.safety_margin_per_side_m) == (0.35, 0.20)
    assert calibration == "perspective_estimate"
    assert near_field == 3.2


def test_a_configured_width_is_marked_as_coming_from_the_environment(monkeypatch):
    monkeypatch.setenv("ARIADNE_ARGUS_WIDTH_M", "0.8")
    vehicle, _, _ = terrain_configuration_from_environment()
    assert vehicle.width_m == 0.8
    assert vehicle.source == "environment"


def test_blank_values_count_as_unset_rather_than_as_zero(monkeypatch):
    # Ein leeres Env-Feld entsteht beim Setzen ohne Wert und darf die Vorgabe
    # nicht auf null ziehen.
    monkeypatch.setenv("ARIADNE_ARGUS_WIDTH_M", "   ")
    vehicle, _, _ = terrain_configuration_from_environment()
    assert vehicle.width_m == 0.35
    assert vehicle.source == "documented_default_assumption"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("ARIADNE_ARGUS_WIDTH_M", "breit", "muss eine Zahl sein"),
        ("ARIADNE_ARGUS_WIDTH_M", "99", "zwischen"),
        ("ARIADNE_ARGUS_WIDTH_M", "0.01", "zwischen"),
        ("ARIADNE_TERRAIN_METRIC_CALIBRATION", "geschaetzt", "calibrated"),
    ],
)
def test_bad_configuration_fails_loudly_in_german(monkeypatch, name, value, message):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        terrain_configuration_from_environment()


def test_iou_is_one_for_identical_boxes_and_zero_for_disjoint_ones():
    box = (10, 10, 30, 30)
    assert _iou(box, box) == pytest.approx(1.0)
    assert _iou(box, (100, 100, 120, 120)) == pytest.approx(0.0)
    # Haelftige Ueberlappung: Schnitt 200, Vereinigung 600.
    assert _iou(box, (20, 10, 40, 30)) == pytest.approx(200 / 600)


def test_iou_survives_a_degenerate_box_without_dividing_by_zero():
    assert _iou((5, 5, 5, 5), (5, 5, 5, 5)) == pytest.approx(0.0)


def test_center_distance_measures_between_box_centres():
    assert _center_distance((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(0)
    assert _center_distance((0, 0, 10, 10), (30, 0, 40, 10)) == pytest.approx(30)


def test_resize_keeps_the_aspect_ratio():
    tall = np.zeros((900, 300, 3), np.uint8)
    resized = _resize(tall, width=640)
    assert resized.shape[1] == 640
    assert resized.shape[0] == round(900 * 640 / 300)


def test_normalised_boxes_are_fractions_of_the_frame():
    assert _normalise_box((0, 0, 320, 240), 640, 480) == [0.0, 0.0, 0.5, 0.5]
