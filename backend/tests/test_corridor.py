import numpy as np
import pytest

from backend.app.corridor import (
    DEFAULT_CLEARANCE_M,
    DEFAULT_GROUND_WIDTH_AT_BOTTOM_M,
    DEFAULT_VEHICLE_WIDTH_M,
    estimate_vanishing_point,
    evaluate_corridors,
)

WIDTH, HEIGHT = 160, 120
VANISHING = (80.0, 40.0)


def _road(bottom_half: float, vanishing=VANISHING, height: int = HEIGHT, width: int = WIDTH):
    """Trapezfoermiger Weg, der exakt auf den Fluchtpunkt zulaeuft."""
    mask = np.zeros((height, width), np.uint8)
    vanishing_x, vanishing_y = vanishing
    for row in range(int(vanishing_y) + 1, height):
        scale = (row - vanishing_y) / ((height - 1) - vanishing_y)
        half = bottom_half * scale
        left = max(0, int(round(vanishing_x - half)))
        right = min(width - 1, int(round(vanishing_x + half)))
        mask[row, left : right + 1] = 1
    return mask


def _status(result, corridor: str):
    return next(item["status"] for item in result["corridors"] if item["corridor"] == corridor)


def _rows(result, corridor: str):
    return next(item["rows"] for item in result["corridors"] if item["corridor"] == corridor)


def test_vanishing_point_is_the_intersection_of_the_path_edges():
    point = estimate_vanishing_point(_road(60).astype(bool))
    assert point["source"] == "path_edge_line_intersection"
    assert point["x"] == pytest.approx(VANISHING[0], abs=1.5)
    assert point["y"] == pytest.approx(VANISHING[1], abs=1.5)
    assert point["residual_px"] < 1.0


def test_vanishing_point_falls_back_visibly_when_there_is_no_path():
    point = estimate_vanishing_point(np.zeros((HEIGHT, WIDTH), bool))
    # Keine stille Falschaussage: die Notloesung benennt sich selbst.
    assert point["source"] == "image_center_assumption"
    assert point["rows_used"] == 0


def test_everything_above_the_vanishing_point_is_skipped():
    result = evaluate_corridors(_road(60))
    zone = result["decomposition"]["irrelevant_zone"]
    assert zone["kind"] == "above_vanishing_point"
    # Der Fluchtpunkt liegt bei Zeile 40, ausgewertet wird erst darunter.
    assert zone["first_evaluated_row"] == pytest.approx(42, abs=2)
    assert zone["rows_skipped"] == zone["first_evaluated_row"]
    assert 0.3 < zone["image_fraction_skipped"] < 0.4
    assert _rows(result, "mitte")["evaluated"] == HEIGHT - zone["first_evaluated_row"]
    triangle = result["decomposition"]["relevant_triangle"]
    assert triangle[0] == [0, HEIGHT - 1] and triangle[1] == [WIDTH - 1, HEIGHT - 1]


def test_narrow_path_leaves_only_the_middle_corridor_free():
    result = evaluate_corridors(_road(60))
    assert _status(result, "mitte") == "free"
    assert _status(result, "rechts") == "blocked"
    assert _status(result, "links") == "blocked"
    assert _rows(result, "mitte")["blocked"] == 0


def test_wide_path_opens_the_side_corridors():
    result = evaluate_corridors(_road(79))
    assert [_status(result, name) for name in ("mitte", "rechts", "links")] == ["free", "free", "free"]


def test_an_obstacle_in_the_middle_blocks_only_the_middle_corridor():
    mask = _road(79)
    # Hindernis mittig, perspektivisch mitskaliert, damit es den Mittelkorridor
    # ueber die gesamte Tiefe verdeckt.
    for row in range(45, HEIGHT):
        scale = (row - VANISHING[1]) / ((HEIGHT - 1) - VANISHING[1])
        half = max(1, int(round(20 * scale)))
        mask[row, 80 - half : 80 + half + 1] = 0
    result = evaluate_corridors(mask)
    assert _status(result, "mitte") == "blocked"
    assert _rows(result, "mitte")["blocked"] > 70
    # Die Seitenkorridore bleiben befahrbar. Einzelne gesperrte Zeilen aus der
    # Pixelrundung an der Wegkante machen sie hoechstens unsicher.
    assert _status(result, "rechts") in {"free", "uncertain"}
    assert _status(result, "links") in {"free", "uncertain"}
    assert _rows(result, "rechts")["blocked"] <= 1


def test_only_the_width_counts_not_how_far_ahead_the_corridor_stays_free():
    near, far = _road(79), _road(79)
    near[100:111, :] = 0
    far[60:71, :] = 0
    near_result, far_result = evaluate_corridors(near), evaluate_corridors(far)

    # Gleich viele gesperrte Zeilen, nur an anderer Stelle in der Tiefe: das
    # Ergebnis ist identisch, weil die Laengsdimension bewusst nicht zaehlt.
    assert _status(near_result, "mitte") == _status(far_result, "mitte") == "blocked"
    assert _rows(near_result, "mitte") == _rows(far_result, "mitte")
    assert _rows(near_result, "mitte")["blocked"] == 11
    # Und es wird auch keine Distanz oder Reichweite berichtet.
    assert "Nur Breitenprüfung" in near_result["limitations"][0]


def test_marginal_grades_make_a_corridor_uncertain_instead_of_free():
    mask = _road(79)
    confident = np.where(mask > 0, 2, 0).astype(np.uint8)
    assert _status(evaluate_corridors(mask, confident), "mitte") == "free"

    # Dieselbe Binaermaske, aber nur "knapp befahrbar" (Stufe 3): der Korridor
    # gilt nicht mehr als frei, ohne deshalb blockiert zu sein.
    marginal = np.where(mask > 0, 3, 0).astype(np.uint8)
    result = evaluate_corridors(mask, marginal)
    assert _status(result, "mitte") == "uncertain"
    assert _rows(result, "mitte")["blocked"] == 0
    assert result["graded_input"] is True


def test_a_wider_vehicle_needs_a_wider_strip():
    narrow = evaluate_corridors(_road(60), vehicle_width_m=0.8)
    wide = evaluate_corridors(_road(60), vehicle_width_m=3.4)
    assert _status(narrow, "mitte") == "free"
    assert _status(wide, "mitte") == "blocked"
    assert wide["strip"]["required_width_m"] == pytest.approx(3.5)
    assert wide["strip"]["required_width_px_at_bottom"] == pytest.approx(3.5 / 4.0 * WIDTH)


def test_calibration_and_defaults_are_reported_with_the_result():
    result = evaluate_corridors(_road(60))
    assert result["strip"]["vehicle_width_m"] == DEFAULT_VEHICLE_WIDTH_M
    assert result["strip"]["clearance_m"] == DEFAULT_CLEARANCE_M
    assert result["strip"]["ground_width_at_bottom_m"] == DEFAULT_GROUND_WIDTH_AT_BOTTOM_M
    assert [item["corridor"] for item in result["corridors"]] == ["mitte", "rechts", "links"]
    assert [item["label"] for item in result["corridors"]] == ["Mitte", "Rechts", "Links"]
    assert any("keine sicherheitsrelevante Fahrfreigabe" in item for item in result["limitations"])


def test_an_empty_mask_is_uncertain_or_blocked_but_never_silently_free():
    result = evaluate_corridors(np.zeros((HEIGHT, WIDTH), np.uint8))
    assert all(item["status"] in {"blocked", "uncertain"} for item in result["corridors"])
    assert result["decomposition"]["vanishing_point"]["source"] == "image_center_assumption"


def test_rejects_mismatched_grade_mask():
    with pytest.raises(ValueError, match="dieselbe Größe"):
        evaluate_corridors(_road(60), np.zeros((10, 10), np.uint8))


def test_rows_the_model_never_judged_are_left_out_instead_of_counted_as_blocked():
    """Realdaten-Befund vom 04.08.2026: knapp unter dem Fluchtpunkt liegen Himmel
    und Ferne, dort steht in der Abstufung ueberall Stufe 0 — "nicht bewertet".
    Das ist keine Aussage ueber Befahrbarkeit und darf kein Hindernis ergeben.
    In echten Waldframes waren so 31 von 37 gesperrten Zeilen gar kein Hindernis.
    """
    mask = _road(79)
    grades = np.where(mask > 0, 2, 0).astype(np.uint8)
    horizon = int(VANISHING[1]) + 1
    grades[horizon : horizon + 25, :] = 0

    result = evaluate_corridors(mask, grades)
    assert _status(result, "mitte") == "free"
    assert _rows(result, "mitte")["blocked"] == 0
    assert _rows(result, "mitte")["undecided"] >= 20


def test_a_judged_obstacle_is_still_blocked():
    """Gegenprobe: bewertete Pixel, die "nicht frei" sagen, bleiben ein Hindernis."""
    mask = _road(79)
    grades = np.where(mask > 0, 2, 0).astype(np.uint8)
    grades[95:117, :] = 5

    result = evaluate_corridors(mask, grades)
    assert _status(result, "mitte") == "blocked"
    assert _rows(result, "mitte")["blocked"] >= 20
