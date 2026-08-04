import numpy as np
import pytest

from backend.app.corridor import (
    DEFAULT_CLEARANCE_M,
    DEFAULT_GROUND_WIDTH_AT_BOTTOM_M,
    DEFAULT_VEHICLE_WIDTH_M,
    evaluate_corridors,
)

WIDTH, HEIGHT = 160, 120


def _road(half_width: float, height: int = HEIGHT, width: int = WIDTH):
    """Rechteckiger, sichtbarer Weg im festen Nahfeld."""
    mask = np.zeros((height, width), np.uint8)
    start = int(round(height * 0.42))
    center = width // 2
    mask[start:, max(0, center - int(half_width)) : min(width, center + int(half_width) + 1)] = 1
    return mask


def _status(result, corridor: str):
    return next(item["status"] for item in result["corridors"] if item["corridor"] == corridor)


def _rows(result, corridor: str):
    return next(item["rows"] for item in result["corridors"] if item["corridor"] == corridor)


def test_result_has_a_fixed_near_field_region_and_no_perspective_decomposition():
    result = evaluate_corridors(_road(60))
    region = result["region"]
    assert region["kind"] == "fixed_near_field_band"
    assert region["first_evaluated_row"] == pytest.approx(round(HEIGHT * 0.42))
    assert region["evaluated_rows"] == HEIGHT - region["first_evaluated_row"]
    assert "decomposition" not in result


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
    mask[int(HEIGHT * 0.42) :, 54:107] = 0
    result = evaluate_corridors(mask)
    assert _status(result, "mitte") == "blocked"
    assert _status(result, "rechts") in {"free", "uncertain"}
    assert _status(result, "links") in {"free", "uncertain"}


def test_marginal_grades_make_a_corridor_uncertain_instead_of_free():
    mask = _road(79)
    confident = np.where(mask > 0, 2, 0).astype(np.uint8)
    assert _status(evaluate_corridors(mask, confident), "mitte") == "free"
    marginal = np.where(mask > 0, 3, 0).astype(np.uint8)
    result = evaluate_corridors(mask, marginal)
    assert _status(result, "mitte") == "uncertain"
    assert _rows(result, "mitte")["blocked"] == 0
    assert result["graded_input"] is True


def test_a_wider_vehicle_needs_a_wider_strip():
    narrow = evaluate_corridors(_road(60), vehicle_width_m=0.8)
    wide = evaluate_corridors(_road(60), vehicle_width_m=3.4)
    assert _status(narrow, "mitte") == "free"
    assert _status(wide, "mitte") in {"blocked", "uncertain"}
    assert wide["strip"]["required_width_m"] == pytest.approx(3.5)
    assert wide["strip"]["required_width_px_at_bottom"] == pytest.approx(3.5 / 4.0 * WIDTH)


def test_calibration_and_defaults_are_reported_with_the_result():
    result = evaluate_corridors(_road(60))
    assert result["strip"]["vehicle_width_m"] == DEFAULT_VEHICLE_WIDTH_M
    assert result["strip"]["clearance_m"] == DEFAULT_CLEARANCE_M
    assert result["strip"]["ground_width_at_bottom_m"] == DEFAULT_GROUND_WIDTH_AT_BOTTOM_M
    assert result["strip"]["scaling"] == "konstante parallele Streifen im sichtbaren Nahfeld"
    assert [item["corridor"] for item in result["corridors"]] == ["mitte", "rechts", "links"]
    assert [item["label"] for item in result["corridors"]] == ["Mitte", "Rechts", "Links"]


def test_an_empty_mask_is_uncertain_or_blocked_but_never_silently_free():
    result = evaluate_corridors(np.zeros((HEIGHT, WIDTH), np.uint8))
    assert all(item["status"] in {"blocked", "uncertain"} for item in result["corridors"])


def test_rejects_mismatched_grade_mask():
    with pytest.raises(ValueError, match="dieselbe Groesse"):
        evaluate_corridors(_road(60), np.zeros((10, 10), np.uint8))


def test_rows_the_model_never_judged_are_left_out_instead_of_counted_as_blocked():
    mask = _road(79)
    grades = np.where(mask > 0, 2, 0).astype(np.uint8)
    start = int(HEIGHT * 0.42)
    grades[start : start + 25, :] = 0
    result = evaluate_corridors(mask, grades)
    assert _status(result, "mitte") == "free"
    assert _rows(result, "mitte")["blocked"] == 0
    assert _rows(result, "mitte")["undecided"] >= 20
