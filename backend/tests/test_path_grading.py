import numpy as np

from backend.app.path_model import (
    GRADE_ONTOLOGY,
    _decode_rle,
    _encode_binary_rle,
    _grade_prediction,
    _grading_summary,
)


def _scores_from_margins(margins, threshold=0.5):
    # Kehrt die Margin-Definition m = (s - t) / (1 - t) exakt um.
    return (threshold + margins * (1.0 - threshold)).astype(np.float32).reshape(-1)


def test_green_bands_grade_by_distance_from_threshold():
    shape = (10, 30)
    margins = np.zeros(shape, np.float32)
    margins[:, :10] = 0.8
    margins[:, 10:20] = 0.4
    margins[:, 20:] = 0.1
    prediction = np.ones(shape, np.uint8)

    grades = _grade_prediction(_scores_from_margins(margins), prediction, 0.5, shape)

    assert grades[5, 4] == GRADE_ONTOLOGY["safe"]["value"]
    assert grades[5, 15] == GRADE_ONTOLOGY["good"]["value"]
    assert grades[5, 25] == GRADE_ONTOLOGY["marginal"]["value"]


def test_grades_partition_matches_binary_prediction():
    shape = (24, 32)
    rng = np.random.default_rng(9)
    scores = rng.normal(0, 1, shape[0] * shape[1]).astype(np.float32)
    prediction = (rng.random(shape) > 0.5).astype(np.uint8)

    grades = _grade_prediction(scores, prediction, 0.2, shape)

    inside = prediction.astype(bool)
    assert np.isin(grades[inside], (1, 2, 3)).all()
    assert np.isin(grades[~inside], (0, 4, 5)).all()


def test_uncertainty_band_outside_path_is_risky_orange():
    shape = (10, 10)
    margins = np.full(shape, -0.05, np.float32)
    prediction = np.zeros(shape, np.uint8)

    grades = _grade_prediction(_scores_from_margins(margins), prediction, 0.5, shape)

    assert (grades == GRADE_ONTOLOGY["risky"]["value"]).all()


def test_problem_zones_require_size_and_adjacency_to_path():
    # 140x140-Raster: Mindestflaeche = int(0.002 * 19600) = 39 Pixel.
    # Weg unten im Bild; drei sicher-negative Bloecke unterscheiden sich nur in
    # Groesse und Abstand zum Weg. Umgebung liegt im Unsicherheitsband (orange).
    shape = (140, 140)
    margins = np.zeros(shape, np.float32)
    margins[120:, :] = 0.8
    margins[100:120, 10:50] = -1.0  # gross und angrenzend -> Problemzone
    margins[5:25, 10:50] = -1.0  # gleich gross, aber fern -> transparent
    margins[116:119, 100:103] = -1.0  # angrenzend, aber winzig -> transparent
    prediction = np.zeros(shape, np.uint8)
    prediction[120:, :] = 1

    grades = _grade_prediction(_scores_from_margins(margins), prediction, 0.5, shape)

    assert grades[110, 30] == GRADE_ONTOLOGY["problem"]["value"]
    assert grades[15, 30] == GRADE_ONTOLOGY["unrated"]["value"]
    assert grades[117, 101] == GRADE_ONTOLOGY["unrated"]["value"]
    assert grades[130, 70] == GRADE_ONTOLOGY["safe"]["value"]
    assert grades[60, 70] == GRADE_ONTOLOGY["risky"]["value"]
    # Auch eine qualifizierte Problemkomponente faerbt nur ihre Pixel im Band um
    # den Fahrbereich (GRADE_PROBLEM_CLIP); ihr ferner Teil bleibt transparent.
    assert grades[101, 30] == GRADE_ONTOLOGY["unrated"]["value"]


def test_grading_is_deterministic():
    shape = (20, 20)
    rng = np.random.default_rng(2)
    scores = rng.normal(0, 1, shape[0] * shape[1]).astype(np.float32)
    prediction = (rng.random(shape) > 0.6).astype(np.uint8)

    first = _grade_prediction(scores, prediction, 0.1, shape)
    second = _grade_prediction(scores, prediction, 0.1, shape)

    assert np.array_equal(first, second)


def test_grade_mask_survives_rle_round_trip():
    shape = (12, 12)
    rng = np.random.default_rng(4)
    scores = rng.normal(0, 1, shape[0] * shape[1]).astype(np.float32)
    prediction = (rng.random(shape) > 0.5).astype(np.uint8)
    grades = _grade_prediction(scores, prediction, 0.0, shape)

    record = {"width": shape[1], "height": shape[0], "rle": _encode_binary_rle(grades)}

    assert np.array_equal(_decode_rle(record), grades)


def test_ontology_and_summary_expose_agreed_colours_and_bands():
    assert [item["value"] for item in GRADE_ONTOLOGY.values()] == [0, 1, 2, 3, 4, 5]
    assert GRADE_ONTOLOGY["safe"]["color"] == "#1e8c46"
    assert GRADE_ONTOLOGY["good"]["color"] == "#55d96f"
    assert GRADE_ONTOLOGY["marginal"]["color"] == "#a3ecb4"
    assert GRADE_ONTOLOGY["risky"]["color"] == "#f08c3a"
    assert GRADE_ONTOLOGY["problem"]["color"] == "#e05b52"

    summary = _grading_summary(0.31)

    assert summary["threshold"] == 0.31
    assert summary["bands"] == {"safe_min_margin": 0.6, "good_min_margin": 0.25, "risky_min_margin": -0.2}
    assert "keine sicherheitsrelevante Fahrfreigabe" in summary["note"]
