import numpy as np
import pytest

from backend.app.eval_cross_mission import (
    POSITION_FEATURE_COUNT,
    _choose_threshold_on,
    _features_for,
    pick_evidence_frames,
    render_report,
    resolve_missions,
)
from backend.app.path_features import pixel_features


def _frame(height=8, width=12, seed=1):
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_position_features_are_exactly_the_last_eight_columns():
    # Pinnt die Annahme, auf der die Laeufe C und D beruhen. Aendert sich die
    # Kanalreihenfolge in pixel_features, muss POSITION_FEATURE_COUNT nachgezogen werden.
    image = _frame()
    full = pixel_features(image)

    reduced = _features_for(image, include_position=False)

    assert full.shape[1] == 22
    assert reduced.shape[1] == 22 - POSITION_FEATURE_COUNT == 14
    assert np.array_equal(reduced, full[:, :14])
    # Die entfernten Spalten sind ortsabhaengig: identische Farbe an zwei
    # verschiedenen Bildpositionen unterscheidet sich nur noch ohne sie nicht mehr.
    flat = np.zeros((4, 4, 3), np.uint8)
    flat[:] = (120, 130, 140)
    positional = pixel_features(flat)
    assert not np.allclose(positional[0], positional[-1])
    assert np.allclose(_features_for(flat, include_position=False)[0], _features_for(flat, include_position=False)[-1])


def _bottom_path_mask(size=20):
    """Untere Bildhaelfte ist Weg — gross genug, dass die Morphologie in
    clean_prediction die Flaeche nicht wegraeumt."""
    mask = np.zeros((size, size), np.uint8)
    mask[size // 2 :] = 1
    return mask


def _scores_for(mask, inverted=False, magnitude=3.0):
    target = 1 - mask if inverted else mask
    return np.where(target.reshape(-1) == 1, magnitude, -magnitude).astype(np.float32)


def test_threshold_selection_minimizes_the_symmetric_penalty():
    mask = _bottom_path_mask()
    frames = [{"mask": mask}, {"mask": mask}]
    scores = [_scores_for(mask), _scores_for(mask)]

    threshold, metrics = _choose_threshold_on(frames, scores)

    # Jede Schwelle echt zwischen den beiden Score-Niveaus trennt perfekt; die
    # Quantile liefern dafuer auch Zwischenwerte. Entscheidend ist der Straffwert.
    assert -3.0 < threshold <= 3.0
    assert metrics["symmetric_penalty_points"] == 0


def test_threshold_selection_never_sees_evaluation_frames():
    # Der Aufrufvertrag ist die Trennung: die Funktion sieht ausschliesslich die
    # uebergebenen Frames. Ein hineingereichter Frame mit gegenlaeufigem Verhaeltnis
    # von Score zu Wahrheit verschlechtert das Optimum nachweisbar — genau das
    # passiert, wenn Evaluationsdaten in die Schwellenwahl geraten.
    mask = _bottom_path_mask()
    training = [{"mask": mask}]
    training_scores = [_scores_for(mask)]
    contaminated = training + [{"mask": mask}]
    contaminated_scores = training_scores + [_scores_for(mask, inverted=True)]

    _clean, clean_metrics = _choose_threshold_on(training, training_scores)
    _dirty, dirty_metrics = _choose_threshold_on(contaminated, contaminated_scores)

    assert clean_metrics["symmetric_penalty_points"] == 0
    assert dirty_metrics["symmetric_penalty_points"] > 0


def test_evidence_selection_takes_three_worst_and_two_best_without_duplicates():
    per_frame = [{"metrics": {"iou": value}} for value in (0.9, 0.1, 0.5, 0.8, 0.2, 0.7)]

    chosen = pick_evidence_frames(per_frame)

    assert [kind for kind, _ in chosen] == ["worst", "worst", "worst", "best", "best"]
    assert [entry["metrics"]["iou"] for _, entry in chosen] == [0.1, 0.2, 0.5, 0.9, 0.8]
    assert len({id(entry) for _, entry in chosen}) == 5


def test_evidence_selection_never_reuses_a_frame_when_there_are_too_few():
    per_frame = [{"metrics": {"iou": value}} for value in (0.4, 0.6)]

    chosen = pick_evidence_frames(per_frame)

    assert len(chosen) == 2
    assert len({id(entry) for _, entry in chosen}) == 2


def test_report_lists_every_run_and_keeps_the_safety_caveat():
    metrics = {"iou": 0.312, "precision": 0.48, "recall": 0.52}
    rows = [
        {
            "key": "A",
            "train_mission": "Mission 1",
            "eval_mission": "Mission 2",
            "include_position": True,
            "feature_count": 22,
            "eval_frames": 120,
            "metrics": metrics,
            "evidence": [{"kind": "worst", "file": "A-0-worst-abc-0000010.jpg", "iou": 0.05, "frame_index": 9}],
        },
        {
            "key": "C",
            "train_mission": "Mission 1",
            "eval_mission": "Mission 2",
            "include_position": False,
            "feature_count": 14,
            "eval_frames": 120,
            "metrics": metrics,
            "evidence": [],
        },
    ]
    baseline = [
        {
            "label": "Aktives globales Modell",
            "train": "200 Frames",
            "validation": "51 Frames",
            "metrics": {"iou": 0.796, "precision": 0.9, "recall": 0.9},
        }
    ]

    report = render_report(rows, baseline, "2026-08-03T20:00:00+00:00")

    assert "| A | Mission 1 | Mission 2 | ja | 22 | 0.312 | 0.480 | 0.520 | 120 |" in report
    assert "| C | Mission 1 | Mission 2 | nein | 14 |" in report
    assert "0.796" in report
    assert "systematisch optimistisch" in report
    assert "keine sicherheitsrelevante" in report
    assert "A-0-worst-abc-0000010.jpg" in report


def test_mission_selection_prefers_label_count_over_recency():
    # Regression: die erste Fassung nahm die beiden juengsten Missionen und
    # verglich dadurch eine frisch angelegte Mission mit 9 Labels statt der
    # eigentlich gemeinten mit 152.
    candidates = [
        {"id": "c", "name": "Misson 3", "confirmed": 9},
        {"id": "b", "name": "Misson  2", "confirmed": 99},
        {"id": "a", "name": "Mission 1", "confirmed": 152},
    ]

    first, second = resolve_missions(candidates)

    assert (first["name"], second["name"]) == ("Mission 1", "Misson  2")


def test_mission_selection_accepts_explicit_names_or_ids():
    candidates = [
        {"id": "a", "name": "Mission 1", "confirmed": 152},
        {"id": "b", "name": "Misson  2", "confirmed": 99},
        {"id": "c", "name": "Misson 3", "confirmed": 9},
    ]

    by_name = resolve_missions(candidates, ("Misson 3", "Mission 1"))
    by_id = resolve_missions(candidates, ("c", "a"))

    assert [item["id"] for item in by_name] == ["c", "a"]
    assert [item["id"] for item in by_id] == ["c", "a"]
    with pytest.raises(ValueError, match="nicht gefunden"):
        resolve_missions(candidates, ("Mission 1", "Mission 9"))
    with pytest.raises(ValueError, match="zwei verschiedene"):
        resolve_missions(candidates, ("Mission 1", "a"))


def test_report_names_the_data_basis_and_marks_the_selected_missions():
    missions = {
        "first": {"name": "Mission 1", "confirmed_frames": 152},
        "second": {"name": "Misson  2", "confirmed_frames": 99},
        "available": [
            {"name": "Mission 1", "confirmed_frames": 152},
            {"name": "Misson  2", "confirmed_frames": 99},
            {"name": "Misson 3", "confirmed_frames": 9},
        ],
    }

    report = render_report([], [], "2026-08-03T20:00:00+00:00", missions)

    assert "| Mission 1 | 152 | ja |" in report
    assert "| Misson 3 | 9 | nein |" in report


def test_reduced_features_still_feed_the_existing_classifier():
    from backend.app.path_features import RANDOM_FEATURES, RIDGE_LAMBDA, fit_kernel_classifier, predict_scores

    image = _frame(6, 6, seed=3)
    reduced = _features_for(image, include_position=False)
    labels = (np.arange(len(reduced)) % 2).astype(np.uint8)

    model = fit_kernel_classifier(reduced, labels, RANDOM_FEATURES, RIDGE_LAMBDA, seed=42)

    assert model["mean"].shape[0] == 14
    assert predict_scores(reduced, model).shape == (len(reduced),)
    with pytest.raises(ValueError):
        # Ein Modell mit 14 Merkmalen darf 22-spaltige Merkmale nicht stillschweigend annehmen.
        predict_scores(_features_for(image, include_position=True), model)
