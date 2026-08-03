import json

import numpy as np

from backend.app.path_model import (
    _apply_refinements,
    _comparison_mask,
    _encode_binary_rle,
    _frame_split,
    _polygon_mask,
    confusion_counts,
    symmetric_metrics,
)


def test_symmetric_score_penalizes_missed_and_invented_area_equally():
    missed = symmetric_metrics({"tp": 50, "tn": 100, "fp": 0, "fn": 50})
    invented = symmetric_metrics({"tp": 100, "tn": 50, "fp": 50, "fn": 0})

    assert missed["symmetric_penalty_points"] == 25
    assert invented["symmetric_penalty_points"] == 25
    assert symmetric_metrics({"tp": 100, "tn": 100, "fp": 0, "fn": 0})["symmetric_score"] == 100


def test_polygon_becomes_binary_training_mask():
    record = {"polygons": [{"points": [[0.2, 0.8], [0.5, 0.2], [0.8, 0.8]]}]}
    mask = _polygon_mask(record, 100, 60)

    assert mask.dtype == np.uint8
    assert mask[35, 50] == 1
    assert mask[2, 2] == 0
    counts = confusion_counts(mask, mask)
    assert counts["fp"] == counts["fn"] == 0


def test_validation_split_contains_no_training_frame():
    records = [{"video_id": "v1", "frame_index": index} for index in range(20)]
    train, validation = _frame_split(records)
    train_keys = {(item["video_id"], item["frame_index"]) for item in train}
    validation_keys = {(item["video_id"], item["frame_index"]) for item in validation}

    assert len(validation) == 4
    assert train_keys.isdisjoint(validation_keys)


def test_binary_prediction_mask_is_encoded_losslessly():
    mask = np.array([[0, 0, 1], [1, 0, 1]], dtype=np.uint8)

    encoded = _encode_binary_rle(mask)
    decoded = np.concatenate(
        [np.full(length, value, dtype=np.uint8) for value, length in zip(encoded[::2], encoded[1::2])]
    )

    assert encoded == [0, 2, 1, 2, 0, 1, 1, 1]
    assert np.array_equal(decoded.reshape(mask.shape), mask)


def test_comparison_mask_marks_correct_missed_and_invented_path():
    truth = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    prediction = np.array([[1, 0], [1, 0]], dtype=np.uint8)

    assert _comparison_mask(truth, prediction).tolist() == [[1, 2], [3, 0]]


def test_saved_refinement_overrides_effective_training_truth(tmp_path):
    directory = tmp_path / "path_refinements" / "video-1"
    directory.mkdir(parents=True)
    record = {"items": [{"correct_value": 1, "region_mask": {"width": 2, "height": 2, "rle": [0, 2, 1, 1, 0, 1]}}]}
    (directory / "000000007.json").write_text(json.dumps(record), encoding="utf-8")

    refined = _apply_refinements(np.zeros((2, 2), dtype=np.uint8), tmp_path, "video-1", 7)

    assert refined.tolist() == [[0, 0], [1, 0]]
