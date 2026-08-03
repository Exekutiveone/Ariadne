import numpy as np
import pytest

from backend.app.path_model import (
    RANDOM_FEATURES,
    RIDGE_LAMBDA,
    _clean_prediction,
    _decode_rle,
    _encode_binary_rle,
    _features,
    _fit_kernel_classifier,
    _frame_split,
    _predict_scores,
    _random_projection,
    _standardize,
)


def _synthetic_frame(height: int = 12, width: int = 16):
    return np.random.default_rng(7).integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_feature_extraction_is_deterministic_and_finite():
    image = _synthetic_frame()

    first = _features(image)
    second = _features(image)

    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert first.dtype == np.float32


def test_feature_vector_width_is_pinned_to_the_model_schema():
    # Gespeicherte Modelle enthalten Gewichte fuer genau diese Merkmalszahl.
    # Aendert sie sich, sind alle Laeufe unter derived/path_model_runs ungueltig
    # und MODEL_SCHEMA_VERSION muss angehoben werden.
    assert _features(_synthetic_frame(12, 16)).shape == (12 * 16, 22)


def test_classifier_learns_a_separable_split_and_scores_by_sign():
    rng = np.random.default_rng(3)
    positive = rng.normal(1.0, 0.25, (200, 4)).astype(np.float32)
    negative = rng.normal(-1.0, 0.25, (200, 4)).astype(np.float32)
    samples = np.vstack([positive, negative])
    labels = np.concatenate([np.ones(200, np.uint8), np.zeros(200, np.uint8)])

    model = _fit_kernel_classifier(samples, labels, RANDOM_FEATURES, RIDGE_LAMBDA, seed=42)
    scores = _predict_scores(samples, model)

    accuracy = ((scores >= 0).astype(np.uint8) == labels).mean()
    assert accuracy > 0.95


def test_classifier_is_reproducible_for_a_fixed_seed():
    rng = np.random.default_rng(11)
    samples = rng.normal(0, 1, (120, 4)).astype(np.float32)
    labels = (samples[:, 0] > 0).astype(np.uint8)

    first = _fit_kernel_classifier(samples, labels, RANDOM_FEATURES, RIDGE_LAMBDA, seed=42)
    same_seed = _fit_kernel_classifier(samples, labels, RANDOM_FEATURES, RIDGE_LAMBDA, seed=42)
    other_seed = _fit_kernel_classifier(samples, labels, RANDOM_FEATURES, RIDGE_LAMBDA, seed=43)

    assert np.array_equal(first["weights"], same_seed["weights"])
    assert not np.array_equal(first["weights"], other_seed["weights"])


def test_chunked_prediction_matches_a_single_pass():
    rng = np.random.default_rng(5)
    samples = rng.normal(0, 1, (500, 4)).astype(np.float32)
    labels = (samples[:, 1] > 0).astype(np.uint8)
    model = _fit_kernel_classifier(samples, labels, RANDOM_FEATURES, RIDGE_LAMBDA, seed=42)

    whole = _predict_scores(samples, model, chunk_size=len(samples))
    chunked = _predict_scores(samples, model, chunk_size=64)

    assert np.allclose(whole, chunked, atol=1e-5)


def test_cleanup_drops_isolated_pixels_but_keeps_connected_path():
    noise = np.zeros((20, 20), np.float32)
    noise[5, 5] = 1.0
    block = np.zeros((20, 20), np.float32)
    block[4:13, 4:13] = 1.0

    assert _clean_prediction(noise.reshape(-1), (20, 20), 0.5).sum() == 0
    assert _clean_prediction(block.reshape(-1), (20, 20), 0.5)[6:11, 6:11].all()


def test_frame_split_holds_back_one_frame_per_short_video():
    records = [{"video_id": video, "frame_index": index} for video in ("v1", "v2") for index in range(3)]

    train, validation = _frame_split(records)

    assert len(train) == 4
    assert len(validation) == 2
    assert {item["video_id"] for item in validation} == {"v1", "v2"}


def test_frame_split_never_validates_on_a_single_labelled_frame():
    train, validation = _frame_split([{"video_id": "v1", "frame_index": 0}])

    assert len(train) == 1
    assert validation == []


def test_rle_survives_an_encode_decode_round_trip():
    mask = np.array([[0, 1, 1], [1, 1, 0]], dtype=np.uint8)

    record = {"width": 3, "height": 2, "rle": _encode_binary_rle(mask)}

    assert np.array_equal(_decode_rle(record), mask)


def test_vectorized_rle_matches_naive_reference_and_handles_edges():
    def naive(values):
        encoded = []
        current, length = int(values[0]), 1
        for value in values[1:]:
            value = int(value)
            if value == current:
                length += 1
            else:
                encoded.extend([current, length])
                current, length = value, 1
        encoded.extend([current, length])
        return encoded

    mask = np.random.default_rng(6).integers(0, 6, (37, 23)).astype(np.uint8)

    encoded = _encode_binary_rle(mask)

    assert encoded == naive(mask.reshape(-1))
    assert all(isinstance(entry, int) for entry in encoded)
    assert _encode_binary_rle(np.empty((0,), np.uint8)) == []
    assert _encode_binary_rle(np.ones((4, 4), np.uint8)) == [1, 16]


def test_fused_score_path_matches_reference_formula():
    rng = np.random.default_rng(8)
    samples = rng.normal(0, 1, (300, 5)).astype(np.float32)
    labels = (samples[:, 2] > 0).astype(np.uint8)
    model = _fit_kernel_classifier(samples, labels, 32, 0.05, seed=1)

    normalized = _standardize(samples, model["mean"], model["scale"])
    reference = (
        _random_projection(normalized, model["projection"], model["phase"]) @ model["weights"][:-1]
        + model["weights"][-1]
    )

    assert np.allclose(_predict_scores(samples, model), reference, atol=1e-4)


def test_rle_with_a_truncated_run_is_rejected():
    with pytest.raises(ValueError):
        _decode_rle({"width": 2, "height": 2, "rle": [0, 1]})
