import cv2
import numpy as np

from backend.app.terrain import TerrainAnalyzer, decode_mask, encode_mask

IDENTITY_MOTION = np.array([[1, 0, 0], [0, 1, 0]], np.float32)


def _road_scene(shift_px=0, obstacle_bounds=None):
    """Deterministic textured road frame with matching vegetation evidence."""
    height, width = 360, 640
    rng = np.random.default_rng(17)
    image = np.full((height, width, 3), (45, 95, 45), np.uint8)
    road = np.array(
        [
            [280 + shift_px, 122],
            [360 + shift_px, 122],
            [600 + shift_px, 359],
            [40 + shift_px, 359],
        ],
        np.int32,
    )
    cv2.fillConvexPoly(image, road, (122, 118, 103))
    noise = rng.normal(0, 7, (height, width, 1))
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    vegetation = np.full((height, width), 255, np.uint8)
    cv2.fillConvexPoly(vegetation, road, 0)

    detections = []
    if obstacle_bounds is not None:
        x1, x2 = obstacle_bounds
        polygon = [[x1, 0.56], [x2, 0.56], [x2 + 0.02, 0.84], [x1 - 0.02, 0.84]]
        pixels = np.asarray([[round(x * width), round(y * height)] for x, y in polygon], np.int32)
        cv2.fillPoly(image, [pixels], (35, 35, 35))
        detections.append({"class_id": "unknown_obstacle", "polygon": polygon})
    return image, vegetation, detections


def _analyze(analyzer, image, vegetation, detections, timestamp_ms=0, motion=None):
    return analyzer.analyze(
        image,
        vegetation,
        detections,
        IDENTITY_MOTION if motion is None else motion,
        motion_inliers=100,
        timestamp_ms=timestamp_ms,
    )


def _corridor_mask(shape, polygon):
    height, width = shape
    mask = np.zeros(shape, np.uint8)
    points = np.asarray([[round(x * width), round(y * height)] for x, y in polygon], np.int32)
    if len(points) >= 3:
        cv2.fillPoly(mask, [points], 255)
    return mask


def test_rle_roundtrip_preserves_native_class_map():
    mask = np.array(
        [
            [0, 0, 1, 1, 1, 2, 2],
            [0, 3, 3, 1, 2, 2, 2],
            [3, 3, 0, 0, 2, 1, 1],
            [1, 2, 3, 0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )

    encoded = encode_mask(mask, output_width=mask.shape[1])

    assert encoded["width"] == mask.shape[1]
    assert encoded["height"] == mask.shape[0]
    assert sum(encoded["rle"][1::2]) == mask.size
    np.testing.assert_array_equal(decode_mask(encoded), mask)


def test_frame_content_changes_source_hash_and_traversability_mask():
    left_frame = _road_scene(obstacle_bounds=(0.37, 0.44))
    right_frame = _road_scene(obstacle_bounds=(0.56, 0.63))
    left, _ = _analyze(TerrainAnalyzer(), *left_frame)
    right, _ = _analyze(TerrainAnalyzer(), *right_frame)

    left_mask = decode_mask(left["traversability"]["mask"])
    right_mask = decode_mask(right["traversability"]["mask"])

    assert left["ground"]["source"] == "current_video_frame_inference"
    assert left["source_frame_hash"] != right["source_frame_hash"]
    assert not np.array_equal(left_mask, right_mask)


def test_unobservable_or_fully_blurred_frame_never_becomes_green():
    vegetation = np.zeros((360, 640), np.uint8)
    frames = [
        np.zeros((360, 640, 3), np.uint8),
        np.full((360, 640, 3), 128, np.uint8),
        np.full((360, 640, 3), 255, np.uint8),
    ]

    for image in frames:
        terrain, raw = _analyze(TerrainAnalyzer(metric_calibration="calibrated"), image, vegetation, [])
        assert terrain["quality"]["blur_score"] == 1.0
        assert not np.any(raw["class_map"] == 1)
        assert terrain["traversability"]["overall_class"] != "likely_traversable"
        assert terrain["corridor"]["status"] != "available"


def test_corridor_polygon_does_not_enter_red_or_unknown_pixels():
    scene = _road_scene(obstacle_bounds=(0.47, 0.55))
    terrain, raw = _analyze(TerrainAnalyzer(metric_calibration="calibrated"), *scene)
    corridor = _corridor_mask(raw["class_map"].shape, terrain["corridor"]["polygon"])

    assert terrain["corridor"]["polygon"]
    assert np.any(raw["class_map"] == 3), "fixture must contain an evaluated obstacle"
    unsafe = (raw["class_map"] == 0) | (raw["class_map"] == 3)
    assert not np.any((corridor > 0) & unsafe)


def test_class_coverage_is_a_bounded_partition_of_the_evaluated_roi():
    scenes = [
        _road_scene(),
        (
            np.zeros((360, 640, 3), np.uint8),
            np.zeros((360, 640), np.uint8),
            [],
        ),
    ]

    for scene in scenes:
        terrain, _ = _analyze(TerrainAnalyzer(), *scene)
        coverage = terrain["traversability"]["class_coverage"]
        assert set(coverage) == {
            "likely_traversable",
            "limited",
            "not_traversable",
            "unknown",
        }
        assert all(0 <= value <= 1 for value in coverage.values())
        assert abs(sum(coverage.values()) - 1.0) <= 0.001


def test_all_green_pixels_form_one_bottom_seed_connected_component():
    analyzer = TerrainAnalyzer()
    terrain, raw = _analyze(analyzer, *_road_scene())
    green = np.where(raw["class_map"] == 1, 1, 0).astype(np.uint8)

    assert terrain["traversability"]["class_coverage"]["likely_traversable"] > 0
    component_count, labels = cv2.connectedComponents(green)
    assert component_count == 2, "expected background plus exactly one green component"

    height, width = green.shape
    bottom_seed = np.zeros_like(green)
    centre_prior = analyzer._centre_prior((height, width))
    start, stop = round(height * 0.69), round(height * 0.96)
    bottom_seed[start:stop] = centre_prior[start:stop]
    seed_labels = set(labels[(bottom_seed > 0) & (green > 0)].tolist())

    assert seed_labels == {1}
    assert np.all(labels[green > 0] == 1)


def test_corridor_remains_stable_under_known_camera_translation():
    analyzer = TerrainAnalyzer(metric_calibration="calibrated")
    centerlines = []
    for index, shift_px in enumerate((0, 6, 12)):
        image, vegetation, detections = _road_scene(shift_px=shift_px)
        motion = np.array([[1, 0, 0 if index == 0 else 6], [0, 1, 0]], np.float32)
        terrain, _ = _analyze(
            analyzer,
            image,
            vegetation,
            detections,
            timestamp_ms=index * 250,
            motion=motion,
        )
        assert terrain["corridor"]["centerline"]
        centerlines.append(np.asarray(terrain["corridor"]["centerline"], np.float32))

    baseline_x = centerlines[0][:, 0]
    for shifted, shift_px in zip(centerlines[1:], (6, 12)):
        compensated_x = shifted[:, 0] - shift_px / 640
        assert np.max(np.abs(compensated_x - baseline_x)) < 0.015

    assert terrain["corridor"]["stable_frames"] == 3
    assert terrain["corridor"]["stability_px"] < 0.01 * 640
