import cv2
import numpy as np

from backend.app.segmentation import ONTOLOGY, ForestCvAdapter, MultiFrameTracker, _iou


def test_forest_adapter_emits_normalized_polygon_and_supported_class():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(image, (30, 20), (285, 185), (35, 150, 45), -1)
    detections = ForestCvAdapter().infer(image)
    assert detections
    detection = detections[0]
    assert detection["class_id"] in ONTOLOGY
    assert 0 <= detection["confidence"] <= 1
    assert all(0 <= value <= 1 for value in detection["bbox"])
    assert len(detection["polygon"]) >= 3
    assert all(0 <= value <= 1 for point in detection["polygon"] for value in point)


def test_tracking_iou_is_bounded_and_symmetric():
    a = [0.1, 0.1, 0.5, 0.5]
    b = [0.2, 0.2, 0.6, 0.6]
    assert 0 < _iou(a, b) < 1
    assert _iou(a, b) == _iou(b, a)
    assert _iou(a, a) == 1


def test_two_visible_trunks_become_two_tree_instances():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (639, 235), (35, 135, 45), -1)
    cv2.rectangle(image, (145, 80), (175, 310), (42, 76, 116), -1)
    cv2.rectangle(image, (465, 65), (495, 305), (46, 82, 125), -1)
    trees = [item for item in ForestCvAdapter().infer(image) if item["class_id"] == "tree"]
    assert len(trees) == 2
    assert all(item["geometry_basis"] == "visible_trunk" for item in trees)


def test_unstructured_green_frame_does_not_invent_tree():
    image = np.full((360, 640, 3), (35, 135, 45), dtype=np.uint8)
    assert not [item for item in ForestCvAdapter().infer(image) if item["class_id"] == "tree"]


def test_motion_compensated_track_reacquires_and_confirms():
    tracker = MultiFrameTracker("video-123", max_age=2, confirmation_hits=3)
    identity = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    shift = np.array([[1, 0, 20], [0, 1, 0]], np.float32)

    def detection(index, box):
        return {"detection_id": f"d-{index}", "class_id": "tree", "confidence": 0.7, "bbox": box}

    first = detection(1, [0.1, 0.1, 0.2, 0.5])
    tracker.update([first], 0, 0, identity, (100, 100))
    tracker.update([], 1, 250, identity, (100, 100))
    second = detection(2, [0.3, 0.1, 0.4, 0.5])
    tracker.update([second], 2, 500, shift, (100, 100))
    third = detection(3, [0.31, 0.1, 0.41, 0.5])
    tracker.update([third], 3, 750, identity, (100, 100))
    frames = [{"detections": [first]}, {"detections": []}, {"detections": [second]}, {"detections": [third]}]
    tracks = tracker.finalise(frames)
    assert first["track_id"] == second["track_id"] == third["track_id"]
    assert second["tracking_status"] == "reacquired"
    assert tracks[0]["countable"] is True
    assert all(item["instance_status"] == "confirmed" for frame in frames for item in frame["detections"])
