import numpy as np

from backend.app.reconstruction import _anchor, _resample


def test_visual_trajectory_is_resampled_and_endpoint_anchored_without_becoming_linear():
    visual = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [1.0, 3.0], [0.0, 4.0]])
    anchored = _resample(_anchor(visual, 100), 32)
    assert np.linalg.norm(anchored[0]) < 1e-6
    assert 99.9 < np.linalg.norm(anchored[-1]) < 100.1
    assert np.max(np.abs(anchored[:, 0])) > 10
