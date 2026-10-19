"""Camera initializers do not establish rink localization or real accuracy."""
import cv2
import numpy as np

import rinkspace.model_localization as localization


def test_camera_features_exclude_scoreboard_timer_and_coarse_foreground(monkeypatch):
    monkeypatch.setattr(localization, 'ice_mask', lambda image: {'foreground_boxes': [[360, 180, 500, 440]]})
    image = np.random.default_rng(12).integers(0, 255, (540, 960, 3), dtype=np.uint8)
    keys, desc = localization._features(image)
    assert desc is not None and len(keys) > 100
    assert {point.class_id for point in keys} == {0, 7, 11}
    for point in keys:
        x, y = (round(value) for value in point.pt)
        assert not (x < 350 and y < 175)
        assert not (x >= 780 and y >= 435)
        assert not (355 <= x < 505 and 175 <= y < 445)


def test_camera_similarity_retains_role_and_does_not_count_duplicate_orientations():
    coordinates = np.array([[100, 220], [170, 224], [250, 212], [340, 226],
                            [120, 250], [200, 257], [280, 245], [365, 258]], dtype=float)
    current = coordinates * 1.1 + [40, -20]
    descriptors = np.random.default_rng(17).uniform(1, 100, (len(coordinates), 128)).astype(np.float32)
    reference_keys = [cv2.KeyPoint(float(x), float(y), 4, class_id=0) for x, y in coordinates]
    current_keys = [cv2.KeyPoint(float(x), float(y), 4, class_id=0) for x, y in current]
    result = localization._proposal(reference_keys, descriptors, current_keys, descriptors.copy())
    assert result is not None and result['matches'] == len(coordinates)
    assert not result['plane_geometry_supported']
    assert result['proposal_role'] == 'background_camera_hypothesis_only_not_ice_correspondence'
    np.testing.assert_allclose(result['H'], [[1.1, 0, 40], [0, 1.1, -20], [0, 0, 1]], atol=1e-4)
    duplicate_reference = [cv2.KeyPoint(100, 220, 4, class_id=0) for _ in range(8)]
    duplicate_current = [cv2.KeyPoint(150, 222, 4, class_id=0) for _ in range(8)]
    assert localization._proposal(duplicate_reference, descriptors, duplicate_current, descriptors.copy()) is None
