import cv2
import numpy as np

from rinkspace.landmark_diagnostics import correlate_patch


def test_independent_patch_measurement_recovers_translation_without_any_homography():
    rng = np.random.default_rng(38)
    reference = cv2.GaussianBlur(rng.integers(0, 256, (150, 220), dtype=np.uint8), (5, 5), 1.)
    current = cv2.warpAffine(reference, np.array([[1,0,7.25],[0,1,-3.5]], np.float32), (220,150))
    result = correlate_patch(reference, current, [90., 80.], patch_radius=12, search_radius=20)
    assert result["status"] == "candidate"
    assert np.linalg.norm(np.asarray(result["image_xy"]) - [97.25, 76.5]) < .15
    assert not result["human_subpixel_groundtruth"]


def test_textureless_or_off_image_patch_cannot_supply_observations():
    reference = np.full((150, 220), 210, np.uint8)
    assert correlate_patch(reference, reference, [90,80])["status"] == "unavailable"
    assert correlate_patch(reference, reference, [2,3])["status"] == "unavailable"
