import cv2
import numpy as np

from rinkspace.landmark_observations import LandmarkDetectionConfig, detect_landmarks, polygon_mask, visible_segments


def _fixture():
    image = np.full((540, 960, 3), 218, np.uint8)
    # Low-contrast broad gray paint, partial second ring, and a colour stripe.
    cv2.ellipse(image, (295, 325), (170, 42), 5, 0, 360, (187, 188, 190), 12)
    cv2.ellipse(image, (930, 190), (180, 48), -7, 0, 360, (190, 190, 190), 9)
    cv2.line(image, (60, 500), (900, 420), (170, 190, 220), 20)
    return cv2.GaussianBlur(image, (0, 0), 1.8)


def test_visible_gray_and_partial_ring_detected_without_model():
    image = _fixture()
    obs, diag = detect_landmarks(image, np.full(image.shape[:2], 255, np.uint8))
    assert not diag['model_projection_used']
    gray = [x for x in obs if x['colour_class'] == 'gray' and x['association_status'] == 'detected_unidentified']
    points = np.concatenate([x['points_source_px'] for x in gray])
    assert np.count_nonzero((points[:, 0] > 140) & (points[:, 0] < 450) & (points[:, 1] > 270) & (points[:, 1] < 385)) > 20
    assert np.count_nonzero((points[:, 0] > 770) & (points[:, 1] < 250)) > 10
    assert all(not x['used_by_solver'] and x['selected_model_id'] is None for x in obs)


def test_points_are_actual_edges_not_complete_ellipse_samples():
    image = _fixture()
    obs, _ = detect_landmarks(image, np.full(image.shape[:2], 255, np.uint8))
    assert any(x['ellipse_candidate'] is not None for x in obs)
    for x in obs:
        p = np.asarray(x['points_source_px'])
        assert np.array_equal(p, np.rint(p))
        assert not x['fit_ellipse_used_to_generate_observations']
        assert np.all((p[:, 0] >= 0) & (p[:, 0] < 960) & (p[:, 1] >= 0) & (p[:, 1] < 540))


def test_foreground_removal_is_logged_and_gaps_are_not_connected():
    image = _fixture()
    fg = np.zeros(image.shape[:2], np.uint8);fg[:, 240:350] = 255
    conservative = np.zeros_like(fg);conservative[:, 200:700] = 255
    obs, _ = detect_landmarks(image, np.full_like(fg, 255), fg, conservative_mask=conservative)
    assert any(x['mask_fraction'] > 0 for x in obs)
    assert any(x['v09_conservative_mask_fraction'] > x['mask_fraction'] for x in obs)
    for x in obs:
        if x['association_status'] == 'masked_out':
            continue
        p = np.asarray(x['points_source_px']).astype(int)
        assert not np.any(fg[p[:, 1], p[:, 0]])
        for segment in visible_segments(x):
            assert np.max(np.linalg.norm(np.diff(segment, axis=0), axis=1)) <= 25.


def test_raw_sample_coordinate_chain_and_polygon_scaling():
    image = _fixture()
    mask = polygon_mask(image.shape, [[[0, 0], [479, 0], [479, 269], [0, 269]]], [480, 270])
    assert mask[500, 900]
    obs, _ = detect_landmarks(image, mask, sample_size=[480, 270])
    for x in obs:
        assert x['image_scale_from_source'] == [.5, .5]
        assert 'points_sample_px' not in x and 'visible_segments_source_px' not in x
        segments = visible_segments(x)
        assert len(segments) == len(x['visible_segment_ranges'])
        for (start, stop), segment in zip(x['visible_segment_ranges'], segments):
            assert segment == x['points_source_px'][start:stop]


def test_no_structure_is_missing_not_a_fake_circle():
    image = np.full((240, 320, 3), 200, np.uint8)
    obs, _ = detect_landmarks(image, np.full(image.shape[:2], 255, np.uint8))
    assert obs == []
