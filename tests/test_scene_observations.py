import cv2
import numpy as np

from rinkspace.scene_observations import TypedLK, scene_masks, foreground_mask, paint_centerlines


def test_typed_lk_keeps_actual_pixel_pairs_and_persistent_births():
    rng = np.random.default_rng(12)
    image = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
    mask = np.full((180, 320), 255, np.uint8)
    tracker = TypedLK('board_surface', scale=1., maximum=160)
    first, count = tracker.update(image, mask, 0)
    assert not first and count > 40
    moved = cv2.warpAffine(image, np.array([[1., 0., 4.], [0., 1., 2.]]), (320, 180))
    pairs, _ = tracker.update(moved, mask, 1)
    assert len(pairs['track_ids']) > 30
    assert np.median(np.linalg.norm(pairs['current_xy'] - pairs['previous_xy'] - [4, 2], axis=1)) < .2
    assert np.all(pairs['birth_frame_index'] == 0)
    assert np.allclose(pairs['birth_xy'], pairs['previous_xy'])


def test_board_points_do_not_require_or_create_ground_positions():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
    tracker = TypedLK('board_surface', scale=.5)
    tracker.update(image, np.full(image.shape[:2], 255, np.uint8), 100)
    pairs, _ = tracker.update(image, np.full(image.shape[:2], 255, np.uint8), 101)
    assert len(pairs['track_ids']) > 20
    assert set(pairs) == {'previous_xy', 'current_xy', 'track_ids', 'birth_xy', 'birth_frame_index', 'fb_error_source_px', 'lk_patch_error'}
    assert 'H_model_from_image' not in pairs


def test_contact_detector_uses_lower_yellow_edge_and_leaves_it_candidate():
    image = np.full((300, 600, 3), 225, np.uint8)
    image[70:76] = (150, 70, 30)  # upper blue rail
    image[150:155] = (100, 210, 225)
    masks, contact, provenance = scene_masks(image, 158)
    assert contact is not None
    assert abs(contact['line_y_equals_mx_plus_b'][1] - 154.5) < 1
    assert contact['status'].startswith('candidate_')
    assert contact['physical_group_count'] == 1
    assert not provenance['contact_prior_is_observation']
    assert not np.any((masks['board_surface'] > 0) & (masks['ice_marking'] > 0))


def test_foreground_mask_rescales_original_segmentation():
    mask = foreground_mask((200, 400, 3), [[[10, 10], [20, 10], [20, 20], [10, 20]]], [200, 100])
    assert mask[30, 30] == 255
    assert mask[100, 100] == 0


def test_colour_centre_is_measured_between_actual_edges():
    image = np.full((400, 600, 3), 220, np.uint8)
    cv2.line(image, (100, 60), (430, 350), (165, 160, 215), 15)
    mask = np.full(image.shape[:2], 255, np.uint8)
    lines = paint_centerlines(image, mask)
    red = [line for line in lines if line['colour'] == 'red']
    assert red
    points = np.array(red[0]['observed_centres_source_px'])
    direction = np.array([330, 290]) / np.linalg.norm([330, 290])
    normal = np.array([-direction[1], direction[0]])
    assert np.median(abs((points - [100, 60]) @ normal)) < 2
    assert red[0]['median_paint_width_source_px'] > 10


def test_blue_score_graphic_cannot_become_board_rail():
    image = np.full((300, 600, 3), 225, np.uint8)
    image[20:35, :180] = (160, 60, 20)  # screen score graphic
    image[100:106, 250:390] = (160, 60, 20)  # visible rail fragment
    image[150:155] = (100, 210, 225)
    _, contact, info = scene_masks(image, 158)
    assert contact is not None
    assert 99 <= info['board_upper_rail_y_source_px'] <= 108


def test_old_ice_roi_cannot_turn_lower_advert_into_ground():
    image = np.full((400, 700, 3), 225, np.uint8)
    image[110:119] = (160, 60, 20)
    image[255:260] = (100, 210, 225)
    cv2.putText(image, 'BOARD', (150, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 10, 10), 4)
    masks, contact, info = scene_masks(image, 125)
    assert contact is not None and contact['line_y_equals_mx_plus_b'][1] > 258
    assert masks['board_surface'][220, 320] == 255
    assert masks['ice_texture'][220, 320] == 0
    assert masks['ice_texture'][310, 320] == 255
    assert not info['contact_search_bounded_by_old_ice_roi']


def test_missing_contact_does_not_invent_ground_below_prior():
    image = np.full((300, 600, 3), 225, np.uint8)
    image[85:93] = (160, 60, 20)
    masks, contact, _ = scene_masks(image, 110)
    assert contact is None
    assert not np.any(masks['ice_texture'])
    assert not np.any(masks['ice_marking'])


def test_old_leg_seed_excludes_upper_body_uncertainty():
    image = np.full((400, 600, 3), 225, np.uint8)
    cv2.rectangle(image, (260, 100), (310, 330), (50, 50, 60), -1)
    diagnostics = {}
    result = foreground_mask(image.shape, [[[250, 240], [320, 240], [320, 345], [250, 345]]], [600, 400], image=image, diagnostics=diagnostics)
    assert result[140, 280] == 255
    assert result[300, 280] == 255
    assert diagnostics['full_body_box_is_uncertainty_exclusion']


def test_warm_foreground_exclusion_cannot_move_measured_board_edge():
    image = np.full((400, 700, 3), 225, np.uint8)
    image[90:99] = (160, 60, 20)
    image[250:255] = (100, 210, 225)
    foreground = np.zeros(image.shape[:2], np.uint8)
    foreground[235:270] = 255  # colour-only exclusion may include warm strip
    masks, contact, _ = scene_masks(image, 120, foreground, contact_foreground=np.zeros_like(foreground))
    assert contact is not None
    assert 253 < contact['line_y_equals_mx_plus_b'][1] < 256
    assert masks['ice_texture'][260, 350] == 0
    assert masks['ice_texture'][300, 350] == 255
