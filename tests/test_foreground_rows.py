import cv2
import numpy as np
import pytest

from rinkspace.foreground_rows import RowForegroundConfig, row_foreground_exclusion


def test_invalid_uncertainty_limits_are_rejected():
    with pytest.raises(ValueError):
        RowForegroundConfig(pale_skate_extension_px=-1)
    with pytest.raises(ValueError):
        RowForegroundConfig(maximum_component_width_fraction=1.1)


def test_empty_ice_and_uniform_ice_do_not_create_central_exclusion():
    image = np.full((540, 960, 3), 220, np.uint8)
    full = np.full(image.shape[:2], 255, np.uint8)
    result = row_foreground_exclusion(image, full)
    assert not result['components']
    assert not result['foreground'].any()
    assert np.array_equal(result['usable'], full)
    empty = row_foreground_exclusion(image, np.zeros_like(full))
    assert not empty['usable'].any()
    assert not empty['foreground'].any()


def test_two_leg_tips_receive_bounded_downward_pale_skate_uncertainty():
    image = np.full((540, 960, 3), 220, np.uint8)
    # One connected clothed silhouette, two legs ending at different heights.
    cv2.rectangle(image, (460, 200), (495, 300), (35, 35, 35), -1)
    cv2.line(image, (474, 288), (441, 351), (35, 35, 35), 16)
    cv2.line(image, (488, 290), (520, 326), (35, 35, 35), 16)
    domain = np.full(image.shape[:2], 255, np.uint8)
    domain[:190] = 0
    result = row_foreground_exclusion(image, domain)
    # Both pale-shoe candidates, including the higher bent leg, are excluded.
    assert result['person'][385, 432] > 0
    assert result['person'][361, 545] > 0
    assert result['person'][420, 432] == 0
    assert result['person'][361, 600] == 0
    assert np.all(result['usable'][domain == 0] == 0)
    # The person's horizontal position does not create a full-height strip.
    assert result['usable'][500, 480] == 255
    assert result['precise_person_segmentation'] is False


def test_row_spans_reconstruct_actual_mask_and_keep_text_separate():
    image = np.full((540, 960, 3), 220, np.uint8)
    cv2.rectangle(image, (420, 230), (450, 340), (35, 35, 35), -1)
    cv2.rectangle(image, (870, 480), (915, 494), (35, 35, 35), -1)
    domain = np.full(image.shape[:2], 255, np.uint8)
    result = row_foreground_exclusion(image, domain)
    rebuilt = np.zeros_like(domain)
    for row in result['row_spans']:
        for left, right in row['excluded_intervals_x']:
            rebuilt[row['y'], left:right] = 255
    assert np.array_equal(rebuilt, result['foreground'])
    assert result['screen_text'][487, 890] == 255
    assert result['person'][487, 890] == 0
    assert result['person'][280, 435] == 255
    assert result['screen_text'][280, 435] == 0


@pytest.mark.parametrize('image,ice', [
    (np.zeros((4, 4, 3), float), np.ones((4, 4))),
    (np.zeros((4, 4, 3), np.uint8), np.ones((3, 4))),
    (np.zeros((4, 4, 3), np.uint8), np.full((4, 4), np.nan)),
])
def test_invalid_coordinate_or_pixel_domain_is_rejected(image, ice):
    with pytest.raises(ValueError):
        row_foreground_exclusion(image, ice)
