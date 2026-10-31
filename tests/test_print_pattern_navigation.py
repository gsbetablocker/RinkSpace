"""Coarse navigation may recover searches, but cannot admit a source pattern."""
import json

import cv2
import numpy as np
import pytest

from rinkspace.session.print_patterns import PrintPattern, project
from rinkspace.session.print_pattern_navigation import coarse_print_registration


def translated_pattern(polarity='light'):
    reference = np.full((80, 240, 3), [150, 60, 140], np.uint8)
    paths = []
    for name, x, height in [('A', 20, 40), ('O', 90, 35), ('M', 160, 30)]:
        cv2.rectangle(reference, (x, 20), (x + 40, 20 + height), (230, 230, 230), -1)
        paths.append(dict(path_id='Ad1.print.' + name,
            pixels=[[x, 20], [x + 40, 20], [x + 40, 20 + height], [x, 20 + height], [x, 20]]))
    cv2.rectangle(reference, (101, 29), (119, 46), (150, 60, 140), -1)
    background = (150, 60, 140)
    if polarity == 'dark':
        reference = 255 - reference
        background = tuple(255 - np.array(background))
    pattern = PrintPattern(dict(paths=paths, ink_polarity=polarity), reference)
    initial = np.array([[1., 0., 100.], [0., 1., 50.], [0., 0., 1.]])
    actual = initial.copy()
    actual[:2, 2] += [60, 70]
    image = cv2.warpPerspective(reference, actual, (520, 280), borderValue=tuple(map(int, background)))
    mask = np.zeros(image.shape[:2], np.uint8)
    return pattern, image, mask, initial, actual


@pytest.mark.parametrize('polarity', ['light', 'dark'])
def test_source_search_recovers_translation_beyond_original_local_ecc_window(polarity):
    pattern, image, mask, initial, actual = translated_pattern(polarity)
    original = pattern.register(image, mask, initial)
    assert original['status'] != 'source_pattern_registered'
    result = coarse_print_registration(pattern, image, mask, initial)
    assert result['status'] == 'source_pattern_registered'
    assert result['correlation'] > .95
    np.testing.assert_allclose(result['coarse_search']['translation_px'], [60, 70], atol=1)
    np.testing.assert_allclose(project(np.array(result['H']), pattern.points), project(actual, pattern.points), atol=.5)
    assert result['coarse_search']['search_only'] is True
    assert result['coarse_search']['source_UV_generated'] is False


def test_coarse_match_cannot_bypass_original_foreground_qualification():
    pattern, image, mask, initial, _ = translated_pattern()
    mask[:] = 255
    result = coarse_print_registration(pattern, image, mask, initial)
    assert result['coarse_search']['score'] > .99
    assert result['status'] == 'occluded_or_clipped'
    assert 'H' not in result


@pytest.mark.parametrize('case,expected_status,reason', [
    ('distant', 'outside_search', 'projected_pattern_outside_search_range'),
    ('oversized', 'coarse_search_unavailable', 'complete_template_larger_than_source_image'),
    ('partial', 'coarse_search_unavailable', 'no_complete_template_search_range'),
])
def test_unsearchable_projection_allocates_no_warp_canvas(monkeypatch, case, expected_status, reason):
    pattern, image, mask, initial, _ = translated_pattern()
    if case == 'distant':
        initial[:2, 2] = 1e12
    elif case == 'oversized':
        initial[0, 0] = initial[1, 1] = 20
        initial[:2, 2] = [-400, -400]
    else:
        initial[:2, 2] = [-170, 50]
    monkeypatch.setattr(cv2, 'warpPerspective', lambda *a, **kw: pytest.fail('An unsearchable projection allocated a canvas'))
    result = coarse_print_registration(pattern, image, mask, initial)
    assert result['status'] == expected_status
    assert result['reason'] == reason
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('case', ['nan_H', 'singular_H', 'projective_pole', 'bad_mask', 'nan_margin', 'negative_margin', 'constant_ink'])
def test_bad_input_is_explicitly_unrecoverable(case):
    pattern, image, mask, initial, _ = translated_pattern()
    margin = 80.
    if case == 'nan_H':
        initial[0, 0] = np.nan
    elif case == 'singular_H':
        initial[:] = 0
    elif case == 'projective_pole':
        initial[2] = [.01, 0, -1]
    elif case == 'bad_mask':
        mask = mask[:-1]
    elif case == 'nan_margin':
        margin = np.nan
    elif case == 'negative_margin':
        margin = -1
    else:
        pattern.ink[:] = 0
    result = coarse_print_registration(pattern, image, mask, initial, margin)
    assert result['status'] == 'coarse_search_unavailable'
    assert result['reason']
    json.dumps(result, allow_nan=False)


def test_nonfinite_match_scores_cannot_win_navigation(monkeypatch):
    pattern, image, mask, initial, _ = translated_pattern()
    monkeypatch.setattr(cv2, 'matchTemplate', lambda *a, **kw: np.full((2, 2), np.nan, np.float32))
    result = coarse_print_registration(pattern, image, mask, initial)
    assert result['status'] == 'coarse_search_unavailable'
    assert result['reason'] == 'no_finite_template_match'
