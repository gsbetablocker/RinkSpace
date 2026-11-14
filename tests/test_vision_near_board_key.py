"""Source-image transforms exercise directional symmetry and protected support."""
import json
import numpy as np
import pytest
from rinkspace.vision_near_board_key import profiles, profiles_at_source_keys


def edge(normal=(1., 0.), gray=100, ice=190, hw=(320, 320), query=(160., 160.)):
    y, x = np.indices(hw)
    side = (x-query[0])*normal[0]+(y-query[1])*normal[1] >= 0
    image = np.repeat(np.where(side, ice, gray)[..., None], 3, axis=2).astype(np.uint8)
    return image, np.asarray(query), np.asarray(normal)


def transformed(image, point, normal, operation):
    h, w = image.shape[:2]
    if operation == 'left_right':
        return np.ascontiguousarray(image[:, ::-1]), np.array([w-1-point[0], point[1]]), normal*np.array([-1, 1])
    if operation == 'up_down':
        return np.ascontiguousarray(image[::-1]), np.array([point[0], h-1-point[1]]), normal*np.array([1, -1])
    if operation == 'rotate_clockwise':
        return np.ascontiguousarray(np.rot90(image, -1)), np.array([h-1-point[1], point[0]]), np.array([-normal[1], normal[0]])
    return np.ascontiguousarray(np.rot90(image)), np.array([point[1], w-1-point[0]]), np.array([normal[1], -normal[0]])


def read(image, point, normal, guard=None):
    if guard is None:
        guard = np.zeros(image.shape[:2], bool)
    return profiles_at_source_keys(image, guard, query_uv=[point], iceward_normals=[normal])


@pytest.mark.parametrize('normal', [(1., 0.), (.6, .8)])
@pytest.mark.parametrize('operation', ['left_right', 'up_down', 'rotate', 'rotate_clockwise'])
def test_image_transform_preserves_the_same_measured_transition(normal, operation):
    image, query, n = edge(normal)
    rows, missing = read(image, query, n)
    assert not missing and len(rows) == 1
    original = rows[0]
    changed, new_query, new_n = transformed(image, query, n, operation)
    rows, missing = read(changed, new_query, new_n)
    assert not missing and len(rows) == 1
    row = rows[0]
    _, expected_uv, expected_normal = transformed(image, np.asarray(original['uv']), n, operation)
    np.testing.assert_allclose(row['uv'], expected_uv, atol=2e-4, rtol=0)
    np.testing.assert_allclose(row['normal'], expected_normal, atol=1e-7)
    for actual, expected in zip(row['interval'], original['interval']):
        _, mapped, _ = transformed(image, np.asarray(expected), n, operation)
        np.testing.assert_allclose(actual, mapped, atol=2e-4, rtol=0)
    np.testing.assert_allclose(row['gray_plateaus'], original['gray_plateaus'], atol=1e-4)
    assert row['normal_transition_width'] == pytest.approx(original['normal_transition_width'], abs=2e-4)
    assert row['direct'] is original['direct'] is True
    assert row['fit_eligible'] is row['physical_identity_qualified'] is False


def test_four_axis_directions_have_the_same_true_tangent_five_tap_width():
    image, query, normal = edge()
    widths = []
    for _ in range(4):
        rows, missing = read(image, query, normal)
        assert not missing and rows[0]['direct']
        widths.append(rows[0]['normal_transition_width'])
        assert np.dot(rows[0]['normal'], rows[0]['tangent']) == pytest.approx(0, abs=1e-12)
        image, query, normal = transformed(image, query, normal, 'rotate')
    np.testing.assert_allclose(widths, widths[0], atol=1e-5)


@pytest.mark.parametrize('operation', ['left_right', 'up_down', 'rotate', 'rotate_clockwise'])
def test_four_border_intersection_retains_short_visible_gray_plateau(operation):
    image, query, normal = edge(query=(12., 160.))
    original, missing = read(image, query, normal)
    assert not missing and len(original) == 1
    assert 5 <= original[0]['left_plateau_extent'] < 12 and not original[0]['direct']
    image, query, normal = transformed(image, query, normal, operation)
    rows, missing = read(image, query, normal)
    assert not missing and len(rows) == 1
    assert rows[0]['left_plateau_extent'] == pytest.approx(original[0]['left_plateau_extent'])
    assert rows[0]['direct'] is False
    json.dumps(rows)


@pytest.mark.parametrize('normal', [(1., 0.), (0., 1.), (-1., 0.), (0., -1.)])
@pytest.mark.parametrize('normal_offset,tangent_offset', [(-10, 2), (0, -2), (0, 5)])
def test_person_guard_blocks_outer_tangent_taps_and_filter_support(normal, normal_offset, tangent_offset):
    image, query, n = edge(normal)
    t = np.array([-n[1], n[0]])
    guard = np.zeros(image.shape[:2], bool)
    pixel = np.rint(query+normal_offset*n+tangent_offset*t).astype(int)
    guard[pixel[1], pixel[0]] = True
    rows, missing = read(image, query, n, guard)
    assert rows == [] and len(missing) == 1


@pytest.mark.parametrize('point', [(-200., 160.), (160., 1.), (160., 318.)])
def test_outside_support_cannot_create_padded_measurements(point):
    image, _, normal = edge()
    rows, missing = read(image, point, normal)
    assert not rows and len(missing) == 1


@pytest.mark.parametrize('gray,ice,expected', [(130, 136, None), (150, 170, None), (140, 150, False), (130, 142, True)])
def test_original_weak_contrast_and_near_white_rules(gray, ice, expected):
    image, query, normal = edge(gray=gray, ice=ice)
    rows, missing = read(image, query, normal)
    if expected is None:
        assert not rows and len(missing) == 1
    else:
        assert not missing and rows[0]['direct'] is expected


def test_source_shape_without_a_programme_y_crop():
    image, query, normal = edge(hw=(100, 240), query=(120., 20.))
    rows, missing = read(image, query, normal)
    assert not missing and rows[0]['direct']
    np.testing.assert_allclose(rows[0]['uv'], [119.5, 20.], atol=1e-4)


@pytest.mark.parametrize('normals', [[[0, 0]], [[2, 0]], [[np.nan, 0]], [[1, 0], [1, 0]]])
def test_invalid_unit_normal_contract_is_rejected(normals):
    image, query, _ = edge()
    with pytest.raises(ValueError, match='iceward_normals'):
        profiles_at_source_keys(image, np.zeros(image.shape[:2], bool), query_uv=[query], iceward_normals=normals)


def test_shapes_and_empty_queries():
    image, query, normal = edge()
    with pytest.raises(ValueError, match='guard'):
        read(image, query, normal, np.zeros((12, 12), bool))
    with pytest.raises(ValueError, match='query_uv'):
        profiles_at_source_keys(image, np.zeros(image.shape[:2], bool), query_uv=[[np.inf, 10]], iceward_normals=normal)
    assert profiles_at_source_keys(image, np.zeros(image.shape[:2], bool), query_uv=[], iceward_normals=normal) == ([], [])


def test_legacy_profiles_retains_its_original_positive_x_interface():
    image, _, _ = edge(hw=(800, 800), query=(400., 400.))
    rows, missing = profiles(image, 0., 400., np.zeros(image.shape[:2], bool), query_ys=[400.])
    assert not missing and len(rows) == 1 and rows[0]['direct']
    np.testing.assert_allclose(rows[0]['uv'], [399.5, 400.], atol=1e-4)
    assert 'fit_eligible' not in rows[0]


def test_queries_remain_independent_and_polarity_is_not_automatically_flipped():
    image, _, _ = edge()
    guard = np.zeros(image.shape[:2], bool)
    guard[120, 160] = True
    rows, missing = profiles_at_source_keys(image, guard,
        query_uv=[[160, 120], [160, 220]], iceward_normals=[1., 0.])
    assert [r['source_query_index'] for r in rows] == [1]
    assert [r['source_query_index'] for r in missing] == [0]
    np.testing.assert_allclose(rows[0]['uv'], [159.5, 220], atol=1e-4)
    wrong, unknown = read(image, [160, 220], [-1., 0.])
    assert wrong == [] and len(unknown) == 1
