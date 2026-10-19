import hashlib
import json

import cv2
import numpy as np
import pytest

from rinkspace.vision_source_shapes import SourceMaterialShapeReader


def _packet(rgb):
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    return dict(RGB_sha256=digest, source_binding=dict(
        source_sha256='a'*64, source_pts=100, time_base='1/15360',
        shot_id='test_raw_source', native_hw=list(rgb.shape[:2]),
        decoded_native_rgb_sha256=digest))


def _read(rgb, xs, center=90.):
    points = np.c_[xs, np.full(len(xs), center)]
    return SourceMaterialShapeReader().read_connected_material_transition_profiles(
        rgb, _packet(rgb), search_points_uv=points,
        normals=np.tile([0., 1.], (len(xs), 1)),
        search_intervals_px=np.tile([35.-center, 135.-center], (len(xs), 1)),
        transition='dark_to_light')


def test_dominant_narrow_candidate_preserves_unresolved_broad_material_context():
    rgb = np.full((160, 120, 3), 135, np.uint8)
    rgb[70:] = 175
    rgb[110:] = 188
    rgb = cv2.GaussianBlur(rgb, (0, 0), 1.5)
    first = _read(rgb, np.arange(10., 111., 2), center=80.)
    second = _read(rgb, np.arange(10., 111., 2), center=100.)
    assert not len(first['actual_source_uv'])
    candidates = np.asarray([r['actual_source_transition_candidate_uv'] for r in first['readings']])
    assert np.allclose(candidates[:, 1], 69.5, atol=.8)
    assert np.allclose(candidates, [r['actual_source_transition_candidate_uv'] for r in second['readings']], atol=.1)
    assert all(r['source_context_profile']['source_flags']['separated_same_sign_transitions'] for r in first['readings'])
    assert all(r['source_context_resolution_required'] for r in first['readings'])
    assert all(len(r['all_actual_source_material_candidates']) == 2 for r in first['readings'])
    # Returned ledgers must not contain a selected row's self-reference.
    json.dumps(first, default=lambda a: a.tolist() if isinstance(a, np.ndarray) else str(a))
    assert not first['source_interpolation_used_as_measurement']


def test_equal_parallel_material_borders_remain_unknown_without_source_authority():
    rgb = np.full((160, 120, 3), 140, np.uint8)
    rgb[65:] = 160
    rgb[105:] = 180
    rgb = cv2.GaussianBlur(rgb, (0, 0), 1.5)
    result = _read(rgb, np.arange(10., 111., 2))
    assert not len(result['actual_source_uv'])
    assert all(len(r['all_actual_source_material_candidates']) == 2 for r in result['readings'])
    assert all(r['disposition'] == 'source_material_role_unknown' for r in result['readings'])


def test_ambiguous_local_border_needs_two_independent_source_neighbors():
    rgb = np.full((160, 120, 3), 140, np.uint8)
    rgb[70:] = 180
    rgb[110:] = 190
    # A second stronger local source rise makes the middle profile ambiguous.
    rgb[70:, 48:53] = 166
    rgb[110:, 48:53] = 188
    rgb = cv2.GaussianBlur(rgb, (0, 0), 1.2)
    result = _read(rgb, np.arange(10., 111., 2))
    middle = result['readings'][20]
    assert middle['source_branch_selection_rule'] == 'one_actual_transition_connected_to_two_unambiguous_RGB_neighbors'
    assert len(middle['source_connection_anchor_indices']) == 2
    assert abs(middle['actual_source_transition_candidate_uv'][1]-69.5) < .8
    assert not middle['actual_source_uv'].size
    assert middle['source_context_profile']['source_flags']['separated_same_sign_transitions']
    assert middle['source_connection_interpolated_UV_is_not_measurement']


def test_truncated_strong_source_border_is_not_erased_to_promote_codec_terrace():
    rgb = np.full((160, 120, 3), 135, np.uint8)
    rgb[35:] = 175
    rgb[110:] = 183
    rgb = cv2.GaussianBlur(rgb, (0, 0), 1.5)
    result = _read(rgb, np.arange(10., 111., 2))
    assert not len(result['actual_source_uv'])
    for row in result['readings']:
        candidates = row['all_actual_source_material_candidates']
        assert len(candidates) == 2
        assert candidates[0]['source_flags']['search_interval_truncated']
        assert not candidates[1]['local_source_flags']['search_interval_truncated']
        assert candidates[1]['source_context_profile']['source_flags']['search_interval_truncated']
        assert candidates[1]['disposition'] == 'source_material_role_unknown'
        assert row['disposition'] == 'source_material_role_unknown'


def test_dark_stroke_center_is_not_one_material_border_and_never_auto_assigns_role():
    rgb = np.full((160, 120, 3), 200, np.uint8)
    rgb[60:66] = 160
    rgb[100:110] = 140
    rgb = cv2.GaussianBlur(rgb, (0, 0), 1.2)
    result = SourceMaterialShapeReader().read_dark_stroke_center_candidates(
        rgb, _packet(rgb), search_points_uv=[[60., 90.]], normals=[[0., 1.]],
        search_intervals_px=[[-55., 45.]], keep_profiles=True)
    row = result['readings'][0]
    assert not row['source_candidate_selected_automatically']
    candidates = row['all_actual_source_dark_stroke_candidates']
    assert len(candidates) == 2
    assert np.allclose([c['actual_source_uv'][1] for c in candidates], [62.5, 104.5], atol=.3)
    assert all(c['measurement_operator'] == 'geometric_midpoint_of_two_actual_material_interfaces' for c in candidates)
    assert all(not c['darkness_mass_quantile_used_as_geometric_center'] for c in candidates)
    assert all(not c['source_physical_role_verified'] for c in candidates)
    assert all(c['actual_RGB_entering_interface']['actual_source_uv'][1] < c['actual_source_uv'][1]
               < c['actual_RGB_leaving_interface']['actual_source_uv'][1] for c in candidates)
    assert all(c['footprint_is_not_centerline_covariance_or_solver_sigma'] for c in candidates)


def test_compound_frame_cache_converts_once_and_reauthenticates_later_mutable_RGB(monkeypatch):
    rgb = np.full((160, 120, 3), 200, np.uint8)
    rgb[60:66] = 160
    rgb[100:110] = 140
    rgb = cv2.GaussianBlur(rgb, (0, 0), 1.2)
    packet = _packet(rgb)
    convert = cv2.cvtColor
    calls = []

    def observed_convert(*args, **kwargs):
        calls.append(1)
        return convert(*args, **kwargs)

    monkeypatch.setattr(cv2, 'cvtColor', observed_convert)
    reader = SourceMaterialShapeReader()
    search = dict(search_points_uv=[[20., 90.], [60., 90.], [100., 90.]],
                  normals=[[0., 1.]]*3, search_intervals_px=[[-55., 45.]]*3)
    first = reader.read_dark_stroke_center_candidates(rgb, packet, **search)
    assert len(calls) == 1  # Twelve local borders share the same immutable proof.
    assert len(first['readings']) == 3
    assert rgb.flags.writeable
    rgb[0, 0] = 0
    with pytest.raises(ValueError, match='bound original RGB differ'):
        reader.read_dark_stroke_center_candidates(rgb, packet, **search)
