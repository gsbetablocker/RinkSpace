from types import SimpleNamespace as NS
import json
import numpy as np
import pytest

from calibration.program_tracking.current_source_point_collection import collect_current_active_source_points
from calibration.program_tracking.current_source_points import (
    CurrentSourcePoints, freeze_current_source_point_inputs, save_current_source_point_binding)


def active_input():
    row = lambda uv, pts=100: dict(uv=np.asarray(uv, float), pts=pts, actual_pts=100,
                                  split='fit', id='actual')
    bundle = NS(a=np.array([[1, 2], [99, 98]]), b=np.array([[3, 4], [97, 96]]),
                ia=np.array([0, 1]), ib=np.array([1, 0]), stamps=[100, 103], motion_rows=4)
    block = NS(name='cut', bundle=bundle, quadratic=np.ones(12), groups=[(100, [
        row([[80, 81]]), dict(row([[40, 41]]), split='development'),
        dict(row([[0, 0]]), source_uv_is_image_boundary_context_only=True,
             curve_observation_kind='clipped_horizontal_extent'),
        dict(row([[900, 900]]), curve_observation_kind='horizontal_extent',
             source_y_role='context only; y ignored by whole component extent')])])
    state = NS(original=NS(aliases={100: 100, 101: 100, 103: 103}),
               bypts={100: block, 103: block}, blocks=[block], external=None,
               cross_fit=[dict(factor_source=dict(native_support_bounds_xy_px=[[1, 2], [3, 4]]))],
               patterns=NS(fit=[dict(source_actual_pts=103, physical_ad_id=13,
                                    source_curve_support=[dict(uv=[50, 51])])], held=[{}]))
    source = NS(static_motion_role_retirement=NS(global_rows=[2, 3]),
                factors=[NS(contract=dict(direct_manual_centre_secondary_objective=True),
                            rows=[row([[40, 41]])])],
                original_role_retirement=NS(active_global_rows=[4, 5]),
                native_ice_factor=NS(rows=[row([[60, 61], [90, 91]])], held=[{}],
                    contract=dict(schema='ice'), _qualified_active_fit_view=lambda:
                        NS(rows=[row([[60, 61]])])),
                native_board_track_factor=NS(contract=dict(schema='tracks'), held=[{}],
                    anchor_rows=[{}], fit_evaluation_rows=[
                        dict(source_actual_pts=100, target_actual_pts=103,
                             uv0=[[20, 21]], uv1=[[22, 23]]),
                        dict(row([[30, 31]], 101), _anchor=True)]))
    wide = NS(contract=dict(source_pts=90), uvs=[[[100, 101]]],
              ad_packet=dict(rows=[dict(source_frame_uv=[102, 103])]), trace_uv=[[[104, 105]]])
    return state, source, wide


def test_collector_uses_real_current_active_views_and_both_correspondence_ends():
    groups, receipts, pts = collect_current_active_source_points(*active_input())
    points = {p: {tuple(v) for v in uv} for p, uv in groups}
    assert points == {100: {(1., 2.), (40., 41.), (60., 61.), (20., 21.), (30., 31.)},
                      103: {(3., 4.), (22., 23.), (50., 51.)}}
    np.testing.assert_array_equal(pts, [100, 103])
    byfamily = {r['family']: r for r in receipts}
    assert byfamily['programme:cut:motion']['excluded_retired_pixel_count'] == 2
    assert byfamily['programme:cut:absolute']['excluded_nonpixel_rows'] == 2
    assert byfamily['native_ice_factor']['excluded_retired_pixel_count'] == 1
    assert byfamily['native_ice_factor']['qualified_active_fit_view_used']
    assert byfamily['native_board_track_factor']['explicit_current_anchor_rows_included'] == 1
    assert byfamily['wide_source']['outside_programme_pixel_count'] == 3
    assert byfamily['direct_manual_centre_secondary']['pixel_input_count'] == 1
    assert all(r['complete_active_input'] for r in receipts)


def test_bad_source_clock_or_unenumerated_installed_pixel_family_fails():
    state, source, wide = active_input()
    source.native_ice_factor._qualified_active_fit_view = lambda: NS(rows=[
        dict(uv=[[60, 61]], pts=101, actual_pts=103, split='fit')])
    with pytest.raises(ValueError, match='exposure alias'):
        collect_current_active_source_points(state, source, wide)
    state, source, wide = active_input()
    source.future_pixel_factor = NS(rows=[dict(uv=[[1, 2]])])
    with pytest.raises(ValueError, match='Unenumerated'):
        collect_current_active_source_points(state, source, wide)


def test_installed_material_factor_alias_is_collected_once_by_object_identity():
    state, source, wide = active_input()
    factor = NS(contract=dict(schema='native-material'),
                rows=[dict(uv=[[70., 71.]], pts=100, actual_pts=100, split='fit')], held=[])
    source.native_board_material_point_factor = factor
    wrapper = NS(base=source, material_factor=factor)
    # Real production wrappers delegate canonical factor names to their base.
    class DelegatingSource:
        def __init__(self, value):self.__dict__.update(vars(value))
        def __getattr__(self, name):return getattr(self.base, name)
    groups, receipts, _ = collect_current_active_source_points(state, DelegatingSource(wrapper), wide)
    assert sum(np.count_nonzero(np.all(uv == [70., 71.], axis=1)) for _, uv in groups) == 1
    alias = next(r for r in receipts if r['family'] == 'material_factor')
    assert alias['shared_object_alias'] and alias['alias_of'] == 'native_board_material_point_factor'
    assert alias['pixel_input_count'] == 0
    wrapper.material_factor = NS(contract=dict(schema='native-material'), rows=factor.rows)
    with pytest.raises(ValueError, match='Unenumerated'):
        collect_current_active_source_points(state, DelegatingSource(wrapper), wide)


def test_states_bind_one_frozen_UV_file_and_tampering_is_rejected(tmp_path):
    groups, receipts, pts = collect_current_active_source_points(*active_input())
    manifest = tmp_path / 'input.json'
    manifest.write_text('{}')
    shared = tmp_path / 'active_pixels.npz'
    frozen = freeze_current_source_point_inputs(shared, groups, actual_pts=pts,
        input_manifest=manifest, family_receipts=receipts)
    untouched = shared.read_bytes()
    camera = tmp_path / 'camera.npz'
    np.savez(camera, pts=np.array([100, 101, 103]),
             exposure_representative_pts=np.array([100, 100, 103]),
             metadata_json=json.dumps(dict(source_sha256='a'*64, source_time_base='1/15360')))
    for q in ('b'*64, 'c'*64):
        binding = tmp_path / (q[0]+'_points.npz')
        save_current_source_point_binding(binding, camera, frozen, q_sha256=q, input_manifest=manifest)
        with np.load(binding, allow_pickle=False) as data:
            assert 'uv' not in data and 'offsets' not in data
        view = CurrentSourcePoints(binding, camera, q_sha256=q)
        assert len(view.points(100)) == 5 and len(view.points(103)) == 3
        assert shared.read_bytes() == untouched
    shared.write_bytes(untouched+b'changed')
    with pytest.raises(ValueError, match='Frozen current source UV'):
        CurrentSourcePoints(binding, camera, q_sha256=q)
