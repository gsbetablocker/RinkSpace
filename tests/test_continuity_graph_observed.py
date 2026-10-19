import copy
import runpy

import numpy as np
import pytest

from rinkspace.continuity_graph_observed import (_actual_anchor_pairs, _anchor, _chain_xy,
    ContinuityGraphConfig, optimize_continuity)
from rinkspace.continuity_graph import optimize_continuity as optimize_legacy


def fixture(native_scale=1., radial=False, wrong_transfer=0.):
    helpers = runpy.run_path('tests/test_continuity_graph.py')
    rows, points = helpers['fixture'](count=5, anchors=(0, 4))
    for i in (0, 4):
        projection = rows[i]['model_anchor_candidate']['projection_chain']
        projection.update(reference_sample_size=[int(960*native_scale), int(540*native_scale)],
            reference_image_domain='raw_px_distortion_unchecked',
            H_reference_undistorted_from_model=np.diag([native_scale, native_scale, 1.]).tolist(),
            H_current_sample_from_reference_raw=(helpers['translation'](3*i+wrong_transfer) @
                np.diag([1/native_scale, 1/native_scale, 1.])).tolist())
        raw_processing = points.copy()
        if radial:
            projection['reference_radial_distortion'] = {'k1': -.08, 'centre_image': [480*native_scale, 270*native_scale], 'scale_image': 960*native_scale}
            delta = points-[480, 270]
            raw_processing = np.array([480, 270])+delta*(1-.08*np.sum((delta/960)**2, axis=1)[:, None])
        rows[i]['model_anchor_candidate']['fit'] = {'fit_reference_xy': raw_processing.tolist(),
            'fit_current_xy': (raw_processing+[3*i, 0.]).tolist()}
    return rows, points


@pytest.mark.parametrize('native_scale,radial', [(1., False), (2., False), (2., True)])
def test_actual_reference_processing_pixels_map_through_native_size_and_one_lens_inverse(native_scale, radial):
    rows, points = fixture(native_scale, radial)
    before = copy.deepcopy(rows)
    model_xy, current_xy = _actual_anchor_pairs(rows[0], _anchor(rows[0]))
    np.testing.assert_allclose(model_xy, points, atol=1e-8)
    assert rows == before
    actual, valid = _chain_xy(rows[0]['model_anchor_candidate']['projection_chain'], model_xy)
    np.testing.assert_allclose(actual, current_xy, atol=1e-8)
    assert valid.all()


def test_actual_observations_not_dense_extrapolated_pose_define_anchor_factor():
    rows, points = fixture(wrong_transfer=24.)
    config = ContinuityGraphConfig(compare_constrained_plane=False)
    old = optimize_legacy(rows, points, config)
    result = optimize_continuity(rows, points, config)
    old_xy, _ = _chain_xy(old['frames'][2]['projection_chain'], points)
    new_xy, _ = _chain_xy(result['frames'][2]['projection_chain'], points)
    np.testing.assert_allclose(old_xy, points+[30., 0.], atol=.02)
    np.testing.assert_allclose(new_xy, points+[6., 0.], atol=.1)
    diagnostic = result['diagnostics']['components'][0]['camera_hypothesis_comparisons'][0]
    assert diagnostic['actual_anchor_count'] == 2
    assert diagnostic['legacy_synthetic_anchor_prior_count'] == 0
    assert all(c['actual_observed_pair_count'] == len(points) for c in diagnostic['anchor_constraints'])


@pytest.mark.parametrize('fault', ['missing_one', 'wrong_shape', 'nan', 'size_disagrees', 'undistorted', 'outside'])
def test_invalid_actual_observations_cannot_silently_become_synthetic_pose_prior(fault):
    rows, _ = fixture()
    row = rows[0]; anchor = row['model_anchor_candidate']; fit = anchor['fit']
    if fault == 'missing_one':
        fit.pop('fit_current_xy')
    elif fault == 'wrong_shape':
        fit['fit_current_xy'] = fit['fit_current_xy'][:-1]
    elif fault == 'nan':
        fit['fit_reference_xy'][0][0] = float('nan')
    elif fault == 'size_disagrees':
        anchor['model_reference'] = {'sample_size': [1920, 1080]}
    elif fault == 'undistorted':
        anchor['projection_chain']['reference_image_domain'] = 'undistorted'
    else:
        fit['fit_reference_xy'][0][0] = 1000.
    with pytest.raises(ValueError):
        _actual_anchor_pairs(row, _anchor(row))


def test_folded_radial_domain_is_rejected_instead_of_taking_a_second_inverse_branch():
    rows, _ = fixture(radial=True)
    rows[0]['model_anchor_candidate']['projection_chain']['reference_radial_distortion']['k1'] = -10.
    with pytest.raises(ValueError, match='monotone radial'):
        _actual_anchor_pairs(rows[0], _anchor(rows[0]))


def test_legacy_missing_both_arrays_is_explicit_pose_prior_only():
    helpers = runpy.run_path('tests/test_continuity_graph.py')
    rows, points = helpers['fixture'](count=5, anchors=(0, 4))
    result = optimize_continuity(rows, points, ContinuityGraphConfig(compare_constrained_plane=False))
    diagnostic = result['diagnostics']['components'][0]['camera_hypothesis_comparisons'][0]
    assert diagnostic['legacy_synthetic_anchor_prior_count'] == 2
    assert diagnostic['actual_anchor_count'] == 0


def test_invalid_actual_anchor_is_excluded_from_every_propagation_hypothesis():
    rows, points = fixture()
    for row in (rows[0], rows[4]):
        row['model_anchor_candidate']['fit'].pop('fit_current_xy')
    result = optimize_continuity(rows, points)
    assert all(row['projection_chain'] is None for row in result['frames'])
    assert all(row['status'] == 'unavailable' for row in result['frames'])
    component = result['diagnostics']['components'][0]
    assert component['real_anchor_count'] == 0
    assert len(component['rejected_actual_anchor_observations']) == 2
