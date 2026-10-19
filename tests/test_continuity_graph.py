import copy
import json

import numpy as np
import pytest

from rinkspace.continuity_graph import (ContinuityGraphConfig, optimize_continuity,
    _chain_xy, _correction, _normalizer)


def translation(x, y=0.):
    return np.array([[1., 0., x], [0., 1., y], [0., 0., 1.]])


def chain(x, radial=False):
    return {'H_reference_undistorted_from_model': np.eye(3).tolist(),
        'H_current_sample_from_reference_raw': translation(x).tolist(),
        'H_source_image_from_sample': np.diag([2., 2., 1.]).tolist(),
        'reference_visible_branch_sign': 1, 'raw_transfer_visible_branch_sign': 1,
        'reference_branch_probe_raw': [400., 250.],
        'reference_radial_distortion': {'k1': -.04, 'centre_image': [480., 270.], 'scale_image': 960.} if radial else None}


def fixture(*, count=11, anchors=(0, 10), measured_step=3., anchor_step=3., conflict=0., radial=False):
    xx, yy = np.meshgrid(np.linspace(130., 730., 8), np.linspace(130., 400., 6))
    points = np.c_[xx.ravel(), yy.ravel()]
    rows = []
    for i in range(count):
        current = points+[measured_step*i, 0.]
        parent = points+[measured_step*(i-1), 0.]
        local = {'status': 'candidate', 'H_reference_from_current': translation(-measured_step*i).tolist(),
            'graph_component_id': 'local_000', 'reference_frame_only': i == 0,
            'train_inliers': len(points), 'support_fraction': .30,
            'inlier_p95_reference_px': .4, 'support_polygon': current[[0, 7, -1, -8]].tolist()}
        if i:
            local.update(edge_keyframe_sample_index=i-1, H_keyframe_from_current=translation(-measured_step).tolist(),
                fit_current_xy=current.tolist(), fit_keyframe_xy=parent.tolist())
        row = {'sample_index': i, 'source_frame_index': 100+i, 'source_sha256': 'a'*64,
            'source_time_s': i*.2, 'pts': i*20, 'time_base': '1/100', 'shot_id': 'one',
            'camera_component_id': 'one:component', 'sample_size': [960, 540],
            'width': 1920, 'height': 1080, 'local_registration': local,
            'status': 'unavailable', 'projection_chain': None, 'anchor_id': None,
            'ice_polygons_sample': [[[0, 0], [960, 0], [960, 540], [0, 540]]]}
        if i in anchors:
            projection = chain(anchor_step*i+(conflict if i == max(anchors) else 0.), radial=radial)
            row.update(status='model_reobserved', anchor_id=f'ref_{i}', projection_chain=projection,
                model_anchor_candidate={'reference_id': f'ref_{i}', 'projection_chain': projection,
                    'pts': i*20, 'time_base': '1/100', 'source_time_s': i*.2})
        rows.append(row)
    return rows, points


def run(rows, points, **extra):
    return optimize_continuity(rows, points, ContinuityGraphConfig(compare_constrained_plane=False, **extra))


def test_actual_edges_use_future_anchor_past_point_six_seconds_without_changing_pts():
    rows, probes = fixture(anchors=(10,))
    frozen = copy.deepcopy(rows)
    output = run(rows, probes)
    assert rows == frozen
    assert output['frames'][0]['status'] == 'image_tracked'
    assert output['frames'][0]['future_anchor_used']
    assert output['frames'][0]['anchor_age_s'] == pytest.approx(2.)
    for before, after in zip(rows, output['frames']):
        assert (before['pts'], before['time_base']) == (after['pts'], after['time_base'])
        actual, valid = _chain_xy(after['projection_chain'], probes)
        np.testing.assert_allclose(actual, probes+[3*before['sample_index'], 0.], atol=.02)
        assert valid.all()
        assert not after['independent_validation']
    json.dumps(output, allow_nan=False)


@pytest.mark.parametrize('field,value', [('shot_id', 'cut'), ('source_sha256', 'b'*64), ('camera_component_id', 'new-local')])
def test_cuts_source_and_unanchored_components_never_inherit_global_identity(field, value):
    rows, probes = fixture(anchors=(0,))
    for row in rows[5:]:
        row[field] = value
    output = run(rows, probes)
    assert all(row['projection_chain'] is None for row in output['frames'][5:])
    assert all(row['status'] == 'unavailable' for row in output['frames'][5:])


def test_missing_point_pairs_are_a_connectivity_barrier_even_with_local_candidate_matrix():
    rows, probes = fixture(anchors=(10,))
    rows[5]['local_registration'].pop('fit_keyframe_xy')
    output = run(rows, probes)
    assert all(row['projection_chain'] is None for row in output['frames'][:5])
    assert output['frames'][5]['status'] == 'image_tracked'
    assert any(item['reason'] == 'missing_actual_relative_point_pairs'
               for component in output['diagnostics']['components'] for item in component['rejected_relative_edges'])


def test_keyframe_changes_accumulate_budget_instead_of_refreshing_absolute_truth():
    rows, probes = fixture(count=31, anchors=(0,))
    output = run(rows, probes)
    early = output['frames'][2]['camera_uncertainty']['engineering_range_sample_px']
    late = output['frames'][-1]['camera_uncertainty']['engineering_range_sample_px']
    assert late > early > 2.
    assert len(output['frames'][-1]['anchor_lineage']) == 1
    assert output['frames'][-1]['anchor_lineage'][0]['observed_edge_count'] == 30
    assert not output['frames'][-1]['camera_uncertainty']['keyframe_replacement_resets_budget']


def test_incompatible_anchors_remain_multiple_hypotheses_across_neighborhood():
    rows, probes = fixture(conflict=60.)
    # A conflict observation is still a real candidate constraint, even if the
    # forward renderer cleared its selected projection as it should.
    rows[-1]['status'], rows[-1]['projection_chain'] = 'ambiguous', None
    output = run(rows, probes)
    assert all(row['status'] == 'ambiguous' for row in output['frames'])
    assert all(row['projection_chain'] is None for row in output['frames'])
    assert all(len(row['model_anchor_hypotheses']) == 2 for row in output['frames'])
    assert output['diagnostics']['components'][0]['selected_camera_strategy'] is None


def test_small_cumulative_drift_can_be_explained_by_joint_real_image_constraints():
    rows, probes = fixture(measured_step=3.2, anchor_step=3.)
    before = run(rows[:1]+[{**r, 'model_anchor_candidate': None, 'status': 'unavailable', 'projection_chain': None} for r in rows[1:]], probes)
    output = run(rows, probes)
    after_xy, _ = _chain_xy(output['frames'][-1]['projection_chain'], probes)
    before_xy, _ = _chain_xy(before['frames'][-1]['projection_chain'], probes)
    truth = probes+[30., 0.]
    assert np.mean(np.linalg.norm(after_xy-truth, axis=1)) < np.mean(np.linalg.norm(before_xy-truth, axis=1))*.4
    checks = output['diagnostics']['components'][0]['camera_hypothesis_comparisons'][0]
    assert checks['accepted'] and checks['actual_relative_edges_optimized'] == 10
    assert checks['edge_holdout_worst_increase_sample_px'] > .01
    assert not checks['withheld_points_are_independent_geometry_validation']
    assert len(output['frames'][5]['anchor_lineage']) == 2


def test_radial_chain_is_preserved_once_and_never_replaced_with_fake_matrix():
    rows, probes = fixture(radial=True)
    output = run(rows, probes)
    for row in output['frames']:
        assert row['H_image_from_model'] is None
        assert row['projection_chain']['reference_radial_distortion']['k1'] == -.04
        xy, _ = _chain_xy(row['projection_chain'], probes)
        expected, _ = _chain_xy(chain(3*row['sample_index'], radial=True), probes)
        np.testing.assert_allclose(xy, expected, atol=.02)


def test_two_camera_families_are_scored_without_treating_matrix_elements_as_states():
    rows, probes = fixture()
    output = optimize_continuity(rows, probes)
    comparisons = output['diagnostics']['components'][0]['camera_hypothesis_comparisons']
    assert [item['camera_degrees_of_freedom'] for item in comparisons] == [4, 8]
    assert output['diagnostics']['components'][0]['selected_camera_strategy'] == 'similarity'
    assert not output['diagnostics']['homography_coefficient_interpolation_used']
    similarity = _correction([.01, -.02, .03, .04], _normalizer([960, 540]))
    assert np.linalg.det(similarity) > 0


def test_engineering_budget_can_withhold_long_observed_path_without_age_timeout():
    rows, probes = fixture(count=41, anchors=(0,))
    output = run(rows, probes, max_engineering_range_px_at_960=2.5)
    assert output['frames'][0]['status'] == 'model_reobserved'
    assert output['frames'][-1]['status'] == 'unavailable'
    assert output['frames'][-1]['reason'] == 'accumulated_image_path_uncertainty_exceeded'
    assert output['frames'][-1]['projection_chain'] is None
