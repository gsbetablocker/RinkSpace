from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.blade_contact_modes import (
    blade_axes_diagnostics, inner_edge_y_sign, source_edge_schedule)


def test_axes_neutral_pitch_and_roll_have_distinct_geometric_meanings():
    # Right-handed +y rotation pitches the heel-to-toe axis downward.
    R = Rotation.from_euler('xyz', [[0, 0, 0], [0, -30, 0], [40, 0, 0]], degrees=True).as_matrix()
    diagnostics = blade_axes_diagnostics(R.reshape(1, 3, 3, 3))
    np.testing.assert_allclose(diagnostics['fore_aft_pitch_degrees'], [[0, 30, 0]], atol=1e-12)
    np.testing.assert_allclose(diagnostics['edge_bank_degrees'], [[0, 0, 40]], atol=1e-12)
    np.testing.assert_allclose(diagnostics['transverse_inclination_degrees'], [[0, 0, 40]], atol=1e-12)
    np.testing.assert_allclose(diagnostics['total_lean_degrees'], [[0, 30, 40]], atol=1e-12)
    np.testing.assert_allclose(diagnostics['plane_normal_to_ice_normal_degrees'], [[90, 90, 50]], atol=1e-12)
    np.testing.assert_allclose(diagnostics['oriented_plane_normal_to_ice_normal_degrees'], [[90, 90, 50]], atol=1e-12)
    for name, column in [('forward_world', 0), ('lateral_world', 1), ('up_world', 2)]:
        np.testing.assert_array_equal(diagnostics[name], R[None, ..., :, column])
    assert diagnostics['ice_normal_world'].shape == (1, 3, 3)
    negative_bank = blade_axes_diagnostics(Rotation.from_euler('x', -40, degrees=True).as_matrix())
    assert all(isinstance(value, np.ndarray) for value in negative_bank.values())
    assert negative_bank['edge_bank_degrees'] == pytest.approx(-40)
    assert negative_bank['plane_normal_to_ice_normal_degrees'] == pytest.approx(50)
    assert negative_bank['oriented_plane_normal_to_ice_normal_degrees'] == pytest.approx(130)


def test_axes_use_the_supplied_world_ice_normal_and_reject_degenerate_input():
    R = Rotation.from_euler('xyz', [15, -20, 45], degrees=True).as_matrix()
    scene_rotation = Rotation.from_euler('y', 37, degrees=True).as_matrix()
    base = blade_axes_diagnostics(R)
    changed = blade_axes_diagnostics(scene_rotation@R, 3*scene_rotation[:, 2])
    for key in base:
        if key.endswith('_degrees'):
            np.testing.assert_allclose(changed[key], base[key], atol=1e-12)
    with pytest.raises(ValueError, match='nonzero'):
        blade_axes_diagnostics(R, [0, 0, 0])


def test_inner_side_uses_neutral_anatomy_and_final_registration_only():
    joints = np.zeros((22, 3))
    joints[7, 0], joints[8, 0] = 1, -1
    # x=forward native z, y=lateral native x, z=up native y.
    fixed = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    rotations = np.stack([fixed, fixed])
    binding = SimpleNamespace(parameters=lambda: (rotations, None, None))
    reference = SimpleNamespace(body=SimpleNamespace(J=joints))
    np.testing.assert_array_equal(inner_edge_y_sign(reference, binding), [-1, 1])
    rotations[:] = Rotation.from_euler('y', 180, degrees=True).as_matrix()@fixed
    np.testing.assert_array_equal(inner_edge_y_sign(reference, binding), [1, -1])
    joints[8] = joints[7]
    np.testing.assert_array_equal(inner_edge_y_sign(reference, binding), [0, 0])
    np.testing.assert_array_equal(inner_edge_y_sign(SimpleNamespace(), binding), [0, 0])


def interval(start=100, end=180, state='supported', name='episode'):
    return dict(id=name, foot=1, start_pts=start, end_pts=end,
                role='fit', optimization_enabled=True, state=state)


def evidence(pts, name, inside, outside, **changes):
    row = dict(pts=pts, foot=1, id=name, role='fit', visibility='visible',
               semantic='source_edge_mode', source_id='source',
               mode_costs=dict(inside_edge=inside, outside_edge=outside))
    row.update(changes)
    return row


def policy(rows=(), **changes):
    result = dict(source_id='source', intervals=[interval()],
                  edge_switch_cost=5., edge_evidence=list(rows))
    result.update(changes)
    return result


def test_no_source_edge_evidence_stays_unknown_even_with_interval_mode():
    p = policy(intervals=[dict(interval(), mode='inside_edge')])
    result = source_edge_schedule(np.arange(100, 181, 10), 1, p)
    assert set(result['modes']) == {'flat_or_unknown'}
    assert result['switches'] == []
    assert result['transition_cost'] == 0
    assert result['physical_contact_truth'] == 'unknown'


def test_finite_source_switch_cost_rejects_weak_isolated_reversal():
    rows = [evidence(100, 'a', 0, 20), evidence(140, 'weak', 2, 0), evidence(180, 'b', 0, 20)]
    result = source_edge_schedule(np.arange(100, 181, 10), 1, policy(rows))
    assert set(result['modes']) == {'inside_edge'}
    assert result['switches'] == []
    assert result['transition_cost'] == 0
    assert result['admitted_evidence_ids'] == ['a', 'weak', 'b']


def test_strong_source_reversal_switches_only_at_real_evidence_pts():
    rows = [evidence(100, 'a', 0, 20), evidence(140, 'strong', 30, 0), evidence(180, 'b', 0, 20)]
    pts = np.arange(100, 181, 10)
    result = source_edge_schedule(pts, 1, policy(rows))
    np.testing.assert_array_equal(result['modes'], ['inside_edge']*4+['outside_edge']*4+['inside_edge'])
    assert [r['pts'] for r in result['switches']] == [140, 180]
    assert [r['source_evidence_ids'] for r in result['switches']] == [['strong'], ['b']]
    assert all(r['pts'] in pts for r in result['switches'])
    assert result['transition_cost'] == 10


@pytest.mark.parametrize('change', [dict(source_id=None), dict(source_id='other'),
                                    dict(visibility='unknown'), dict(role='review'),
                                    dict(semantic='model_lowest_feature'),
                                    dict(mode_costs={'inside_edge': float('nan')})])
def test_unknown_or_unidentified_source_rows_cannot_upgrade_mode(change):
    row = evidence(100, 'unadmitted', 0, 100, **change)
    result = source_edge_schedule(np.arange(100, 181, 10), 1, policy([row]))
    assert set(result['modes']) == {'flat_or_unknown'}
    assert not result['switches']
    assert not result['admitted_evidence_ids']


def test_unknown_episode_and_missing_policy_source_id_cannot_upgrade():
    rows = [evidence(100, 'a', 0, 20)]
    for p in [policy(rows, source_id=None), policy(rows, intervals=[interval(state='unknown')])]:
        result = source_edge_schedule([100, 140, 180], 1, p)
        assert set(result['modes']) == {'flat_or_unknown'}


def test_airborne_boundary_does_not_carry_edge_prior_between_episodes():
    intervals = [interval(100, 120, name='before'), interval(130, 150, 'airborne', 'flight'),
                 interval(160, 180, name='after')]
    rows = [evidence(110, 'before', 0, 20)]
    result = source_edge_schedule(np.arange(100, 181, 10), 1, policy(rows, intervals=intervals))
    np.testing.assert_array_equal(result['modes'], ['inside_edge']*3+['flat_or_unknown']*6)
    assert result['switches'] == []
    assert result['transition_cost'] == 0


def test_toe_pick_or_heel_requires_its_own_source_cost_and_exact_sample():
    rows = [evidence(100, 'heel', 0, 20, mode_costs={'heel': 0}),
            evidence(135, 'not-native', 0, 20, mode_costs={'toe_pick': 0})]
    result = source_edge_schedule(np.arange(100, 181, 10), 1, policy(rows))
    assert set(result['modes']) == {'heel'}
    assert result['admitted_evidence_ids'] == ['heel']
    assert result['rejected_evidence'][0]['id'] == 'not-native'


def test_evidence_requires_explicit_finite_switch_cost():
    p = policy([evidence(100, 'a', 0, 20)])
    del p['edge_switch_cost']
    with pytest.raises(ValueError, match='Explicit edge_switch_cost'):
        source_edge_schedule([100], 1, p)
    p['edge_switch_cost'] = np.inf
    with pytest.raises(ValueError, match='finite'):
        source_edge_schedule([100], 1, p)
