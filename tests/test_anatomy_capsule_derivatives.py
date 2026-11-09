"""CPU regressions for the analytic capsule-pair branch of collision rows."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.programme_anatomy import Capsule, ProgrammeAnatomy
from calibration.program_tracking.body_tracking.skate_physics import fk_state


RADIUS = 0.08


def _capsule_pair(*, ratio=1.0, gap=0.05, parallel=False):
    # Joints 3 and 4 carry the capsules. Joint 1 moves both, while joint 2
    # moves only the first capsule and is distinct from its attachment joint.
    parents = np.zeros(22, dtype=int)
    parents[:5] = [-1, 0, 1, 2, 1]
    joints = np.zeros((22, 3))
    joints[:5] = [[0, 0, 0], [0.08, -0.14, 0.10], [-0.18, -0.08, 0.24],
                  [-0.16, 0.20, 0.15], [0.31, -0.12, 0.21]]

    def shoe_transforms(P, Q, native, rotations):
        return np.broadcast_to(np.eye(3), (len(P), 2, 3, 3)), np.zeros((len(P), 2, 3))

    ref = SimpleNamespace(body=SimpleNamespace(J=joints / ratio, parents=parents, scale=0.92),
                          binding=SimpleNamespace(transforms=shoe_transforms))
    n = 3
    base = dict(P=np.array([[0.02, 0.03, -0.04], [-0.01, 0.0, 0.02], [0.04, -0.05, 0.01]]) / ratio,
                Q=Rotation.from_rotvec([[0.12, 0.08, -0.21], [0.0, -0.09, 0.15], [0.09, -0.16, 0.30]]).as_matrix(),
                local_R=np.broadcast_to(np.eye(3), (n, 21, 3, 3)).copy())
    base['local_R'][:, :4] = Rotation.from_rotvec(
        [[0.13, 0.01, -0.12], [0.03, -0.05, 0.17], [0.09, 0.18, -0.08], [-0.16, 0.03, 0.04]]).as_matrix()
    delta = np.zeros((n, 69))
    delta[:, 3:18] = [0.07, 0.06, -0.03, -0.09, 0.04, 0.12, 0.23, -0.19, 0.09,
                      -0.16, 0.08, 0.21, 0.13, 0.12, -0.22]
    state = fk_state(ref, base, delta)

    # Specify the desired centreline geometry at sample zero and express it
    # in joint-local metric coordinates. The other poses differ rigidly.
    endpoints_a = [[-0.26, 0.10, 0.12], [0.32, 0.10, 0.12]]
    endpoints_b = ([[-0.18, 0.10, 0.12 + gap], [0.27, 0.10, 0.12 + gap]] if parallel else
                   [[0.14, -0.22, 0.12 + gap], [0.14, 0.46, 0.12 + gap]])

    def capsule(joint, endpoints):
        frame = state['Q'][0] @ state['frames'][0, joint]
        origin = state['native_world'][0, joint] * ratio
        start, end = (np.asarray(endpoints) - origin) @ frame
        return Capsule(joint, start, end, RADIUS, ())

    anatomy = object.__new__(ProgrammeAnatomy)
    anatomy.ratio = ratio
    anatomy.times = np.arange(n) * 0.02
    anatomy.capsules = {'a': capsule(3, endpoints_a), 'b': capsule(4, endpoints_b)}
    anatomy.boxes = {}
    anatomy.pairs = [('a', 'b')]
    anatomy.part_joint_ancestry = {'a': {1, 2, 3}, 'b': {1, 4}}
    anatomy.left_rear_grip_seconds = None
    return anatomy, ref, base, delta


@pytest.mark.parametrize('ratio', [1.0, 0.01])
def test_capsule_envelope_derivative_matches_fk_central_difference(ratio):
    anatomy, ref, base, delta = _capsule_pair(ratio=ratio)
    all_samples = np.arange(len(delta))
    active = np.array([0, 2])
    lookup = np.arange(len(active))
    pairs = np.zeros(len(active), dtype=int)
    weights = np.array([0.45, 1.3, 0.8])
    local = np.zeros((len(active), 1, 69))

    def weighted_depth(parameters):
        state = fk_state(ref, base, parameters)
        depth, _, _ = anatomy.collision_values(state, all_samples)
        return weights[active] * depth[active, 0]

    state = fk_state(ref, base, delta)
    handled = anatomy.capsule_local_jacobian(
        state, delta[active], active, active, pairs, lookup, weights, local)

    np.testing.assert_array_equal(handled, [True, True])
    assert np.all(weighted_depth(delta) > 0.01)
    step = 1e-6
    numerical = np.column_stack([(weighted_depth(delta + bump) - weighted_depth(delta - bump)) / (2 * step)
                                for bump in step * np.eye(69)])
    np.testing.assert_allclose(local[:, 0], numerical, rtol=2e-6, atol=2e-8)
    for first in (9, 12, 15):
        assert np.all(np.linalg.norm(local[:, 0, first:first + 3], axis=1) > 1e-4)
    # Root translation, root rotation and the common ancestor all act rigidly.
    np.testing.assert_array_equal(local[:, 0, :9], 0.0)
    rigid = delta.copy()
    rigid[:, :9] += [0.1, -0.03, 0.07, 0.20, -0.15, 0.12, -0.15, 0.25, 0.08]
    np.testing.assert_allclose(weighted_depth(rigid), weighted_depth(delta), rtol=0, atol=2e-14)


@pytest.mark.parametrize(('gap', 'parallel', 'expected_singular'), [
    pytest.param(0.05, True, 0, id='parallel'),
    pytest.param(2.5e-7, False, 1, id='near-coincident'),
    pytest.param((2.0 - 0.12) * RADIUS - 5e-6, False, 0, id='near-zero-penetration'),
])
def test_capsule_singular_and_threshold_rows_leave_local_jacobian_to_fd(gap, parallel, expected_singular):
    anatomy, ref, base, delta = _capsule_pair(gap=gap, parallel=parallel)
    state = fk_state(ref, base, delta)
    sample = np.array([0])
    local = np.full((1, 1, 69), np.nan)
    depth, _, singular = anatomy.collision_values(state, np.arange(len(delta)))
    assert depth[0, 0] > 0.0
    assert singular == expected_singular * len(delta)

    handled = anatomy.capsule_local_jacobian(
        state, delta[sample], sample, sample, sample, sample, np.ones(len(delta)), local)

    np.testing.assert_array_equal(handled, [False])
    assert np.isnan(local).all()
