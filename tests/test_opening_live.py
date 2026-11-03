"""Live opening-factor regressions on small articulated states, without assets."""
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.body_execution import BodyExecution, BodyExecutionConfig
from calibration.program_tracking.body_tracking.fullbody_state import matrices
from calibration.program_tracking.body_tracking.opening_live import OpeningLiveFactors
from calibration.program_tracking.body_tracking.opening_motion import START
from calibration.program_tracking.body_tracking.opening_timing import QUIET_POSITION_SCALE
from calibration.program_tracking.body_tracking.skate_integration import SkateReference


class TwoTimeState:
    """Two independent time controls expose the temporal scatter explicitly."""
    active = np.array([0, 6, 51, 66])
    d = 4
    m = 2
    size = 8

    def basis(self, times):
        t = np.asarray(times)/.2
        return np.c_[1-t, t]

    def increments(self, controls, times):
        delta = np.zeros((len(times), 69))
        delta[:, self.active] = self.basis(times)@np.asarray(controls).reshape(self.m, self.d)
        return delta


class TinySkin:
    """One waist sample and 24 hand samples, driven by the wrist proxy."""
    def __init__(self):
        self.body = SimpleNamespace(J=np.zeros((55, 3)), scale=1.)
        self.shaped = np.zeros((25, 3))
        self.shaped[0] = [-.1, .1, 0.]
        self.shaped[1:, 0] = np.linspace(-.01, .01, 24)
        self.weights = np.zeros((25, 55))
        self.weights[0, 0] = 1.
        self.weights[1:, 21] = 1.

    def sample_vertices(self, joints, frames, vertex_ids, double_precision=False):
        vertices = np.broadcast_to(self.shaped, (len(joints), 25, 3)).copy()
        vertices[:, 1:] += joints[:, 21, None]
        return vertices[:, vertex_ids]


def parent_state(times):
    n = len(times)
    return dict(P=np.zeros((n, 3)), Q=np.tile(np.eye(3), (n, 1, 1)),
                local_R=np.tile(np.eye(3), (n, 21, 1, 1)), times=np.asarray(times))


def articulated_world(ref, delta):
    """Minimal rotational state and skin driver; no copied factor formulas."""
    local = ref['local_R']@matrices(delta[:, 6:].reshape(-1, 21, 3))
    P = ref['P']+delta[:, :3]
    Q = ref['Q']@matrices(delta[:, 3:6])
    native = np.broadcast_to(P[:, None], (len(P), 22, 3)).copy()
    arm = Rotation.from_matrix(local[:, 15]).as_rotvec()[:, 0]
    wrist = Rotation.from_matrix(local[:, 20]).as_rotvec()[:, 0]
    native[:, 21, 0] += arm+wrist
    return dict(P=P, Q=Q, local_R=local, native=native,
                frames=np.concatenate((Q[:, None], local), axis=1))


def make_body(saved_displacement=0.):
    state = TwoTimeState()
    parent = SimpleNamespace(at=parent_state,
        p={'pts': np.rint((START+np.array([0., .2]))*12800).astype(np.int64)})
    reference = SkateReference.__new__(SkateReference)
    reference.parent = parent
    reference.new_refinements = []
    reference.binding = SimpleNamespace(surface=TinySkin())
    reference.world = articulated_world
    # The production SkateReference.at composes two saved layers before the
    # live factor applies the current trial's controls through world().
    for fraction in (.4, .6):
        controls = np.zeros((2, 4))
        controls[1, 0] = saved_displacement*fraction
        reference.new_refinements.append((state, controls.ravel()))
    return SimpleNamespace(reference=reference, state=state, size=state.size,
        frames=np.arange(2), mid_times=np.array([0., .2]),
        execution=BodyExecution(BodyExecutionConfig()))


class OpeningLiveTests(unittest.TestCase):
    def test_quiet_common_translation_is_free_and_single_time_has_two_sided_derivative(self):
        body = make_body()
        factor = OpeningLiveFactors(body)
        value = factor.evaluate(np.zeros(body.size), True)
        quiet = factor.rows['source_confirmed_quiet_body']
        J = value['jac'][quiet].toarray()
        scale = np.sqrt(.02/2)/QUIET_POSITION_SCALE
        np.testing.assert_allclose(J[0, [0, 4]], [-scale, scale], atol=1e-10)
        self.assertEqual(np.count_nonzero(abs(J[0]) > 1e-10), 2)
        common = np.zeros((2, 4))
        common[:, 0] = .13
        shifted = factor.evaluate(common.ravel(), True)
        np.testing.assert_allclose(shifted['residual'][quiet], value['residual'][quiet], atol=1e-12)
        np.testing.assert_allclose(J@common.ravel(), 0., atol=1e-12)
        one_time = np.zeros((2, 4))
        one_time[1, 0] = .03
        moved = factor.evaluate(one_time.ravel())
        np.testing.assert_allclose(moved['residual'][quiet], value['residual'][quiet]+J@one_time.ravel(), atol=1e-12)
        self.assertAlmostEqual(moved['metrics']['quiet_position_range_m'], .03)

    def test_quiet_evaluates_composed_saved_layers_plus_current_final_state(self):
        body = make_body(saved_displacement=.05)
        factor = OpeningLiveFactors(body)
        quiet = factor.rows['source_confirmed_quiet_body']
        inherited = factor.evaluate(np.zeros(body.size))
        self.assertAlmostEqual(inherited['metrics']['quiet_position_range_m'], .05)
        self.assertGreater(np.linalg.norm(inherited['residual'][quiet]), .49)
        controls = np.zeros((2, 4))
        controls[1, 0] = -.05
        final = factor.evaluate(controls.ravel(), True)
        np.testing.assert_allclose(final['residual'][quiet], 0., atol=1e-12)
        self.assertAlmostEqual(final['metrics']['quiet_position_range_m'], 0.)
        self.assertEqual(final['source_pts'], body.reference.p['pts'].tolist())
        # Re-evaluation must follow the next live trial rather than the last
        # supplied x, the initial seed, or just the current correction alone.
        again = factor.evaluate(np.zeros(body.size))
        np.testing.assert_allclose(again['residual'], inherited['residual'], atol=1e-12)

    def test_arms_are_excluded_from_quiet_but_live_hand_relation_changes(self):
        body = make_body()
        factor = OpeningLiveFactors(body)
        quiet = factor.rows['source_confirmed_quiet_body']
        before = factor.evaluate(np.zeros(body.size), True)
        controls = np.zeros((2, 4))
        controls[1, 2:] = [.12, .18]
        after = factor.evaluate(controls.ravel(), True)
        np.testing.assert_allclose(after['residual'][quiet], before['residual'][quiet], atol=1e-12)
        arm_columns = [2, 3, 6, 7]
        np.testing.assert_allclose(after['jac'][quiet][:, arm_columns].toarray(), 0., atol=1e-10)
        hand = factor.rows['observed_right_hand_waist_relation']
        self.assertGreater(np.linalg.norm(after['residual'][hand]-before['residual'][hand]), .1)
        self.assertNotEqual(after['metrics']['hand_waist_max_distance_m'], before['metrics']['hand_waist_max_distance_m'])

    def test_empty_opening_range_has_no_state_or_surface_requirement(self):
        body = SimpleNamespace(reference=SimpleNamespace(p={'pts': np.array([0, 256])}),
                               frames=np.arange(2), mid_times=np.array([0., .02]), size=5)
        factor = OpeningLiveFactors(body)
        for jacobian in (False, True):
            result = factor.evaluate(np.zeros(body.size), jacobian)
            self.assertEqual(result['residual'].shape, (0,))
            self.assertEqual(result['source_pts'], [])
            self.assertEqual(result['parts'], {})
            self.assertEqual(result['metrics'], {})
            if jacobian:
                self.assertEqual(result['jac'].shape, (0, body.size))
                self.assertEqual(result['jac'].nnz, 0)
            else:
                self.assertIsNone(result['jac'])


if __name__ == '__main__':
    unittest.main()