"""Small mathematical checks; no model assets, neural inference, or native DLL."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from calibration.program_tracking.body_tracking.absolute_motion_curve import AbsoluteMotionCurve, clamped_cubic_knots, greville_times, actual_angular_motion


class CurveChecks(unittest.TestCase):
    def curve(self):
        knots = clamped_cubic_knots(np.r_[0., .08, .19, .31, .45, .57, .73, .89, 1.])
        g = greville_times(knots)
        p = np.c_[2*g, -g, 1.+.3*g]
        r = Rotation.from_rotvec(np.c_[0*g, 0*g, 4*np.pi*g]).as_matrix()[:, None]
        return AbsoluteMotionCurve(knots, p, r)

    def test_nonuniform_affine_position_and_two_full_turns(self):
        c = self.curve()
        t = np.linspace(0, 1, 103)
        s = c.evaluate(t)
        np.testing.assert_allclose(s['P'], np.c_[2*t, -t, 1.+.3*t], atol=2e-14)
        expected = Rotation.from_rotvec(np.c_[0*t, 0*t, 4*np.pi*t]).as_matrix()
        np.testing.assert_allclose(s['Q'], expected, atol=5e-14)
        motion = actual_angular_motion(s['Q'], t)
        np.testing.assert_allclose(motion['omega'][:, 2], 4*np.pi, atol=2e-12)
        self.assertLess(np.max(np.abs(motion['alpha'])), 1e-9)

    def test_actual_controls_can_change_without_frozen_reference_added_back(self):
        c = self.curve()
        dp = np.ones_like(c.position_controls)*.1
        dr = np.zeros(c.rotation_controls.shape[:-2]+(3,))
        dr[:, :, 2] = .2
        s0 = c.evaluate([.5]); s1 = c.retracted(dp, dr).evaluate([.5])
        np.testing.assert_allclose(s1['P']-s0['P'], .1, atol=2e-14)
        relative = Rotation.from_matrix(s0['Q'][0].T @ s1['Q'][0]).as_rotvec()
        np.testing.assert_allclose(relative, [0., 0., .2], atol=1e-13)

    def test_local_support_and_chain_jacobian(self):
        c = self.curve(); second = .423
        blocks = c.rotation_control_jacobian(second)
        self.assertLessEqual(len(blocks), 4)
        rng = np.random.default_rng(361)
        d = rng.normal(size=c.rotation_controls.shape[:-2]+(3,))
        eps = 1e-6
        rp = c.retracted(np.zeros_like(c.position_controls), eps*d).evaluate([second])['Q'][0]
        rm = c.retracted(np.zeros_like(c.position_controls), -eps*d).evaluate([second])['Q'][0]
        r0 = c.evaluate([second])['Q'][0]
        fd = (Rotation.from_matrix(r0.T @ rp).as_rotvec()-Rotation.from_matrix(r0.T @ rm).as_rotvec())/(2*eps)
        predicted = sum(j @ d[k, 0] for k, j in blocks.items())
        np.testing.assert_allclose(predicted, fd, atol=2e-9, rtol=2e-8)

    def test_world_rotation_covariance_and_domain(self):
        c = self.curve(); a = Rotation.from_rotvec([.2, -.3, .4]).as_matrix()
        d = AbsoluteMotionCurve(c.knots, c.position_controls @ a.T, a @ c.rotation_controls)
        t = np.linspace(.1, .9, 7)
        np.testing.assert_allclose(d.evaluate(t)['Q'], a @ c.evaluate(t)['Q'], atol=3e-14)
        with self.assertRaises(ValueError): c.evaluate([-.01])
        with self.assertRaises(ValueError): clamped_cubic_knots([0, .1, .1, 1])

    def test_sparse_seed_cannot_silently_reverse_two_turns(self):
        def reference(t):
            t = np.asarray(t)
            q = Rotation.from_rotvec(np.c_[0*t, 0*t, 4*np.pi*t]).as_matrix()
            return dict(P=np.zeros((len(t), 3)), Q=q, local_R=np.tile(np.eye(3), (len(t), 1, 1, 1)))
        native = np.linspace(0, 1, 51)
        with self.assertRaisesRegex(ValueError, 'misses observed rotation'):
            AbsoluteMotionCurve.seed_from_reference(reference, [0, 1], source_seconds=native)
        curve = AbsoluteMotionCurve.seed_from_reference(reference, np.linspace(0, 1, 11), source_seconds=native)
        np.testing.assert_allclose(curve.evaluate(native)['Q'], reference(native)['Q'], atol=2e-14)

    def test_bad_rotations_are_not_silently_normalized(self):
        r = np.tile(np.eye(3), (3, 1, 1)); r[1, 0, 0] = 2.
        with self.assertRaises(ValueError): actual_angular_motion(r, [0, .02, .04])


if __name__ == '__main__':
    unittest.main()
