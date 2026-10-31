"""Independent native probes for diagnostic coordinates and evidence scope."""
import os
import unittest

import numpy as np

from rinkspace.core.native import NativeCore
from rinkspace.session.camera_diagnostics import inspect_lens_forward, inspect_shot_motion


@unittest.skipUnless(os.environ.get('RINKSPACE_G3_LIBRARY'), 'explicit native build required')
class CameraDiagnosticAuditTests(unittest.TestCase):
    def setUp(self):
        self.core = NativeCore(os.environ['RINKSPACE_G3_LIBRARY'])
        self.rig = np.zeros(38)
        self.rig[:9] = [.36, -.48, .8, .8, .6, 0., -.48, .64, .6]
        self.rig[9:12] = [0., 0., 1.]
        self.rig[12:15] = [.6, 0., -.8]
        self.rig[15:18] = [1., -2., .5]
        self.rig[18:21] = [.02, -.1, .08]
        self.rig[24:28] = [300., 2., 160., 90.]
        self.rig[31:37] = [.08, .01, .03, -.005, .001, -.002]
        self.rig[37] = .02
        self.first, self.dt = -.04, .02
        t = self.first + np.arange(20)*self.dt
        self.controls = np.c_[.3*t, .1-.05*t, np.log(300.)+.1*t]
        self.times = np.array([.05, .1, .15])

    def lens(self, rig=None, grid=(5, 3)):
        rig = self.rig if rig is None else rig
        state = self.core.evaluate_motion_camera(
            rig, self.controls, self.first, self.dt, self.times)
        return inspect_lens_forward(
            self.core, rig, self.controls, self.first, self.dt, self.times,
            self.times+51., state, 321, 181, grid_shape=grid), state

    def test_nontrivial_world_to_camera_rotation_matches_independent_brown_oracle(self):
        (summary, arrays), state = self.lens()
        grid = arrays['lens_grid']
        x = (grid[None, :, 0]-state['cx'][:, None])/state['fx'][:, None]
        y = (grid[None, :, 1]-state['cy'][:, None])/state['fy'][:, None]
        radius = x*x+y*y
        k1, k2, p1, p2 = (state[name][:, None] for name in ('k1', 'k2', 'p1', 'p2'))
        radial = 1+k1*radius+k2*radius**2
        xd = x*radial+2*p1*x*y+p2*(radius+2*x*x)
        yd = y*radial+p1*(radius+2*y*y)+2*p2*x*y
        predicted = np.stack((state['fx'][:, None]*xd+state['cx'][:, None],
                              state['fy'][:, None]*yd+state['cy'][:, None]), axis=-1)
        expected = np.linalg.norm(predicted-grid, axis=-1)
        np.testing.assert_allclose(arrays['lens_displacement_px'], expected, atol=2e-12)
        self.assertAlmostEqual(summary['maximum_displacement_pixels'], expected.max(), places=11)
        self.assertTrue(summary['mathematical_optical_domain_certificate_on_all_probes'])

    def test_lens_probe_preserves_input_and_is_independent_of_signed_readout(self):
        original_rig = self.rig.copy()
        original_controls = self.controls.copy()
        (positive, pa), _ = self.lens()
        negative_rig = self.rig.copy()
        negative_rig[37] *= -1
        (negative, na), _ = self.lens(negative_rig)
        np.testing.assert_array_equal(self.rig, original_rig)
        np.testing.assert_array_equal(self.controls, original_controls)
        np.testing.assert_array_equal(pa['lens_displacement_px'], na['lens_displacement_px'])
        np.testing.assert_array_equal(pa['lens_jacobian_determinant'], na['lens_jacobian_determinant'])
        self.assertTrue(positive['temporary_readout_zero_only_for_lens_diagnostic'])
        self.assertEqual(positive, negative)

    def test_native_certificate_failure_is_counted_without_claiming_a_fold(self):
        rig = self.rig.copy()
        rig[31:37] = [0., 0., 0., 0., .4, 0.]
        (summary, arrays), _ = self.lens(rig, grid=(3, 3))
        self.assertGreater(summary['native_domain_failure_probes'], 0)
        self.assertFalse(summary['mathematical_optical_domain_certificate_on_all_probes'])
        self.assertEqual(summary['nonpositive_sampled_jacobians'], 0)
        self.assertTrue(np.isnan(arrays['lens_jacobian_determinant']).any())

    def test_native_continuous_derivative_peak_keeps_shifted_source_clock(self):
        speed, dt = .4, .2
        controls = np.tile([0., .1, np.log(300.)], (4, 1))
        controls[:, 0] = np.array([4/3, -2/3, 4/3, -2/3])*speed*dt
        times = np.array([.21, .25, .39])
        rig = self.rig.copy()
        rig[37] = -.01
        summary, _, _ = inspect_shot_motion(
            self.core, rig, controls, 0., dt, times, times+56., 60.)
        peak = summary['channels']['pan']['speed']['continuous_peak']
        self.assertAlmostEqual(peak['signed_value'], np.degrees(speed), places=12)
        self.assertAlmostEqual(peak['source_seconds'], 56.3, places=12)
        self.assertGreater(peak['signed_value'], summary['channels']['pan']['speed']['sampled_peak']['signed_value'])
        np.testing.assert_allclose(summary['exposure_interval'], [56.205, 56.395], atol=1e-12)
        self.assertEqual(summary['centre']['derivative_method'],
                         'finite differences of native centres at original exposure times')


if __name__ == '__main__':
    unittest.main()
