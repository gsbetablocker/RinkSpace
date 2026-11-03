"""Run against an explicitly built CPU binary; never build or launch fitting.

Set RINKSPACE_TEST_NATIVE_LIBRARY to the new library. Optionally set
RINKSPACE_TEST_LEGACY_NATIVE_LIBRARY to an unchanged ABI2 binary.
"""
import os
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from rinkspace.core.native import NativeCore


def synthetic_motion(wide=False, readout=.02):
    """Full nonzero physical rig: axes, lever, pupil/zoom and Brown terms."""
    rotation = Rotation.from_rotvec([.07, -.04, .03]).as_matrix()
    pan = np.array([.05, 1., .1]); pan /= np.linalg.norm(pan)
    axis = rotation @ pan
    tilt = np.array([1., 0., 0.]) - axis * axis[0]; tilt /= np.linalg.norm(tilt)
    rig = np.r_[rotation.ravel(), pan, tilt, [.1, -.1, .3], [.01, -.025, .015],
                [.004, .002, -.0005, 1100., 2.5, 960., 540., .02, 5., -3.,
                 -.015, .003, .001, -.0002, .0007, -.0004, readout]]
    if wide:
        rig = np.r_[rig, .03, .01, -.001, .0002, -.0003]
    x = np.arange(12)
    controls = np.c_[.05 * np.sin(.3 * x), .02 * np.cos(.27 * x),
                     np.log(1100.) + .02 * np.sin(.24 * x)]
    return rig, controls, -.5, .1


@unittest.skipUnless(os.environ.get('RINKSPACE_TEST_NATIVE_LIBRARY'), 'explicit CPU native test binary required')
class NativeBodyProjectionJacobianTests(unittest.TestCase):
    def setUp(self):
        self.core = NativeCore(os.environ['RINKSPACE_TEST_NATIVE_LIBRARY'])
        self.assertTrue(self.core.supports_motion_xyz_jacobian)

    def points(self, args):
        times = np.linspace(-.18, .28, 19)
        camera = self.core.evaluate_motion_camera(*args, times)
        self.assertFalse(camera['status'].any())
        rng = np.random.default_rng(781103)
        xyz_camera = np.c_[rng.uniform(-1.2, 1.2, len(times)),
                           rng.uniform(-.9, .9, len(times)), rng.uniform(3., 6., len(times))]
        xyz_camera[-1] = [0., 4., 5.]  # finite projected point outside the raster
        points = camera['centre'] + np.einsum('nji,nj->ni', camera['rotation'], xyz_camera)
        return times, points

    def test_full_rig_rs_roots_and_analytic_spatial_derivatives(self):
        for wide in (False, True):
            for readout in (0., .02, -.02):
                with self.subTest(wide=wide, readout=readout):
                    args = synthetic_motion(wide, readout); times, points = self.points(args)
                    call = (*args, 1080, times)
                    result = self.core.project_motion_xyz_jacobian(*call, points)
                    original = self.core.project_motion(*call, points)
                    np.testing.assert_array_equal(result['status'], original['status'])
                    self.assertFalse(result['status'].any())
                    for key in ('uv', 'row', 'depth', 'root_equality_px', 'contraction_bound'):
                        np.testing.assert_array_equal(result[key], original[key])
                    self.assertGreater(result['uv'][-1, 1], 1079.)
                    h = 2e-5; numeric = np.empty_like(result['jacobian_xyz'])
                    for axis in range(3):
                        direction = np.zeros_like(points); direction[:, axis] = h
                        plus = self.core.project_motion(*call, points + direction)
                        minus = self.core.project_motion(*call, points - direction)
                        self.assertFalse(plus['status'].any() or minus['status'].any())
                        numeric[:, :, axis] = (plus['uv'] - minus['uv']) / (2 * h)
                    np.testing.assert_allclose(result['jacobian_xyz'], numeric, rtol=2e-6, atol=2e-4)

    def test_parallel_rows_are_bitwise_identical(self):
        args = synthetic_motion(True); times, points = self.points(args)
        call = (*args, 1080, times)
        self.core.motion_projection_threads = 1
        serial = self.core.project_motion_xyz_jacobian(*call, points)
        self.core.motion_projection_threads = 4
        parallel = self.core.project_motion_xyz_jacobian(*call, points)
        for key in serial:
            np.testing.assert_array_equal(serial[key], parallel[key])

    def test_failed_rows_keep_original_status_and_nan_derivatives(self):
        args = synthetic_motion(True); times, points = self.points(args)
        camera = self.core.evaluate_motion_camera(*args, times)
        points[0] = np.nan
        points[1] = camera['centre'][1] + camera['rotation'][1].T @ np.array([0., 0., -2.])
        times[2] = 10.
        times[3] = np.nan
        result = self.core.project_motion_xyz_jacobian(*args, 1080, times, points)
        original = self.core.project_motion(*args, 1080, times, points)
        np.testing.assert_array_equal(result['status'], original['status'])
        self.assertTrue(np.all(result['status'][:4] != 0))
        self.assertTrue(np.isnan(result['uv'][:4]).all())
        self.assertTrue(np.isnan(result['jacobian_xyz'][:4]).all())

    def test_empty_and_invalid_inputs(self):
        args = synthetic_motion()
        result = self.core.project_motion_xyz_jacobian(*args, 1080, np.empty(0), np.empty((0, 3)))
        self.assertEqual(result['jacobian_xyz'].shape, (0, 2, 3))
        for value in (0, -1, True, 2147483648):
            self.core.motion_projection_threads = value
            with self.assertRaises(ValueError):
                self.core.project_motion_xyz_jacobian(*args, 1080, 0., np.zeros((1, 3)))
        self.core.motion_projection_threads = 1
        with self.assertRaises(ValueError):
            self.core.project_motion_xyz_jacobian(*args, 1080, 0., np.zeros((1, 2)))


@unittest.skipUnless(os.environ.get('RINKSPACE_TEST_LEGACY_NATIVE_LIBRARY'), 'optional legacy ABI2 binary required')
class NativeLegacyCapabilityTests(unittest.TestCase):
    def test_historical_abi2_still_loads_and_reports_missing_capability(self):
        core = NativeCore(os.environ['RINKSPACE_TEST_LEGACY_NATIVE_LIBRARY'])
        self.assertFalse(core.supports_motion_xyz_jacobian)
        self.assertFalse(core.supports_parallel_motion_projection)
        args = synthetic_motion()
        with self.assertRaisesRegex(ValueError, 'lacks analytic'):
            core.project_motion_xyz_jacobian(*args, 1080, 0., np.zeros((1, 3)))


if __name__ == '__main__':
    unittest.main()
