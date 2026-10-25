"""A one-exposure snapshot must preserve the whole rolling readout, not one pose."""
from copy import deepcopy
import os
import unittest

import numpy as np

from rinkspace.session.camera_exposure import FAMILY, exposure_controls
from rinkspace.core.native import NativeCore


class ExposureControls(unittest.TestCase):
    def setUp(self):
        self.rig = [1.,0.,0.,0.,1.,0.,0.,0.,1., 0.,0.,1., 1.,0.,0.,
                    0.,0.,0., .1,.2,.3, .04,.02,-.01, 100.,2.,
                    50.,50.,.01, 2.,3., .01,.002,.001,.0001,.001,.002,.02]
        self.controls = np.column_stack((np.linspace(-.01,.01,12),
            np.linspace(-.03,.03,12), np.linspace(4.5,4.7,12)))
        self.camera = {**FAMILY, "rig": self.rig, "first_control_time": -.04,
                       "control_dt": .02, "reference_pts": 10**12,
                       "time_base_num": 1, "time_base_den": 12800,
                       "first_pts": 10**12, "stop_pts_exclusive": 10**12+2048}

    def test_trim_keeps_original_coefficients_for_both_readout_signs(self):
        for tau in (-.02, 0., .02):
            self.camera["rig"][-1] = tau
            local, controls, _ = exposure_controls(self.camera, self.controls, np.int64(10**12+768))
            start, stop = local["snapshot_origin"]["control_indices"]
            np.testing.assert_array_equal(controls, self.controls[start:stop])
            self.assertLess(len(controls), len(self.controls))
            self.assertEqual(local["first_pts"], 10**12+768)
            self.assertEqual(local["rig"], self.camera["rig"])

    def test_rejects_non_native_pts_unsupported_family_and_missing_readout_support(self):
        for pts in (True, float(10**12), 10**12-1):
            with self.assertRaises(ValueError):
                exposure_controls(self.camera, self.controls, pts)
        invalid = deepcopy(self.camera); invalid["lens_family"] = "unknown"
        with self.assertRaisesRegex(ValueError, "family"):
            exposure_controls(invalid, self.controls, 10**12)
        self.camera["rig"][-1] = .2
        with self.assertRaisesRegex(ValueError, "full readout"):
            exposure_controls(self.camera, self.controls, 10**12)

    @unittest.skipUnless(os.environ.get("RINKSPACE_CORE_LIBRARY"), "explicit native library required")
    def test_native_projection_and_row_camera_survive_exact_support_trim(self):
        core = NativeCore(os.environ["RINKSPACE_CORE_LIBRARY"])
        for tau in (-.02, 0., .02):
            self.camera["rig"][-1] = tau
            local, controls, t = exposure_controls(self.camera, self.controls, 10**12+768)
            times = t+np.array([-.5,0.,.5])*tau
            a = core.evaluate_motion_camera(self.rig, self.controls, -.04, .02, times)
            b = core.evaluate_motion_camera(self.rig, controls, local["first_control_time"], .02, times)
            self.assertTrue(np.all(a["status"] == 0)); np.testing.assert_array_equal(a["status"], b["status"])
            np.testing.assert_allclose(a["values"], b["values"], atol=2e-12, rtol=1e-13)
            a = core.project_motion(self.rig, self.controls, -.04, .02, 101, t, [[0.,0.,10.],[1.,1.,10.]])
            b = core.project_motion(self.rig, controls, local["first_control_time"], .02, 101, t, [[0.,0.,10.],[1.,1.,10.]])
            np.testing.assert_array_equal(a["status"], b["status"])
            self.assertTrue(np.all(a["status"] == 0))
            np.testing.assert_allclose(a["uv"], b["uv"], atol=2e-10, rtol=1e-13)
