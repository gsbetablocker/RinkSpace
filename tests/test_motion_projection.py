"""Native physical-camera semantics; synthetic optics are not fit evidence."""
import os
import unittest
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.compat.d78_state import decode_camera, project_native_pts


@unittest.skipUnless(os.environ.get("RINKSPACE_CORE_LIBRARY"), "explicit native library required")
class MotionProjection(unittest.TestCase):
    def setUp(self):
        self.core = NativeCore(os.environ["RINKSPACE_CORE_LIBRARY"])
        # Identity head: pan Z, tilt X. Constant pinhole, no readout skew.
        self.rig = [1.,0.,0.,0.,1.,0.,0.,0.,1., 0.,0.,1., 1.,0.,0., 0.,0.,0., 0.,0.,1.,
                    0.,0.,0., 100.,1., 50.,50.,0., 0.,0., 0.,0.,0.,0.,0.,0.,0.]
        self.controls = np.tile([0.,0.,np.log(100.)], (8,1))

    def test_pinhole_and_nonzero_lever_are_preserved(self):
        a = self.core.project_motion(self.rig,self.controls,-2.,1.,101,0.,[[1.,0.,10.]])
        self.assertEqual(a["status"].tolist(),[0])
        self.assertAlmostEqual(a["uv"][0,0],50+100/9,places=10)
        self.assertAlmostEqual(a["depth"][0],9.,places=10)
        moved = list(self.rig);moved[18]=.7
        b = self.core.project_motion(moved,self.controls,-2.,1.,101,0.,[[1.,0.,10.]])
        self.assertAlmostEqual(b["uv"][0,0],50+100*.3/9,places=10)

    def test_native_pts_difference_and_predicted_row(self):
        self.rig[-1]=.02
        self.controls[:,1]=np.linspace(-.02,.02,8)
        camera={"rig":self.rig,"first_control_time":-2.,"control_dt":1.,"image_height":101,
                "reference_pts":10**12,"first_pts":10**12,"stop_pts_exclusive":10**12+1024,
                "time_base_num":1,"time_base_den":12800}
        a=project_native_pts(self.core,camera,self.controls,np.int64(10**12+256),[[0.,.5,10.]])
        self.assertEqual(a["status"].tolist(),[0])
        self.assertLess(abs(a["root_equality_px"][0]),1e-8)
        self.assertLess(a["contraction_bound"][0],1.)
        with self.assertRaises(ValueError):
            project_native_pts(self.core,camera,self.controls,float(10**12+256),[[0.,.5,10.]])

    def test_support_and_sensor_failures_are_explicit(self):
        a=self.core.project_motion(self.rig,self.controls,-2.,1.,101,[100.,0.],[[0.,0.,10.],[0.,8.,10.]])
        self.assertEqual(a["status"].tolist(),[3,4])
        self.assertTrue(np.isnan(a["uv"]).all())

    def test_unknown_camera_extensions_cannot_be_ignored(self):
        model={"camera":{"unsupported_lens_model":1},"blocks":[]}
        with self.assertRaisesRegex(ValueError,"unsupported d78 camera fields"):
            decode_camera(model,self.core)


if __name__=="__main__":
    unittest.main()
