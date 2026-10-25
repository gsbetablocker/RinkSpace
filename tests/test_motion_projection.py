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

    def test_instant_snapshot_preserves_nonzero_installation_and_zoom_optics(self):
        self.rig[15:24]=[.2,-.3,.4, .1,-.2,.5, .07,.03,.04]
        self.rig[26:38]=[51.,49.,.02, 2.,3., .01,.02,.003,.004,.001,.002,.02]
        self.controls[:,2]=np.log(100.)+.1
        result=self.core.evaluate_motion_camera(self.rig,self.controls,-2.,1.,0.)
        self.assertEqual(result['status'].tolist(),[0])
        self.assertEqual(result['values'].shape,(1,24))
        np.testing.assert_allclose(result['rotation'][0],np.eye(3),atol=1e-15)
        np.testing.assert_allclose(result['centre'][0],[.3,-.43,.9034],atol=1e-15)
        np.testing.assert_allclose(result['values'][0,16:],
            [100*np.exp(.11),100*np.exp(.09),51.2,49.3,.012,.0034,.001,.002],rtol=1e-14)
        self.assertAlmostEqual(result['zoom_log_ratio'][0],.1,places=14)

    def test_instant_at_solved_row_time_reproduces_complete_rs_projection(self):
        self.rig[-1]=.02
        self.rig[18:24]=[.02,-.03,.8,.01,.04,-.02]
        self.rig[26:37]=[50.,50.,.01,.3,-.2,.01,.02,.001,.002,.0003,-.0002]
        self.controls[:,0]=np.linspace(-.02,.02,8)
        self.controls[:,1]=np.linspace(-.03,.03,8)
        self.controls[:,2]+=np.linspace(-.05,.05,8)
        xyz=np.array([[.2,.4,10.]])
        moving=self.core.project_motion(self.rig,self.controls,-2.,1.,101,0.,xyz)
        self.assertEqual(moving['status'].tolist(),[0])
        row_time=self.rig[-1]*(moving['row'][0]/100.-.5)
        instant=self.core.evaluate_motion_camera(self.rig,self.controls,-2.,1.,row_time)
        self.assertEqual(instant['status'].tolist(),[0])
        v=instant['values'][0]
        static=[*v[4:16],*v[16:20],0.,*v[20:24],0.]
        uv,status=self.core.project(static,xyz)
        self.assertEqual(status,[0])
        np.testing.assert_allclose(uv,moving['uv'],rtol=0.,atol=1e-12)
        centre=self.core.evaluate_motion_camera(self.rig,self.controls,-2.,1.,0.)
        self.assertGreater(np.max(abs(instant['values']-centre['values'])),1e-7)

    def test_instant_failed_time_rows_retain_nan_and_status(self):
        result=self.core.evaluate_motion_camera(self.rig,self.controls,-2.,1.,[0.,100.,np.nan])
        self.assertEqual(result['status'].tolist(),[0,3,1])
        self.assertTrue(np.isfinite(result['values'][0]).all())
        self.assertTrue(np.isnan(result['values'][1:]).all())

    def test_old_library_snapshot_capability_is_explicit(self):
        old=NativeCore.__new__(NativeCore)
        old._lib=object()
        with self.assertRaisesRegex(ValueError,'lacks instantaneous motion-camera snapshot capability'):
            old.evaluate_motion_camera(self.rig,self.controls,-2.,1.,0.)


if __name__=="__main__":
    unittest.main()
