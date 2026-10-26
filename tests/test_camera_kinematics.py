"""Native time derivatives are checked against independent polynomial oracles."""
import os,unittest
import numpy as np
from rinkspace.core.native import NativeCore

@unittest.skipUnless(os.environ.get('RINKSPACE_G3_LIBRARY'),'explicit native build required')
class CameraKinematics(unittest.TestCase):
    def setUp(self):
        self.core=NativeCore(os.environ['RINKSPACE_G3_LIBRARY'])

    def test_linear_quadratic_and_cubic_physical_time(self):
        for dt in (.02,.17):
            first=-.25;grid=first+dt*np.arange(16)
            controls=np.column_stack((.7+2.3*grid,grid**2-dt**2/3,grid**3-grid*dt**2))
            times=first+dt*np.array([1.,1.25,2.,4.3,11.8,14.])
            got=self.core.evaluate_motion_kinematics(controls,first,dt,times)
            self.assertFalse(got['status'].any())
            want=np.stack((np.column_stack((.7+2.3*times,times**2,times**3)),
                np.column_stack((times*0+2.3,2*times,3*times**2)),
                np.column_stack((times*0,times*0+2,6*times)),
                np.column_stack((times*0,times*0,times*0+6))),axis=1)
            np.testing.assert_allclose(got['values'],want,atol=8e-10,rtol=2e-12)

    def test_same_values_as_camera_and_c2_knots(self):
        rng=np.random.default_rng(206);controls=rng.normal(size=(12,3))*.01;controls[:,2]+=7
        rig=np.zeros(38);rig[:9]=np.array([[1,0,0],[0,0,-1],[0,1,0]]).ravel()
        rig[9:12]=[0,0,1];rig[12:15]=[1,0,0];rig[15:18]=[0,-1,.03];rig[18:21]=[0,-.002,.001];rig[24:28]=[1000,2,960,540]
        times=np.arange(0,.16,.0037);kin=self.core.evaluate_motion_kinematics(controls,-.04,.02,times)
        pose=self.core.evaluate_motion_camera(rig,controls,-.04,.02,times)
        np.testing.assert_array_equal(kin['position'],pose['values'][:,:3])
        for t in (.02,.04,.08,.12):
            pair=self.core.evaluate_motion_kinematics(controls,-.04,.02,[t-1e-10,t+1e-10])['values']
            np.testing.assert_allclose(pair[0,:3],pair[1,:3],atol=3e-6,rtol=0)

    def test_outside_support_stays_invalid(self):
        got=self.core.evaluate_motion_kinematics(np.zeros((8,3)),-.04,.02,[-.021,-.02,.08,.081])
        np.testing.assert_array_equal(got['status'],[3,0,0,3])
        self.assertTrue(np.isnan(got['values'][[0,3]]).all())

    def test_exact_extrema_respect_interval_and_hidden_interior_peak(self):
        dt=.02;speed=.2;first=-dt;controls=np.zeros((4,3))
        controls[:,0]=np.array([4/3,-2/3,4/3,-2/3])*speed*dt
        full=self.core.motion_derivative_extrema(controls,first,dt,[0,dt])
        np.testing.assert_allclose(full['value'][:,0],[speed,4*speed/dt,-8*speed/dt**2],rtol=2e-14)
        self.assertAlmostEqual(full['time'][0,0],dt/2,places=14)
        partial=self.core.motion_derivative_extrema(controls,first,dt,[0,dt/4])
        self.assertAlmostEqual(partial['value'][0,0],.75*speed,places=14)
        self.assertAlmostEqual(partial['time'][0,0],dt/4,places=14)
        endpoint=self.core.motion_derivative_extrema(controls,first,dt,[dt,dt])
        self.assertAlmostEqual(endpoint['value'][0,0],0.,places=13)
        with self.assertRaises(ValueError):self.core.motion_derivative_extrema(controls,first,dt,[-.001,dt])

if __name__=='__main__':unittest.main()
