"""Independent cubic spline checks for native continuous motion coordinates."""
import os,unittest
import numpy as np
from scipy.interpolate import BSpline
from rinkspace.domain.motion_chart import UniformMotionChart,cubic_samples
from rinkspace.core.native import NativeCore


class MotionChart(unittest.TestCase):
    def test_exact_refinement_and_reference_gauge(self):
        rng=np.random.default_rng(901)
        for stride in (1,2,5,10):
            chart=UniformMotionChart.create(258,-.04,.02,stride)
            c=rng.normal(size=(chart.refinement.shape[1],3));fine=chart.refinement@c
            times=np.linspace(-.02,5.08,1001);idx,w=cubic_samples(len(fine),chart.first,chart.dt,times)
            actual=np.einsum('ij,ijk->ik',w,fine[idx])
            knots=chart.coarse_first+chart.dt*stride*np.arange(-2,len(c)+2)
            expected=BSpline(knots,c,3)(times)
            np.testing.assert_allclose(actual,expected,atol=8e-14,rtol=0)
            directions,_=chart.directions();i,b=cubic_samples(len(fine),chart.first,chart.dt,[0.])
            np.testing.assert_allclose(np.einsum('i,ijk->jk',b[0],directions[i[0]])[:2],0,atol=2e-16)
            np.testing.assert_allclose(chart.initialize(fine),fine,atol=3e-12,rtol=0)

    @unittest.skipUnless(os.environ.get('RINKSPACE_CORE_LIBRARY'),'explicit native library required')
    def test_native_derivatives_match_independent_coarse_cubic(self):
        core=NativeCore(os.environ['RINKSPACE_CORE_LIBRARY']);rng=np.random.default_rng(902)
        chart=UniformMotionChart.create(258,-.04,.02,10);c=rng.normal(size=(chart.refinement.shape[1],3))
        fine=chart.refinement@c;times=np.linspace(-.019,5.079,701)
        actual=core.evaluate_motion_kinematics(fine,chart.first,chart.dt,times)
        np.testing.assert_array_equal(actual['status'],0)
        spline=BSpline(chart.coarse_first+chart.dt*10*np.arange(-2,len(c)+2),c,3)
        for order,key,tolerance in [(0,'position',1e-13),(1,'velocity',2e-11),(2,'acceleration',2e-9),(3,'jerk',2e-7)]:
            np.testing.assert_allclose(actual[key],spline(times,nu=order),atol=tolerance,rtol=1e-11)

    def test_invalid_grid_and_fractional_stride_are_rejected(self):
        for stride in (0,-1,1.5,True):
            with self.assertRaises(ValueError):UniformMotionChart.create(20,-.04,.02,stride)
        with self.assertRaises(ValueError):UniformMotionChart.create(20,-.04,.02,2,reference_time=20.)

    def test_initial_zoom_stays_inside_old_valid_range_without_limiting_directions(self):
        chart=UniformMotionChart.create(258,-.04,.02,10);t=np.arange(258)*.02
        controls=np.c_[.1*t,.02*t,8+.1*np.sin(3*t)]
        initialized=chart.initialize(controls,bound_zoom=True)
        times=np.linspace(-.02,5.08,1001);i,w=cubic_samples(258,-.04,.02,times)
        actual=np.einsum('ij,ijk->ik',w,initialized[i])
        knots_i,knots_w=cubic_samples(258,-.04,.02,-.04+.02*np.arange(1,257))
        old=np.einsum('ij,ijk->ik',knots_w,controls[knots_i])
        self.assertGreaterEqual(actual[:,2].min(),old[:,2].min()-1e-13)
        self.assertLessEqual(actual[:,2].max(),old[:,2].max()+1e-13)
        i,w=cubic_samples(258,-.04,.02,[0.]);np.testing.assert_allclose(w[0]@initialized[i[0],:2],w[0]@controls[i[0],:2],atol=1e-14)
        directions,labels=chart.directions()
        self.assertEqual(sum(c==2 for k,c in labels),chart.refinement.shape[1])


if __name__=='__main__':unittest.main()
