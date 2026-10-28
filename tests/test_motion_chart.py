"""Independent cubic spline checks for native continuous motion coordinates."""
import os,unittest
import numpy as np
from scipy.interpolate import BSpline
from rinkspace.domain.motion_chart import UniformMotionChart,cubic_samples
from rinkspace.core.native import NativeCore


class MotionChart(unittest.TestCase):
    def test_halo_direction_closure_keeps_interior_space_and_shots_separate(self):
        from rinkspace.domain.multi_shot_motion import ShotMotionTimeline,continue_exposure_halos
        shots=[]
        for i,camera in enumerate((0,1,0)):
            shots.append(dict(id=str(i),camera_index=camera,control_begin=i*81,control_stop=(i+1)*81,
                first_control_time=-.4,knot_stride=10,motion_reference_time=0.,reference_pts=i*100000,
                first_pts=i*100000,last_pts=i*100000+12800,time_base_num=1,time_base_den=12800))
        tl=ShotMotionTimeline(shots,.02);old,old_labels=tl.directions();new,labels=tl.directions(continue_endpoint_cubics=True)
        lookup={x:i for i,x in enumerate(old_labels)};keep=[lookup[x] for x in labels]
        times=np.linspace(0.,1.,257);idx,w=cubic_samples(81,-.4,.02,times)
        for shot in shots:
            sl=slice(shot['control_begin'],shot['control_stop'])
            before=np.einsum('ij,ijkl->ikl',w,old[sl][idx]);after=np.einsum('ij,ijkl->ikl',w,new[sl][idx])
            np.testing.assert_array_equal(after,before[:,:,keep])
            removed=np.setdiff1d(np.arange(len(old_labels)),keep)
            np.testing.assert_array_equal(before[:,:,removed],0.)
            for j,label in enumerate(labels):
                if label[0]!=shot['id']:np.testing.assert_array_equal(new[sl,:,j],0.)
        self.assertLess(new.shape[2],old.shape[2])
        rng=np.random.default_rng(447);delta=rng.normal(size=len(labels))*1000
        controls=np.einsum('ijk,k->ij',new,delta)
        for shot in shots:
            sl=slice(shot['control_begin'],shot['control_stop'])
            np.testing.assert_allclose(continue_exposure_halos(controls[sl],-.4,.02,[0.,1.]),controls[sl],atol=2e-9,rtol=2e-12)

    @unittest.skipUnless(os.environ.get('RINKSPACE_CORE_LIBRARY'),'explicit native library required')
    def test_native_halo_continues_endpoint_cubic_after_updates(self):
        from rinkspace.domain.multi_shot_motion import continue_exposure_halos
        core=NativeCore(os.environ['RINKSPACE_CORE_LIBRARY']);chart=UniformMotionChart.create(81,-.4,.02,10,0.)
        directions,_=chart.directions();closed=continue_exposure_halos(directions,-.4,.02,[0.,1.])
        rng=np.random.default_rng(448);state=np.einsum('ijk,k->ij',closed,rng.normal(size=closed.shape[2]))
        times=np.array([-.01,.01,.99,1.01]);kin=core.evaluate_motion_kinematics(state,-.4,.02,times)
        self.assertFalse(kin['status'].any())
        # The derivative polynomial is the same across both frame endpoints,
        # even though each endpoint lies at a coarse knot before closure.
        for a,b in ((0,1),(2,3)):
            h=times[b]-times[a]
            np.testing.assert_allclose(kin['jerk'][a],kin['jerk'][b],rtol=2e-10,atol=2e-8)
            np.testing.assert_allclose(kin['acceleration'][b],kin['acceleration'][a]+h*kin['jerk'][a],rtol=2e-10,atol=2e-8)

    def test_exposure_halo_continuation_preserves_entire_frame_interval(self):
        from rinkspace.domain.multi_shot_motion import continue_exposure_halos
        x=np.arange(24,dtype=float)
        original=np.c_[.1+.03*x-.001*x*x,2.-.02*x+.0001*x**3,5.+.003*x]
        corrupt=original.copy();corrupt[:5]+=np.array([2.,-.5,.2]);corrupt[18:]-=np.array([1.,.7,.4])
        repaired=continue_exposure_halos(corrupt,-.12,.02,[0.,.2])
        np.testing.assert_allclose(repaired,original,atol=2e-12,rtol=0)
        i,w=cubic_samples(24,-.12,.02,np.linspace(0.,.2,1001))
        np.testing.assert_array_equal(np.einsum('ij,ijk->ik',w,repaired[i]),np.einsum('ij,ijk->ik',w,corrupt[i]))
        with self.assertRaisesRegex(ValueError,'knot grid'):
            continue_exposure_halos(corrupt,-.12,.02,[.001,.2])

    def test_only_shared_installation_reference_has_angle_gauges(self):
        chart=UniformMotionChart.create(48,-.04,.02,5,0.)
        first,_=chart.directions();later,_=chart.directions(reference_angle_gauge=False)
        self.assertEqual(later.shape[2],first.shape[2]+2)
        i,w=cubic_samples(48,-.04,.02,[0.])
        reference=np.einsum('i,ijk->jk',w[0],later[i[0]])
        self.assertEqual(np.linalg.matrix_rank(reference[:2]),2)

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
