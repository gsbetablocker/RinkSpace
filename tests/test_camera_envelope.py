"""Continuous physical limits constrain results, not duplicate head heights."""
import ctypes as c
import os
import unittest
import numpy as np
from rinkspace.core.native import NativeCore,D,N
from rinkspace.core.gpu import ResidentResidualProblem
import test_g2_gpu as fixture


@unittest.skipUnless(os.environ.get('RINKSPACE_G3_LIBRARY'),'explicit native build required')
class CameraEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.core=NativeCore(os.environ['RINKSPACE_G3_LIBRARY'])
        self.rig=np.zeros(38)
        self.rig[:9]=[1,0,0,0,0,-1,0,1,0]
        self.rig[9:12]=[0,0,1];self.rig[12:15]=[1,0,0]
        self.rig[15:18]=[0,-3,1.5];self.rig[18:21]=[0,-.15,.1]
        self.rig[24:28]=[100.,1.,50.,50.]
        self.controls=np.c_[np.zeros(8),np.full(8,.15),np.log(100.)+(np.arange(8)-1)/5]
        self.limits=np.r_[[-.1,.3,-.1,.1],np.radians([3,5,-20,5,-3,3])]
        self.fn=self.core._lib.rs_camera_envelope_evaluate
        self.fn.argtypes=[D,N,D,N,c.c_double,D,N,D,N];self.fn.restype=c.c_int

    def evaluate(self,rig=None,controls=None,limits=None,dt=1.):
        rig=np.ascontiguousarray(self.rig if rig is None else rig)
        controls=np.ascontiguousarray(self.controls if controls is None else controls)
        limits=np.ascontiguousarray(self.limits if limits is None else limits);out=np.empty(4)
        self.assertEqual(self.fn(rig.ctypes.data_as(D),rig.size,controls.ctypes.data_as(D),len(controls),dt,limits.ctypes.data_as(D),limits.size,out.ctypes.data_as(D),out.size),0)
        return out

    def motion_limits(self,channel,order,value):
        limits=np.r_[self.limits,np.full(9,np.inf)];limits[10+3*(order-1)+channel]=value
        return limits

    def test_interior_speed_peak_cannot_hide_between_exposures(self):
        dt=.02;speed=.2;controls=self.controls[:4].copy()
        # Independent polynomial oracle: x'(u)=4*speed*u*(1-u), zero at
        # both cell endpoints, peak at u=.5. Frame-only checks would miss it.
        controls[:,0]=np.array([4/3,-2/3,4/3,-2/3])*speed*dt
        limits=self.motion_limits(0,1,.15)
        got=self.core.evaluate_camera_envelope(self.rig,controls,-dt,dt,limits)
        self.assertEqual((got['kind'],got['channel'],got['derivative_order']),(7,0,1))
        self.assertAlmostEqual(got['u'],.5,places=13)
        self.assertAlmostEqual(got['margin'],1-speed/.15,places=12)
        self.assertAlmostEqual(got['derivative'],speed,places=12)
        self.assertAlmostEqual(got['time'],.01,places=13)

    def test_acceleration_and_jerk_use_seconds_for_every_channel(self):
        for dt in (.02,.13):
            for channel in range(3):
                for order in (2,3):
                    with self.subTest(dt=dt,channel=channel,order=order):
                        controls=self.controls[:4].copy();controls[:]=[0,.15,np.log(100)]
                        controls[:,channel]+=np.array([4/3,-2/3,4/3,-2/3])*.002
                        expected=.008/dt**2 if order==2 else .016/dt**3
                        got=self.core.evaluate_camera_envelope(self.rig,controls,0,dt,
                            self.motion_limits(channel,order,expected/2))
                        self.assertEqual((got['kind'],got['channel']),(6+order,channel))
                        self.assertAlmostEqual(got['margin'],-1,places=10)
                        self.assertAlmostEqual(abs(got['derivative'])/expected,1,places=10)

    def test_motion_limit_is_explicit_and_cannot_silently_freeze(self):
        np.testing.assert_array_equal(self.evaluate(),self.evaluate(limits=np.r_[self.limits,np.full(9,np.inf)]))
        for value in (0.,-1.,np.nan,-np.inf):
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.core.evaluate_camera_envelope(self.rig,self.controls,0,1,self.motion_limits(0,1,value))

    def test_gpu_trial_changes_motion_inside_declared_speed_acceleration_and_jerk(self):
        f=fixture.G2GPU();f.setUp();self.rig[27]=540.
        first=-.04;dt=.02;controls=np.tile([0,.15,np.log(100.)],(8,1))
        kw={**f.kw,'rig':self.rig,'controls':controls,'image_height':1080,
            'first_control_time':first,'control_dt':dt,'times':np.array([.001,.021,.041,.061])}
        # A true cubic trajectory basis with all three derivatives nonzero.
        t=first+np.arange(8)*dt
        direction=t**3-t*dt**2
        target=controls.copy();target[:,0]+=100*direction
        with ResidentResidualProblem(self.core,**{**kw,'controls':target}) as p:
            observed=p.evaluate('cuda')['values'][:,3:5]
        for order,limit in ((1,.02),(2,.4),(3,5.)):
            with self.subTest(order=order):
                limits=self.motion_limits(0,order,limit)
                with ResidentResidualProblem(self.core,**{**kw,'observed':observed}) as p:
                    maxima={['max_speed','max_acceleration','max_jerk'][order-1]:[limit,np.inf,np.inf]}
                    p.configure_camera_envelope(forward_bounds=limits[:2],pupil_bounds=limits[2:4],
                        max_pan_tilt=limits[4],max_tilt_deviation=limits[5],
                        elevation_bounds=limits[6:8],roll_bounds=limits[8:10],**maxima)
                    directions=np.zeros((8,3,1));directions[:,0,0]=direction
                    p.configure_pixel_linearization(directions=directions,local_ids=np.full(4,-1,np.int32),local_scales=np.zeros(0))
                    p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
                    result=p.lm_step()['summary'];self.assertTrue(result['accepted'],result)
                    state=p.export_state();self.assertGreater(np.max(abs(state['controls']-controls)),1e-6)
                    check=self.core.evaluate_camera_envelope(state['rig'],state['controls'],first,dt,limits)
                    self.assertGreaterEqual(check['margin'],-4e-14,check)
                    values=p.evaluate('cuda');self.assertFalse(values['status'].any())
                    self.assertAlmostEqual(float(.5*np.sum(values['values'][:,9:11]**2)),result['trial_cost'],places=9)

    def test_pupil_quadratic_interior_extremum_is_checked(self):
        self.assertGreater(self.evaluate()[0],0.)
        rig=self.rig.copy();rig[22:24]=[.6,-.6]
        check=self.evaluate(rig=rig)
        self.assertAlmostEqual(check[0],-.05,places=12)
        self.assertEqual(check[1],5)

    def test_full_forward_sum_cannot_evade_separate_base_limit(self):
        rig=self.rig.copy();rig[20]=.28;rig[22]=.06
        check=self.evaluate(rig=rig)
        self.assertAlmostEqual(check[0],-.04,places=12)
        self.assertEqual(check[1],4)

    def test_actual_elevation_and_roll_are_checked(self):
        controls=self.controls.copy();controls[:,1]=np.radians(25.)
        self.assertLess(self.evaluate(controls=controls)[0],-.08)
        rig=self.rig.copy();angle=np.radians(7.)
        turn=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        rig[:9]=(turn@rig[:9].reshape(3,3)).ravel()
        rig[12:15]=turn@rig[12:15]
        limits=self.limits.copy();limits[5]=np.radians(10.)
        self.assertLess(self.evaluate(rig=rig,limits=limits)[0],0.)

    def test_gpu_trial_obeys_complete_envelope_without_freezing_optics(self):
        f=fixture.G2GPU();f.setUp()
        self.rig[27]=540.
        kw={**f.kw,'rig':self.rig,'controls':self.controls,'image_height':1080}
        target=self.rig.copy();target[22]=.25;target[26]+=3.
        with ResidentResidualProblem(self.core,**{**kw,'rig':target}) as p:
            observed=p.evaluate('cuda')['values'][:,3:5]
        with ResidentResidualProblem(self.core,**{**kw,'observed':observed}) as p:
            rd=np.zeros((27,2));rd[13,0]=.2;rd[15,1]=1.
            p.configure_camera_parameters(directions=rd,max_readout=.1)
            p.configure_camera_envelope(forward_bounds=self.limits[:2],pupil_bounds=self.limits[2:4],
                max_pan_tilt=self.limits[4],max_tilt_deviation=self.limits[5],
                elevation_bounds=self.limits[6:8],roll_bounds=self.limits[8:10])
            p.configure_pixel_linearization(directions=np.zeros((8,3,2)),local_ids=np.full(4,-1,np.int32),local_scales=np.zeros(0))
            p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
            result=p.lm_step()['summary'];self.assertTrue(result['accepted'],result)
            state=p.export_state()
            self.assertGreater(state['rig'][22],0.)
            self.assertNotEqual(state['rig'][26],self.rig[26])
            self.assertGreaterEqual(self.evaluate(rig=state['rig'],controls=state['controls'])[0],-4e-14)

    def test_envelope_boundary_step_matches_coupled_quadratic_oracle(self):
        # Pupil displacement is linear here: z ranges from 0 to 1, hence
        # .2 * delta[0] <= .1. A constrained optimum must also move cx through
        # the off-diagonal Hessian, although cx is absent from the constraint.
        f=fixture.G2GPU();f.setUp();rig=self.rig.copy();rig[27]=540.
        kw={**f.kw,'rig':rig,'controls':self.controls,'image_height':1080}
        target=rig.copy();target[22]=.25;target[26]+=3.
        with ResidentResidualProblem(self.core,**{**kw,'rig':target}) as p:
            observed=p.evaluate('cuda')['values'][:,3:5]
        with ResidentResidualProblem(self.core,**{**kw,'observed':observed}) as p:
            rd=np.zeros((27,2));rd[13,0]=.2;rd[15,1]=1.
            p.configure_camera_parameters(directions=rd,max_readout=.1)
            p.configure_camera_envelope(forward_bounds=self.limits[:2],pupil_bounds=self.limits[2:4],
                max_pan_tilt=self.limits[4],max_tilt_deviation=self.limits[5],
                elevation_bounds=self.limits[6:8],roll_bounds=self.limits[8:10])
            p.configure_pixel_linearization(directions=np.zeros((8,3,2)),
                local_ids=np.full(4,-1,np.int32),local_scales=np.zeros(0))
            p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),initial_lambda=1e-4,
                device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
            result=p.lm_step()['summary'];system=p.export_step_system()
            S=system['S'];rhs=system['rhs'];free=np.linalg.solve(S,rhs)
            self.assertGreater(free[0],.5)
            expected=np.array([.5,(rhs[1]-S[1,0]*.5)/S[1,1]])
            np.testing.assert_allclose(system['step'],expected,rtol=0,atol=1e-8)
            self.assertTrue(result['accepted'],result)
            self.assertGreater(result['predicted_decrease'],0.)

    def test_projected_shared_step_completes_scalar_locals_without_refactor_cycle(self):
        f=fixture.G2GPU();f.setUp();rig=self.rig.copy();rig[27]=540.
        kw={**f.kw,'rig':rig,'controls':self.controls,'image_height':1080}
        target=rig.copy();target[22]=.25;target[26]+=3.
        with ResidentResidualProblem(self.core,**{**kw,'rig':target}) as p:
            observed=p.evaluate('cuda')['values'][:,3:5]
        with ResidentResidualProblem(self.core,**{**kw,'observed':observed}) as p:
            rd=np.zeros((27,2));rd[13,0]=.2;rd[15,1]=1.
            p.configure_camera_parameters(directions=rd,max_readout=.1)
            p.configure_camera_envelope(forward_bounds=self.limits[:2],pupil_bounds=self.limits[2:4],
                max_pan_tilt=self.limits[4],max_tilt_deviation=self.limits[5],
                elevation_bounds=self.limits[6:8],roll_bounds=self.limits[8:10])
            p.configure_pixel_linearization(directions=np.zeros((8,3,2)),
                local_ids=np.array([0,1,-1,-1]),local_scales=np.full(2,.001))
            p.linearize();J=p.probe_rows(np.arange(4));residual=p.evaluate('cuda')['values'][:,9:11]
            damping=1e-4;lower=kw['stations'][:2]-.001;upper=kw['stations'][:2]+.001
            p.configure_solver(lower=lower,upper=upper,initial_lambda=damping,
                device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
            result=p.lm_step()['summary'];state=p.export_state();shared=p.export_step_system()['step'][:2]
            self.assertTrue(result['accepted'],result)
            self.assertEqual(result['reason_code'],5)
            self.assertEqual(result['bound_passes'],1)
            # Independent scalar quadratic minimum for the returned shared step.
            for row in range(2):
                tangent=J[row,:,2]
                gradient=tangent@(residual[row]+J[row,:,:2]@shared)
                expected=np.clip(-gradient/(tangent@tangent+damping),-1.,1.)
                self.assertAlmostEqual((state['stations'][row]-kw['stations'][row])/.001,expected,places=9)
            self.assertTrue(np.all(state['stations'][:2]>=lower))
            self.assertTrue(np.all(state['stations'][:2]<=upper))
            self.assertGreaterEqual(self.evaluate(rig=state['rig'],controls=state['controls'])[0],-4e-14)
            after=p.evaluate('cuda');self.assertFalse(after['status'].any())
            np.testing.assert_allclose(after['values'][:,9:11],after['values'][:,3:5]-observed,atol=1e-12)
            self.assertLess(result['trial_cost'],result['old_cost']*.05)

    def test_both_zoom_endpoints_solve_the_same_constrained_quadratic(self):
        # Two pupil coefficients couple the opposite zoom endpoints. Construct
        # pixel residuals from an independent KKT solution with both constraints
        # active; merely projecting to a feasible point is insufficient.
        f=fixture.G2GPU();f.setUp();rig=self.rig.copy();rig[27]=540.
        controls=self.controls.copy()
        controls[:,2]=np.log(100.)+2*(np.arange(8)-1)/5-1
        kw={**f.kw,'rig':rig,'controls':controls,'image_height':1080,
            'times':np.array([-.8,.4,1.9,3.8])}
        rd=np.zeros((27,3));rd[13,0]=.002;rd[14,1]=.002;rd[15,2]=1.
        plan=dict(directions=np.zeros((8,3,3)),local_ids=np.full(4,-1,np.int32),local_scales=np.zeros(0))
        with ResidentResidualProblem(self.core,**kw) as p:
            p.configure_camera_parameters(directions=rd,max_readout=.1)
            p.configure_pixel_linearization(**plan)
            uv=p.evaluate('cuda')['values'][:,3:5]
            p.linearize()
            J=p.probe_rows(np.arange(4))[:,:,:3].reshape(8,3)
        damping=1e-4;S=J.T@J+damping*np.eye(3)
        normals=np.array([[-.002,-.002,0.],[.002,-.002,0.]])
        expected=np.array([0.,.5,.02])
        rhs=S@expected-normals.T@np.array([.0001,.0003])
        observed=uv+np.linalg.lstsq(J.T,rhs,rcond=None)[0].reshape(4,2)
        with ResidentResidualProblem(self.core,**{**kw,'observed':observed}) as p:
            p.configure_camera_parameters(directions=rd,max_readout=.1)
            p.configure_camera_envelope(forward_bounds=self.limits[:2],pupil_bounds=(-.001,.001),
                max_pan_tilt=self.limits[4],max_tilt_deviation=self.limits[5],
                elevation_bounds=self.limits[6:8],roll_bounds=self.limits[8:10])
            p.configure_pixel_linearization(**plan)
            p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),initial_lambda=damping,
                device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
            result=p.lm_step()['summary'];system=p.export_step_system()
            np.testing.assert_allclose(system['S'],S,rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(system['rhs'],rhs,rtol=1e-9,atol=1e-12)
            np.testing.assert_allclose(system['step'],expected,rtol=0,atol=1e-8)
            self.assertTrue(result['accepted'],result)
            self.assertGreater(result['predicted_decrease'],0.)

    def test_nonlinear_pupil_zoom_cut_does_not_exclude_feasible_origin(self):
        # The accepted state is the KKT optimum of the linearized problem.
        # A trial-state tangent to pupil1 * log(f/fref) can exclude that state
        # and force an ascent step, even though the physical origin is legal.
        f=fixture.G2GPU();f.setUp();rig=self.rig.copy();rig[27]=540.;rig[22]=.1
        controls=np.tile([0.,.15,np.log(100.)+1],(8,1))
        kw={**f.kw,'rig':rig,'controls':controls,'image_height':1080}
        rd=np.zeros((27,2));rd[13,0]=.02
        directions=np.zeros((8,3,2));directions[:,2,1]=.2
        plan=dict(directions=directions,local_ids=np.full(4,-1,np.int32),local_scales=np.zeros(0))
        with ResidentResidualProblem(self.core,**kw) as p:
            p.configure_camera_parameters(directions=rd,max_readout=.1)
            p.configure_pixel_linearization(**plan);uv=p.evaluate('cuda')['values'][:,3:5];p.linearize()
            J=p.probe_rows(np.arange(4))[:,:,:2].reshape(8,2)
        damping=1e-4;S=J.T@J+damping*np.eye(2);normal=np.array([-.02,-.02]);rhs=-.05*normal
        free=np.linalg.solve(S,rhs);self.assertLess(float(np.prod(free)),0.)
        observed=uv+np.linalg.lstsq(J.T,rhs,rcond=None)[0].reshape(4,2)
        with ResidentResidualProblem(self.core,**{**kw,'observed':observed}) as p:
            p.configure_camera_parameters(directions=rd,max_readout=.1)
            p.configure_camera_envelope(forward_bounds=self.limits[:2],pupil_bounds=self.limits[2:4],
                max_pan_tilt=self.limits[4],max_tilt_deviation=self.limits[5],
                elevation_bounds=self.limits[6:8],roll_bounds=self.limits[8:10])
            p.configure_pixel_linearization(**plan)
            p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),initial_lambda=damping,
                device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
            result=p.lm_step()['summary'];system=p.export_step_system()
            np.testing.assert_allclose(system['step'],np.zeros(2),rtol=0,atol=1e-7)
            self.assertGreaterEqual(result['predicted_decrease'],-1e-10)

    def test_selected_exposures_do_not_constrain_unused_allocation_but_include_readout(self):
        # A remote allocated halo has a bad attitude. The actual exposures
        # use only later, constant controls. Expanding readout back into the
        # bad cell must restore the violation; it cannot be silently omitted.
        f=fixture.G2GPU();f.setUp();rig=self.rig.copy();rig[27]=540.
        controls=np.tile([0.,.15,np.log(100.)],(8,1));controls[1,1]=2.
        kw={**f.kw,'rig':rig,'controls':controls,'image_height':1080,
            'first_control_time':0.,'control_dt':1.,'times':np.linspace(3.2,3.8,4)}
        envelope=dict(forward_bounds=self.limits[:2],pupil_bounds=self.limits[2:4],
            max_pan_tilt=self.limits[4],max_tilt_deviation=self.limits[5],
            elevation_bounds=self.limits[6:8],roll_bounds=self.limits[8:10])
        with ResidentResidualProblem(self.core,**kw) as p:
            with self.assertRaisesRegex(RuntimeError,'violates complete envelope'):
                p.configure_camera_envelope(**envelope)
            p.configure_camera_envelope(**envelope,frame_time_bounds=(3.2,3.8))
            self.assertFalse(p.evaluate('cuda')['status'].any())
        with ResidentResidualProblem(self.core,**kw) as p:
            with self.assertRaisesRegex(ValueError,'contain every source exposure'):
                p.configure_camera_envelope(**envelope,frame_time_bounds=(3.3,3.8))
        for readout in (-.8,.8):
            with self.subTest(readout=readout),ResidentResidualProblem(self.core,**{**kw,'rig':np.r_[rig[:37],readout]}) as p:
                with self.assertRaisesRegex(RuntimeError,'violates complete envelope'):
                    p.configure_camera_envelope(**envelope,frame_time_bounds=(3.2,3.8))
        # The public evaluator uses the same optional grid-coordinate window.
        limits=np.r_[self.limits,np.full(9,np.inf),3.2,3.8]
        self.assertGreaterEqual(self.core.evaluate_camera_envelope(rig,controls,0.,1.,limits)['margin'],0.)


if __name__=='__main__':unittest.main()
