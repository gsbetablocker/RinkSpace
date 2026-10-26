"""Physical rig unknowns must differentiate, update and survive export on CUDA."""
import ctypes as c
import os
import unittest
import numpy as np
from rinkspace.core.native import NativeCore, D, N
from rinkspace.core.gpu import ResidentResidualProblem
import test_g2_gpu as fixture

@unittest.skipUnless(os.environ.get("RINKSPACE_G3_LIBRARY"), "explicit CUDA/cuDSS build required")
class JointCameraGPU(unittest.TestCase):
    def setUp(self):
        f=fixture.G2GPU();f.setUp();self.kw=f.kw
        self.core=NativeCore(os.environ["RINKSPACE_G3_LIBRARY"])
        rig=np.array(self.kw["rig"]);rig[18:24]=[.02,-.01,.05,.004,.01,-.02]
        rig[28:37]=[.01,.4,-.7,.02,.03,-.01,.02,.004,.002]
        self.kw["rig"]=rig
        self.kw["controls"][:,2]+=np.linspace(-.05,.08,8)
        self.scale=np.array([.001]*6+[.01]*9+[.2,.2,.01,.2,.2,.1,.1,.1,.1,.01,.01,.005])
        fn=self.core._lib.rs_camera_retract
        fn.argtypes=[D,N,D,N,D,N];fn.restype=c.c_int
        self.retract_fn=fn

    def retract(self, delta):
        rig=self.kw["rig"];delta=np.ascontiguousarray(delta,np.float64);out=np.empty(38)
        assert self.retract_fn(rig.ctypes.data_as(D),38,delta.ctypes.data_as(D),27,out.ctypes.data_as(D),38)==0
        return out

    def plan(self, q):
        return dict(directions=np.zeros((8,3,q)),local_ids=np.full(4,-1,np.int32),local_scales=np.zeros(0),
            group_offsets=np.arange(5,dtype=np.int32),members=np.arange(4,dtype=np.int32),
            ranks=np.full(4,2,np.int32),weight_offsets=np.arange(5,dtype=np.int32)*4,
            weights=np.tile(np.eye(2).ravel(),4),huber_delta=100.)

    def attach(self,p):
        p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),device_budget_bytes=p.creation["free_after_alloc_bytes"],host_budget_bytes=2**30)

    def test_all_physical_directions_match_projection_and_cpu_with_rolling_shutter(self):
        rig_dirs=np.diag(self.scale);plan=self.plan(27)
        # A retained column may jointly carry a trajectory and a rig direction.
        plan["directions"][:,0,0]=.0003
        with ResidentResidualProblem(self.core,**self.kw) as p:
            p.configure_camera_parameters(directions=rig_dirs,max_readout=.08)
            p.configure_linearization(**plan);p.linearize();gpu=p.probe_rows(np.arange(4))
            p.linearize("cpu");cpu=p.probe_rows(np.arange(4),"cpu")
            np.testing.assert_allclose(gpu,cpu,atol=1e-11,rtol=1e-10)
            epsilon=.002
            for j in range(27):
                delta=rig_dirs[:,j]*epsilon;values=[]
                for sign in (-1,1):
                    rig=self.retract(sign*delta)
                    np.testing.assert_allclose(rig[:9].reshape(3,3)@rig[:9].reshape(3,3).T,np.eye(3),atol=1e-13)
                    self.assertLess(abs(rig[12:15]@(rig[:9].reshape(3,3)@rig[9:12])),1e-13)
                    controls=self.kw["controls"]+sign*epsilon*plan["directions"][:,:,j]
                    with ResidentResidualProblem(self.core,**{**self.kw,"rig":rig,"controls":controls}) as sample:
                        result=sample.evaluate("cuda");np.testing.assert_array_equal(result["status"],0)
                        values.append(result["values"][:,3:5])
                derivative=(values[1]-values[0])/(2*epsilon)
                np.testing.assert_allclose(gpu[:,:,j],derivative,rtol=3e-4,atol=3e-7,err_msg=f"chart coordinate {j}")
            self.assertGreater(np.linalg.norm(gpu[:,:,26]),1e-7,"readout direction was lost")

    def test_real_lm_updates_installation_lever_pupil_and_optics(self):
        for coordinate in (6,10,12,13,15,20,26):
            with self.subTest(coordinate=coordinate):
                direction=np.zeros((27,1));direction[coordinate,0]=self.scale[coordinate]
                target_rig=self.retract(direction[:,0]*.2)
                with ResidentResidualProblem(self.core,**{**self.kw,"rig":target_rig}) as target:
                    observed=target.evaluate("cpu")["values"][:,3:5]
                with ResidentResidualProblem(self.core,**{**self.kw,"observed":observed}) as p:
                    p.configure_camera_parameters(directions=direction,max_readout=.08)
                    p.configure_linearization(**self.plan(1));self.attach(p)
                    start=p.linearize()["summary"]["huber_cost"];step=p.lm_step()["summary"]
                    self.assertTrue(step["accepted"],step)
                    self.assertLess(step["trial_cost"],start*.01)
                    state=p.export_state();self.assertGreater(np.linalg.norm(state["rig"]-self.kw["rig"]),1e-10)
                    np.testing.assert_array_equal(state["controls"],self.kw["controls"])
                    np.testing.assert_allclose(p.evaluate("cpu")["values"],p.evaluate("cuda")["values"],atol=1e-10)
                    np.testing.assert_array_equal(state["rig"],p.export_state(trial=True)["rig"])

    def test_temporary_physical_boundary_can_release_after_coupled_move(self):
        # Unconstrained optimum violates both upper bounds. Fixing the second
        # bound makes the first coordinate's optimum INTERIOR. Permanently
        # retaining both faces returns a zero step despite a descent direction.
        f=fixture.G2GPU();f.setUp();kw=f.kw;rig=np.array(kw['rig']);rig[37]=0.
        kw.update(rig=rig,controls=np.c_[np.zeros((8,2)),np.linspace(np.log(80.),np.log(160.),8)])
        directions=np.zeros((27,2));directions[6,0]=.03;directions[15,1]=1.
        plan=dict(directions=np.zeros((8,3,2)),local_ids=np.full(4,-1,np.int32),local_scales=np.zeros(0))
        with ResidentResidualProblem(self.core,**kw) as p:
            p.configure_camera_parameters(directions=directions,max_readout=.08)
            p.configure_pixel_linearization(**plan);uv=p.evaluate('cuda')['values'][:,3:5];p.linearize()
            J=p.probe_rows(np.arange(4))[:,:,:2].reshape(8,2)
        damping=1e-4;S=J.T@J+damping*np.eye(2);rhs=S@np.array([.1,.2])
        expected=np.array([rhs[0]/S[0,0],0.]);self.assertLess(expected[0],0.)
        observed=uv+np.linalg.lstsq(J.T,rhs,rcond=None)[0].reshape(4,2)
        with ResidentResidualProblem(self.core,**{**kw,'observed':observed}) as p:
            lo=np.full(38,-np.inf);hi=np.full(38,np.inf);hi[15]=rig[15];hi[26]=rig[26]
            p.configure_camera_parameters(directions=directions,max_readout=.08,lower=lo,upper=hi)
            p.configure_pixel_linearization(**plan)
            p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),initial_lambda=damping,
                device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
            result=p.lm_step()['summary'];system=p.export_step_system()
            np.testing.assert_allclose(system['rhs'],rhs,rtol=1e-9,atol=1e-11)
            self.assertTrue(result['accepted'],result)
            np.testing.assert_allclose(system['step'],expected,rtol=0,atol=1e-8)
            self.assertLess(p.export_state()['rig'][15],rig[15])
            self.assertLessEqual(p.export_state()['rig'][26],rig[26])

    def test_explicit_fixed_physical_bound_keeps_accepted_rig(self):
        direction=np.zeros((27,1));direction[15,0]=1.
        with ResidentResidualProblem(self.core,**self.kw) as source:observed=source.evaluate("cpu")["values"][:,3:5]
        observed[:,0]+=2.
        with ResidentResidualProblem(self.core,**{**self.kw,"observed":observed}) as p:
            lo=np.full(38,-np.inf);hi=np.full(38,np.inf);lo[26]=hi[26]=self.kw["rig"][26]
            p.configure_camera_parameters(directions=direction,max_readout=.08,lower=lo,upper=hi)
            p.configure_linearization(**self.plan(1));self.attach(p);step=p.lm_step()["summary"]
            self.assertFalse(step["accepted"]);self.assertEqual(step["reason"],"nonpositive_prediction_or_no_feasible_step")
            np.testing.assert_array_equal(p.export_state()["rig"],self.kw["rig"])
            self.assertEqual(p.export_state(trial=True)["rig"][26],self.kw["rig"][26])

    def test_confirmed_mount_relations_accept_only_exterior_bounded_step(self):
        rig=self.kw["rig"].copy();rig[:9]=np.diag([1.,-1.,-1.]).ravel()
        rig[15:18]=[0.,-.3,1.];rig[18:24]=[0.,-.02,.01,0.,0.,0.]
        self.kw["rig"]=rig;direction=np.zeros((27,1));direction[7,0]=.01
        target_rig=self.retract(direction[:,0]*8.)
        with ResidentResidualProblem(self.core,**{**self.kw,"rig":target_rig}) as target:
            result=target.evaluate("cpu");np.testing.assert_array_equal(result["status"],0);observed=result["values"][:,3:5]
        with ResidentResidualProblem(self.core,**{**self.kw,"observed":observed}) as p:
            p.configure_mount_constraints(side="negative_y")
            p.configure_camera_parameters(directions=direction,max_readout=.08)
            p.configure_linearization(**self.plan(1));self.attach(p);step=p.lm_step()["summary"]
            self.assertTrue(step["accepted"],step);self.assertEqual(step["trial_domain_failures"],0)
            after=p.export_state()["rig"]
            self.assertLessEqual(after[16],-self.kw["board"][1])
            self.assertGreater(after[16],rig[16])
            self.assertAlmostEqual(after[16],-self.kw["board"][1],places=11)
            np.testing.assert_array_equal(np.delete(after,16),np.delete(rig,16))
            self.assertLess(step["trial_cost"],step["old_cost"])

    def test_wide_extension_forward_implicit_derivatives_and_coefficients_are_native(self):
        kw={**self.kw,'rig':np.r_[self.kw['rig'],np.log(140.),.2,.6,.01,-.02]}
        rd=np.zeros((31,5));rd[27:31,:4]=np.diag([.5,1.,.01,.01]);directions=np.zeros((8,3,5));directions[:,2,4]=.001
        plan={**self.plan(5),'directions':directions}
        base=self.core.evaluate_motion_camera(kw['rig'][:38],kw['controls'],kw['first_control_time'],kw['control_dt'],[0.,.1])
        wide=self.core.evaluate_motion_camera(kw['rig'],kw['controls'],kw['first_control_time'],kw['control_dt'],[0.,.1]);h=np.maximum(kw['rig'][38]-wide['log_f'],0.)**2
        for i,key in enumerate(('k1','k2','p1','p2')):np.testing.assert_allclose(wide[key],base[key]+kw['rig'][39+i]*h,atol=1e-14)
        with ResidentResidualProblem(self.core,**kw) as p:
            p.configure_camera_parameters(directions=rd,max_readout=.08);p.configure_linearization(**plan)
            p.linearize();jac=p.probe_rows(np.arange(4));gpu=p.evaluate('cuda');cpu=p.evaluate('cpu')
            np.testing.assert_array_equal(gpu['status'],0);np.testing.assert_allclose(gpu['values'],cpu['values'],atol=1e-10,rtol=1e-10)
            for j in range(5):
                values=[];epsilon=.001
                for sign in (-1,1):
                    rig=kw['rig'].copy();rig[39:43]+=sign*epsilon*rd[27:31,j]
                    with ResidentResidualProblem(self.core,**{**kw,'rig':rig,'controls':kw['controls']+sign*epsilon*directions[:,:,j]}) as trial:
                        ev=trial.evaluate('cuda');np.testing.assert_array_equal(ev['status'],0);values.append(ev['values'][:,3:5])
                np.testing.assert_allclose(jac[:,:,j],(values[1]-values[0])/(2*epsilon),rtol=4e-4,atol=2e-8)
                self.assertGreater(np.linalg.norm(jac[:,:,j]),1e-10)
        for j in range(4):
            rig=kw['rig'].copy();rig[39+j]+=.2
            with ResidentResidualProblem(self.core,**{**kw,'rig':rig}) as target:observed=target.evaluate('cpu')['values'][:,3:5]
            with ResidentResidualProblem(self.core,**{**kw,'observed':observed}) as p:
                direction=np.zeros((31,1));direction[27+j,0]=.1;p.configure_camera_parameters(directions=direction,max_readout=.08)
                p.configure_linearization(**self.plan(1));self.attach(p);result=p.lm_step()['summary'];self.assertTrue(result['accepted'],result)
                current=p.export_state()['rig'];self.assertEqual(len(current),43);self.assertEqual(current[38],kw['rig'][38]);self.assertGreater(abs(current[39+j]-kw['rig'][39+j]),.1)
                self.assertLess(result['trial_cost'],result['old_cost']*.01)

    def test_wide_extension_inactive_hull_matches_base_exactly(self):
        rig=np.r_[self.kw['rig'],np.log(50.),.6,1.7,0.,0.]
        with ResidentResidualProblem(self.core,**self.kw) as a,ResidentResidualProblem(self.core,**{**self.kw,'rig':rig}) as b:
            np.testing.assert_array_equal(a.evaluate('cuda')['values'],b.evaluate('cuda')['values'])
