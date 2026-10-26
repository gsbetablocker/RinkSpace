"""Immutable rational source queries retain their implicit root and full W."""
import os
import unittest
import numpy as np
from rinkspace.core.gpu import ResidentResidualProblem
import test_gpu_joint_scene as fixture

@unittest.skipUnless(os.environ.get("RINKSPACE_G3_LIBRARY"),"explicit CUDA/cuDSS library required")
class ProfileQueriesGPU(unittest.TestCase):
    def setUp(self):
        self.f=fixture.JointSceneGPU();self.f.setUp();self.core=self.f.core
        self.base_kw={k:(v.copy() if hasattr(v,"copy") else v) for k,v in self.f.kw.items()}
        self.base_kw["rig"]=np.asarray(self.base_kw["rig"],float)
        self.f.kw=self.base_kw.copy()
        with self.f.make() as p:
            result=p.evaluate("cuda");np.testing.assert_array_equal(result["status"],0);uv=result["values"][:,3:5]
        samples=[]
        for sign in (-1,1):
            self.f.kw={**self.base_kw,"stations":self.base_kw["stations"]+sign*1e-6}
            with self.f.make() as p:samples.append(p.evaluate("cuda")["values"][:,3:5])
        self.f.kw=self.base_kw.copy();tangent=(samples[1]-samples[0])/2e-6
        direction=np.c_[-tangent[:,1],tangent[:,0]];direction/=np.linalg.norm(direction,axis=1)[:,None]
        # Curved, non-affine query, through the known initial projected position.
        den=np.tile([1.,.0003,0.,0.],(30,1));nu=np.zeros((30,4));nv=nu.copy()
        nu[:,0]=uv[:,0];nv[:,0]=uv[:,1]
        nu[:,1]=direction[:,0]+.0003*uv[:,0];nv[:,1]=direction[:,1]+.0003*uv[:,1]
        nu[:,2]=.002*direction[:,1];nv[:,2]=-.002*direction[:,0]
        supports=np.tile([0.,1.],(30,1));supports[10:15]=[0.,3.]
        self.profile=dict(row_ids=np.arange(30,dtype=np.int32),numerator_u=nu,numerator_v=nv,denominator=den,
            alpha_bounds=np.tile([-5.,5.],(30,1)),station_bounds=supports,initial_alpha=np.zeros(30))
        self.dirs=np.zeros((63,3));self.dirs[6,0]=.001;self.dirs[43,1]=.001;self.dirs[1,2]=.001
        self.rig_dirs=np.zeros((27,3));self.rig_dirs[6,1]=.001
        self.plan=self.f.plan(3);self.plan["directions"][:,0,0]=.001

    def make(self, state=None, observed=None, delta=None, profile=None):
        delta=np.zeros(3) if delta is None else delta
        self.f.kw={**self.base_kw,"controls":self.base_kw["controls"]+np.einsum("ikj,j->ik",self.plan["directions"],delta),
            "rig":self.base_kw["rig"].copy()}
        self.f.kw["rig"][15]+=.001*delta[1]
        p=self.f.make(state=self.f.state+self.dirs@delta if state is None else state,directions=self.dirs,observed=observed)
        p.configure_profiles(**(self.profile if profile is None else profile))
        p.configure_camera_parameters(directions=self.rig_dirs,max_readout=.08)
        return p

    def test_intersection_and_joint_implicit_derivatives_cpu_cuda(self):
        with self.make() as p:
            p.configure_linearization(**self.plan);p.linearize();gpu=p.probe_rows(np.arange(30))
            p.linearize("cpu");np.testing.assert_allclose(gpu,p.probe_rows(np.arange(30),"cpu"),atol=2e-9,rtol=1e-8)
            for backend in ("cpu","cuda"):
                roots=p.resolve_profile_roots(backend);np.testing.assert_array_equal(roots["status"],0)
                np.testing.assert_allclose(roots["station"],self.base_kw["stations"],atol=1e-12)
            for j in range(3):
                uv=[];epsilon=.002
                for sign in (-1,1):
                    d=np.zeros(3);d[j]=sign*epsilon
                    with self.make(delta=d) as q:
                        result=q.evaluate("cuda");np.testing.assert_array_equal(result["status"],0);uv.append(result["values"][:,3:5])
                np.testing.assert_allclose(gpu[:,:,j],(uv[1]-uv[0])/(2*epsilon),atol=3e-7,rtol=5e-4)
            np.testing.assert_array_equal(gpu[:,:,-1],0.)

    def test_query_cannot_be_freed_as_a_material_station(self):
        with self.make() as p:
            plan={**self.plan,"local_ids":np.arange(30,dtype=np.int32),"local_scales":np.ones(30)*.001}
            with self.assertRaisesRegex(RuntimeError,"implicit intersection"):
                p.configure_linearization(**plan)

    def test_real_joint_lm_query_updates_and_rejected_domain_keeps_state(self):
        with self.make(delta=np.array([.01,.02,-.01])) as target:
            result=target.evaluate("cuda");np.testing.assert_array_equal(result["status"],0);observed=result["values"][:,3:5]
        for limited in (False,True):
            with self.subTest(limited=limited):
                profile={**self.profile,"alpha_bounds":np.tile([-1e-9,1e-9],(30,1))} if limited else self.profile
                with self.make(observed=observed,profile=profile) as p:
                    p.configure_linearization(**self.plan);p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),
                        device_budget_bytes=p.creation["free_after_alloc_bytes"],host_budget_bytes=2**30)
                    before=p.export_state();step=p.lm_step()["summary"]
                    self.assertEqual(step["accepted"],not limited,step)
                    if limited:
                        self.assertEqual(step["reason"],"trial_domain_failure")
                        for key in ("rig","controls","scene_values","stations"):np.testing.assert_array_equal(before[key],p.export_state()[key])
                    else:self.assertLess(step["trial_cost"],step["old_cost"]*.02)
                    np.testing.assert_array_equal(p.resolve_profile_roots()["status"],0)

    def test_explicit_restoration_keeps_exact_query_failure_visible(self):
        profile={**self.profile,"alpha_bounds":np.tile([-1e-9,1e-9],(30,1)),"mode":"feasibility_restoration"}
        with self.make(delta=np.array([.01,.02,-.01]),profile=profile) as p:
            np.testing.assert_array_equal(p.evaluate("cuda")["status"],0)
            self.assertGreater(np.count_nonzero(p.resolve_profile_roots()["status"]),0)
            plan={**self.plan,"local_ids":np.arange(30,dtype=np.int32),"local_scales":np.ones(30)*.001}
            p.configure_linearization(**plan);p.linearize()
            self.assertEqual(p.profile_mode,"feasibility_restoration")
