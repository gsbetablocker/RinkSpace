"""Multiple physical cameras share one native LM scene and source objective."""
import os
import unittest
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.core.gpu import ResidentResidualProblem
import test_gpu_joint_camera as camera_fixture
import test_gpu_joint_scene as scene_fixture


@unittest.skipUnless(os.environ.get('RINKSPACE_G3_LIBRARY'), 'explicit CUDA/cuDSS build required')
class MultiCameraGPU(unittest.TestCase):
    def setUp(self):
        f=camera_fixture.JointCameraGPU();f.setUp();self.core=f.core;self.single=f.kw
        a=np.r_[self.single['rig'],np.log(140.),.2,.6,.01,-.02]
        b=a.copy();b[26:28]+=[17.,-9.];b[28]=-.02;b[37]=-.01
        self.rigs=np.stack([a,b]);self.ids=np.repeat(np.arange(2,dtype=np.int32),4)
        self.kw={**self.single,'rig':self.rigs,'camera_ids':self.ids}
        for key in ('codes','stations','times','observed'):self.kw[key]=np.concatenate([self.single[key]]*2)

    def plan(self,q,local=False):
        return dict(directions=np.zeros((8,3,q)),local_ids=np.arange(8,dtype=np.int32) if local else np.full(8,-1,np.int32),
                    local_scales=np.full(8,.001) if local else np.zeros(0))

    def attach(self,p,local=0):
        p.configure_solver(lower=np.zeros(local),upper=np.ones(local),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)

    def test_forward_and_all_camera_jacobians_match_separate_cameras(self):
        q=62;rd=np.eye(q)*.001
        with ResidentResidualProblem(self.core,**self.kw) as p:
            p.configure_camera_parameters(directions=rd,max_readout=.08)
            p.configure_pixel_linearization(**self.plan(q));p.linearize();jac=p.probe_rows(np.arange(8))
            p.linearize('cpu');np.testing.assert_allclose(jac,p.probe_rows(np.arange(8),'cpu'),atol=2e-10,rtol=1e-10)
            gpu=p.evaluate('cuda');np.testing.assert_array_equal(gpu['status'],0)
            np.testing.assert_allclose(gpu['values'],p.evaluate('cpu')['values'],atol=2e-10)
            for camera in range(2):
                rows=slice(4*camera,4*camera+4)
                with ResidentResidualProblem(self.core,**{**self.single,'rig':self.rigs[camera]}) as single:
                    np.testing.assert_allclose(gpu['values'][rows],single.evaluate('cuda')['values'],atol=2e-10)
                    single.configure_camera_parameters(directions=np.eye(31)*.001,max_readout=.08)
                    single.configure_pixel_linearization(directions=np.zeros((8,3,31)),local_ids=np.full(4,-1,np.int32),local_scales=np.zeros(0))
                    single.linearize();expected=single.probe_rows(np.arange(4))
                np.testing.assert_allclose(jac[rows,:,31*camera:31*camera+31],expected[:,:,:31],atol=2e-10)
                other=31*(1-camera);np.testing.assert_array_equal(jac[rows,:,other:other+31],0.)
                self.assertTrue(np.linalg.norm(jac[rows,:,31*camera:31*camera+31],axis=(0,1)).min()>0)

    def test_both_cameras_update_and_second_camera_fixed_bound_is_preserved(self):
        with ResidentResidualProblem(self.core,**self.kw) as target:observed=target.evaluate('cuda')['values'][:,3:5]
        observed[:4,0]+=2.;observed[4:,0]-=3.
        rd=np.zeros((62,2));rd[15,0]=1.;rd[31+15,1]=1.
        lo=np.full(86,-np.inf);hi=np.full(86,np.inf);lo[43+26]=hi[43+26]=self.rigs[1,26]
        with ResidentResidualProblem(self.core,**{**self.kw,'observed':observed}) as p:
            p.configure_camera_parameters(directions=rd,max_readout=.08,lower=lo,upper=hi)
            p.configure_pixel_linearization(**self.plan(2));self.attach(p)
            result=p.lm_step()['summary'];self.assertTrue(result['accepted'],result)
            state=p.export_state();self.assertGreater(state['rig'][0,26],self.rigs[0,26]+1.9)
            np.testing.assert_array_equal(state['rig'][1],self.rigs[1]);self.assertEqual(result['trial_domain_failures'],0)
            diagnostic=p.stationarity_diagnostics(include_constraints=True)
            self.assertEqual(diagnostic['constraint_projected_gradient_inf']>=0,True)

    def test_scene_column_has_cross_terms_with_both_independent_cameras(self):
        f=scene_fixture.JointSceneGPU();f.setUp();n=len(f.kw['codes']);rigs=np.stack([f.kw['rig'],f.kw['rig']]);rigs[1,26]+=30.
        kw={**f.kw,'rig':rigs,'camera_ids':np.repeat(np.arange(2,dtype=np.int32),n)}
        for key in ('codes','stations','times','observed'):kw[key]=np.concatenate([f.kw[key]]*2)
        def make():
            p=ResidentResidualProblem(self.core,**kw);paths=[]
            for code in (100,410):
                kind,params=self.core.shared_geometry(f.state[:6],f.state[6:41],code,options=0);paths.append(dict(kind=kind,parameters=params))
            paths.append(dict(kind=7,parameters=[1.]))
            for _ in range(3):paths.append(dict(kind=5,parameters=[.81,.005,.03,.12,.008],vertices=f.data[:10]))
            p.configure_geometry(paths);sd=np.zeros((len(f.state),3));sd[1,2]=.001
            p.configure_scene_parameters(values=f.state,records=f.records,data=f.data,directions=sd)
            rd=np.zeros((54,3));rd[15,0]=.1;rd[27+15,1]=.1;p.configure_camera_parameters(directions=rd,max_readout=.08)
            p.configure_pixel_linearization(directions=np.zeros((8,3,3)),local_ids=np.full(2*n,-1,np.int32),local_scales=np.zeros(0))
            return p
        with make() as p:
            p.linearize();j=p.probe_rows(np.arange(2*n))[:,:,:3].reshape(4*n,3)
            self.assertGreater(abs((j.T@j)[0,2]),1e-12);self.assertGreater(abs((j.T@j)[1,2]),1e-12)
            np.testing.assert_array_equal(j[:2*n,1],0);np.testing.assert_array_equal(j[2*n:,0],0)
            self.attach(p);step=p.lm_step()['summary'];self.assertTrue(step['accepted'],step)
            self.assertLess(step['trial_cost'],step['old_cost']);self.assertEqual(p.export_state()['rig'].shape,(2,38))

    def test_inactive_camera_keeps_its_exact_imported_axis_values(self):
        # Normalizing the zero-step camera used to alter a saved, already valid
        # axis by a few ulps during initialization of the other installation.
        rigs=self.rigs.copy();rigs[0,12:15]*=1.+2e-15
        kwargs={**self.kw,'rig':rigs}
        with ResidentResidualProblem(self.core,**kwargs) as target:observed=target.evaluate('cuda')['values'][:,3:5]
        observed[4:,0]+=2.
        rd=np.zeros((62,1));rd[31+15,0]=1.
        with ResidentResidualProblem(self.core,**{**kwargs,'observed':observed}) as p:
            p.configure_camera_parameters(directions=rd,max_readout=.08)
            p.configure_pixel_linearization(**self.plan(1));self.attach(p)
            step=p.lm_step()['summary'];self.assertTrue(step['accepted'],step)
            state=p.export_state();np.testing.assert_array_equal(state['rig'][0],rigs[0])
            self.assertGreater(state['rig'][1,26],rigs[1,26]+1.9)

    def test_camera_identity_is_part_of_curve_search_cache(self):
        # Same path/time/support in both cameras must not share projected samples.
        kw={**self.kw,'codes':np.full(8,100,np.int32),'stations':np.full(8,.1)}
        desired=np.r_[np.full(4,.3),np.full(4,.7)]
        with ResidentResidualProblem(self.core,**{**kw,'stations':desired}) as target:observed=target.evaluate('cuda')['values'][:,3:5]
        with ResidentResidualProblem(self.core,**{**kw,'observed':observed}) as p:
            p.configure_pixel_linearization(**self.plan(1,True));self.attach(p,8)
            p.refine_curve_stations(mode='global');errors=p.evaluate_errors()
            self.assertLess(errors.max(),1e-5)
            np.testing.assert_allclose(p.export_state()['stations'],desired,atol=1e-6)

    def test_row_without_declared_physical_camera_is_rejected(self):
        for ids in (np.zeros(7,np.int32),np.full(8,2,np.int32),np.zeros(8)):
            with self.assertRaisesRegex(ValueError,'physical camera'):
                ResidentResidualProblem(self.core,**{**self.kw,'camera_ids':ids})
