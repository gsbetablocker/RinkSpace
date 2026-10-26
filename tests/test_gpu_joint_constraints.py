"""Coupled Lane/goal constraints must not lock either physical position."""
import os,unittest
import numpy as np
from rinkspace.core.native import NativeCore
from rinkspace.core.gpu import ResidentResidualProblem
import test_g2_gpu as fixture

@unittest.skipUnless(os.environ.get("RINKSPACE_G3_LIBRARY"),"native GPU/cuDSS required")
class JointConstraints(unittest.TestCase):
    def test_lane_at_goal_boundary_keeps_coupled_positions_estimable(self):
        f=fixture.G2GPU();f.setUp();kw=f.kw;core=NativeCore(os.environ["RINKSPACE_G3_LIBRARY"])
        kw.update(options=6,codes=np.repeat(np.arange(2,dtype=np.int32),4),stations=np.tile([0.,.1,.2,.3],2),times=np.zeros(8),observed=np.zeros((8,2)))
        state=np.r_[kw['board'],kw['layout']];directions=np.zeros((41,2));directions[16,0]=.001;directions[29,1]=.001
        def make(observed=None):
            p=ResidentResidualProblem(core,**{**kw,'observed':kw['observed'] if observed is None else observed})
            paths=[]
            for code in (300,403):
                kind,parameters=core.shared_geometry(kw['board'],kw['layout'],code,options=6);paths.append(dict(kind=kind,parameters=parameters))
            p.configure_geometry(paths);p.configure_scene_parameters(values=state,records=np.array([[1,300,*([0]*10)],[1,403,*([0]*10)]],np.int32),data=np.zeros(1),directions=directions)
            return p
        with make() as p:initial=p.evaluate('cpu')['values'][:,3:5]
        for displacement in (-.1,.1):
            with self.subTest(displacement=displacement):
                observed=initial.copy();observed[:4,0]+=displacement
                with make(observed) as p:
                    p.configure_linearization(directions=np.zeros((8,3,2)),local_ids=np.full(8,-1,np.int32),local_scales=np.zeros(0),group_offsets=np.arange(9),members=np.arange(8),ranks=np.full(8,2),weight_offsets=np.arange(9)*4,weights=np.tile(np.eye(2).ravel(),8),huber_delta=100.)
                    p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
                    result=p.lm_step()['summary'];self.assertTrue(result['accepted'],result);self.assertLess(result['trial_cost'],result['old_cost'])
                    after=p.export_state()['scene_values'];self.assertGreaterEqual(after[16],after[29]);self.assertGreater(abs(after[16]-state[16]),1e-7)
                    if displacement<0:self.assertLess(after[29],state[29]-1e-7)
                    else:self.assertGreater(after[16]-after[29],1e-7)
                    self.assertEqual(p.linearize()['summary']['failed_rows'],0)

    def test_off_image_curve_trial_retains_original_pixel_residual(self):
        f=fixture.G2GPU();f.setUp();kw=f.kw;core=NativeCore(os.environ['RINKSPACE_G3_LIBRARY'])
        kw.update(controls=np.c_[np.zeros((8,2)),np.full(8,np.log(100.))],codes=np.array([0,1],np.int32),stations=np.array([.735,0.]),times=np.zeros(2),observed=np.array([[50.,99.],[50.,70.]]))
        paths=[dict(kind=0,parameters=[0.,0.,0.,0.,2.,0.]),dict(kind=0,parameters=[0.,0.,0.,0.,0.,0.])]
        with ResidentResidualProblem(core,**kw) as p:
            p.configure_geometry(paths)
            direction=np.zeros((27,1));direction[16,0]=1.
            p.configure_camera_parameters(directions=direction,max_readout=.08)
            p.configure_linearization(directions=np.zeros((8,3,1)),local_ids=np.array([0,-1]),local_scales=np.array([1e-6]),group_offsets=np.arange(3),members=np.arange(2),ranks=np.full(2,2),weight_offsets=np.arange(3)*4,weights=np.tile(np.eye(2).ravel(),2),huber_delta=100.)
            p.configure_solver(lower=np.array([0.]),upper=np.array([1.]),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
            # The damped global step moves cy about ten pixels. The tiny local
            # coordinate is damped independently; leaving the raster is valid.
            result=p.lm_step()['summary'];self.assertTrue(result['accepted'],result)
            state=p.export_state();self.assertGreater(state['rig'][27],55.)
            self.assertGreater(state['stations'][0],.73)
            self.assertEqual(result['trial_domain_failures'],0)
            after=p.evaluate('cuda');np.testing.assert_array_equal(after['status'],0)
            self.assertGreater(after['values'][0,4],100.)
            J=np.array([[0.,0.],[1.,200/3*1e-6],[0.,0.],[1.,0.]])
            residual=np.array([0.,0.,0.,-20.])
            step=np.linalg.solve(J.T@J+result['lambda_used']*np.eye(2),-J.T@residual)
            np.testing.assert_allclose(after['values'][:,9:11].ravel(),residual+J@step,atol=1e-10)
            np.testing.assert_allclose(after['values'][:,9:11],after['values'][:,3:5]-kw['observed'],atol=1e-12)
            # The same implicit row equality holds outside the image.
            self.assertLess(np.max(np.abs(after['values'][:,7])),1e-8)
            self.assertLess(result['trial_cost'],result['old_cost']*.51)

    def test_off_image_curve_and_camera_step_matches_complete_damped_system(self):
        f=fixture.G2GPU();f.setUp();kw=f.kw;core=NativeCore(os.environ['RINKSPACE_G3_LIBRARY'])
        kw.update(controls=np.c_[np.zeros((8,2)),np.full(8,np.log(100.))],codes=np.array([0,1],np.int32),stations=np.array([.735,0.]),times=np.zeros(2),observed=np.array([[100.,99.],[50.,70.]]))
        paths=[dict(kind=0,parameters=[-1.,0.,0.,1.,2.,0.]),dict(kind=0,parameters=[0.,0.,0.,0.,0.,0.])]
        with ResidentResidualProblem(core,**kw) as p:
            p.configure_geometry(paths);direction=np.zeros((27,1));direction[16,0]=1.
            p.configure_camera_parameters(directions=direction,max_readout=.08)
            p.configure_linearization(directions=np.zeros((8,3,1)),local_ids=np.array([0,-1]),local_scales=np.array([1e-6]),group_offsets=np.arange(3),members=np.arange(2),ranks=np.full(2,2),weight_offsets=np.arange(3)*4,weights=np.r_[np.eye(2).ravel(),10*np.eye(2).ravel()],huber_delta=1000.)
            p.configure_solver(lower=np.array([0.]),upper=np.array([1.]),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
            result=p.lm_step()['summary'];self.assertTrue(result['accepted'],result)
            state=p.export_state();self.assertGreater(state['rig'][27],65.)
            after=p.evaluate('cuda');np.testing.assert_array_equal(after['status'],0)
            self.assertGreater(after['values'][0,4],100.)
            J=np.array([[0.,200/3*1e-6],[1.,200/3*1e-6],[0.,0.],[1.,0.]])
            residual=np.array([50+100*(-1+2*.735)/3-100,0.,0.,-20.])
            W=np.diag([1.,1.,10.,10.]);A=W@J;b=W@residual
            step=np.linalg.solve(A.T@A+result['lambda_used']*np.eye(2),-A.T@b)
            np.testing.assert_allclose(after['values'][:,9:11].ravel(),residual+J@step,atol=1e-10)
            self.assertAlmostEqual(result['trial_cost'],.5*np.sum((b+A@step)**2),places=9)

    def test_whole_curve_outside_image_does_not_clip_joint_camera_candidate(self):
        f=fixture.G2GPU();f.setUp();kw=f.kw;core=NativeCore(os.environ['RINKSPACE_G3_LIBRARY'])
        kw.update(controls=np.c_[np.zeros((8,2)),np.full(8,np.log(100.))],codes=np.array([0,1],np.int32),stations=np.array([.5,0.]),times=np.zeros(2),observed=np.array([[50.,99.],[70.,70.]]))
        paths=[dict(kind=0,parameters=[-.2,1.47,0.,.2,1.47,0.]),dict(kind=0,parameters=[0.,0.,0.,0.,0.,0.])]
        with ResidentResidualProblem(core,**kw) as p:
            p.configure_geometry(paths);direction=np.zeros((27,2));direction[15,0]=direction[16,1]=1.
            p.configure_camera_parameters(directions=direction,max_readout=.08)
            p.configure_linearization(directions=np.zeros((8,3,2)),local_ids=np.array([0,-1]),local_scales=np.array([.001]),group_offsets=np.arange(3),members=np.arange(2),ranks=np.full(2,2),weight_offsets=np.arange(3)*4,weights=np.tile(np.eye(2).ravel(),2),huber_delta=100.)
            p.configure_solver(lower=np.array([0.]),upper=np.array([1.]),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,damping_metric='jacobi')
            result=p.lm_step()['summary'];self.assertTrue(result['accepted'],result)
            state=p.export_state();self.assertGreater(state['rig'][26],55.);self.assertGreater(state['rig'][27],59.9)
            ev=p.evaluate('cuda');np.testing.assert_array_equal(ev['status'],0);self.assertLess(np.max(np.abs(ev['values'][:,7])),1e-8)
            self.assertLess(result['trial_cost'],result['old_cost']*.85)

    def test_finite_query_boundary_keeps_unrelated_camera_direction_free(self):
        f=fixture.G2GPU();f.setUp();kw=f.kw;core=NativeCore(os.environ['RINKSPACE_G3_LIBRARY'])
        kw['rig']=list(kw['rig']);kw['rig'][-1]=0.
        kw.update(controls=np.c_[np.zeros((8,2)),np.full(8,np.log(100.))],codes=np.array([0,1],np.int32),stations=np.array([.5,0.]),times=np.zeros(2),observed=np.array([[50.,50.05],[55.,51.]]))
        with ResidentResidualProblem(core,**kw) as p:
            p.configure_geometry([dict(kind=0,parameters=[-1.,0.,0.,1.,0.,0.]),dict(kind=4,vertices=[[0.,0.,0.]],parameters=[])])
            p.configure_profiles(row_ids=np.array([0,-1]),numerator_u=np.array([[50.,0.,0.,0.]]),numerator_v=np.array([[50.,1.,0.,0.]]),denominator=np.array([[1.,0.,0.,0.]]),alpha_bounds=np.array([[-.05,.05]]),station_bounds=np.array([[0.,1.]]),initial_alpha=np.array([0.]))
            direction=np.zeros((27,2));direction[15,0]=1.;direction[16,1]=1.
            p.configure_camera_parameters(directions=direction,max_readout=.08)
            p.configure_linearization(directions=np.zeros((8,3,2)),local_ids=np.array([-1,-1]),local_scales=np.zeros(0),group_offsets=np.arange(3),members=np.arange(2),ranks=np.full(2,2),weight_offsets=np.arange(3)*4,weights=np.r_[np.eye(2).ravel(),10*np.eye(2).ravel()],huber_delta=1000.)
            p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
            step=p.lm_step()['summary'];self.assertTrue(step['accepted'],step)
            state=p.export_state();self.assertGreater(state['rig'][26],54.)
            roots=p.resolve_profile_roots();np.testing.assert_array_equal(roots['status'],0)
            self.assertGreater(roots['alpha'][0],.049);self.assertLessEqual(roots['alpha'][0],.05)
            self.assertLess(step['trial_cost'],step['old_cost']*.1)
