"""Raw pixel objective must not compress, rescale, or discount a point."""
import os
import unittest
import numpy as np
from rinkspace.core.gpu import ResidentResidualProblem
import test_g2_gpu as fixture


@unittest.skipUnless(os.environ.get('RINKSPACE_G3_LIBRARY'),'explicit CUDA/cuDSS build required')
class PixelResidualGPU(unittest.TestCase):
    def setUp(self):
        self.f=fixture.G2GPU();self.f.setUp()

    def attach(self,p,local=False):
        return p.configure_pixel_linearization(directions=self.f.directions,
            local_ids=np.arange(4,dtype=np.int32) if local else np.full(4,-1,np.int32),
            local_scales=np.full(4,.001) if local else np.zeros(0))

    def test_trial_profiles_independent_curve_stations_before_acceptance(self):
        rig=np.array(self.f.kw['rig']);rig[37]=0.
        controls=np.tile([0.,0.,np.log(100.)],(8,1))
        stations=np.array([.1,.32,.56,.81]);paths=[dict(kind=1,parameters=[0.,0.,0.,1.,0.,2*np.pi])]
        kw={**self.f.kw,'rig':rig,'controls':controls,'codes':np.zeros(4,np.int32),'stations':stations}
        rd=np.zeros((27,2));rd[15,0]=1.;rd[16,1]=1.
        with ResidentResidualProblem(self.f.core,**kw) as p:
            p.configure_geometry(paths);observed=p.evaluate('cuda')['values'][:,3:5]+[10.,-8.]
        outputs=[]
        for profiling in (False,True):
            with ResidentResidualProblem(self.f.core,**{**kw,'observed':observed}) as p:
                p.configure_geometry(paths);p.configure_camera_parameters(directions=rd,max_readout=.08)
                p.configure_pixel_linearization(directions=np.zeros((8,3,2)),local_ids=np.arange(4,dtype=np.int32),local_scales=np.full(4,.001))
                p.configure_solver(lower=np.zeros(4),upper=np.ones(4),initial_lambda=.001,
                    device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,
                    profile_trial_stations=profiling)
                p.refine_curve_stations(mode='global');summary=p.lm_step()['summary'];trial=p.export_state(trial=True)
                outputs.append((summary,trial))
                if profiling:
                    self.assertTrue(summary['accepted'],summary)
                    self.assertAlmostEqual(.5*sum(p.evaluate_errors()**2),summary['trial_cost'],places=9)
                    refinement=p.refine_curve_stations(mode='global')
                    self.assertLess(refinement['pixel_cost_decrease'],1e-8)
        old,new=outputs
        np.testing.assert_allclose(old[1]['rig'],new[1]['rig'],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(old[1]['controls'],new[1]['controls'])
        self.assertLess(new[0]['trial_cost'],old[0]['trial_cost']-1e-6)


    def test_opposite_large_errors_cannot_cancel_or_be_downweighted(self):
        with ResidentResidualProblem(self.f.core,**self.f.kw) as p:
            prediction=p.evaluate('cuda')['values'][:,3:5]
        displacement=np.array([[1000.,2000.],[-1000.,-2000.],[3.,4.],[-3.,-4.]])
        kw={**self.f.kw,'observed':prediction-displacement}
        with ResidentResidualProblem(self.f.core,**kw) as p:
            self.attach(p)
            for backend in ('cpu','cuda'):
                summary=p.linearize(backend)['summary']
                self.assertAlmostEqual(summary['huber_cost'],.5*np.sum(displacement**2),places=7)
                self.assertEqual(summary['downweighted_groups'],0)
                for i in range(4):
                    group=p.probe_group(i,backend)
                    np.testing.assert_allclose(group['white_r'],displacement[i],atol=1e-12)
                    self.assertEqual(group['loss'][1],1.)
                    self.assertAlmostEqual(group['loss'][2],.5*sum(displacement[i]**2),places=7)
            self.assertEqual(p.linearization_info['whitened_rank'],8)
            self.assertEqual(p.linearization_info['objective'],'unweighted_squared_pixel_distance')

    def test_cpu_gpu_raw_blocks_and_derivatives_agree(self):
        with ResidentResidualProblem(self.f.core,**self.f.kw) as p:
            self.attach(p,True);p.linearize();g=p.export_blocks();j=p.probe_rows(np.arange(4))
            p.linearize('cpu');c=p.export_blocks('cpu')
            for k in c:np.testing.assert_allclose(g[k],c[k],rtol=1e-11,atol=1e-11)
            for i in range(4):
                group=p.probe_group(i)
                # Each component has the two camera columns and its own station.
                np.testing.assert_allclose(group['white_J'],j[i],rtol=1e-12,atol=1e-12)

    def test_many_camera_columns_fit_without_dense_gpu_normal_block_copies(self):
        # Independent pixels store two Jacobian rows, not a dense or triangular
        # outer product per point. Check memory and the independent dense CPU
        # equations together: reducing memory cannot change the normal blocks.
        n=128;q=27
        kw=dict(self.f.kw)
        for key in ('codes','stations','times','observed'):
            value=np.asarray(kw[key]);kw[key]=np.tile(value,(n//4,1) if value.ndim==2 else n//4)
        with ResidentResidualProblem(self.f.core,**kw) as p:
            p.configure_camera_parameters(directions=np.eye(q)*.001,max_readout=.02)
            plan=p.configure_pixel_linearization(directions=np.zeros((len(kw['controls']),3,q)),
                local_ids=np.arange(n,dtype=np.int32),local_scales=np.full(n,.001))
            self.assertLess(plan['device_bytes'],.4*8*plan['H_value_count'])
            p.linearize('cuda');gpu=p.export_blocks('cuda')
            p.linearize('cpu');cpu=p.export_blocks('cpu')
            for key in cpu:np.testing.assert_allclose(gpu[key],cpu[key],rtol=1e-11,atol=1e-11)

    def test_lm_acceptance_uses_actual_unweighted_pixel_sum(self):
        with ResidentResidualProblem(self.f.core,**self.f.kw) as p:
            self.attach(p,True)
            p.configure_solver(lower=np.zeros(4),upper=np.ones(4),
                device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
            before=p.evaluate('cuda')['values'][:,9:11]
            step=p.lm_step()['summary'];self.assertTrue(step['accepted'],step)
            after=p.evaluate('cuda')['values'][:,9:11]
            self.assertAlmostEqual(step['old_cost'],.5*np.sum(before**2),places=9)
            self.assertAlmostEqual(step['trial_cost'],.5*np.sum(after**2),places=9)
            self.assertLess(np.sum(after**2),np.sum(before**2))

    def test_native_curve_search_escapes_wrong_arc_and_endpoint_without_moving_model(self):
        path=dict(kind=1,parameters=[0.,0.,0.,.3,0.,2*np.pi])
        kw={**self.f.kw,'codes':np.zeros(4,np.int32),'stations':np.array([.05,.3,.6,.95])}
        with ResidentResidualProblem(self.f.core,**kw) as p:
            p.configure_geometry([path]);observed=p.evaluate('cuda')['values'][:,3:5]
        kw.update(observed=observed,stations=np.array([1.,.1,.5,.2]))
        with ResidentResidualProblem(self.f.core,**kw) as p:
            p.configure_geometry([path]);self.attach(p,True)
            p.configure_solver(lower=np.zeros(4),upper=np.ones(4),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
            before=p.export_state();r0=p.evaluate('cuda')['values'][:,9:11];info=p.refine_curve_stations()
            after=p.export_state();r1=p.evaluate('cuda')['values'][:,9:11]
            self.assertEqual(info['failed_rows'],0);self.assertEqual(info['changed_rows'],4)
            self.assertLess(np.max(np.linalg.norm(r1,axis=1)),1e-7)
            self.assertTrue(np.all(np.sum(r1**2,axis=1)<=np.sum(r0**2,axis=1)))
            self.assertAlmostEqual(info['pixel_cost_decrease'],.5*(np.sum(r0**2)-np.sum(r1**2)),places=9)
            for k in ('rig','controls','board','layout'):np.testing.assert_array_equal(before[k],after[k])
            # A subsequent linearization must observe the corrected stations.
            self.assertLess(p.linearize()['summary']['huber_cost'],1e-13)

    def test_curve_search_does_not_independently_move_shared_material_points(self):
        with ResidentResidualProblem(self.f.core,**self.f.kw) as p:
            p.configure_pixel_linearization(directions=self.f.directions,local_ids=np.array([0,1,1,-1]),local_scales=np.full(2,.001))
            p.configure_solver(lower=np.zeros(2),upper=np.ones(2),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
            before=p.export_state();p.refine_curve_stations();after=p.export_state()
            np.testing.assert_array_equal(before['stations'][1:],after['stations'][1:])

    def test_near_horizontal_edge_keeps_nearest_point_instead_of_sliding_hundreds_of_pixels(self):
        rig=np.asarray(self.f.kw['rig']).copy();rig[24:28]=[1000.,2.,0.,0.]
        kw={**self.f.kw,'rig':rig,'controls':np.tile([0.,0.,np.log(1000.)],(8,1)),
            'image_height':1080,'codes':np.zeros(1,np.int32),'stations':np.array([1/6]),
            'times':np.zeros(1),'observed':np.array([[400.,1071.]])}
        # Independent pinhole image line (1000*s, 1078+6*s). Its nearest
        # point is 1.4 px outside the image; clipping at v=1079 adds >230 px.
        station=(1000*400+6*(1071-1078))/(1000**2+6**2)
        expected=np.array([1000*station,1078+6*station])
        clipped=np.array([1000/6,1079.])
        self.assertGreater(np.linalg.norm(clipped-kw['observed'][0]),230.)
        with ResidentResidualProblem(self.f.core,**kw) as p:
            p.configure_geometry([dict(kind=0,parameters=[0.,3.234,0.,3.,3.252,0.])])
            p.configure_projection_time_support(half_seconds=.02)
            p.configure_pixel_linearization(directions=np.zeros((8,3,1)),local_ids=np.zeros(1,np.int32),local_scales=np.ones(1))
            p.configure_solver(lower=np.zeros(1),upper=np.ones(1),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
            before=p.export_state();result=p.refine_curve_stations();after=p.export_state();ev=p.evaluate('cuda')
            self.assertEqual(result['failed_rows'],0);self.assertEqual(ev['status'].tolist(),[0])
            np.testing.assert_allclose(ev['values'][0,3:5],expected,atol=1e-7,rtol=0)
            self.assertGreater(ev['values'][0,4],1079.)
            self.assertLess(np.linalg.norm(ev['values'][0,9:11]),10.)
            self.assertAlmostEqual(after['stations'][0],station,places=9)
            for key in ('rig','controls','board','layout'):np.testing.assert_array_equal(before[key],after[key])
            np.testing.assert_allclose(ev['values'][:,9:11],ev['values'][:,3:5]-kw['observed'],atol=1e-12)

    def test_off_image_root_crossing_knot_keeps_complete_motion_derivative(self):
        rig=np.asarray(self.f.kw['rig']).copy();rig[25]=2.
        controls=np.tile([0.,0.,np.log(100.)],(10,1));frame=-.011
        kw={**self.f.kw,'rig':rig,'controls':controls,'first_control_time':-.06,'control_dt':.02,
            'codes':np.zeros(1,np.int32),'stations':np.zeros(1),'times':np.array([frame]),'observed':np.array([[50.,99.]])}
        xyz=np.array([[0.,3.9,0.]])
        directions=np.zeros((10,3,1));directions[5,1,0]=1.
        for half in (None,.03):
            with self.subTest(support=half),ResidentResidualProblem(self.f.core,**kw) as p:
                p.configure_geometry([dict(kind=4,parameters=[],vertices=xyz.tolist())])
                if half is not None:p.configure_projection_time_support(half_seconds=half)
                p.configure_pixel_linearization(directions=directions,local_ids=np.array([-1]),local_scales=np.zeros(0))
                p.linearize('cuda');jac=p.probe_rows([0])[0,:,0]
                p.linearize('cpu');np.testing.assert_allclose(p.probe_rows([0],'cpu')[0,:,0],jac,atol=1e-10)
                root=p.evaluate('cuda')['values'][0,6];row_time=frame+rig[37]*(root/100.-.5)
                self.assertLess(frame+abs(rig[37])/2,0.);self.assertGreater(row_time,0.)
                h=1e-6;uv=[]
                for sign in (-1.,1.):
                    value=self.f.core.project_motion(rig,controls+sign*h*directions[:,:,0],-.06,.02,101,frame,xyz)
                    self.assertEqual(value['status'].tolist(),[0]);uv.append(value['uv'][0])
                expected=(uv[1]-uv[0])/(2*h)
                self.assertGreater(np.linalg.norm(expected),1.)
                np.testing.assert_allclose(jac,expected,atol=2e-6,rtol=2e-7)

        with ResidentResidualProblem(self.f.core,**kw) as p:
            p.configure_geometry([dict(kind=4,parameters=[],vertices=xyz.tolist())])
            p.configure_projection_time_support(half_seconds=.01)
            self.assertEqual(p.evaluate('cuda')['status'].tolist(),[4])
            p.configure_pixel_linearization(directions=directions,local_ids=np.array([-1]),local_scales=np.zeros(0))
            with self.assertRaisesRegex(RuntimeError,'invalid rows retained'):p.linearize()

    def test_off_image_closest_point_matches_unconstrained_pixel_oracle(self):
        # Independent convex oracle: pixels (20+40*s, 90+20*s+dy),
        # observed (60,98), plus a fixed point observing dy=0. The free
        # solution puts the curve below the image. The raster must not add
        # a spurious v<=100 constraint to this ordinary pixel least squares.
        A=np.array([[40.,0.],[20.,1.],[0.,1.]])
        b=np.array([40.,8.,0.]);answer=np.linalg.lstsq(A,b,rcond=None)[0]
        for sign in (1.,-1.):
            with self.subTest(sensor_side=sign):
                rig=np.asarray(self.f.kw['rig']).copy();rig[37]=0.
                kw={**self.f.kw,'rig':rig,'controls':np.tile([0,0,np.log(100.)],(8,1)),
                    'codes':np.array([0,1],np.int32),'stations':np.array([.5,0.]),'times':np.zeros(2),
                    'observed':np.array([[60.,50+48*sign],[50.,50.]])}
                paths=[dict(kind=4,parameters=[],vertices=[[-.9,1.2*sign,0.],[.3,1.8*sign,0.]]),
                    dict(kind=4,parameters=[],vertices=[[0.,0.,0.]])]
                with ResidentResidualProblem(self.f.core,**kw) as p:
                    p.configure_geometry(paths);rd=np.zeros((27,1));rd[16,0]=1.
                    p.configure_camera_parameters(directions=rd,max_readout=.02)
                    p.configure_pixel_linearization(directions=np.zeros((8,3,1)),local_ids=np.array([0,-1],np.int32),local_scales=np.ones(1))
                    p.configure_solver(lower=np.zeros(1),upper=np.ones(1),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,damping_metric='jacobi')
                    result=p.solve();self.assertTrue(result['numerically_converged'],result)
                    state=p.export_state();actual=p.evaluate('cuda')
                    self.assertFalse(actual['status'].any())
                    self.assertAlmostEqual(state['stations'][0],answer[0],places=6)
                    self.assertAlmostEqual(state['rig'][27],50+answer[1]*sign,places=5)
                    self.assertAlmostEqual(result['final_cost'],.5*np.sum((A@answer-b)**2),places=5)
                    self.assertAlmostEqual(result['final_cost'],.5*np.sum(actual['values'][:,9:11]**2),places=10)
                    self.assertGreater(sign*(actual['values'][0,4]-50.),50.)

    def test_all_off_image_points_retain_unconstrained_pixel_contributions(self):
        # Both optima cross the bottom. Keep every residual and compare with
        # an independent three-variable least-squares oracle, without clipping.
        A=np.array([[40,0,0],[20,0,1],[0,40,0],[0,20,1],[0,0,0],[0,0,1]],float)
        b=np.array([40,8,33,10,0,0.]);G=np.array([[20,0,1],[0,20,1]],float)
        free=np.linalg.lstsq(A,b,rcond=None)[0];self.assertTrue(np.all(G@free>10))
        answer=free
        rig=np.asarray(self.f.kw['rig']).copy();rig[37]=0.
        kw={**self.f.kw,'rig':rig,'controls':np.tile([0,0,np.log(100.)],(8,1)),
            'codes':np.array([0,0,1],np.int32),'stations':np.array([.5,.5,0.]),'times':np.zeros(3),
            'observed':np.array([[60.,98.],[53.,100.],[50.,50.]])}
        paths=[dict(kind=4,parameters=[],vertices=[[-.9,1.2,0.],[.3,1.8,0.]]),
            dict(kind=4,parameters=[],vertices=[[0.,0.,0.]])]
        with ResidentResidualProblem(self.f.core,**kw) as p:
            p.configure_geometry(paths);rd=np.zeros((27,1));rd[16,0]=1.
            p.configure_camera_parameters(directions=rd,max_readout=.02)
            p.configure_pixel_linearization(directions=np.zeros((8,3,1)),local_ids=np.array([0,1,-1],np.int32),local_scales=np.ones(2))
            p.configure_solver(lower=np.zeros(2),upper=np.ones(2),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,damping_metric='jacobi',initial_lambda=1e-10)
            result=p.solve();self.assertTrue(result['numerically_converged'],result)
            state=p.export_state();got=np.r_[state['stations'][:2],state['rig'][27]-50]
            np.testing.assert_allclose(got,answer[:3],atol=1e-7,rtol=0)
            self.assertAlmostEqual(result['final_cost'],.5*np.sum((A@answer[:3]-b)**2),places=7)

    def test_curve_endpoint_can_release_inside_the_same_joint_step(self):
        A=np.array([[40,0,0],[20,0,1],[0,40,0],[0,20,1],[0,0,0],[0,0,1]],float)
        b=np.array([50,10,41.5,10,0,0.]);free=np.linalg.lstsq(A,b,rcond=None)[0]
        self.assertTrue(np.all(free[:2]>1.))
        E=np.array([[1.,0.,0.]]);answer=np.linalg.solve(np.block([[A.T@A,E.T],[E,np.zeros((1,1))]]),np.r_[A.T@b,1.])
        self.assertLess(answer[1],1.)
        rig=np.asarray(self.f.kw['rig']).copy();rig[37]=0.
        kw={**self.f.kw,'rig':rig,'controls':np.tile([0,0,np.log(100.)],(8,1)),
            'codes':np.array([0,0,1],np.int32),'stations':np.array([.5,.5,0.]),'times':np.zeros(3),
            'observed':np.array([[70.,50.],[61.5,50.],[50.,50.]])}
        paths=[dict(kind=4,parameters=[],vertices=[[-.9,-.3,0.],[.3,.3,0.]]),
            dict(kind=4,parameters=[],vertices=[[0.,0.,0.]])]
        with ResidentResidualProblem(self.f.core,**kw) as p:
            p.configure_geometry(paths);rd=np.zeros((27,1));rd[16,0]=1.
            p.configure_camera_parameters(directions=rd,max_readout=.02)
            p.configure_pixel_linearization(directions=np.zeros((8,3,1)),local_ids=np.array([0,1,-1],np.int32),local_scales=np.ones(2))
            p.configure_solver(lower=np.zeros(2),upper=np.ones(2),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,damping_metric='jacobi',initial_lambda=1e-10)
            result=p.lm_step()['summary'];self.assertTrue(result['accepted'],result)
            state=p.export_state();got=np.r_[state['stations'][:2],state['rig'][27]-50]
            np.testing.assert_allclose(got,answer[:3],atol=1e-7,rtol=0)
            self.assertFalse(p.evaluate('cuda')['status'].any())

    def test_replicated_off_image_points_keep_same_convex_optimum_and_cost(self):
        # Repeating raw points leaves the optimum unchanged and scales cost.
        # No raster-edge active faces should add repeated factorizations.
        A=np.array([[40,0,0],[20,0,1],[0,40,0],[0,20,1],[0,0,0],[0,0,1]],float)
        b=np.array([40,8,33,10,0,0.]);E=np.array([[20.,0.,1.]])
        answer=np.linalg.lstsq(A,b,rcond=None)[0]
        passes=[]
        for copies in (1,32):
            with self.subTest(copies=copies):
                rig=np.asarray(self.f.kw['rig']).copy();rig[37]=0.
                kw={**self.f.kw,'rig':rig,'controls':np.tile([0,0,np.log(100.)],(8,1)),
                    'codes':np.tile([0,0,1],copies).astype(np.int32),
                    'stations':np.tile([.5,.5,0.],copies),'times':np.zeros(3*copies),
                    'observed':np.tile([[60.,98.],[53.,100.],[50.,50.]],(copies,1))}
                paths=[dict(kind=4,parameters=[],vertices=[[-.9,1.2,0.],[.3,1.8,0.]]),
                    dict(kind=4,parameters=[],vertices=[[0.,0.,0.]])]
                local=np.c_[2*np.arange(copies),2*np.arange(copies)+1,np.full(copies,-1)].ravel().astype(np.int32)
                with ResidentResidualProblem(self.f.core,**kw) as p:
                    p.configure_geometry(paths);rd=np.zeros((27,1));rd[16,0]=1.
                    p.configure_camera_parameters(directions=rd,max_readout=.02)
                    p.configure_pixel_linearization(directions=np.zeros((8,3,1)),local_ids=local,local_scales=np.ones(2*copies))
                    p.configure_solver(lower=np.zeros(2*copies),upper=np.ones(2*copies),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,damping_metric='jacobi',initial_lambda=1e-10)
                    step=p.lm_step()['summary'];self.assertTrue(step['accepted'],step);passes.append(step['bound_passes'])
                    state=p.export_state();evaluated=p.evaluate('cuda')
                    np.testing.assert_allclose(state['stations'].reshape(copies,3)[:,:2],np.tile(answer[:2],(copies,1)),atol=1e-7,rtol=0)
                    self.assertAlmostEqual(state['rig'][27]-50,answer[2],places=6)
                    self.assertAlmostEqual(step['trial_cost'],copies*.5*np.sum((A@answer[:3]-b)**2),places=5)
                    self.assertAlmostEqual(step['trial_cost'],.5*np.sum(evaluated['values'][:,9:11]**2),places=8)
                    self.assertFalse(evaluated['status'].any())
        self.assertLessEqual(passes[1],passes[0]+1)

    def test_projected_shared_step_can_complete_local_cycle_without_damping_retry(self):
        # Two shared coordinates give dx=a, dy=a+b with dy>=0. Projection of
        # the unconstrained shared solution changes dx as well as dy. The free
        # station then crosses its endpoint, but that endpoint's multiplier
        # changes sign after the next shared projection: a deterministic cycle.
        # The joint constrained optimum is dy=0, dx=.4, s=1. Checking only
        # the conditional station optimum would miss an incorrect shared step.
        rig=np.asarray(self.f.kw['rig']).copy();rig[37]=0.
        kw={**self.f.kw,'rig':rig,'controls':np.tile([0,0,np.log(100.)],(8,1)),
            'codes':np.array([0,1],np.int32),'stations':np.array([.5,0.]),
            'times':np.zeros(2),'observed':np.array([[60.8,48.8],[50.,48.8]])}
        paths=[dict(kind=4,parameters=[],vertices=[[-.9,0.,0.],[.3,0.,0.]]),
               dict(kind=4,parameters=[],vertices=[[0.,0.,0.]])]
        with ResidentResidualProblem(self.f.core,**kw) as p:
            p.configure_geometry(paths)
            rd=np.zeros((27,2));rd[15,0]=1.;rd[16,:]=1.
            lower=np.full(38,-np.inf);upper=np.full(38,np.inf);lower[27]=50.
            p.configure_camera_parameters(directions=rd,lower=lower,upper=upper,max_readout=.02)
            p.configure_pixel_linearization(directions=np.zeros((8,3,2)),
                local_ids=np.array([0,-1],np.int32),local_scales=np.ones(1))
            p.configure_solver(lower=np.zeros(1),upper=np.ones(1),
                device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,
                damping_metric='jacobi',initial_lambda=1e-10)
            result=p.lm_step()['summary']
            self.assertTrue(result['accepted'],result)
            self.assertEqual(result['reason_code'],6)
            self.assertEqual(result['lambda_used'],1e-10)
            state=p.export_state();evaluated=p.evaluate('cuda')
            dx=state['rig'][26]-50.
            self.assertAlmostEqual(dx,.4,places=7)
            self.assertAlmostEqual(state['stations'][0],1.,places=9)
            self.assertAlmostEqual(state['stations'][0],np.clip((40.8-dx)/40.,0.,1.),places=8)
            self.assertGreaterEqual(state['rig'][27],50.-1e-10)
            self.assertEqual(state['stations'][1],0.)
            self.assertFalse(evaluated['status'].any())
            self.assertAlmostEqual(result['trial_cost'],.5*np.sum(evaluated['values'][:,9:11]**2),places=9)
            self.assertGreater(result['actual_decrease'],200.)


if __name__=='__main__':unittest.main()
