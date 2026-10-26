"""Joint native scene derivatives and real accepted-state updates on CUDA."""
import os
import ctypes as c
import unittest
import numpy as np
from rinkspace.core.native import NativeCore, D, I, N
from rinkspace.core.gpu import ResidentResidualProblem
import test_g2_gpu as fixture

@unittest.skipUnless(os.environ.get("RINKSPACE_G3_LIBRARY"),"explicit CUDA/cuDSS library required")
class JointSceneGPU(unittest.TestCase):
    def setUp(self):
        f=fixture.G2GPU();f.setUp();self.kw=f.kw
        self.kw["board"][4]=1.84
        self.core=NativeCore(os.environ["RINKSPACE_G3_LIBRARY"])
        self.state=np.r_[self.kw["board"],self.kw["layout"],np.log([.12,.008]),[.81,.005,.03],
            [0.,0.,.02,-.01,.01,.02,-.02,-.01,.02,.01],[.5,.5,np.log(.3)],[.5,.5,np.log(.6),np.log(.5)]]
        self.data=np.array([0.,0.,1.,0.,1.,1.,0.,1.,0.,0.,0.,0.,1.,1.])
        self.records=np.array([[1,100,0,0,0,0,0,0,0,0,0,0],[1,410,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],[2,0,41,43,46,10,1,0,0,0,5,10],
            [2,0,41,43,56,3,2,0,0,0,5,0],[2,0,41,43,59,4,3,1,1,0,5,0]],np.int32)
        self.kw.update(codes=np.repeat(np.arange(6,dtype=np.int32),5),stations=np.tile([.071,.183,.391,.647,.913],6),
            times=np.linspace(0.,.1,30),observed=np.zeros((30,2)))
        self.kw["stations"][10:15]=[.1,.805,.87,1.02,1.3]
        self.scales=np.full(63,.001);self.scales[[4,41,42,*range(48,56),58,61,62]]=.02

    def make(self, state=None, directions=None, observed=None, charts=None):
        state=self.state if state is None else state
        kwargs={**self.kw,"board":state[:6],"layout":state[6:41]}
        if observed is not None:kwargs["observed"]=observed
        p=ResidentResidualProblem(self.core,**kwargs);paths=[]
        for code in (100,410):
            kind,params=self.core.shared_geometry(state[:6],state[6:41],code,options=0)
            paths.append(dict(kind=kind,parameters=params))
        paths.append(dict(kind=7,parameters=[1.]))
        for _ in range(3):paths.append(dict(kind=5,parameters=[.81,.005,.03,.12,.008],vertices=self.data[:10]))
        p.configure_geometry(paths)
        p.configure_scene_parameters(values=state,records=self.records,data=self.data,
            directions=np.zeros((len(state),1)) if directions is None else directions,charts=charts)
        return p

    def plan(self, q):
        return dict(directions=np.zeros((8,3,q)),local_ids=np.full(30,-1,np.int32),local_scales=np.zeros(0),
            group_offsets=np.arange(31,dtype=np.int32),members=np.arange(30,dtype=np.int32),ranks=np.full(30,2,np.int32),
            weight_offsets=np.arange(31,dtype=np.int32)*4,weights=np.tile(np.eye(2).ravel(),30),huber_delta=100.)

    def test_shared_geometry_print_shape_and_board_derivatives(self):
        indices=[*range(1,6),6,7,9,10,11,12,13,29,30,35,36,37,38,*range(41,63)]
        dirs=np.eye(63)[:,indices]*self.scales[indices]
        with self.make(directions=dirs) as p:
            p.configure_linearization(**self.plan(len(indices)));p.linearize();gpu=p.probe_rows(np.arange(30))
            p.linearize("cpu");np.testing.assert_allclose(gpu,p.probe_rows(np.arange(30),"cpu"),atol=2e-10,rtol=1e-9)
            for j,index in enumerate(indices):
                epsilon=.002;uv=[]
                for sign in (-1,1):
                    with self.make(state=self.state+sign*epsilon*dirs[:,j]) as sample:
                        result=sample.evaluate("cuda");np.testing.assert_array_equal(result["status"],0);uv.append(result["values"][:,3:5])
                np.testing.assert_allclose(gpu[:,:,j],(uv[1]-uv[0])/(2*epsilon),atol=3e-7,rtol=2e-4,err_msg=f"scene coordinate {index}")

    def test_real_lm_updates_scene_and_exported_board_is_consistent(self):
        for index in (1,4,6,41,43,46,50,58,62):
            with self.subTest(index=index):
                dirs=np.zeros((63,1));dirs[index,0]=self.scales[index]
                with self.make(state=self.state+.2*dirs[:,0]) as target:observed=target.evaluate("cpu")["values"][:,3:5]
                with self.make(directions=dirs,observed=observed) as p:
                    p.configure_linearization(**self.plan(1));p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),
                        device_budget_bytes=p.creation["free_after_alloc_bytes"],host_budget_bytes=2**30)
                    step=p.lm_step()["summary"];self.assertTrue(step["accepted"],step)
                    state=p.export_state();self.assertGreater(abs(state["scene_values"][index]-self.state[index]),1e-10)
                    np.testing.assert_array_equal(state["board"],state["scene_values"][:6]);np.testing.assert_array_equal(state["layout"],state["scene_values"][6:41])
                    np.testing.assert_allclose(p.evaluate("cpu")["values"],p.evaluate("cuda")["values"],atol=2e-10)
                    after=p.linearize()["summary"]["huber_cost"];self.assertLess(after,step["old_cost"]*.01)

    def test_scene_attachment_cannot_overwrite_selected_board(self):
        with ResidentResidualProblem(self.core,**self.kw) as p:
            paths=[dict(kind=1,parameters=[0.,0.,0.,.1,0.,6.28])]*6
            paths[2]=dict(kind=7,parameters=[1.])
            p.configure_geometry(paths)
            other=self.state.copy();other[1]+=.001
            with self.assertRaisesRegex(RuntimeError,"cannot replace selected native"):
                p.configure_scene_parameters(values=other,records=self.records,data=self.data,directions=np.zeros((63,1)))

    def test_positive_and_ordered_charts_match_native_retraction(self):
        charts=np.zeros(63,np.int32);charts[[1,2,3,5,9,10]]=1;charts[4]=2;charts[11:14]=[3,4,5]
        indices=[1,4,9,10,11,12,13];dirs=np.eye(63)[:,indices]*.01
        fn=self.core._lib.rs_scene_retract;fn.argtypes=[D,N,I,D,D];fn.restype=c.c_int
        def retract(delta):
            delta=np.ascontiguousarray(delta);result=np.empty_like(self.state)
            self.assertEqual(fn(self.state.ctypes.data_as(D),len(self.state),charts.ctypes.data_as(I),delta.ctypes.data_as(D),result.ctypes.data_as(D)),0)
            self.assertTrue(0<result[11]<result[12]<result[13]<1);self.assertGreater(result[4],1)
            return result
        with self.make(directions=dirs,charts=charts) as p:
            p.configure_linearization(**self.plan(len(indices)));p.linearize();jac=p.probe_rows(np.arange(30))
            for j,index in enumerate(indices):
                epsilon=.002;uv=[]
                for sign in (-1,1):
                    with self.make(state=retract(sign*epsilon*dirs[:,j]),charts=charts) as sample:uv.append(sample.evaluate("cuda")["values"][:,3:5])
                np.testing.assert_allclose(jac[:,:,j],(uv[1]-uv[0])/(2*epsilon),atol=3e-7,rtol=2e-4,err_msg=f"chart {index}")
        delta=dirs@np.array([.2,-.1,.1,-.3,.2,-.1,.3])
        with self.make(state=retract(delta),charts=charts) as target:uv=target.evaluate("cpu")["values"][:,3:5]
        with self.make(directions=dirs,charts=charts,observed=uv) as p:
            p.configure_linearization(**self.plan(len(indices)));p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),
                device_budget_bytes=p.creation["free_after_alloc_bytes"],host_budget_bytes=2**30)
            result=p.lm_step()["summary"];self.assertTrue(result["accepted"],result)
            state=p.export_state()["scene_values"];self.assertTrue(0<state[11]<state[12]<state[13]<1);self.assertGreater(state[4],1)
            p.linearize("cpu");p.linearize("cuda")
            np.testing.assert_allclose(p.probe_rows(np.arange(30),"cpu"),p.probe_rows(np.arange(30),"cuda"),atol=3e-10,rtol=1e-8)

    def test_superellipse_join_first_derivatives_remain_finite(self):
        breaks=np.asarray(self.core.board_course_chart(self.state[:6]))
        # Exact straight/corner joins, including the periodic closing join.
        self.kw["stations"][10:15]=breaks[[0,1,2,4,8]]
        dirs=np.eye(63)[:,:6]*.001
        with self.make(directions=dirs) as p:
            p.configure_linearization(**self.plan(6));p.linearize();jac=p.probe_rows(np.arange(10,15))
            self.assertTrue(np.isfinite(jac).all())
            p.linearize("cpu");np.testing.assert_allclose(jac,p.probe_rows(np.arange(10,15),"cpu"),atol=1e-10,rtol=1e-9)

    def test_scene_rig_and_motion_share_the_same_retained_step(self):
        scene_dirs=np.zeros((63,1));scene_dirs[43,0]=.001
        rig_dirs=np.zeros((27,1));rig_dirs[6,0]=.003
        plan=self.plan(1);plan["directions"][:,0,0]=.001
        rig=np.asarray(self.kw["rig"],np.float64);delta=np.ascontiguousarray(rig_dirs[:,0]*.2);target_rig=np.empty(38)
        fn=self.core._lib.rs_camera_retract;fn.argtypes=[D,N,D,N,D,N];fn.restype=c.c_int
        self.assertEqual(fn(rig.ctypes.data_as(D),38,delta.ctypes.data_as(D),27,target_rig.ctypes.data_as(D),38),0)
        controls=self.kw["controls"].copy();self.kw["rig"]=target_rig;self.kw["controls"]=controls+.2*plan["directions"][:,:,0]
        with self.make(state=self.state+.2*scene_dirs[:,0]) as target:uv=target.evaluate("cpu")["values"][:,3:5]
        self.kw["rig"]=rig;self.kw["controls"]=controls
        with self.make(directions=scene_dirs,observed=uv) as p:
            p.configure_camera_parameters(directions=rig_dirs,max_readout=.08)
            p.configure_linearization(**plan);p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),
                device_budget_bytes=p.creation["free_after_alloc_bytes"],host_budget_bytes=2**30)
            result=p.lm_step()["summary"];self.assertTrue(result["accepted"],result);self.assertLess(result["trial_cost"],result["old_cost"]*.01)
            state=p.export_state();self.assertNotEqual(state["scene_values"][43],self.state[43]);self.assertNotEqual(state["rig"][15],rig[15])
            self.assertFalse(np.array_equal(state["controls"],controls))

    def test_unobserved_print_cannot_veto_shared_shape_update(self):
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from rinkspace.session.gpu_geometry import NativeSceneGPU
        state=np.r_[self.state[:41],np.log([.12,.008]),[.81,.005,0.],np.zeros(10),[.81,.000001,0.]]
        records=np.array([[2,0,41,43,46,10,1,0,0,0,5,10],
                          [2,0,41,56,46,10,1,0,0,0,5,10]],np.int32)
        vertices=self.data[:10].reshape(-1,2)
        camera=dict(rig=self.kw['rig'],reference_pts=0,time_base_num=1,time_base_den=12800,
                    first_control_time=self.kw['first_control_time'],control_dt=self.kw['control_dt'],
                    image_height=self.kw['image_height'])
        stations=np.linspace(.05,.95,12);dirs=np.zeros((len(state),1));dirs[47,0]=.1
        def make(values,observed):
            native=NativeSceneGPU.__new__(NativeSceneGPU)
            native.core=self.core;native.scene_path=Path('.')
            native.scene=dict(board=values[:6],shared_layout=values[6:41],shared_geometry_options=0)
            native.parameter_state=SimpleNamespace(values=values)
            native.paths={name:dict(kind=5,parameters=[.81,float(values[i+1]),0.,.12,.008],vertices=vertices)
                          for name,i in [('seen',43),('unseen',56)]}
            def compile_records(scene,path,parameters,order,**kwargs):
                return records[[0 if name=='seen' else 1 for name in order]],self.data
            with patch('rinkspace.session.gpu_geometry.compile_joint_scene_records',side_effect=compile_records), \
                 patch('rinkspace.session.gpu_geometry.scene_coordinate_charts',return_value=np.zeros(len(state),np.int32)):
                return native.resident(camera=camera,controls=self.kw['controls'],path_ids=['seen']*len(stations),
                    stations=stations,pts=np.zeros(len(stations),np.int64),observed=observed,scene_directions=dirs)
        target=state-dirs[:,0]
        with make(target,np.zeros((len(stations),2))) as p:
            ev=p.evaluate('cuda');np.testing.assert_array_equal(ev['status'],0);observed=ev['values'][:,3:5]
        with make(state,observed) as p:
            p.configure_pixel_linearization(directions=np.zeros((8,3,1)),local_ids=np.full(len(stations),-1,np.int32),local_scales=np.zeros(0))
            p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30)
            result=p.lm_step()['summary'];self.assertTrue(result['accepted'],result)
            self.assertEqual(result['trial_domain_failures'],0)
            self.assertLess(result['trial_cost'],result['old_cost']*.01)
            after=p.export_state()['scene_values']
            # The same material change would cross the unseen panel's bottom.
            # Its fixed initial placement is not an observation or constraint.
            self.assertLess(after[57]+np.exp(after[42])*after[47],0.)
            np.testing.assert_array_equal(after[56:],state[56:])

    def test_observed_print_bottom_boundary_preserves_camera_descent(self):
        self.state[44]=.000001;self.state[45]=0.;self.state[46:56]=0.
        dirs=np.zeros((63,2));dirs[47,0]=.1
        rig_dirs=np.zeros((27,2));rig_dirs[15,1]=1.
        def configure(p):
            p.configure_camera_parameters(directions=rig_dirs,max_readout=.08)
            p.configure_pixel_linearization(directions=np.zeros((8,3,2)),local_ids=np.full(30,-1,np.int32),local_scales=np.zeros(0))
        with self.make(directions=dirs) as p:
            configure(p);p.linearize('cuda');jac=p.probe_rows(np.arange(30))
            observed=p.evaluate('cuda')['values'][:,3:5]-jac[:,:,0]+5*jac[:,:,1]
        with self.make(directions=dirs,observed=observed) as p:
            configure(p)
            p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,damping_metric='jacobi',initial_lambda=.001)
            result=p.lm_step()['summary'];self.assertTrue(result['accepted'],result)
            self.assertEqual(result['trial_domain_failures'],0)
            after=p.export_state();values=after['scene_values']
            bottom=values[44]+np.exp(values[42])*values[47]
            self.assertGreaterEqual(bottom,-1e-12);self.assertLess(bottom,1e-8)
            self.assertGreater(after['rig'][26]-self.kw['rig'][26],4.)
            self.assertLess(result['trial_cost'],result['old_cost']*.1)
            np.testing.assert_array_equal(p.evaluate('cuda')['status'],0)

    def test_physical_panel_contains_artwork_without_blocking_joint_camera(self):
        # A visible rectangular poster contains its printing. That relation is
        # independent of the much larger board's top/bottom domain.
        self.records[3,1]=1
        self.state[45]=0.;self.state[46:56]=0.;self.state[48:50]=np.log(.8)
        for axis,sign in [(0,-1.),(0,1.),(1,-1.),(1,1.)]:
            with self.subTest(axis=axis,sign=sign):
                dirs=np.zeros((63,2));dirs[46+axis,0]=.1
                rig_dirs=np.zeros((27,2));rig_dirs[15+axis,1]=1.
                def configure(p):
                    p.configure_camera_parameters(directions=rig_dirs,max_readout=.08)
                    p.configure_pixel_linearization(directions=np.zeros((8,3,2)),local_ids=np.full(30,-1,np.int32),local_scales=np.zeros(0))
                with self.make(directions=dirs) as p:
                    configure(p);p.linearize('cuda');jac=p.probe_rows(np.arange(30))
                    observed=p.evaluate('cuda')['values'][:,3:5]+2*sign*jac[:,:,0]+5*jac[:,:,1]
                with self.make(directions=dirs,observed=observed) as p:
                    configure(p)
                    p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,damping_metric='jacobi',initial_lambda=.001)
                    step=p.lm_step()['summary'];self.assertTrue(step['accepted'],step)
                    self.assertEqual(step['trial_domain_failures'],0)
                    after=p.export_state();translation=after['scene_values'][46+axis]
                    self.assertLessEqual(abs(translation),.1+1e-10)
                    self.assertAlmostEqual(translation,sign*.1,places=8)
                    self.assertGreater(after['rig'][26+axis]-self.kw['rig'][26+axis],1.)
                    self.assertGreater(step['actual_decrease'],0.)

    def test_unknown_print_material_domain_fails_explicitly(self):
        self.records[3,1]=2
        with self.assertRaisesRegex(RuntimeError,'unsupported print material domain'):
            with self.make():pass

    def test_rink_exterior_boundary_keeps_joint_board_and_camera_directions(self):
        self.kw['controls']=np.c_[np.zeros((8,2)),np.full(8,np.log(100.))]
        for side,sign,positive_width in [('negative_y',-1.,False),('positive_y',1.,False),
                                         ('negative_y',-1.,True),('positive_y',1.,True)]:
            with self.subTest(side=side,positive_width=positive_width):
                rig=np.asarray(self.kw['rig'],float).copy();rig[:9]=np.diag([1.,-1.,-1.]).ravel()
                rig[15:18]=[0.,sign*(self.state[1]+1e-6),3.];rig[18:21]=[0.,0.,.001];rig[37]=0.
                self.kw['rig']=rig
                dirs=np.zeros((63,3));dirs[1,0]=.01/self.state[1] if positive_width else .01
                charts=np.zeros(63,np.int32);charts[1]=int(positive_width)
                rig_dirs=np.zeros((27,3));rig_dirs[7,1]=-sign*.01;rig_dirs[15,2]=1.
                def configure(p):
                    p.configure_camera_parameters(directions=rig_dirs,max_readout=.08)
                    p.configure_mount_constraints(side=side)
                    p.configure_pixel_linearization(directions=np.zeros((8,3,3)),local_ids=np.full(30,-1,np.int32),local_scales=np.zeros(0))
                with self.make(directions=dirs,charts=charts) as p:
                    configure(p);p.linearize('cuda');jac=p.probe_rows(np.arange(30))
                    # Keep the retraction in the local linear regime so the
                    # constrained quadratic optimality check is meaningful.
                    observed=p.evaluate('cuda')['values'][:,3:5]+.005*jac[:,:,0]+.01*jac[:,:,1]+.05*jac[:,:,2]
                with self.make(directions=dirs,charts=charts,observed=observed) as p:
                    configure(p)
                    p.configure_solver(lower=np.zeros(0),upper=np.zeros(0),device_budget_bytes=p.creation['free_after_alloc_bytes'],host_budget_bytes=2**30,damping_metric='jacobi',initial_lambda=.001)
                    result=p.lm_step()['summary'];self.assertTrue(result['accepted'],result)
                    self.assertEqual(result['trial_domain_failures'],0)
                    after=p.export_state();gap=sign*after['rig'][16]-after['board'][1]
                    self.assertGreaterEqual(gap,-1e-12);self.assertLess(gap,1e-8)
                    self.assertGreater(after['rig'][26]-rig[26],.04)
                    self.assertNotEqual(after['board'][1],self.state[1]);self.assertNotEqual(after['rig'][16],rig[16])
                    self.assertLess(result['trial_cost'],result['old_cost']*.1)
                    # At the exterior boundary the quadratic gradient must be
                    # normal to the same physical constraint used by retraction.
                    # A second log-chart pullback corrupts this direction while
                    # a repeated feasibility projection may still look valid.
                    system=p.export_step_system();d=system['applied'][:3]
                    gradient=system['S']@d-system['rhs']
                    normal=np.array([-dirs[1,0]*(after['board'][1] if positive_width else 1.),
                                     sign*rig_dirs[7,1],0.])
                    tangent=gradient-normal*np.dot(gradient,normal)/np.dot(normal,normal)
                    self.assertLess(np.linalg.norm(tangent)/np.linalg.norm(gradient),.01)
