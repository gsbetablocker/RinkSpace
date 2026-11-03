"""Shared-column replay regression; no video, model weights, CUDA or cv2 needed.

This unit harness executes the production classes/functions from their AST with
synthetic kinematics and a pinhole camera. It tests objective/Jacobian equivalence
and eliminated body/surface calls, not the accuracy of the native camera/model.
"""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack, block_diag
from scipy.spatial.transform import Rotation
from calibration.program_tracking.body_tracking.footwear_layout import OBSERVATION_LAYOUT

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
BODY=ROOT/'calibration/program_tracking/body_tracking'


def load_execution_module():
    name='_joint_boot_body_execution'
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,BODY/'body_execution.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def load_definitions(filename,names,namespace):
    tree=ast.parse((BODY/filename).read_text(encoding='utf8'))
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    if len(selected)!=len(names):raise AssertionError('Missing test target in '+filename)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(BODY/filename),'exec'),namespace)


def production_namespace():
    ns=dict(np=np,Rotation=Rotation,Path=Path,json=json,time=time,
        coo_matrix=coo_matrix,csr_matrix=csr_matrix,hstack=hstack,vstack=vstack,block_diag=block_diag,
        WindowState=object,OBSERVATION_LAYOUT=OBSERVATION_LAYOUT)
    load_definitions('fullbody_state.py',{'matrices'},ns)
    load_definitions('footwear_geometry.py',{'local_vertices','decode','boot_alignment','equipment_world'},ns)
    ns['NAMES']=('shoe_rear','shoe_front','shoe_width','ankle_to_sole','upper_height','blade_drop',
        'blade_rear_extension','blade_front_extension','rocker_rise','toe_rise','boot_align_x','boot_align_y','boot_align_z','blade_pitch')
    load_definitions('fullbody_state.py',{'FrozenReference'},ns)
    load_definitions('footwear_binding.py',{'FootwearBinding','IntegratedBodyState'},ns)
    load_definitions('rebuild_exposure_body.py',{'row_weights','moving_projection'},ns)
    load_definitions('footwear_fit.py',{'sample_polyline'},ns)
    load_definitions('footwear_pixel_factors.py',{'nearest','nearest_segment','_boot_edge_rows','pixel_factors'},ns)
    load_definitions('footwear_observations.py',{'visible_shell_edges','_edge_midpoint_occluder'},ns)
    load_definitions('fullbody_fit.py',{'skew','right_jacobian','FullBodyProblem'},ns)
    load_definitions('boot_integration.py',{'camera_centres','edges_for','BootWindowProblem','JointBootProblem'},ns)
    load_definitions('opening_repair.py',{'OpeningProblem'},ns)
    return ns


class SyntheticBody:
    def __init__(self):self.calls=0
    def forward_rotations(self,local,with_frames=False):
        self.calls+=1;n=len(local)
        frames=np.broadcast_to(np.eye(3),(n,22,3,3)).copy();frames[:,1:]=local
        base=np.c_[np.linspace(-.3,.3,22),np.linspace(-.7,.6,22),np.linspace(-.1,.1,22)]
        joints=np.broadcast_to(base,(n,22,3)).copy()
        # Nonzero local-rotation dependencies; root placement remains external.
        joints+=np.einsum('njab,b->nja',frames,np.array([.02,.03,.01]))
        out=(joints[:,:17],joints,np.zeros((n,2,5,3)))
        return (*out,frames) if with_frames else out


class SyntheticSurface:
    def __init__(self):self.calls=0
    def sample_vertices(self,joints,frames,ids=None,double_precision=False):
        self.calls+=1;points=[]
        for ankle in (7,8):
            for offset in ([0.,-.04,-.03],[.01,-.04,.04],[-.01,-.03,.10]):
                points.append(joints[:,ankle]+np.einsum('nab,b->na',frames[:,ankle],offset))
        return np.stack(points,1)


class Pinhole:
    def __init__(self,n):
        self.scale=1.;self.p=dict(rig=np.zeros(43),controls=np.zeros((4,3)),
            first_control_time=0.,control_dt=1.,motion_times=np.arange(n)*.02)
        self.core=SimpleNamespace(evaluate_motion_camera=lambda rig,controls,first,dt,t:
            dict(centre=np.zeros((len(t),3)),status=np.zeros(len(t),int)))
    def project(self,world,indices):
        return np.stack((960.+400.*world[...,0]/world[...,2],540.+400.*world[...,1]/world[...,2]),-1)
    def with_jacobian(self,world,indices):
        uv=self.project(world,indices);J=np.zeros((*uv.shape,3));z=world[...,2]
        J[...,0,0]=400./z;J[...,1,1]=400./z
        J[...,0,2]=-400.*world[...,0]/z**2;J[...,1,2]=-400.*world[...,1]/z**2
        return uv,J


class SplineFixture:
    def __init__(self):
        self.active=np.array([0,1,2,3,4,5,6,7,8,9,10,11]);self.m=2;self.d=len(self.active)
        self.size=self.m*self.d;self.seconds=(0.,.1);self.mode='full'
    def basis(self,t,derivative=0):
        t=np.asarray(t)
        if derivative==1:return np.broadcast_to([-1.,1.],(len(t),2)).copy()
        if derivative!=0:raise ValueError('Unsupported synthetic derivative')
        return np.stack((1-t,t),1)
    def increments(self,x,t):
        out=np.zeros((len(t),69));out[:,self.active]=self.basis(t)@x.reshape(self.m,self.d);return out


def make_window(ns):
    n=3;ref=ns['IntegratedBodyState'].__new__(ns['IntegratedBodyState']);ref.body=SyntheticBody()
    ref.geometry=np.r_[np.log([.065,.165,.082,.082,.045,.040,.012,.015,.012,.012]),np.zeros(4)]
    binding=ns['FootwearBinding'].__new__(ns['FootwearBinding'])
    original,_=ns['local_vertices'](ref.geometry);p=ns['decode'](ref.geometry)
    binding.c0=np.array([0.,-p['ankle_to_sole'],(p['shoe_front']-p['shoe_rear'])/2])
    binding.template=original-binding.c0;binding.R0=np.broadcast_to(np.eye(3),(2,3,3)).copy()
    binding.t0=np.tile(binding.c0,(2,1));binding.R_BL=np.eye(3);binding.t_BL=np.zeros(3)
    binding.x=np.zeros(14);binding.deformation_enabled=True
    binding.shank_radius=.026;binding.cuff_clearance=.004
    binding.mode_regularizer=np.array([.35,.5,.8]);binding.maximum_modes=np.array([.07,.04,.025])
    binding.lining=.003;binding.heel_bed=.022;binding.ball_bed=.007
    binding.surface=SyntheticSurface();binding.sample_ids=np.arange(6)
    binding.foot_indices=[np.arange(3),np.arange(3,6)]
    binding.patch_indices=[[np.array([0]),np.array([2])],[np.array([3]),np.array([5])]]
    from rinkspace.viewer3d.open_boot_topology import open_boot_shell
    binding.faces,binding.groups=open_boot_shell(binding.template[:14]);ref.binding=binding
    w=ns['BootWindowProblem'].__new__(ns['BootWindowProblem']);w.reference=ref;w.binding=binding
    w.layout=OBSERVATION_LAYOUT
    w.state=SplineFixture();w.size=w.state.size;w.n=n;w.frames=np.arange(n);w.projector=Pinhole(n)
    w.times=(np.arange(n)[:,None]*.02+np.linspace(-.001,.001,5)).ravel()
    w.mid_times=w.times.reshape(n,5)[:,2]
    def at(t):
        count=len(t);return dict(P=np.tile([0.,0.,4.],(count,1)),
            Q=np.broadcast_to(np.eye(3),(count,3,3)).copy(),
            local_R=np.broadcast_to(np.eye(3),(count,21,3,3)).copy(),times=np.array(t))
    w.ref=at(w.times);w.mid_ref=at(w.mid_times);w.B=w.state.basis(w.times).reshape(n,5,2)
    w.eval_basis=w.B.reshape(-1,w.state.m)
    execution=load_execution_module()
    w.execution=execution.BodyExecution(execution.BodyExecutionConfig())
    w.fi=np.repeat(np.arange(n),113);w.pi=np.tile(np.arange(113),n)
    w.uv_rows=(w.fi*113+w.pi)[:,None]*2+np.arange(2)
    w.spline_scatter=[]
    for node in range(5):
        obs,knot=np.nonzero(w.B[w.fi,node]);shape=(len(obs),2,w.state.d)
        rr=np.broadcast_to(w.uv_rows[obs,:,None],shape).ravel().copy()
        cc=np.broadcast_to(knot[:,None,None]*w.state.d+np.arange(w.state.d),shape).ravel().copy()
        weight=w.B[w.fi[obs],node,knot,None,None]
        w.spline_scatter.append((obs,rr,cc,weight))
    w.spline_rows=np.concatenate([v[1] for v in w.spline_scatter]);w.spline_cols=np.concatenate([v[2] for v in w.spline_scatter])
    w.of,w.oj=np.nonzero(np.ones((n,17),bool))
    w.obs=dict(sigma_px=np.full((n,17),24.),uv=np.zeros((n,17,2)))
    w.prior_matrices=[('fixture_prior',csr_matrix(np.eye(w.size)/.7))]
    w.cache=None;w.cache_x=None;w.binding_key=None;w.start=time.perf_counter();w.evaluations=0;w.log=None
    w.camera_centres=ns['camera_centres'](w.projector,w.frames)
    d=w.state.increments(np.zeros(w.size),w.times);s=ref.world(w.ref,d)
    uv=w.projector.project(s['world'].reshape(n,5,113,3)[:,2],w.frames)
    w.obs['uv']=uv[:,96:]+.4
    w.gear=SimpleNamespace(
        visible_rows=[dict(frame=0,foot=0,observed=uv[0,[0,1,2]]+.3,sigma_px=2.,reverse_boundary=False)],
        blade_rows=[dict(frame=1,foot=1,uv=uv[1,[62,78]]+.2,sigma_px=1.5,partial=True,pts=256)],
        landmark_rows=[dict(frame=2,foot=0,heel=uv[2,0]+.2,toe=uv[2,1]+.3,sigma_px=2.)])
    return w


def make_opening(ns):
    w=make_window(ns);w.__class__=ns['OpeningProblem']
    w.standing=np.array([True,True,False]);w.velocity_scale=np.array([.1,.1,.15])
    w.reference_velocity=np.zeros((w.n,3));w.reference.anchor=np.zeros(3)
    return w


class SharedRegistrationReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.ns=production_namespace()

    def test_all_shared_columns_match_full_evaluator_without_body_or_surface_repetition(self):
        w=make_window(self.ns);x=np.linspace(-.008,.009,w.size)
        base=w.evaluate(x);packet=w.prepare_registration_replay(x,base)
        body_calls=w.reference.body.calls;surface_calls=w.binding.surface.calls
        replay=[];legacy=[];initial_binding=w.binding.x.copy()
        for k in range(14):
            for sign in (1.,-1.):
                shared=initial_binding.copy();shared[k]+=sign*1e-6
                replay.append(w.registration_residual_from_replay(shared,packet))
        self.assertEqual(w.reference.body.calls,body_calls)
        self.assertEqual(w.binding.surface.calls,surface_calls)
        np.testing.assert_array_equal(w.binding.x,initial_binding)
        self.assertIs(w.cache,base)
        for k in range(14):
            for sign in (1.,-1.):
                shared=initial_binding.copy();shared[k]+=sign*1e-6;w.binding.x=shared
                legacy.append(w.evaluate(x)['residual'].copy())
        self.assertEqual(w.reference.body.calls-body_calls,56)
        self.assertEqual(w.binding.surface.calls-surface_calls,28)
        np.testing.assert_allclose(replay,legacy,rtol=0,atol=2e-12)
        replay=np.asarray(replay);legacy=np.asarray(legacy)
        np.testing.assert_allclose((replay[::2]-replay[1::2])/2e-6,
            (legacy[::2]-legacy[1::2])/2e-6,rtol=1e-8,atol=1e-6)

    def test_extra_residual_subclasses_decline_replay(self):
        class ExtraResidual(self.ns['BootWindowProblem']):pass
        w=make_window(self.ns);w.__class__=ExtraResidual
        self.assertIsNone(w.prepare_registration_replay(np.zeros(w.size),{}))

    def test_fun_then_jac_matches_fresh_jac_without_duplicate_augmentation(self):
        w=make_window(self.ns);fresh=make_window(self.ns);x=np.linspace(-.008,.009,w.size)
        first=w.evaluate(x,False);initial_residual=first['residual'].copy();initial_rows=len(initial_residual)
        upgraded=w.evaluate(x,True);independent=fresh.evaluate(x,True)
        self.assertTrue(upgraded['forward_reused'])
        self.assertIsNot(upgraded,first)
        self.assertEqual(len(upgraded['residual']),initial_rows)
        np.testing.assert_array_equal(upgraded['residual'],initial_residual)
        np.testing.assert_allclose(upgraded['residual'],independent['residual'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(upgraded['jac'].toarray(),independent['jac'].toarray(),rtol=0,atol=1e-12)
        self.assertEqual(upgraded['parts'],independent['parts'])
        evaluations=w.evaluations;calls=w.reference.body.calls
        for flag in (True,False,True):
            self.assertIs(w.evaluate(x,flag),upgraded)
            self.assertEqual(len(w.cache['residual']),initial_rows)
        self.assertEqual(w.evaluations,evaluations);self.assertEqual(w.reference.body.calls,calls)
        # A different shoe binding must invalidate the complete forward packet.
        w.binding.x[0]=.001;changed=w.evaluate(x,False)
        self.assertFalse(changed['forward_reused'])
        self.assertEqual(len(changed['residual']),initial_rows)
        self.assertGreater(np.max(abs(changed['residual']-initial_residual)),1e-5)

    def test_binding_is_restored_on_shared_domain_failure(self):
        cls=self.ns['JointBootProblem'];joint=cls.__new__(cls);w=make_window(self.ns)
        joint.binding=w.binding;joint.windows=[w];joint.sizes=[w.size];joint.offsets=np.array([14,14+w.size])
        joint.cached=joint.cached_x=None;joint.calls=0;joint.start=time.perf_counter()
        class FailingRegistration:
            last_parts={}
            def residual(self,shared):
                if shared[0]!=0:raise ValueError('fixture domain failure')
                return shared.copy()
        joint.registration=FailingRegistration()
        with tempfile.TemporaryDirectory() as directory:
            joint.output=Path(directory);x=np.zeros(14+w.size)
            with self.assertRaisesRegex(ValueError,'fixture domain failure'):joint.evaluate(x,True)
            np.testing.assert_array_equal(joint.binding.x,x[:14])

    def test_opening_fun_to_jac_and_batch_schedule_do_not_duplicate_or_drop_rows(self):
        execution=load_execution_module();x=None
        for factory in (make_window,make_opening):
            with self.subTest(window=factory.__name__):
                ordinary=factory(self.ns);batched=factory(self.ns);fresh=factory(self.ns)
                x=np.linspace(-.008,.009,ordinary.size)
                first=ordinary.evaluate(x,False);old_residual=first['residual'].copy()
                result=ordinary.evaluate(x,True);direct=fresh.evaluate(x,True)
                self.assertTrue(result['forward_reused'])
                np.testing.assert_array_equal(result['residual'],old_residual)
                np.testing.assert_allclose(result['jac'].toarray(),direct['jac'].toarray(),rtol=0,atol=1e-12)
                self.assertIs(ordinary.evaluate(x,True),result)
                # NumPy synthetic state: exercises CUDA batching semantics only.
                batched.execution=execution.BodyExecution(execution.BodyExecutionConfig(
                    backend='cuda',perturbation_batch=4,max_batch_samples=80))
                gpu_schedule=batched.evaluate(x,True)
                np.testing.assert_allclose(gpu_schedule['residual'],result['residual'],rtol=0,atol=1e-12)
                np.testing.assert_allclose(gpu_schedule['jac'].toarray(),result['jac'].toarray(),rtol=1e-8,atol=1e-6)


if __name__=='__main__':unittest.main()
