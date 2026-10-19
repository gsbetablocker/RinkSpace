import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from calibration.program_tracking.measured_board_motion import (
    SCHEMA, BoardMotionPacket, MeasuredBoardMotionObjective, boundary_differentials)
from calibration.program_tracking.first_shot_source_transition import reference


def dump(path, value):
    path.write_text(json.dumps(value), encoding='utf8')
    return reference(path)


@pytest.fixture(autouse=True)
def synthetic_material_support(monkeypatch):
    # Source qualification has its own real original-packet tests. This fixture
    # isolates owner assembly with a known synthetic board-body witness.
    from calibration.program_tracking import measured_board_motion as motion
    def load(ref):
        data=json.loads(open(ref['path'],encoding='utf8').read())
        class Support:
            source_video_sha256=data['source']['source_video_sha256']
            time_base=data['source']['time_base']
            def require(self,frame,pts,uv,radius):
                if data['source']['native_pts'][frame-1]!=pts:raise ValueError('source PTS differs')
                field='source'if frame==1 else'target'
                candidates=[r for r in data['records']if np.array_equal(r[field+'_uv'],uv)]
                if len(candidates)!=1:raise ValueError('exact measured pixels differ')
                poly=np.asarray(candidates[0][field+'_support']['convex_board_body_polygon_uv'])
                if np.any(uv-radius<poly.min(axis=0))or np.any(uv+radius>poly.max(axis=0)):
                    raise ValueError('Entire motion support leaves synthetic board')
                return 'known-synthetic-wall'
        return Support()
    monkeypatch.setattr(motion,'_load_material_support',load)


def packet_files(tmp_path, u=None, v=None):
    u=np.array([[950.,510.],[970.,530.]]) if u is None else u
    v=u+np.array([.2,-.1]) if v is None else v
    frame=tmp_path/'original_frame.bin';frame.write_bytes(b'synthetic original RGB witness')
    review=dump(tmp_path/'review.json',dict(synthetic_known_material=True))
    support=dict(material='board_body',review_method='original_rgb_direct_review',
        source_frame=reference(frame),review_evidence=review,support_radius_px=2.,
        convex_board_body_polygon_uv=[[0.,0.],[1920.,0.],[1920.,1080.],[0.,1080.]],
        entire_polygon_board_body_confirmed=True)
    source=dict(source_video_sha256='synthetic-video',time_base='1/12800',native_pts=[256256,256512])
    records=[dict(source_point_id='known-board-'+str(i),a=1,b=2,native_pts=source['native_pts'],
        source_uv=left.tolist(),target_uv=right.tolist(),source_support=deepcopy(support),
        target_support=deepcopy(support))for i,(left,right)in enumerate(zip(u,v,strict=True))]
    admission=dict(schema='synthetic-known-wall-fixture',source=source,records=records)
    admission_ref=dump(tmp_path/'admission.json',admission)
    arrays=tmp_path/'pairs.npz';np.savez(arrays,p_previous_uv=u,p_current_uv=v)
    qarrays=tmp_path/'quality.npz';np.savez(qarrays,p_source_2d_identifiable=np.ones(len(u),bool))
    qref=dump(tmp_path/'quality.json',dict(camera_model_or_residual_used=False,source_track_pixels_modified=False))
    packet=dict(schema=SCHEMA,source=source,material_support=admission_ref,arrays=reference(arrays),tracking_support_radius_px=2.,source_qualification=qref,source_qualification_arrays=reference(qarrays),
        loss=dict(residual_unit='native_px',working_unit_native_px=1.,integration_weight=1.,
                  robust_loss='public_component_Huber2',calibrated_probability=False),
        pairs=[dict(a=1,b=2,native_pts=source['native_pts'],array_prefix='p',source_indices=[0,1],
                    source_point_ids=[r['source_point_id']for r in records],split='fit')])
    return dump(tmp_path/'packet.json',packet),packet,admission


def test_requires_direct_material_support_and_exact_pixels(tmp_path):
    ref,packet,admission=packet_files(tmp_path)
    value=BoardMotionPacket(ref);assert value.rows==4
    assert value.receipt['unsupported_frames']==[]
    with pytest.raises(ValueError):value.fit_pairs[0].source_uv[0,0]=0.
    legacy=dump(tmp_path/'legacy.json',dict(schema='rinkspace.source-only-2d-patch-identifiability.v3'))
    with pytest.raises(ValueError,match='legacy flow'):BoardMotionPacket(legacy)
    admission['records'][0]['source_support']['convex_board_body_polygon_uv']=[[940.,500.],[951.,500.],[951.,540.],[940.,540.]]
    packet['material_support']=dump(tmp_path/'admission.json',admission)
    with pytest.raises(ValueError,match='Entire motion support'):
        BoardMotionPacket(dump(tmp_path/'packet.json',packet))
    admission['records'][0]['source_uv'][0]+=.001
    packet['material_support']=dump(tmp_path/'admission.json',admission)
    with pytest.raises(ValueError,match='exact measured pixels'):
        BoardMotionPacket(dump(tmp_path/'packet.json',packet))


def test_clock_and_missing_admission_reject(tmp_path):
    ref,packet,_=packet_files(tmp_path)
    del packet['material_support']
    with pytest.raises(ValueError,match='Missing authenticated'):
        BoardMotionPacket(dump(tmp_path/'packet.json',packet))
    ref,packet,_=packet_files(tmp_path)
    packet['pairs'][0]['native_pts']=[256256,256513]
    with pytest.raises(ValueError,match='source identities differ|exact native'):
        BoardMotionPacket(dump(tmp_path/'packet.json',packet))


def test_complete_owner_both_camera_and_surface_chain_and_checkpoint(tmp_path,monkeypatch):
    from calibration.program_tracking.first_shot_source_kernel import _interfaces
    _interfaces()
    from z190_streaming_source_normal import SourceRowBlock
    from calibration.program_tracking import first_shot_row_camera as rc
    from calibration.program_tracking.projection import project_fast
    template=np.zeros(25);template[:3]=[0.,0.,3.];template[3]=np.deg2rad(95)
    template[6:8]=np.log([1200.,1210.]);template[8:10]=[960.,540.]
    class Course:
        def sample_perimeter(self,cq,*,with_jacobian):
            bottom=np.array([[-10.,-10.,0.],[10.,-10.,0.],
                [10.,10.+cq[0],0.],[-10.,10.+cq[0],0.]])
            J=np.zeros(bottom.shape+(6,));J[2:,1,0]=1.
            return bottom,bottom+[0.,0.,4.],J,J,[]
    def course_binding(g):
        cq=np.zeros(6);cq[0]=g[0];J=np.zeros((6,2));J[0,0]=1.
        return Course(),cq,J
    class State:
        pts=np.array([256256,256512]);source_clock=dict(time_base='1/12800')
        world_parameter_indices=np.array([2,3]);provider=SimpleNamespace(fd_step=1e-5,parameter_count=2,
            current=SimpleNamespace(print_course_binding=course_binding))
        binding_reference=dump(tmp_path/'binding.json',dict(source=dict(sha256='synthetic-video')))
        def guard_geometry(self,p):
            if p[3]!=0.:raise ValueError('unrelated print at active material domain')
            return True
        def scene(self,p):
            bottom=np.array([[-10.,-10.,0.],[10.,-10.,0.],[10.,10.+p[2],0.],[-10.,10.+p[2],0.]])
            return dict(ice_boundary=dict(points_xyz=bottom),board_surface=dict(bottom_edge_xyz=bottom,top_edge_xyz=bottom+[0.,0.,4.]))
    state=State()
    class Trajectory:
        def __init__(self,p,index):self.p=p;self.index=index
        def camera15_mapping(self,row):
            p=self.p;c=template.copy();M=np.zeros((15,2))
            if self.index==1:
                c[0]+=p[0]+1e-5*row*p[1];M[0]=[.01,1e-7*row]
            else:
                c[1]+=.2*p[0];c[4]+=p[1]+1e-6*row*p[0]
                M[1]=[.002,0.];M[4]=[1e-6*row,1.]
            return c,np.array([0,1]),M
        def camera25_at_row(self,row):return self.camera15_mapping(row)[0]
        def camera25_at_rows(self,rows):return np.array([self.camera25_at_row(y)for y in rows])
    monkeypatch.setattr(rc,'prepare_frame_row_camera',lambda state,p,index:Trajectory(p,index))
    world=np.array([[-1.,10.,1.6],[1.,10.,2.4]])
    u=project_fast(template,world)[0];v=u+np.array([.1,-.2])
    ref,_,_=packet_files(tmp_path,u,v)
    class Base:
        programme=SimpleNamespace(state=state);parameter_count=5;programme_count=4
        start=np.array([.01,.002,.015,0.,0.]);fixed=np.array([4]);shared=np.array([0,1,2,3])
        scales=np.array([1.,2.]);quadratic=np.array([False,True]);source_producers=[];recovery_references=[]
        def effective_q_and_chain(self,q):
            p=np.array([q[0]+.1*q[2]**2,q[1],q[2],q[3]])
            return p,csr_matrix([[1.,0.,.2*q[2],0.,0.],[0.,1.,0.,0.,0.],[0.,0.,1.,0.,0.],[0.,0.,0.,1.,0.]])
        def source_blocks(self,q,jacobian):
            yield SourceRowBlock(np.array([q[0]-.1,q[2]+.2]),csr_matrix([[1.,0.,0.,0.,0.],[0.,0.,1.,0.,0.]])if jacobian else None,
                                 self.quadratic,self.scales,dict(family='original-unchanged'))
        def guard(self,q):
            if q[4]!=0.:raise ValueError('original fixed state')
            state.guard_geometry(self.effective_q_and_chain(q)[0])
            return True
        def checkpoint_arrays(self):return dict(original_aux=np.arange(7))
        def checkpoint_identity(self):return dict(original_identity=True)
        def complete_state_payload(self,q):return dict(q=q.copy(),aux__original_aux=np.arange(7))
        def solver_kwargs(self):return dict(original_option='retained',fixed_parameter_indices=self.fixed.copy())
    base=Base();owner=MeasuredBoardMotionObjective(base,ref,entry_reference=reference(__file__))
    q=owner.start;r,J=owner.evaluate(q,True)
    assert r.shape==(6,)and J.shape==(6,5)
    np.testing.assert_array_equal(r[:2],next(base.source_blocks(q,False)).r)
    np.testing.assert_array_equal(owner.scales[:2],base.scales)
    for column in range(3):
        d=np.zeros(5);d[column]=1e-6
        finite=(owner.evaluate(q+d)-owner.evaluate(q-d))/(2e-6)
        np.testing.assert_allclose(J[:,column].toarray().ravel(),finite,rtol=3e-5,atol=2e-5)
    first=list(owner.source_blocks(q,True))[-1]
    second=list(owner.source_blocks(q+np.array([1e-5,0.,0.,0.,0.]),False))[-1]
    assert first.row_ledger==second.row_ledger
    diagnostic=owner.directional_check(q,direction=np.array([.3,.1,.2,.4,0.]),epsilon=1e-6)
    assert diagnostic['max_absolute']<2e-5 and diagnostic['relative_norm']<3e-5
    assert diagnostic['full_motion_or_original_source_J_constructed']is False
    assert owner.last_directional_direction[3]==0.
    assert diagnostic['probe']['removed_unrelated_nonzero_columns']==[3]
    original_guard=base.guard
    def one_sided(q):
        original_guard(q)
        if q[0]<base.start[0]:raise ValueError('active one-sided camera inequality')
        return True
    base.guard=one_sided
    one_side=owner.directional_check(q,direction=np.array([1.,0.,0.,0.,0.]),epsilon=1e-6)
    assert one_side['probe']['method']=='second_order_one_sided_feasible_canonical'
    assert one_side['max_absolute']<2e-5 and one_side['relative_norm']<3e-5
    base.guard=original_guard
    # A random direction crosses two active faces in opposite signs. A known
    # accepted secant must be tried at the original, measurable step size.
    def two_active_faces(candidate):
        original_guard(candidate)
        if candidate[0]<q[0]or candidate[1]<q[1]:
            raise ValueError('two active camera faces')
        return True
    base.guard=two_active_faces
    reference_probe=owner.directional_check(q,direction=np.array([1.,-1.,0.,0.,0.]),
        fallback_direction=np.array([1.,1.,0.,0.,0.]),epsilon=1e-6)
    assert reference_probe['probe']['previous_accepted_secant_fallback']is True
    assert reference_probe['epsilon']==1e-6
    assert reference_probe['max_absolute']<3e-5
    # A pan-only probe avoids a shared/tilt boundary without changing the
    # optimizer's fixed set. All original nonlinear guards still execute.
    original_arrays=base.checkpoint_arrays
    base.checkpoint_arrays=lambda:dict(original_arrays(),native_time_control_slots=np.array([[0],[1],[2]]))
    def pan_face(candidate):
        original_guard(candidate)
        if candidate[0]<q[0]or candidate[1]>q[1]:raise ValueError('active pan/tilt faces')
        return True
    base.guard=pan_face
    pan_probe=owner.directional_check(q,direction=np.array([1.,1.,0.,0.,0.]),epsilon=1e-6)
    assert pan_probe['probe']['pan_control_feasible_direction_fallback']is True
    assert pan_probe['epsilon']==1e-6
    assert pan_probe['max_absolute']<3e-5
    np.testing.assert_array_equal(base.fixed,[4])
    base.guard=original_guard;base.checkpoint_arrays=original_arrays
    assert owner.guard(q)is True
    with pytest.raises(ValueError,match='original fixed'):owner.guard(q+np.array([0.,0.,0.,0.,1.]))
    payload=owner.complete_state_payload(q)
    np.testing.assert_array_equal(payload['aux__original_aux'],np.arange(7))
    np.testing.assert_array_equal(payload['aux__measured_board_motion_source_uv'],u)
    assert owner.solver_kwargs()['original_option']=='retained'
    assert owner.checkpoint_identity()['original_identity']
    assert owner.solver_kwargs()['source_block_factory'].__self__ is owner


def test_actual_course_chain_never_perturbs_unrelated_active_print_domain():
    calls=[]
    def bottom(cq):
        return np.array([[-10.,-10.,0.],[10.,-10.,0.],
            [10.,10.+cq[0],0.],[-10.,10.+cq[0],0.]])
    class Course:
        def sample_perimeter(self,cq,*,with_jacobian):
            assert with_jacobian
            value=bottom(cq);J=np.zeros(value.shape+(6,));J[2:,1,0]=1.
            return value,value+[0.,0.,4.],J,J,[]
    def binding(q):
        cq=np.array([q[0],0.,0.,0.,0.,q[1]])
        J=np.zeros((6,2));J[0,0]=1.;J[5,1]=1.
        return Course(),cq,J
    class State:
        world_parameter_indices=np.array([0,1,2])
        provider=SimpleNamespace(parameter_count=2,current=SimpleNamespace(print_course_binding=binding))
        def scene(self,q):return dict(ice_boundary=dict(points_xyz=bottom([q[0]])))
        def guard_geometry(self,q):
            calls.append(q.copy())
            if np.any(q[1:]!=0.):raise ValueError('active unrelated print domain')
    state=State();q=np.array([.02,0.,0.]);columns=np.arange(3)
    _,J,reports=boundary_differentials(state,q,columns,fd_step=1e-6)
    assert len(calls)==1
    np.testing.assert_array_equal(calls[0],q)
    np.testing.assert_array_equal(J[:,:,1:],0.)
    numerical=(bottom([q[0]+1e-5])-bottom([q[0]-1e-5]))[:,:2]/2e-5
    np.testing.assert_allclose(J[:,:,0],numerical,rtol=0.,atol=5e-11)
    assert all(r['method']=='analytic_current_course_sample'for r in reports)
    assert reports[1]['boundary_XY_derivative_identically_zero']
    assert reports[2]['boundary_XY_derivative_identically_zero']
    state.scene=lambda q:dict(ice_boundary=dict(points_xyz=bottom([q[0]+.001])))
    with pytest.raises(ValueError,match='exactly'):
        boundary_differentials(state,q,columns,fd_step=1e-6)
