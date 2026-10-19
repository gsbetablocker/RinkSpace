"""Real frozen-parent, real camera/W/RS checks; never constructs a fit owner."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import importlib.util
import json
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix,eye,hstack,load_npz

from calibration.solver_runtime import install
install()
from calibration.program_tracking.first_shot_time_detail_stage import (
    CenteredNativeTimeDetailPlan, NativeTimeDetailVenueStage)
from calibration.program_tracking.first_shot_safe_frame_export import load_source_frames
from calibration.program_tracking.first_shot_fit import _CameraState,FrozenWhitening
from calibration.program_tracking.first_shot_row_source import evaluate_row_timed_frame
from calibration.program_tracking.first_shot_physical_state import FirstShotPhysicalState
from calibration.program_tracking.first_shot_derived_venue_stage import _reference
from z190_streaming_source_normal import SourceRowBlock

DATA=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/shot1_physical_fit_20261014/zoom_poly_72ac/id_ad5')
ACCEPTED=DATA/'fixed_head_time8d/accepted_states/0107_220e21438cd2'


def equal(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


@pytest.fixture(scope='module')
def real_parent():
    if not (ACCEPTED/'canonical.npz').exists():
        pytest.skip('The preserved real Ankara parent is unavailable on this host')
    with np.load(ACCEPTED/'canonical.npz',allow_pickle=False) as z:
        parent=dict(q=z['q'].copy(),aux={k[5:]:z[k].copy() for k in z.files if k.startswith('aux__')},
            identity=json.loads(str(z['identity_json'])),damping=float(z['damping']),
            fraction=float(z['next_line_search_fraction']),checkpoint=_reference(ACCEPTED/'canonical.npz'))
    with np.load(ACCEPTED/'complete_state.npz',allow_pickle=False) as z:
        parent.update(physical_q=z['physical_q_augmented'].copy(),expanded_q=z['native_time_expanded_base_canonical_q'].copy())
    payload,_=load_source_frames(ACCEPTED/'review/programme/frames.json')
    sys.path[:0]=[str(DATA),str(DATA.parent),str(DATA.parent.parent/'head_axis_audit')]
    model_ref=payload['model_reference']
    assert _reference(model_ref['path'])==model_ref
    spec=importlib.util.spec_from_file_location('time_detail_real_test_model',model_ref['path'])
    model=importlib.util.module_from_spec(spec);sys.modules[spec.name]=model;spec.loader.exec_module(model);model.install_dispatch()
    old=_CameraState.__new__(_CameraState);old.__setstate__(payload['camera_state']);camera=model.TransverseNativePrincipalState(old)
    exported=load_npz(ACCEPTED/'review/effective_chain.npz').tocsr()
    native=parent['aux']['native_time_native_columns'].ravel()
    other=np.setdiff1d(np.arange(7700),native)
    assert exported[other][:,native].nnz==0
    # The saved export includes B8 already. The original inner native chart is
    # identity on typed PTZ coordinates; retain its exact saved non-native block.
    # Do not invert the compressed B8 block to invent missing derivatives.
    C=exported.tolil();C[:,native]=0.
    for col in native:C[col,col]=1.
    context=dict(parent=parent,payload=payload,model=model,camera=camera,C=C.tocsr(),full_state=None)
    return context


class NativeBase:
    """Real saved physical map on native coordinates; real original source/W."""
    parameter_count=7700
    programme_count=7687
    source_producers=[]
    recovery_references=[]
    def __init__(self,context):
        self.context=context;self.parent=context['parent'];self.start=self.parent['expanded_q'].copy()
        self.shared=np.asarray(self.parent['identity']['global_shared_indices'],int)
        native=self.parent['aux']['native_time_native_columns'].ravel()
        self.fixed=np.setdiff1d(self.parent['identity']['fixed_parameter_indices'],native)
        assert len(self.fixed)==3784
        self.programme=SimpleNamespace(state=context['camera']);self.physical_stage=self;self.factor=SimpleNamespace()
        self.state=SimpleNamespace(world_parameter_indices=context['camera'].world_parameter_indices.copy())
        self.item=next(item for item in context['payload']['frames'] if item['frame']['index']==842)
        self.whitening=FrozenWhitening(self.item['frame'])
        self.scales=self.whitening.scales.copy();self.quadratic=np.zeros(len(self.scales),bool)
        self.guard_calls=[]
    def checkpoint_arrays(self):return {k:v.copy() for k,v in self.parent['aux'].items()}
    def checkpoint_identity(self):return dict(native_base_test=True)
    def chart_identity(self):return dict(existing_saved_derived_chain=True)
    def effective_q_and_chain(self,q):
        delta=q-self.parent['expanded_q']
        result=self.parent['physical_q'].copy()
        if np.any(delta):result+=self.context['C']@delta
        return result,self.context['C'].copy()
    def guard(self,q):
        physical,_=self.effective_q_and_chain(q);self.guard_calls.append(q.copy())
        if self.context['full_state'] is None:
            old=FirstShotPhysicalState.from_binding(self.context['payload']['binding']['path'])
            state=self.context['model'].TransverseNativePrincipalState(old)
            state.original.mount_bounds_active=False;self.context['full_state']=state
        return self.context['full_state'].guard(physical[:7687])
    def source_blocks(self,q,jacobian):
        physical,C=self.effective_q_and_chain(q);world=self.item['worlds']
        count=len(self.context['camera'].world_parameter_indices)
        zero={k:np.zeros(v.shape+(count,)) for k,v in world.items()} if jacobian else None
        residual,J=evaluate_row_timed_frame(self.context['camera'],physical[:7687],self.item['frame'],self.whitening,world,zero)
        pulled=(hstack((J,csr_matrix((len(residual),13))),format='csr')@C).tocsr() if jacobian else None
        yield SourceRowBlock(residual,pulled,self.quadratic,self.scales,dict(original_frame=842,full_W=True))
    def complete_state_payload(self,q):
        physical,_=self.effective_q_and_chain(q)
        return dict(q_augmented=q.copy(),physical_q_augmented=physical,qbase=q[:7687].copy())
    def solver_kwargs(self):return dict(initial_damping=.001,initial_line_search_fraction=1.,cache_native_source_replay=True)
    def recover(self,parent,q):return dict(q=q.copy(),report=dict(original_native_guard_recovery=True))
    def linear_constraints(self,q,selected):
        matrix=csr_matrix(([1.],([0],[0])),shape=(1,172))
        return dict(matrix=matrix,lower=np.array([0.]),report=dict(original=True))
    def close(self):pass


def make_stage(context,active=None,parent=None):
    parent=context['parent'] if parent is None else parent
    return NativeTimeDetailVenueStage(NativeBase(context),parent=parent,
        entry_reference=_reference(__file__),transition_reference=_reference(__file__),active_details=active)


def active_at_real_source(plan):
    # Explicit test directions cover the real frame842 row-time neighbourhood.
    ids=np.flatnonzero(np.asarray(abs(plan.D[840:843]).sum(axis=0)).ravel()>0)
    return np.array([j*156+int(ids[len(ids)//2]) for j in range(3)],np.int64)


def test_real_complete_parent_and_every_row_clock_are_preserved(real_parent):
    stage=make_stage(real_parent);parent=real_parent['parent']
    assert len(parent['aux'])==386 and equal(stage.start,parent['q'])
    expanded,_=stage.expanded_q_and_chain(stage.start);physical,_=stage.effective_q_and_chain(stage.start)
    assert equal(expanded,parent['expanded_q']) and equal(physical,parent['physical_q'])
    assert len(stage.active_details)==0 and equal(stage.shared,np.asarray(parent['identity']['global_shared_indices'],int))
    for key,value in parent['aux'].items():assert equal(stage.checkpoint_arrays()[key],value),key
    payload=stage.complete_state_payload(stage.start)
    assert equal(payload['q_augmented'],parent['q']) and equal(payload['physical_q_augmented'],parent['physical_q'])
    assert equal(payload['native_time_expanded_base_canonical_q'],parent['expanded_q'])
    kwargs=stage.solver_kwargs()
    assert kwargs['initial_damping']==parent['damping'] and kwargs['initial_line_search_fraction']==parent['fraction']
    assert kwargs['reuse_source_jacobian'] is False and stage.source_cache_policy()['topology_or_source_change_invalidates_all_old_source_J']
    assert stage.guard(stage.start)['passed']
    assert equal(stage.base.guard_calls[-1],parent['expanded_q'])


def test_nonzero_details_match_real_source_whitening_and_row_time_FD(real_parent):
    plan=CenteredNativeTimeDetailPlan.from_parent(real_parent['parent']);active=active_at_real_source(plan)
    stage=make_stage(real_parent,active);q=stage.start.copy()
    columns=stage.plan.slots.ravel()[active]
    q[columns]+=np.array([2e-6,-1e-6,3e-6])
    r,J=stage.evaluate(q,True)
    direction=np.zeros(7700);direction[columns]=[.3,-.2,.1]
    for h in (1e-8,1e-9):
        numeric=(stage.evaluate(q+h*direction)-stage.evaluate(q-h*direction))/(2*h)
        analytic=np.asarray(J@direction).ravel()
        relative=np.linalg.norm(numeric-analytic)/max(np.linalg.norm(numeric),1e-30)
        maximum=np.max(abs(numeric-analytic))
        assert relative<=1e-4 or maximum<=1e-5,(h,relative,maximum)
    expanded,T=stage.expanded_q_and_chain(q)
    assert not equal(expanded,real_parent['parent']['expanded_q'])
    # The actual source has row-specific cameras; derivative support reaches
    # both native neighbours and is checked against real UV/query/full W above.
    native=stage.plan.native
    changed=np.flatnonzero(np.any(expanded[native]!=real_parent['parent']['expanded_q'][native],axis=1))
    assert len(changed)>1 and changed.min()<=841<=changed.max()
    assert J[:,columns].nnz and not J[:,stage.fixed_native].nnz
    assert T[:,stage.plan.slots.ravel()[np.setdiff1d(np.arange(468),active)]].nnz==0


def test_native_partial_pullback_uses_physical_columns_and_rejects_B8_cache(real_parent):
    # Exercise a nonidentity derived CAMERA chain as well as the real Brown/RS
    # partials, so ignoring the old chart cannot accidentally pass this check.
    context=dict(real_parent);C=real_parent['C'].tolil()
    C[124,int(real_parent['parent']['aux']['native_time_native_columns'][841,0])]=5.
    context['C']=C.tocsr()
    stage=make_stage(context,active_at_real_source(CenteredNativeTimeDetailPlan.from_parent(real_parent['parent'])))
    physical,_=stage.effective_q_and_chain(stage.start);base=stage.base
    count=len(real_parent['camera'].world_parameter_indices)
    zero={k:np.zeros(v.shape+(count,))for k,v in base.item['worlds'].items()}
    _,J=evaluate_row_timed_frame(real_parent['camera'],physical[:7687],base.item['frame'],base.whitening,base.item['worlds'],zero)
    extra=stage.pullback_native_camera_partials(J,input_basis='physical_native_camera',selected=stage.active_details)
    _,actual=stage.evaluate(stage.start,True)
    np.testing.assert_allclose(extra.toarray(),actual[:,stage.plan.slots.ravel()[stage.active_details]].toarray(),atol=1e-10,rtol=1e-12)
    with pytest.raises(ValueError,match='compressed B8'):
        stage.pullback_native_camera_partials(actual,input_basis='B8_canonical')


def test_real_domains_fixed_history_and_native_recovery_are_retained(real_parent):
    stage=make_stage(real_parent,active_at_real_source(CenteredNativeTimeDetailPlan.from_parent(real_parent['parent'])))
    invalid=stage.start.copy();invalid[5167]=20.1
    with pytest.raises(ValueError,match='readout'):stage.guard(invalid)
    invalid=stage.start.copy();invalid[stage.fixed_native[0]]+=1e-8
    with pytest.raises(ValueError,match='Fixed native history'):stage.guard(invalid)
    proposed=stage.start.copy();proposed[stage.plan.slots.ravel()[stage.active_details]]+=1e-7
    assert equal(stage.recover(stage.start,proposed)['q'],proposed)
    result=stage.linear_constraints(proposed,stage.shared)
    assert result['report']['original'] and result['matrix'].shape==(1,172)


def test_nonzero_detail_complete_restore_retains_original_centres(real_parent,tmp_path):
    stage=make_stage(real_parent,active_at_real_source(CenteredNativeTimeDetailPlan.from_parent(real_parent['parent'])))
    q=stage.start.copy();q[stage.plan.slots.ravel()[stage.active_details]]+=1e-6
    expanded,_=stage.expanded_q_and_chain(q);physical,_=stage.effective_q_and_chain(q)
    arrays=stage.checkpoint_arrays();checkpoint=tmp_path/'detail_parent.npz'
    np.savez(checkpoint,q=q,**{'aux__'+k:v for k,v in arrays.items()})
    parent=dict(q=q,aux=arrays,expanded_q=expanded,physical_q=physical,identity=stage.checkpoint_identity(),
        checkpoint=_reference(checkpoint),damping=7e-8,fraction=.125)
    restored=make_stage(real_parent,parent=parent)
    assert equal(restored.start,q) and equal(restored.plan.centres,stage.plan.centres)
    assert equal(restored.expanded_q_and_chain(q)[0],expanded)
    for key,value in arrays.items():assert equal(restored.checkpoint_arrays()[key],value)
    assert restored.solver_kwargs()['initial_line_search_fraction']==.125
    changed=deepcopy(parent);changed['expanded_q']=expanded.copy();changed['expanded_q'][127]+=1e-8
    with pytest.raises(ValueError,match='parent expansion'):make_stage(real_parent,parent=changed)


@pytest.fixture(scope='module')
def rebuilt_print(real_parent,tmp_path_factory):
    """Compile the actual new Ad7 bank with the unchanged public domain tool."""
    from calibration.program_tracking.first_shot_print_halfspaces import PrintLayoutDomain
    root=Path(__file__).resolve().parents[1]
    original_ref=_reference(root/'events/ankara_2026/programmes/okada_mei_sp/ad7_print_1016.json')
    config=json.loads(Path(original_ref['path']).read_text(encoding='utf-8'))
    # A test-owned config snapshots the current public runtime while the live
    # entry/config may be edited concurrently; the real material is unchanged.
    config['runtime_producer']=_reference(root/'calibration/program_tracking/first_shot_disjoint_print.py')
    config['protocol_fixture_original_configuration']=original_ref
    config_path=tmp_path_factory.getbasetemp()/'live_print_configuration.json'
    assert len(str(config_path).encode('utf-16-le'))//2<260
    assert all(len(part.encode('utf-16-le'))//2<=255 for part in config_path.parts)
    config_path.write_text(json.dumps(config,sort_keys=True),encoding='utf-8')
    config_ref=_reference(config_path)
    material=json.loads(Path(config['material_override']['path']).read_text(encoding='utf-8'))
    physical=FirstShotPhysicalState.from_binding(real_parent['payload']['binding']['path'])
    physical.templates=deepcopy(physical.templates)
    physical.templates['advertisements']['7']=material
    domain=PrintLayoutDomain(physical,7700)
    key='print_H8_QP_domain_identity_json'
    old=json.loads(str(real_parent['parent']['aux'][key]));new=deepcopy(old)
    new['domain']=domain.identity
    new['producers']=[_reference(root/'calibration/program_tracking/first_shot_print_halfspaces.py'),
        _reference(__file__),old['producers'][-1]]
    assert old['domain']!=new['domain']
    assert len([v for v in domain.full_vertex_ledger if v['advertisement_id']==7])==len(material['contours'])
    return dict(key=key,reference=config_ref,config=config,domain=domain,receipt=new,
        value=np.asarray(json.dumps(new,sort_keys=True)))


def new_print_base(context,rebuilt):
    base=NativeBase(context);original_arrays=base.checkpoint_arrays
    base.checkpoint_arrays=lambda:dict(original_arrays(),**{rebuilt['key']:rebuilt['value'].copy()})
    base.checkpoint_identity=lambda:dict(native_base_test=True,print_H8_QP_domain=deepcopy(rebuilt['receipt']))
    base._print_H8_domain=rebuilt['domain']
    base.programme.disjoint_print_receipt=dict(configuration=deepcopy(rebuilt['reference']),
        material_override=deepcopy(rebuilt['config']['material_override']),
        producer=deepcopy(rebuilt['config']['runtime_producer']))
    return base


def detail_with_base(base,parent,**kwargs):
    return NativeTimeDetailVenueStage(base,parent=parent,entry_reference=_reference(__file__),
        transition_reference=_reference(__file__),**kwargs)


def test_rebuilt_print_conflict_requires_explicit_key_and_live_config(real_parent,rebuilt_print):
    parent=real_parent['parent'];base=new_print_base(real_parent,rebuilt_print)
    with pytest.raises(ValueError,match='reconstructed native-base aux print_H8'):
        detail_with_base(base,parent)
    with pytest.raises(ValueError,match='explicit live print configuration'):
        detail_with_base(base,parent,allowed_rebuilt_aux=[rebuilt_print['key']])
    with pytest.raises(ValueError,match='Only the declared print H8'):
        detail_with_base(base,parent,allowed_rebuilt_aux=['native_time_pts'],
            live_model_reference=rebuilt_print['reference'])
    base.programme.disjoint_print_receipt['configuration']=_reference(__file__)
    with pytest.raises(ValueError,match='Installed live print model'):
        detail_with_base(base,parent,allowed_rebuilt_aux=[rebuilt_print['key']],
            live_model_reference=rebuilt_print['reference'])


def test_rebuilt_print_domain_archives_live_value_and_restores_all_parent_aux(real_parent,rebuilt_print,tmp_path):
    parent=real_parent['parent'];kwargs=dict(allowed_rebuilt_aux=[rebuilt_print['key']],
        live_model_reference=rebuilt_print['reference'])
    stage=detail_with_base(new_print_base(real_parent,rebuilt_print),parent,**kwargs)
    arrays=stage.checkpoint_arrays()
    for key,value in parent['aux'].items():assert equal(arrays[key],value),key
    live=stage.checkpoint_identity()['live_base_model_aux']
    assert live['model_configuration']==rebuilt_print['reference']
    archive_key=live['entries'][rebuilt_print['key']]['live_aux_key']
    assert archive_key.startswith('live_print_H8_QP_') and equal(arrays[archive_key],rebuilt_print['value'])
    assert json.loads(str(arrays[archive_key]))['domain']==rebuilt_print['domain'].identity
    assert live['metadata_aux_key'] in arrays
    assert equal(stage.start,parent['q']) and equal(stage.effective_q_and_chain(stage.start)[0],parent['physical_q'])
    checkpoint=tmp_path/'live_domain_parent.npz'
    np.savez(checkpoint,q=stage.start,**{'aux__'+k:v for k,v in arrays.items()})
    restored_parent=dict(parent,q=stage.start.copy(),aux=arrays,identity=stage.checkpoint_identity(),
        checkpoint=_reference(checkpoint))
    restored=detail_with_base(new_print_base(real_parent,rebuilt_print),restored_parent,**kwargs)
    assert set(restored.checkpoint_arrays())==set(arrays)
    for key,value in arrays.items():assert equal(restored.checkpoint_arrays()[key],value),key
    assert restored.checkpoint_identity()['live_base_model_aux']==live


def test_rebuilt_print_permission_does_not_allow_other_parent_aux_changes(real_parent,rebuilt_print):
    base=new_print_base(real_parent,rebuilt_print);original=base.checkpoint_arrays
    changed=real_parent['parent']['aux']['native_time_pts'].copy();changed[0]+=1
    base.checkpoint_arrays=lambda:dict(original(),native_time_pts=changed)
    with pytest.raises(ValueError,match='reconstructed native-base aux native_time_pts'):
        detail_with_base(base,real_parent['parent'],allowed_rebuilt_aux=[rebuilt_print['key']],
            live_model_reference=rebuilt_print['reference'])
