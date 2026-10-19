"""Real saved parent, unfolded shape, source RS/full-W and domain checks.

No fit owner or images. The terminal public 400-aux numerical state and its
real accepted nonzero B4 q are checked against the final complete payload.
"""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import io
import json
import os
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix,load_npz

from calibration.solver_runtime import install
install()
from calibration.program_tracking import first_shot_ad7_shape_chart as art
from calibration.program_tracking import first_shot_disjoint_print as union
from calibration.program_tracking.first_shot_fit import _CameraState,FrozenWhitening
from calibration.program_tracking.first_shot_physical_state import FirstShotPhysicalState
from calibration.program_tracking.first_shot_row_source import row_timed_raw,evaluate_row_timed_frame,_raw_and_local
from calibration.program_tracking.first_shot_safe_frame_export import load_source_frames
from calibration.program_tracking.first_shot_source_snapshot import load_source_snapshot
from calibration.program_tracking.first_shot_source_transition import reference,read
from calibration.program_tracking.first_shot_time_detail_stage import CenteredNativeTimeDetailPlan,NativeTimeDetailVenueStage

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(r'D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/shot1_physical_fit_20261014/zoom_poly_72ac/id_ad5')


def equal(a,b):
    return np.asarray(a).dtype==np.asarray(b).dtype and np.asarray(a).shape==np.asarray(b).shape and np.asarray(a).tobytes()==np.asarray(b).tobytes()


def save_json(path,value):
    assert len(str(path.resolve()).encode('utf-16-le'))//2<260
    assert all(len(x.encode('utf-16-le'))//2<=255 for x in path.parts)
    path.write_text(json.dumps(value,sort_keys=True),encoding='utf-8')
    return reference(path)


def load_camera(record,ref):
    assert reference(ref['path'])==ref
    name='ad7_shape_test_model'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ref['path']);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)
    model=sys.modules[name];model.install_dispatch()
    old=_CameraState.__new__(_CameraState);old.__setstate__(record)
    return model.TransverseNativePrincipalState(old),model


@pytest.fixture(scope='module')
def real(tmp_path_factory):
    if not (DATA/'fixed_head_time4a2/public_solver/latest_accepted_state.npz').exists():
        pytest.skip('Preserved real 400-aux parent is unavailable on this host')
    sys.path[:0]=[str(DATA),str(DATA.parent),str(DATA.parent.parent/'head_axis_audit')]
    from calibration.program_tracking import first_shot_value_execution as value
    parent_installation=value.install()
    current=json.loads((DATA/'fixed_head_time4a2/current.json').read_text(encoding='utf-8'))
    parent_ref=reference(DATA/'fixed_head_time4a2/public_solver/latest_accepted_state.npz')
    with np.load(parent_ref['path'],allow_pickle=False)as z:
        parent=dict(q=z['q'].copy(),aux={k[5:]:z[k].copy()for k in z.files if k.startswith('aux__')},
            checkpoint=parent_ref,identity=json.loads(str(z['identity_json'])),damping=float(z['damping']),
            fraction=float(z['next_line_search_fraction']))
    with np.load(DATA/'fixed_head_time4a2/final_complete_state.npz',allow_pickle=False)as z:
        parent.update(physical_q=z['physical_q_augmented'].copy(),expanded_q=z['native_time_expanded_base_canonical_q'].copy())
    packet,_=load_source_frames(Path(current['canonical_checkpoint']['path']).parent/'review/programme/frames.json')
    camera,model=load_camera(packet['camera_state'],packet['model_reference'])
    physical=FirstShotPhysicalState.from_binding(packet['binding']['path'])
    wrapped=model.TransverseNativePrincipalState(physical);physical.mount_bounds_active=False
    q=parent['physical_q'][:camera.count].copy()
    delta=json.loads((ROOT/'events/ankara_2026/programmes/okada_mei_sp/s1_t8c_source.json').read_text(encoding='utf-8'))
    source,source_receipt=load_source_snapshot(delta['original_source_snapshot'])
    old_frame=source['frames'][750]
    union_ref=reference(ROOT/'events/ankara_2026/programmes/okada_mei_sp/ad7_print_1016.json')
    union.install_execution(union_ref);union.install_material_state(wrapped)
    physical=art.actual_physical_state(wrapped)
    metadata=SimpleNamespace(state=physical,pool=None,transverse_world_cache={},guard_cache={},
        bundle=SimpleNamespace(state=physical,frames=source['frames'],_geometry_pool=None,_source_pool=None,
            _world_cache={},_finite_coverage_cache={}),
        disjoint_print_receipt=dict(source_target_phase='pending_original_source_delta_authentication'))
    union.refresh_source_targets(metadata);frame=metadata.bundle.frames[750]
    descriptors=OrderedDict((r['_target_key'],r['target'])for r in frame['rows'])
    old_worlds=dict(zip(descriptors,wrapped.world_paths(q,list(descriptors.values()))))
    baseline=row_timed_raw(camera,q,frame,old_worlds)
    material_ref=reference(ROOT/'events/ankara_2026/models/ad7_shape_1016.json')
    support_ref=reference(DATA/'staging/final_4a2_source_group_support.json')
    config=art.build_configuration(parent,physical,material_reference=material_ref,source_support_reference=support_ref)
    config_ref=save_json(tmp_path_factory.getbasetemp()/'ad7_groups.json',config)
    art.install_execution(config_ref);art.install_state(wrapped)
    supplemental=[];unobserved=[]
    present=set(group for row in frame['rows']if art._ad7(row['target'])for group in map(art._group,art._keys(row['target'])))
    for needed in sorted(set(range(3))-present):
        found=next((candidate for candidate in metadata.bundle.frames if any(art._ad7(row['target']) and
            needed in set(map(art._group,art._keys(row['target'])))for row in candidate['rows'])),None)
        if found is None:unobserved.append(needed)
        elif all(found is not prior for prior in supplemental):supplemental.append(found)
    targets=OrderedDict((r['_target_key'],r['target'])for r in frame['rows']if r['target'].get('advertisement_id')==7)
    def worlds_at(sample):
        result=dict(old_worlds);paths=wrapped.world_paths(sample,list(targets.values()))
        result.update(zip(targets,paths));return result
    return dict(parent=parent,packet=packet,camera=camera,model=model,physical=physical,wrapped=wrapped,
        q=q,frame=frame,old_frame=old_frame,old_worlds=old_worlds,baseline=baseline,
        config=config,config_ref=config_ref,chart=art._GLOBAL['chart'],targets=targets,worlds_at=worlds_at,supplemental_frames=supplemental,unobserved_source_groups=unobserved,
        tmp=tmp_path_factory.getbasetemp(),source=source,live_frames=metadata.bundle.frames,source_receipt=source_receipt,parent_installation=parent_installation)


def test_real_group_shapes_and_nonlinear_domain_jacobian(real):
    state=real['physical'];q=real['q'];geo=q[state.world_parameter_indices]
    metric,Dmetric=art.canvas_metrics(state,geo);chart=real['chart'];slots=geo[state.print_layout_world_slices[7]].copy()
    offline=art.build_configuration_from_scene(real['parent'],real['packet']['scene'],material_reference=real['config']['material'],source_support_reference=real['config']['source_support_reference'])
    offline_ref=save_json(real['tmp']/'ad7_scene_groups.json',offline)
    proof=art.verify_configuration_against_state(offline_ref,real['parent'],state)
    assert proof['passed'] and proof['parent_aux_count']==len(real['parent']['aux'])
    real['scene_seed_proof']=proof
    slots+=np.array([1e-4,-1e-4,.003,-2e-4,1e-4,2e-4,-1e-4,-.002])
    values,D=chart.paths(slots,metric,Dmetric);params=chart.parameters(slots)
    for key,uv in values.items():
        group=art._group(key);ui,vi,si=((0,1,2),(3,4,7),(5,6,7))[group]
        np.testing.assert_allclose((uv-params[[ui,vi]])*metric,
            np.exp(params[si])*chart.local[key],atol=1e-15,rtol=2e-14)
    margin,J,labels=chart.margins(slots,metric,Dmetric)
    assert len(margin)==15 and J.shape==(15,len(geo)) and chart.guard(slots,metric,Dmetric)['passed']
    for column in (41,43,*range(len(geo)-8,len(geo))):
        step=1e-7;dg=np.zeros(len(geo));dg[column]=step
        ds=np.zeros(8)
        if column>=len(geo)-8:ds[column-(len(geo)-8)]=step
        plus_metric=metric+Dmetric@dg;minus_metric=metric-Dmetric@dg
        numeric=(chart.margins(slots+ds,plus_metric,Dmetric)[0]-chart.margins(slots-ds,minus_metric,Dmetric)[0])/(2*step)
        np.testing.assert_allclose(numeric,J[:,column],atol=3e-9,rtol=2e-5)
    # Log-scale makes this genuinely nonlinear; guard evaluates the real map.
    big=slots.copy();big[2]+=10.
    with pytest.raises(art.Ad7ShapeDomainError,match='Ad7 grouped material domain'):chart.guard(big,metric,Dmetric)
    bad=slots.copy();bad[0]=chart.anchor[0]+1.-chart.seed[0]
    with pytest.raises(ValueError,match='domain'):chart.guard(bad,metric,Dmetric)
    overlap=slots.copy();overlap[5]=overlap[0]+chart.seed[0]-chart.anchor[0]-chart.seed[5]+chart.anchor[5]
    with pytest.raises(ValueError,match='logo_left'):chart.guard(overlap,metric,Dmetric)


def test_actual_world_RS_full_W_and_review_use_one_grouped_shape(real):
    state=real['physical'];q=real['q'].copy();q[5159:5167]+=np.array([1e-4,-1e-4,.002,2e-4,1e-4,-1e-4,2e-4,-.002])
    worlds=real['worlds_at'](q);frame=real['frame'];camera=real['camera'];W=FrozenWhitening(frame)
    # The old ice-only guard can request templates at its neutral shadow. Its
    # result must not be used for actual Ad7 guard/source/review evaluation.
    calls=[];checked=[];original_templates=state.templates_at;original_guard=real['chart'].guard
    def templates_spy(value):
        calls.append(state._geometry(value)[state.print_layout_world_slices[7]].copy())
        return original_templates(value)
    def guard_spy(slots,metrics,Dmetric):
        checked.append(np.asarray(slots).copy())
        return original_guard(slots,metrics,Dmetric)
    state.templates_at=templates_spy;real['chart'].guard=guard_spy
    try:
        assert real['wrapped'].guard(q)['passed']
    finally:
        del state.templates_at;real['chart'].guard=original_guard
    assert checked and all(equal(value,q[5159:5167])for value in checked)
    assert any(not np.any(value)for value in calls)
    metric,Dmetric=art.canvas_metrics(state,q[state.world_parameter_indices])
    margin,_,_=real['chart'].margins(q[5159:5167],metric,Dmetric)
    assert np.all(margin[:12]>=0) and np.all(margin[12:]>0)
    real['actual_q_guard_neutral_spy']=dict(passed=True,neutral_template_calls=sum(not np.any(v)for v in calls),
        all_actual_shape_guards_used_real_q=True,actual_minimum_margin=float(margin.min()))

    ng=len(camera.world_parameter_indices);derivatives={k:np.zeros(w.shape+(ng,))for k,w in worlds.items()}
    layout=state.print_layout_world_jacobians(q[state.world_parameter_indices],list(real['targets'].values()))
    indices=state.print_layout_global_indices_by_ad[7];local=np.flatnonzero(np.isin(camera.world_parameter_indices,indices))
    for key,j in zip(real['targets'],layout):derivatives[key][...,local]=j[...,state.print_layout_tail_slices[7]]
    raw=row_timed_raw(camera,q,frame,worlds)
    mask=np.concatenate([np.full(len(row['uv']),row['target'].get('advertisement_id')==7)for row in frame['rows']])
    np.testing.assert_array_equal(raw.reshape(-1,2)[~mask],real['baseline'].reshape(-1,2)[~mask])
    r,J=evaluate_row_timed_frame(camera,q,frame,W,worlds,derivatives)
    np.testing.assert_array_equal(r,W.apply(raw))
    errors=[];supported=set()
    cases=[(frame,worlds,real['worlds_at'],J)]
    for extra in real['supplemental_frames']:
        descriptors=OrderedDict((row['_target_key'],row['target'])for row in extra['rows'])
        selected=OrderedDict((key,target)for key,target in descriptors.items()if art._ad7(target))
        fixed=dict(zip(descriptors,real['wrapped'].world_paths(q,list(descriptors.values()))))
        def sample_worlds(sample,fixed=fixed,selected=selected):
            value=dict(fixed);value.update(zip(selected,real['wrapped'].world_paths(sample,list(selected.values()))));return value
        extra_world=sample_worlds(q);extra_D={key:np.zeros(path.shape+(ng,))for key,path in extra_world.items()}
        extra_layout=state.print_layout_world_jacobians(q[state.world_parameter_indices],list(selected.values()))
        for key,j in zip(selected,extra_layout):extra_D[key][...,local]=j[...,state.print_layout_tail_slices[7]]
        extra_W=FrozenWhitening(extra)
        _,extra_J=evaluate_row_timed_frame(camera,q,extra,extra_W,extra_world,extra_D)
        cases.append((extra,extra_world,sample_worlds,extra_J))
    for actual,_,sample_worlds,actual_J in cases:
        actual_W=FrozenWhitening(actual)
        for column in indices:
            step=1e-7;d=np.zeros(len(q));d[column]=step
            plus=evaluate_row_timed_frame(camera,q+d,actual,actual_W,sample_worlds(q+d),None)[0]
            minus=evaluate_row_timed_frame(camera,q-d,actual,actual_W,sample_worlds(q-d),None)[0]
            numeric=(plus-minus)/(2*step);analytic=actual_J[:,column].toarray().ravel()
            maximum=float(np.max(abs(numeric-analytic)));relative=float(np.linalg.norm(numeric-analytic)/max(np.linalg.norm(numeric),1e-30))
            assert maximum<=2e-4 or relative<=2e-5,(int(column),maximum,relative)
            if np.any(analytic):supported.add(int(column))
            errors.append(dict(frame_index=actual['index'],column=int(column),maximum=maximum,relative=relative,source_supported=bool(np.any(analytic))))
    groups=set(range(3))-set(real['unobserved_source_groups'])
    expected={int(indices[local])for group in groups for local in ((0,1,2),(3,4,7),(5,6,7))[group]}
    assert supported==expected,(supported,expected)
    real['source_unsupported_columns']=sorted(set(map(int,indices))-supported)
    _,_,columns=_raw_and_local(camera,q,frame,worlds,derivatives)
    column=max(map(int,columns),key=lambda c:np.linalg.norm(J[:,c].data))
    for column,step in ((column,1e-8),(5167,1e-4)):
        d=np.zeros(len(q));d[column]=step
        if column==5167 and abs(q[column])+step>20.:
            # Actual accepted readout is on its original domain boundary.
            # Keep the domain intact and use its second-order inward derivative.
            inward=-np.sign(q[column])*d
            once=evaluate_row_timed_frame(camera,q+inward,frame,W,worlds,None)[0]
            twice=evaluate_row_timed_frame(camera,q+2*inward,frame,W,worlds,None)[0]
            numeric=(-3*r+4*once-twice)/(2*inward[column])
        else:
            plus=evaluate_row_timed_frame(camera,q+d,frame,W,worlds,None)[0]
            minus=evaluate_row_timed_frame(camera,q-d,frame,W,worlds,None)[0]
            numeric=(plus-minus)/(2*step)
        analytic=J[:,column].toarray().ravel()
        assert np.max(abs(numeric-analytic))<=2e-3 or np.linalg.norm(numeric-analytic)<=2e-5*np.linalg.norm(numeric)
    from calibration.program_tracking.render_result import source_conditioned_board_print_paths
    scene=state.review_scene(q[state.world_parameter_indices])
    full=[p for p in source_conditioned_board_print_paths(scene)if p['number']==7]
    assert len(full)==38 and 'ad7_grouped_shape_chart' in scene
    by_key={p['role'].removeprefix('shared_Ad7_print_'):p['xyz']for p in full}
    # Evaluate all 38 real bank contours through the same public world paths,
    # including top outlines that have no currently assigned source target.
    all_targets=[dict(kind='ad_print',advertisement_id=7,contour_key=key)for key in real['chart'].keys]
    geo=q[state.world_parameter_indices];all_paths=state.world_paths(geo,all_targets)
    all_J=state.print_layout_world_jacobians(geo,all_targets)
    for key,path in zip(real['chart'].keys,all_paths):np.testing.assert_array_equal(path,by_key[key])
    world_errors=[]
    for slot,column in enumerate(indices):
        delta=np.zeros(len(geo));delta[state.print_layout_world_slices[7].start+slot]=1e-7
        plus=state.world_paths(geo+delta,all_targets);minus=state.world_paths(geo-delta,all_targets)
        numeric=np.concatenate([(a-b)/(2e-7)for a,b in zip(plus,minus)])
        analytic=np.concatenate([j[...,state.print_layout_tail_slices[7].start+slot]for j in all_J])
        maximum=float(np.max(abs(numeric-analytic)));relative=float(np.linalg.norm(numeric-analytic)/np.linalg.norm(numeric))
        assert np.any(analytic) and (maximum<=2e-6 or relative<=2e-5),(int(column),maximum,relative)
        world_errors.append(dict(column=int(column),maximum=maximum,relative=relative))
    real['world_errors']=world_errors
    real['source_errors']=errors


def test_original_full_physical_domains_are_retained(real):
    state=real['physical'];q=real['q'].copy();wrapped=real['wrapped']
    assert wrapped.guard(q)['passed']
    full=state.guard(q[:state.count],check_mount_bounds=False)
    assert full['shared_print_layout_domain']['7']['old_H8_not_evaluated_as_Ad7_ink']
    for column,value,match in ((5167,20.1,'readout'),(state.view_start+3,-100.,'focal')):
        bad=q.copy();bad[column]=value
        with pytest.raises(ValueError,match=match):wrapped.guard(bad)
    bad=q.copy();bad[41]=-100.
    with pytest.raises(ValueError):wrapped.guard(bad)
    bad=q.copy();bad[5159]+=2.
    with pytest.raises(ValueError,match='Ad7 grouped material'):wrapped.guard(bad)
    # The old Ad2 H8 pole/containment guard still sees exactly its own values.
    bad=q.copy();bad[5125]=10.
    with pytest.raises(ValueError):wrapped.guard(bad)


def test_real_400_aux_nonzero_B4_parent_survives_new_model_continuation(real):
    parent=real['parent'];q=parent['q'].copy();aux=parent['aux'];identity=parent['identity']
    damping,fraction=parent['damping'],parent['fraction']
    assert len(aux)==400 and damping==2.56e-10 and fraction==1.0
    plan=CenteredNativeTimeDetailPlan.from_parent(parent)
    expanded=parent['expanded_q'];physical=parent['physical_q']
    assert equal(plan.expand(q),expanded) and np.any(q[plan.slots]-plan.centres)
    current=json.loads((DATA/'fixed_head_time4a2/current.json').read_text(encoding='utf-8'))
    C=load_npz(Path(current['canonical_checkpoint']['path']).parent/'review/effective_chain.npz').tocsr()
    native=plan.native.ravel();other=np.setdiff1d(np.arange(7700),native)
    assert C[other][:,native].nnz==0
    C=C.tolil();C[:,native]=0.
    for col in native:C[col,col]=1.
    C=C.tocsr()
    class Base:
        parameter_count=7700;programme_count=7687
        def __init__(self):
            self.shared=np.asarray(identity['global_shared_indices'],int)
            self.fixed=np.setdiff1d(identity['fixed_parameter_indices'],native)
            assert len(self.fixed)==3784
            self.programme=SimpleNamespace(state=real['camera'],bundle=SimpleNamespace(frames=real['live_frames']));self.physical_stage=SimpleNamespace(state=real['camera'])
            self.factor=None;self.scales=np.ones(1);self.quadratic=np.zeros(1,bool)
            self.source_producers=[];self.recovery_references=[]
        def checkpoint_arrays(self):return {k:v.copy()for k,v in aux.items()}
        def checkpoint_identity(self):return {}
        def chart_identity(self):return dict(original_inner_native_chart=True)
        def effective_q_and_chain(self,value):
            delta=value-expanded
            return (physical+C@delta if np.any(delta) else physical.copy()),C.copy()
        def solver_kwargs(self):return {}
    stage=NativeTimeDetailVenueStage(Base(),parent=parent,entry_reference=reference(__file__),
        transition_reference=reference(__file__),active_details=None)
    fixed_receipt=art.finalize_stage(stage,configuration_reference=real['config_ref'])
    assert fixed_receipt['parameters']==['top_u','top_v','bottom_u','bottom_v','log_text_scene_scale']
    assert len(stage.shared)==167 and not np.isin(np.arange(5162,5167),stage.shared).any() and np.isin(np.arange(5162,5167),stage.fixed).all()
    for column in range(5162,5167):
        invalid=q.copy();invalid[column]+=1e-9
        with pytest.raises(ValueError,match='Fixed'):stage.expanded_q_and_chain(invalid)
    assert equal(stage.start,q) and equal(stage.plan.centres,aux['time_detail_centres'])
    assert equal(stage.expanded_q_and_chain(q)[0],expanded) and equal(stage.effective_q_and_chain(q)[0],physical)
    assert np.any(q[plan.slots]-plan.centres)
    for key,value in aux.items():assert equal(stage.checkpoint_arrays()[key],value),key
    kwargs=stage.solver_kwargs();assert kwargs['initial_damping']==damping and kwargs['initial_line_search_fraction']==fraction
    receipt=art.verify_parent_continuation(stage,parent)
    real['parent_continuation']=receipt
    assert receipt['parent_aux_count']>=400 and receipt['all_row_time_cameras_byte_exact']
    # A new physical art seed is permitted to change Ad7 only; it must not
    # rebuild zero B4 details from this complete successor canonical q.
    config=art.build_configuration(parent,real['physical'],material_reference=real['config']['material'],source_support_reference=real['config']['source_support_reference'])
    assert config['parent_q_sha256']==art._sha(q) and equal(stage.plan.centres,plan.centres)

    if os.environ.get('RINKSPACE_AD7_QA_OUTPUT'):
        save_json(Path(os.environ['RINKSPACE_AD7_QA_OUTPUT']),dict(
            schema='rinkspace.ad7-grouped-shape-real-QA.v1',passed=True,
            module=reference(art.__file__),tests=reference(__file__),parent_checkpoint=parent['checkpoint'],
            parent_continuation=receipt,initialization_only_fixed=fixed_receipt,scene_seed=real['scene_seed_proof'],source_directions=real['source_errors'],
            all_38bank_world_directions=real['world_errors'],source_unsupported_columns=real['source_unsupported_columns'],
            actual_q_guard_neutral_spy=real['actual_q_guard_neutral_spy'],
            full_38bank_source_review_same_geometry=True,original_nonAd7_raw_byte_exact=True,
            all_original_physical_guards_retained=True,new_physical_material_predictions_not_parent_equivalent=True,
            source_accuracy_or_heldout_improvement_claimed=False,old_J_reused=False))
