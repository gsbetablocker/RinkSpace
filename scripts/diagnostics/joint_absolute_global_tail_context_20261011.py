"""Small q8 global source tails, using the original public operators.

No motion graph, source image decoding, fitting, or source observation writes.
The existing pattern owner is retained so native material and placement factors
continue to share exactly one physical producer.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MethodType
import hashlib
import json
import time
import argparse

import joint_absolute_source_context_20261011 as ctx
a=ctx.a
np=ctx.np
from scipy.sparse import csr_matrix
from calibration.program_tracking.z190_physical_camera import SharedCalibration,ViewState
from z190_physical_camera import SharedCalibration as OriginalSharedCalibration
from structured_programme_source_candidate.current_full_source import CurrentExternal,CurrentPatterns
from structured_programme_source_candidate.paired_programme_chart import PairedProgrammeMountOpticsChart
from shared_board_course_candidate.course_pattern_adapter import CourseMountOpticsPatterns
from all_installation_optics_candidate.all_installation_optics_graph import AllInstallationOpticsPatterns
from secondary_optics_candidate.secondary_optics_graph import OpticsPatternFactor
from z190_external_main_low_source import ExternalMainLowSourceBundle
from z190_joint_shared_board_pattern_adapter import JointSharedBoardPatterns
from z190_external_board_pattern_source import ExternalBoardPatternDistanceSourceFactor,FACTOR_KIND
from z190_shared_board_pattern_source import SharedBoardPatternSourceFactor,SHIELD_ROLES
from structured_programme_source_candidate.batched_absolute_geometry_fd import _feasible_world_stencil
from calibration.solver_runtime.source_loss import weighted_source_cost
from rinkspace.storage import AssetStore
from z190_continuous_source_bundle import SOURCE_SHA


def _file_sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


_IMPORT_SHA=_file_sha(__file__)


def _source_equal(left,right):
    if isinstance(left,dict)and isinstance(right,dict):
        return set(left)==set(right)and all(_source_equal(left[k],right[k])for k in left)
    if isinstance(left,(list,tuple,np.ndarray))and isinstance(right,(list,tuple,np.ndarray)):
        return len(left)==len(right)and all(_source_equal(x,y)for x,y in zip(left,right))
    return bool(left==right)


def _authenticate_png(c,path,expected,receipts):
    actual=AssetStore(a.ROOT).source(path)
    digest=_file_sha(actual)
    if digest!=expected:raise ValueError('Original tail source PNG SHA changed: '+str(path))
    receipts[str(path)]=dict(path=str(path),actual_read_path=str(actual),sha256=digest,
        decoded=False,existing_exact_PNG_bytes=True)


def _install_camera_hooks(c):
    state=c.state;reference=c.q.copy()
    roll=json.loads(str(c.arrays['aux__correlated_roll_prior_contract_json'].item()))
    optical=json.loads(str(c.arrays['aux__all_installation_optics_contract_json'].item()))
    state.ids=list(roll['installation_order'])
    pivots=np.asarray(c.arrays['aux__installation_xyz_slots'],int)
    mounts=np.asarray(c.arrays['aux__current_shared_mount_parameter_indices'],int).reshape(3,3)
    optics=np.asarray(c.arrays['aux__current_all_installation_optics_parameter_indices'],int).reshape(3,3)
    state.slots={name:pivots[i].copy()for i,name in enumerate(state.ids)}
    state.installation_optics={}
    for i,name in enumerate(state.ids):
        item=optical['installations'][name]
        state.installation_optics[name]=PairedProgrammeMountOpticsChart(name,tuple(map(int,optics[i])),
            int(item['original_parameter_count']),tuple(item['physical_derivative_steps']),tuple(map(int,mounts[i])),
            additional_world_parameter_count=c.provider.parameter_count-66)
    state.external_optics_base_calibration=OriginalSharedCalibration.from_manifest(optical['external_base_calibration'])
    for block in state.blocks:
        name=block.bundle.calibration.installation_id;i=state.ids.index(name)
        cal0=block.bundle.calibration
        def current_calibration(value,*,i=i,cal0=cal0):
            value=np.asarray(value);dp=value[pivots[i]]-reference[pivots[i]]
            dm=value[mounts[i]]-reference[mounts[i]];do=value[optics[i]]-reference[optics[i]]
            return replace(cal0,pivot_xyz=tuple(np.asarray(cal0.pivot_xyz)+dp/c.provider.scene_unit_m),
                lever_xyz=tuple(np.asarray(cal0.lever_xyz)+dm/c.provider.scene_unit_m),
                principal_x_fraction=(cal0.principal_x_fraction[0]+do[0]/1920.,),
                principal_y_fraction=(cal0.principal_y_fraction[0]+do[1]/1080.,),k1=(cal0.k1[0]+.001*do[2],))
        def view(value,pts,*,block=block):
            row=np.asarray(value)[block.start+4*block.bundle.index[int(pts)]:][:4]
            return ViewState(tuple(row[:3]),float(row[3]-np.log(1920.)))
        block.bundle.current_calibration=current_calibration;block.bundle.view=view
    state.external_start=int(c.arrays['aux__external_start']);state.external_stop=int(c.arrays['aux__external_stop'])
    return roll


def _external_owner(c):
    source=c.packet['external'];external=ExternalMainLowSourceBundle.__new__(ExternalMainLowSourceBundle)
    for key,value in source['row_sets'].items():setattr(external,key,deepcopy(value))
    external.canonical_pts=np.asarray(source['canonical_pts'],np.int64).copy()
    external.index={int(p):i for i,p in enumerate(external.canonical_pts)}
    external.aliases={int(k):int(v)for k,v in source['aliases'].items()}
    external.held_pts=frozenset(map(int,source['held_pts']));external.local_size=4*len(external.canonical_pts)+len(external.track_sources)
    assert external.local_size==154 and external.held_pts=={15657728}
    external.top_world_scene=np.asarray(source['top_world_scene'],float).copy()
    external.top_world_m=np.asarray(source['top_world_m'],float).copy()
    external.manifest=deepcopy(source['source_manifest']);external.calibration=c.state.external_optics_base_calibration
    external.lever_m=np.asarray(external.calibration.lever_xyz)*c.provider.scene_unit_m
    external.pp=np.array([1920*external.calibration.principal_x_fraction[0],1080*external.calibration.principal_y_fraction[0]])
    external.k1=float(external.calibration.k1[0]);c.state.external=external
    assert np.array_equal(external.canonical_pts,c.arrays['aux__external_canonical_pts'])
    for ref in external.manifest['source_authorities'].values():
        if _file_sha(ref['path'])!=ref['sha256']:raise ValueError('Original external source container changed')
    return external,CurrentExternal(c.state)


def _pattern_frames(c,external,receipts):
    root=Path('D:/CodexData/.codex/visualizations/2026/10/02/private-thread-00008/RinkSpace_tmp/low5_repair/secondary6_Ad13_Ad12_native_patterns_v1')
    frames=[];source_refs=[]
    for ad,expected in ((13,'214448266965e5ee14e8519c0fef8bac03bba356273c8acdaea17c00f84775e5'),
                        (12,'8bc7571bac78ebb6c3ca6fa2d21f9085ba43b8ac9277021e53883293f3d8e28f')):
        path=root/f'secondary6_Ad{ad}_native_pattern_QUALIFIED_FROZEN.json';source=a.read(path)
        assert _file_sha(path)==expected and source['physical_ad_id']==ad
        assert source['source_video_sha256']==SOURCE_SHA
        source_refs.append(dict(path=str(path),sha256=expected))
        for record in source['frames']:
            frame=deepcopy(record);native=int(frame['source_native_pts']);actual=int(frame['source_actual_pts'])
            assert c.state.original.aliases[native]==actual and c.state.bypts[actual].name=='secondary'
            assert frame['split']in('fit','held')and frame['frozen_before_fit']is True
            assert frame['model_projection_used_to_generate_or_select_uv']is False
            if frame['raw_png']not in receipts:_authenticate_png(c,frame['raw_png'],frame['raw_png_sha256'],receipts)
            frame['observer_state_owner']='programme';frames.append(frame)
    ref=external.manifest['source_authorities']['patterns'];data=a.read(ref['path'])
    assert _source_equal(data['rows'],external.pattern_source_records)
    source_refs.append(dict(ref));admitted={};excluded=0
    for original in data['rows']:
        source=deepcopy(original);native=int(source['source_native_pts']);sr=source['source_ref']
        if sr['raw_png']not in receipts:_authenticate_png(c,sr['raw_png'],sr['raw_png_sha256'],receipts)
        assert source['source_visible']is True and source['person_occlusion']is False
        assert source['manually_qualified_source_identity']is True and source['model_projection_used']is False
        if source['physical_id']=='Ad12':
            assert native not in external.aliases and source['outside_actual38']is True and source['fit_admitted']is False
            excluded+=1;continue
        actual=external.aliases[native];split='held'if actual in external.held_pts else'fit'
        assert source['physical_id']=='Ad13'and source['source_actual_pts']==actual
        assert source['split']==('held'if split=='held'else'development_source')
        if actual not in admitted:
            admitted[actual]=dict(source_native_pts=native,source_actual_pts=actual,physical_ad_id=13,
                physical_pattern_id='Ad13_outermost_circular_seal',split=split,engineering_sigma_native_px=[4.,4.],
                source_factor_kind=FACTOR_KIND,source_curve_support=[],observer_state_owner='external',
                frozen_before_fit=True,model_projection_used_to_generate_or_select_uv=False)
        source.update(uv=deepcopy(source['uv_native']),source_support_manually_qualified=True,
            source_factor_kind=FACTOR_KIND,native_support_bounds_xy_px=None,source_original_row=deepcopy(original))
        admitted[actual]['source_curve_support'].append(source)
    assert set(admitted)=={15653888,15655168,15657728,15661568}and excluded==12
    assert all(len(frame['source_curve_support'])==16 for frame in admitted.values())
    frames.extend(admitted[p]for p in sorted(admitted))
    return frames,source_refs


def _pattern_camera(owner,q,frame,*,held=False):
    if frame['observer_state_owner']=='programme':
        block=owner.state.bypts[int(frame['source_actual_pts'])];cal=block.bundle.current_calibration(q)
        camera=block.bundle.camera(q,int(frame['source_actual_pts']))
        return camera,block.xyz_slots,block.start+4*block.bundle.index[int(frame['source_actual_pts'])],np.asarray(cal.lever_xyz)*owner.state.geometry.scene_unit_m
    return AllInstallationOpticsPatterns.camera(owner,q,frame,held=held)


def _install_pattern_owner(c,external):
    owner=c.state.patterns;assert owner is c.material.patterns and owner is c.placement.patterns
    receipts={};frames,refs=_pattern_frames(c,external,receipts)
    owner.frames=frames;owner.fit=[f for f in frames if f['split']=='fit'];owner.held=[f for f in frames if f['split']=='held']
    owner.installation_optics=c.state.installation_optics;owner.course_indices=np.asarray(c.world_indices[44:50],int)
    owner.course_fd_steps=np.full(6,1e-6)
    steps=next(iter(owner.installation_optics.values())).physical_steps
    base=owner._original_course_pattern_geometry
    owner._original_course_pattern_factor=OpticsPatternFactor(SharedBoardPatternSourceFactor(base),steps)
    owner._original_course_external_pattern_factor=OpticsPatternFactor(ExternalBoardPatternDistanceSourceFactor(base),steps)
    owner.factor=owner._original_course_pattern_factor;owner.external_factor=owner._original_course_external_pattern_factor
    for name,method in [('camera',_pattern_camera),('chart_for_frame',AllInstallationOpticsPatterns.chart_for_frame),
        ('frame_factor',JointSharedBoardPatterns.frame_factor),('_evaluate_current',CourseMountOpticsPatterns._evaluate_current),
        ('_restore_camera_owners',CourseMountOpticsPatterns._restore_camera_owners),
        ('_course_fd_stencil',CurrentPatterns._course_fd_stencil),('evaluate',CourseMountOpticsPatterns.evaluate)]:
        setattr(owner,name,MethodType(method,owner))
    owner._joint_global_tail_sources=refs;owner._joint_global_tail_source_auth=receipts
    return owner


@contextmanager
def _raw_pattern_frames(owner):
    old=owner.fit;raw=deepcopy(old)
    for frame in raw:
        key='source_curve_support'if frame['physical_ad_id']in(13,26)else'source_pattern_points'
        for source in frame[key]:
            if frame.get('source_factor_kind')==FACTOR_KIND:source['normal_support_interval_px']=[0.,0.]
            else:
                uv=np.asarray(source['uv'],float);source['native_support_bounds_xy_px']=np.column_stack((uv,uv)).tolist()
    owner.fit=raw
    try:yield
    finally:owner.fit=old


def _evaluate_external(c,factor,value,jacobian=False,return_ledger=False):
    core=c.state.structured_programme_coordinate_chart.core_count
    result=factor.evaluate(np.asarray(value)[:core],jacobian=jacobian)
    r=result.residual.copy();ledger=result.source_row_ledger
    sigma=np.deg2rad(c._global_tail_roll['component_sigma_degrees']);old_sigma=np.deg2rad(.5)
    bias=int(c.state.structured_programme_coordinate_chart.bias_indices[0]);selected=[]
    for item in ledger:
        if item['id'].startswith('roll_')and item['actual_pts']not in c.state.external.held_pts:
            begin,end=item['residual_slice'];assert end==begin+1;selected.append(begin)
    assert len(selected)==37
    r[selected]=(r[selected]-value[bias]/old_sigma)*(old_sigma/sigma)
    if not jacobian:return (r,ledger)if return_ledger else r
    j=result.global_jacobian.tocsr(copy=True);j.resize((len(r),len(value)))
    if sigma!=old_sigma:
        for i in selected:j.data[j.indptr[i]:j.indptr[i+1]]*=old_sigma/sigma
    j+=csr_matrix((np.full(len(selected),-1/sigma),(selected,np.full(len(selected),bias))),shape=j.shape)
    return r,j,ledger


def _evaluate_pattern(c,owner,value,jacobian=False,raw=False):
    core=c.state.structured_programme_coordinate_chart.core_count
    with c.placement.pattern_placement_context(value):
        if raw:
            with _raw_pattern_frames(owner):result=owner.evaluate(np.asarray(value)[:core],jacobian)
        else:result=owner.evaluate(np.asarray(value)[:core],jacobian)
    if not jacobian:return np.asarray(result)
    r,j=result;j=j.tocsr(copy=False);j.resize((len(r),len(value)));return r,j


def _column_labels(c,columns):
    labels={int(col):dict(column=int(col),kind='shared_world',logical_world_coordinate=i)
        for i,col in enumerate(c.world_indices)}
    for name in c.state.ids:
        for kind,indices in [('installation_pivot',c.state.slots[name]),
            ('installation_optics',c.state.installation_optics[name].indices),
            ('installation_mount',c.state.installation_optics[name].mount_indices)]:
            for k,col in enumerate(indices):labels[int(col)]=dict(column=int(col),kind=kind,installation_id=name,coordinate=k)
    for block in c.state.blocks:
        for pts,i in block.bundle.index.items():
            for k in range(4):
                col=block.start+4*i+k
                if col in columns:labels[col]=dict(column=col,kind='programme_RF4',actual_pts=int(pts),block=block.name,coordinate=k)
    external=c.state.external
    for pts,i in external.index.items():
        for k in range(4):
            col=c.state.external_start+4*i+k
            labels[col]=dict(column=col,kind='external_RF4',actual_pts=int(pts),coordinate=k,
                fixed_held=pts in external.held_pts)
    start=c.state.external_start+4*len(external.canonical_pts)
    for k in range(len(external.track_sources)):
        labels[start+k]=dict(column=start+k,kind='external_unknown_anchor_log_depth',coordinate=k)
    for k,col in enumerate(c.state.structured_programme_coordinate_chart.bias_indices):
        labels[int(col)]=dict(column=int(col),kind='statistical_roll_bias',installation_id=c.state.ids[k])
    for k in range(9):labels[c.state.patterns.start+k]=dict(column=c.state.patterns.start+k,kind='shared_pattern_q9',coordinate=k)
    return [labels[int(col)]for col in columns]


def _saved_pattern_values(c,owner):
    records=a.read(c.directory/'source_review_UNACCEPTED.json')['independent_board_pattern_source']['fit']
    assert len(records)==len(owner.fit)
    values=[]
    for record,frame in zip(records,owner.fit):
        assert record['physical_ad_id']==frame['physical_ad_id']and record['actual_pts']==frame['source_actual_pts']
        for item in record['source_reports']:
            if 'native_sigma_scaled_excess'in item:values.append(float(item['native_sigma_scaled_excess']))
            elif 'distance_native_px'in item:values.append(float(item['distance_native_px'])/float(item['source_sigma_px']))
            else:values.extend(np.asarray(item['outside_support_native_px'])/frame['engineering_sigma_native_px'])
    return np.asarray(values,float)


def _producer_signature(c,factor,family,world):
    g=c.provider.current;cq=g.current_course6(world)
    if family=='shared_board_patterns':return np.asarray(cq,float).tobytes()
    values=[np.asarray(factor.material.evaluate(cq),float).ravel()]
    values.extend(np.asarray(g.corner(14,row['source_corner_role'],world),float).ravel()for row in factor.ad14)
    return np.concatenate(values).tobytes()


@contextmanager
def _public_derivative_widths(c,multiplier):
    charts=c.state.installation_optics;old=dict(charts);owner=c.state.patterns
    originals=[owner._original_course_pattern_factor,owner._original_course_external_pattern_factor]
    widths=[factor.steps for factor in originals];course=owner.course_fd_steps.copy()
    for name,chart in old.items():charts[name]=replace(chart,physical_steps=tuple(np.asarray(chart.physical_steps)*multiplier))
    for factor,width in zip(originals,widths):factor.steps=tuple(np.asarray(width)*multiplier)
    owner.course_fd_steps=course*multiplier
    try:yield
    finally:
        charts.clear();charts.update(old);owner.course_fd_steps=course
        for factor,width in zip(originals,widths):factor.steps=width


def prepare_tails(c):
    if hasattr(c,'global_tails'):return c.global_tails
    started=time.perf_counter();c._global_tail_roll=_install_camera_hooks(c)
    external,ef=_external_owner(c);pf=_install_pattern_owner(c,external)
    ext,ledger=_evaluate_external(c,ef,c.q,False,True)
    pattern=_evaluate_pattern(c,pf,c.q,False);raw_pattern=_evaluate_pattern(c,pf,c.q,False,True)
    saved_pattern=_saved_pattern_values(c,pf);assert pattern.shape==saved_pattern.shape==(104,)
    saved_pattern_error=float(np.max(abs(pattern-saved_pattern)));assert saved_pattern_error<1e-12
    roll=c._global_tail_roll;bias=np.asarray(c.state.structured_programme_coordinate_chart.bias_indices,int)
    sigma=np.deg2rad(roll['component_sigma_degrees'])
    core=c.state.structured_programme_coordinate_chart.core_count
    ecols=np.unique(np.r_[c.world_indices,c.state.slots[c.state.ids[0]],c.state.slots[c.state.ids[2]],
        c.state.installation_optics[c.state.ids[0]].indices,c.state.installation_optics[c.state.ids[0]].mount_indices,
        np.arange(c.state.external_start,c.state.external_stop),bias[0]]).astype(int)
    pcols=np.unique(np.r_[c.world_indices,np.arange(pf.start,pf.start+9),
        np.concatenate([np.r_[slots,np.arange(view,view+4),pf.chart_for_frame(frame).indices,pf.chart_for_frame(frame).mount_indices]
            for frame in pf.fit for camera,slots,view,lever in [pf.camera(c.q[:core],frame)]])]).astype(int)
    elabels=[]
    for item in ledger:
        for i in range(*item['residual_slice']):elabels.append(dict(id=item['id'],actual_pts=item['actual_pts'],
            source_native_pts=item['source_native_pts'],scalar_in_source=i-item['residual_slice'][0],
            role='engineering_roll'if item['id'].startswith('roll_')else'original_source',
            raw_support_retained_no_measured_centre=not item['id'].startswith(('roll_','external_door_'))))
    plabels=[]
    for frame in pf.fit:
        sources=frame['source_curve_support']if frame['physical_ad_id']in(13,26)else sorted(frame['source_pattern_points'],key=lambda r:SHIELD_ROLES.index(r['physical_pattern_point_role']))
        for source in sources:
            for axis in range(1 if frame['physical_ad_id']in(13,26)else 2):
                plabels.append(dict(id=source['id'],actual_pts=frame['source_actual_pts'],source_native_pts=frame['source_native_pts'],
                    physical_ad_id=frame['physical_ad_id'],observer_state_owner=frame['observer_state_owner'],scalar_in_source=axis,role='original_source'))
    definitions=[('external_main_to_low',ef,ext,ext.copy(),ecols,elabels),
        ('shared_board_patterns',pf,pattern,raw_pattern,pcols,plabels),
        ('installation_correlated_roll_hardware_bias',None,c.q[bias]/sigma,c.q[bias]/sigma,bias,
            [dict(id='statistical_bias_'+name,role='engineering_bias')for name in c.state.ids])]
    tails=[]
    for family,factor,value,raw,columns,labels in definitions:
        stored=c.ledgers[family][0];l=stored['original_row_ledger'];lo,hi=l['original_start'],l['original_stop']
        scales=c.arrays['scales'][lo:hi].copy();quad=c.arrays['quadratic'][lo:hi].copy()
        assert len(value)==len(scales)==len(labels)
        cost=weighted_source_cost(value,quad,None);saved=stored['normalized_cost']
        if abs(cost-saved)>=2e-8:raise ValueError('Tail baseline differs from actual saved source: '+str((family,cost,saved)))
        tail=dict(family=family,factor=factor,actual=value,raw=raw,columns=columns,row_labels=labels,
            column_labels=_column_labels(c,columns),
            scales=scales,quadratic=quad,loss=None,baseline_receipt=dict(family=family,scalar_rows=len(value),
                actual_cost=cost,saved_cost=saved,cost_error=cost-saved,source_images_decoded=0,source_images_written=0,
                shared_pattern_owner_preserved=pf is c.material.patterns is c.placement.patterns,
                parameter_columns=columns.tolist(),module_import_sha256=_IMPORT_SHA,
                raw_semantics=('original_support_retained_when_no_measured_centre_exists'if family=='external_main_to_low'else
                    'zero_width_at_original_source_UV_diagnostic_only'if family=='shared_board_patterns'else'original_engineering_bias'),
                saved_scalar_residual_max_error=saved_pattern_error if family=='shared_board_patterns'else None,
                saved_scalar_residual_available=family=='shared_board_patterns'))
        if family=='external_main_to_low':
            tail['evaluate']=lambda value,factor=factor:_evaluate_external(c,factor,value,False)
            tail['raw_evaluate']=tail['evaluate']
        elif family=='shared_board_patterns':
            tail['evaluate']=lambda value,factor=factor:_evaluate_pattern(c,factor,value,False)
            tail['raw_evaluate']=lambda value,factor=factor:_evaluate_pattern(c,factor,value,False,True)
        else:
            tail['evaluate']=lambda value:np.asarray(value)[bias]/sigma
            tail['raw_evaluate']=tail['evaluate']
        tails.append(tail)
    assert _file_sha(__file__)==_IMPORT_SHA
    c.global_tails=tails;c.global_tail_baseline_seconds=time.perf_counter()-started
    return tails


def evaluate_tail_derivatives(c,multiplier=1.):
    tails=prepare_tails(c);answers={};started=time.perf_counter()
    if not np.isfinite(multiplier)or multiplier<=0:raise ValueError('Positive derivative step multiplier required')
    for tail in tails:
        family=tail['family'];columns=tail['columns'];factor=tail['factor']
        unchanged_producer_stencil_values=0;evaluated_producer_stencil_values=0
        if family=='installation_correlated_roll_hardware_bias':
            j=np.eye(3)/np.deg2rad(c._global_tail_roll['component_sigma_degrees']);rawj=j.copy()
        else:
            with _public_derivative_widths(c,multiplier):
                if family=='external_main_to_low':_,public,_=_evaluate_external(c,factor,c.q,True);rawpublic=public
                else:
                    _,public=_evaluate_pattern(c,factor,c.q,True)
                    _,rawpublic=_evaluate_pattern(c,factor,c.q,True,True)
            j=public[:,columns].toarray();rawj=rawpublic[:,columns].toarray()
            # Include all current world coordinates, rather than treating the
            # inherited six course slots as the complete dependency set.
            evaluate=(lambda value,raw=False:_evaluate_external(c,factor,value,False))if family=='external_main_to_low'else(
                lambda value,raw=False:_evaluate_pattern(c,factor,value,False,raw))
            signature=_producer_signature(c,factor,family,c.q[c.world_indices])
            for k,col in enumerate(c.world_indices):
                points,_,stencil=_feasible_world_stencil(c.provider,c.q[c.world_indices],k,1e-6*multiplier)
                actual_values=[];raw_values=[]
                for world in points:
                    value=c.q.copy();value[c.world_indices]=world
                    if _producer_signature(c,factor,family,world)==signature:
                        actual_values.append(tail['actual']);raw_values.append(tail['raw']);unchanged_producer_stencil_values+=1
                    else:
                        rv=evaluate(value);actual_values.append(rv)
                        raw_values.append(rv if family=='external_main_to_low'else evaluate(value,True));evaluated_producer_stencil_values+=1
                if stencil['method']=='central':
                    aj=(actual_values[0]-actual_values[1])/(2*stencil['step']);rj=(raw_values[0]-raw_values[1])/(2*stencil['step'])
                else:
                    aj=stencil['side']*(4*(actual_values[1]-actual_values[0])-(actual_values[2]-actual_values[0]))/(2*stencil['step'])
                    rj=stencil['side']*(4*(raw_values[1]-raw_values[0])-(raw_values[2]-raw_values[0]))/(2*stencil['step'])
                slot=int(np.flatnonzero(columns==col)[0]);j[:,slot]=aj;rawj[:,slot]=rj
        answers[family]=dict(columns=columns,actualJ=j,rawJ=rawj,receipt=dict(family=family,
            step_multiplier=float(multiplier),small_public_tail_J_only=True,full_current_world_columns_retained=True,
            scalar_rows=len(tail['actual']),columns=len(columns),seconds=time.perf_counter()-started,
            unchanged_exact_physical_producer_stencil_values_reused=unchanged_producer_stencil_values,
            evaluated_changed_physical_producer_stencil_values=evaluated_producer_stencil_values,
            module_import_sha256=_IMPORT_SHA))
    assert _file_sha(__file__)==_IMPORT_SHA
    return answers


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-J',action='store_true');args=parser.parse_args()
    c=ctx.load_context();tails=prepare_tails(c)
    report=dict(schema='rinkspace.q8-global-tail-small-factory-baseline.v1',script=a.ref(__file__),
        state=a.ref(c.path),q_sha256=a.CHILD,checks=[t['baseline_receipt']for t in tails],
        source_container_references=c.state.patterns._joint_global_tail_sources,
        original_PNG_SHA_reuse=list(c.state.patterns._joint_global_tail_source_auth.values()),
        no_motion_graph_or_fit=True,no_source_image_decode_or_write=True,seconds=c.global_tail_baseline_seconds)
    if args.check_J:
        derivatives=evaluate_tail_derivatives(c,1.)
        report['small_J_checks']=[dict(result['receipt'],actual_shape=list(result['actualJ'].shape),
            raw_shape=list(result['rawJ'].shape),all_finite=bool(np.all(np.isfinite(result['actualJ']))and np.all(np.isfinite(result['rawJ']))))for result in derivatives.values()]
    path=a.H/'joint_absolute_global_tail_q8_baseline_20261011.json'
    path.write_text(json.dumps(report,default=a.plain,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(path=str(path),sha256=a.sha(path),checks=report['checks']),default=a.plain),flush=True)
