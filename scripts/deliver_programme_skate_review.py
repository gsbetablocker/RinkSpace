"""Deliver a replayed programme candidate without changing the historical player."""
from argparse import ArgumentParser
from collections import defaultdict
from copy import copy, deepcopy
from pathlib import Path
from shutil import copyfileobj
from zipfile import ZipFile, ZIP_DEFLATED
import json

import numpy as np

from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, curve_from_packet
from calibration.program_tracking.body_tracking.body_clock import time_base
from calibration.program_tracking.body_tracking.continuous_physics import ContinuousMassGeometry
from calibration.program_tracking.body_tracking.fullbody_fit import body_joint_assignment
from calibration.program_tracking.body_tracking.programme_com import acceleration_operator, export_com_diagnostics
from calibration.program_tracking.body_tracking.programme_projection import generate, read, render_comparison, SOURCE_FIELDS
from calibration.program_tracking.body_tracking.programme_skate_evidence import programme_observation_path
from calibration.program_tracking.body_tracking.programme_skate_replay import save, write
from calibration.program_tracking.body_tracking.rebuild_prepare import digest, load
from calibration.program_tracking.body_tracking.sequence_skate_state import shoe_curve_from_packet
from calibration.program_tracking.body_tracking.skate_checkpoint import validate_checkpoint_identity
from calibration.program_tracking.body_tracking.skate_delivery import export_candidate
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference

PROGRAMME=Path('events/ankara_2026/programmes/okada_mei_sp')
NEW='programme_skate_evidence_1109'
PARENT='programme_s09_evidence_1109'
EARLIER='programme_lsp_continuity_1109'
OLD_CONFIG='viewer3d_'+EARLIER+'.json'
RATE=12800


def binding(path):
    return dict(path=str(path),sha256=digest(path))


def open_project_file(path, mode, **kwargs):
    units=lambda s:len(str(s).encode('utf-16-le'))//2
    if units(path.resolve())>=260 or units(path.name)>255:
        raise ValueError('OneDrive path exceeds its limit: '+str(path))
    return path.open(mode,**kwargs)


def copy_checked(source,target):
    if not target.exists():
        with source.open('rb') as i, open_project_file(target,'xb') as o:copyfileobj(i,o)
    if digest(source)!=digest(target):raise ValueError('Existing parameter has a different binding')


def register_json(path,record,*,project_file=False):
    if path.exists():
        if read(path)!=record:raise ValueError('Existing review record has a different binding: '+str(path))
    elif project_file:
        with open_project_file(path,'x',encoding='utf8') as handle:json.dump(record,handle,ensure_ascii=False,indent=2)
    else:write(path,record)


def affected(knots,count,changed,times):
    span=np.clip(np.searchsorted(knots,times,side='right')-4,0,count-4)
    return np.isin(span[...,None]+np.arange(4),changed).any(axis=-1)


def support(saved,parent):
    # Differences below floating point retraction noise cannot expand the
    # exact spline support into otherwise untouched shots.
    delta=lambda k,axes:np.any(abs(saved[k]-parent[k])>2.e-12,axis=axes)
    b=np.flatnonzero(delta('absolute_position_controls',1)|delta('absolute_rotation_controls',(1,2,3)))
    edges=np.flatnonzero(delta('absolute_rotation_edge_reference_quaternion',(1,2)))
    b=np.union1d(b,np.r_[edges,edges+1])
    s=np.flatnonzero(delta('relative_skate_rotation_controls',(1,2,3)))
    return lambda t:(affected(saved['absolute_knots'],len(saved['absolute_position_controls']),b,t) |
        affected(saved['relative_skate_knots'],len(saved['relative_skate_rotation_controls']),s,t))


def export(project,prepared,source,parameter,parent_param,parent_result,output,*,protect_s01=True):
    parent=load(parent_param);saved=load(parameter)
    ref,camera,info,doc,_=load_reference(project,prepared,source)
    validate_checkpoint_identity(saved,ref.fixed_checkpoint_identity)
    observation_path=programme_observation_path(prepared,doc)
    for key,file in [('prepared_input_sha256',prepared/'input.npz'),('prepared_observations_sha256',observation_path),
                     ('source_evidence_sha256',source)]:
        if str(saved[key])!=digest(file):raise ValueError('Candidate export binding changed: '+key)
    for k in ('absolute_knots','relative_skate_knots','sequence_contact_modes','sequence_contact_status'):
        np.testing.assert_array_equal(saved[k],parent[k],err_msg=k)
    def camera_binding(packet):
        value=json.loads(str(packet['camera_binding_json']))
        for entry in (value,*value['shots'].values()):entry.pop('requested_native_library',None)
        return value
    if camera_binding(saved)!=camera_binding(parent):
        raise ValueError('Camera binding differs beyond the requested runtime alias')
    np.testing.assert_array_equal(saved['x'][:17],parent['x'][:17])
    for k in ('absolute_rotation_controls','absolute_rotation_edge_reference_quaternion'):
        np.testing.assert_allclose(saved[k][:,0],parent[k][:,0],rtol=0,atol=2.e-12,err_msg=k)

    current=absolute_reference(ref,curve_from_packet(saved),root=prepared)
    current.relative_skate_curve=shoe_curve_from_packet(saved)
    current.binding=copy(ref.binding);current.binding.reference=current
    current.binding.x=saved['x'][:17].copy();current.binding._parameter_key=current.binding._packet=None
    pts=ref.p['pts'];times=ref.times;tb=time_base(ref.p);obs=load(observation_path)
    np.testing.assert_array_equal(pts,obs['pts'])
    np.testing.assert_array_equal(pts,np.arange(283136,2415104+256,256))
    assert tuple(tb)==(1,RATE)
    mask=support(saved,parent)
    native=mask(times)
    nodes=times[:,None]+camera.readout_times(np.arange(len(times)))[:,None]*np.linspace(-.5,.5,5)
    rolling=mask(nodes).any(axis=1)
    if not native.any() or not rolling.any():raise ValueError('New variant must have a distinct saved motion')
    # Reviewed root rotation follows the same lifted branch at every node,
    # including complete LSp turns between native exposures.
    old_curve=curve_from_packet(parent)
    from calibration.program_tracking.body_tracking.absolute_motion_state import evaluate_curve
    for t in np.array_split(nodes.ravel(),9):
        np.testing.assert_allclose(evaluate_curve(old_curve,t)['rotations'][:,0],
                                   evaluate_curve(current.absolute_curve,t)['rotations'][:,0],rtol=0,atol=3.e-10)
    print(json.dumps(dict(event='complete_native_export',exposures=len(pts),changed_support=int(native.sum()))),flush=True)
    output.mkdir(parents=True);fitted=output/'fitted';fitted.mkdir()
    packet=export_candidate(prepared,output,current,camera,info,doc.get('supports',[]),write=False)
    samples=current.at(times);zero=np.zeros((len(times),69))
    body=fk_state(current,samples,zero);mass=ContinuousMassGeometry(current)
    centers,axes=mass.evaluate(samples,zero)
    packet.update(export_com_diagnostics(mass,times,centers,axes))
    q=load(parent_result.parents[1]/'com_physics_quadrature.npz')
    qt=q['evaluation_time_s'];changed=mask(qt)
    centers,_=mass.evaluate(current.at(qt[changed]),np.zeros((changed.sum(),69)))
    q['com_world_m'][changed]=mass.com(centers)
    q['acceleration_world_mps2']=acceleration_operator(qt)@q['com_world_m']
    q['parameter_sha256']=np.array(digest(parameter));q['source_evidence_sha256']=np.array(digest(source))
    qfile=output/'com_physics_quadrature.npz';save(qfile,**q)
    packet.update(native_body_local_R_mid=body['local_R'],native_body_FK_world_R_mid=body['Q'][:,None]@body['frames'],
        P_velocity_world_mps=current.root_derivative(times,1)*current.binding.ratio,source_shot_ids=ref.p['shot_ids'],
        trajectory_evidence_status=np.where(obs['use'].sum(1)>=4,'source_constrained_estimate','temporal_inference'),
        position_uncertainty_status=np.full(len(pts),'conditional_scale_and_depth; no independent accuracy acceptance'),
        ice_plane_known=np.array(ref.ice_plane_known),sequence_contact_modes=saved['sequence_contact_modes'],
        sequence_contact_status=saved['sequence_contact_status'],shared_betas=ref.p['shared_betas'],
        shared_body_scale=ref.p['body_scale'],source_body_model_joint_ids=body_joint_assignment(obs),
        body_joint_assignment_provenance_json=obs['body_joint_assignment_provenance_json'],
        source_observation_use=obs['use'].copy(),
        source_action_ledger_sha256=np.array(digest(source)),source_skate_revision_parameter_sha256=np.array(digest(parameter)),
        source_skate_revision_support_mask=native,com_physics_quadrature_path=np.array(str(qfile)),
        com_physics_quadrature_sha256=np.array(digest(qfile)))
    np.testing.assert_array_equal(packet['pts'],pts)
    np.testing.assert_array_equal(packet['source_pts'],pts)
    np.testing.assert_allclose(packet['native_body_local_R_mid'],packet['local_R_mid'],atol=2.e-12,rtol=0)
    assert json.loads(str(packet['camera_binding_json']))['state_sha256']==info['state_sha256']
    mids=('P_world_mid','G_world_mid','Q_world_mid','native_body_local_R_mid','native_body_FK_world_R_mid','COCO_world_mid',
          'skate_R_world_mid','skate_lower_world_mid','skate_upper_world_mid','skate_reference_world_mid','com_world_mid',
          'P_velocity_world_mps','sequence_contact_modes','sequence_contact_status','hand_local_R_mid')
    pixels=('COCO_uv','equipment_uv','world_at_row','source_row_time')
    protected=(pts>=int(77.60*RATE))&(pts<=int(79.*RATE))
    if protect_s01:
        protected|=ref.p['shot_ids']=='s01'
    if np.any(rolling & protected):raise ValueError('An explicitly protected exposure received changed spline support')
    diff={}
    with np.load(parent_result,allow_pickle=False) as archived:
        for fields,keep,tol in ((mids,~native,2.e-9),(pixels,~rolling,3.e-7)):
            for name in fields:
                a,b=packet[name][keep],archived[name][keep]
                if a.dtype.kind in 'f':
                    np.testing.assert_allclose(a,b,rtol=0,atol=tol,err_msg=name)
                    diff[name]=float(np.max(abs(a-b)))
                else:
                    np.testing.assert_array_equal(a,b,err_msg=name)
                    diff[name]=0.
    result=fitted/'result.npz';save(result,**packet)
    intervals=np.split(np.flatnonzero(native),np.flatnonzero(np.diff(np.flatnonzero(native))>1)+1)
    receipt=dict(result=binding(result),parameter=binding(parameter),source=binding(source),
        exact_source_exposures=len(pts),source_time_base=list(tb),camera_sha256=info['state_sha256'],
        source_first_last_pts=pts[[0,-1]].tolist(),affected_intervals_pts=[pts[i[[0,-1]]].tolist() for i in intervals],
        affected_exposures=int(native.sum()),affected_rolling_exposures=int(rolling.sum()),
        protected_S01_and_2A=int(protected.sum()) if protect_s01 else 0,
        protected_exposures=int(protected.sum()),s01_footwear_refit_enabled=not protect_s01,
        off_support_maximum_differences=diff,
        fixed_root_at_all_five_exposure_nodes=bool(np.array_equal(saved['absolute_position_controls'],parent['absolute_position_controls'])),
        fixed_root_orientation_at_all_five_exposure_nodes=True,user_accepted=False,accuracy_accepted=False)
    write(output/'export_receipt.json',receipt)
    return receipt


def metrics_rows(comparison,label):
    for role,groups in comparison[label].items():
        for family,rows in groups.items():
            for row in rows:yield role,family,row


def replay_fit(evidence,receipt,comparison):
    bound={r['raw_group_id']:r for name in ('ordered_contours','manual_landmarks','native_support_bands') for r in evidence[name]}
    rows=[r for role,family,r in metrics_rows(comparison,'candidate') if role=='fit'
          and r.get('status')!='no_visible_same_semantic_model_path']
    audit=[dict(pts=r['pts'],foot=r['foot'],id=r['id'],role='fit',structure_feature=bound[r['id']]['semantic']) for r in rows]
    support_ids={r['raw_group_id'] for r in evidence['native_support_bands']}
    return dict(description_zh='按全片重放登记。check 只用于核对；支撑带推得的平面位置不重复加权。',
                parameter_sha256=receipt['parameter_sha256'],source_evidence_sha256=receipt['source_evidence_sha256'],
                feature_rows=[r for r in audit if r['id'] not in support_ids],
                support_rows=[r for r in audit if r['id'] in support_ids],unassigned_source_sets=[])


def source_shapes(evidence):
    grouped={}
    for r in evidence['ordered_contours']:
        key=r['pts'],r['foot'],r['track_id']
        q=grouped.setdefault(key,dict(pts=r['pts'],foot=r['foot'],source_track_id=r['track_id'],features={}))
        feature_key=r['semantic']
        if feature_key in q['features']:
            feature_key+=':'+r['raw_group_id']
        q['features'][feature_key]=dict(points_uv=r['polyline_uv'],source_id=r['raw_group_id'],semantic=r['semantic'],
            role=r['role'],status='observed',visible_range=r['visible_range'],
            completeness=r['visibility'],pixel_uncertainty_radius_px=r['radius_px'])
    for r in evidence['manual_landmarks']:
        key=r['pts'],r['foot'],r['track_id']
        q=grouped.setdefault(key,dict(pts=r['pts'],foot=r['foot'],source_track_id=r['track_id'],features={}))
        phi=np.linspace(0.,2*np.pi,13)[:-1]
        ring=np.asarray(r['uv'])+r['radius_px']*np.stack((np.cos(phi),np.sin(phi)),axis=1)
        q['features']['anatomical_landmark_region_'+str(r['joint'])]=dict(points_uv=ring.tolist(),closed=True,
                source_id=r['raw_group_id'],role=r['role'],status='observed',pixel_uncertainty_radius_px=r['radius_px'])
    for r in evidence['native_support_bands']:
        if r['role']=='unknown':continue
        key=r['pts'],r['foot'],r['track_id']
        q=grouped.setdefault(key,dict(pts=r['pts'],foot=r['foot'],source_track_id=r['track_id'],features={}))
        q['features']['support_neighborhood']=dict(points_uv=r['polyline_uv'],source_id=r['raw_group_id'],
                role=r['role'],status='observed',pixel_uncertainty_radius_px=r['radius_px'])
    return list(grouped.values())


def review_summary(evidence,comparison):
    """Keep the raw fit/check roles, finite pixel bands, and side/part totals."""
    table={}
    for label in comparison:
        bins=defaultdict(list)
        for role,family,r in metrics_rows(comparison,label):
            key=r['event_id'],int(r['foot']),role,r.get('semantic',family)
            bins[key].append(r)
        output=[]
        for (event,foot,role,semantic),rows in sorted(bins.items()):
            values=[];worst=None;visibility=defaultdict(int)
            for row in rows:
                visibility[row.get('visibility','finite_source_region')]+=1
                x=row.get('distance_px',row.get('outside_candidate_band_px'))
                if x is None:continue
                x=float(np.percentile(np.atleast_1d(x),95));values.append(x)
                if worst is None or x>worst['pixels']:worst=dict(pts=row['pts'],pixels=x,id=row['id'])
            output.append(dict(event=event,foot=foot,role=role,semantic=semantic,rows=len(rows),
                scored=len(values),missing_model_path=len(rows)-len(values),visibility=dict(visibility),worst=worst,
                median_px=float(np.median(values)) if values else None,
                p95_px=float(np.percentile(values,95)) if values else None))
        table[label]=output
    totals={}
    for key in ('manual_landmarks','native_support_bands','ordered_contours','plane_xy_diagnostics'):
        counts=defaultdict(int)
        for row in evidence[key]:counts[row['role']]+=1
        totals[key]=dict(counts)
    return dict(candidates=table,source_row_totals=totals,source_video_sha256=evidence['source_video_sha256'],
        evidence_coverage=dict(reviewed_native_exposures=len(evidence['exposure_review']),
             all_other_exposures='not newly hand-annotated; inherited observations and continuous model remain'),
        source_pixel_measure='p95 within each finite observed contour/region, then aggregate over exposures',
        held_out_role_is_development_check_not_independent_accuracy=True)


def temporal_review(result,earlier,previous,evidence,depth_reference=None):
    """Check whole-clock identity and audit conflicting visible left/right marks."""
    with np.load(result,allow_pickle=False) as f, np.load(earlier,allow_pickle=False) as old, np.load(previous,allow_pickle=False) as parent:
        pts=f['pts'];tb=tuple(map(int,f['time_base']))
        for other in (old,parent):
            np.testing.assert_array_equal(pts,other['pts'])
            np.testing.assert_array_equal(f['time_base'],other['time_base'])
            np.testing.assert_array_equal(f['sequence_contact_modes'],other['sequence_contact_modes'])
            np.testing.assert_array_equal(f['sequence_contact_status'],other['sequence_contact_status'])
        lookup={int(p):i for i,p in enumerate(pts)}
        marks=defaultdict(dict)
        for r in evidence['manual_landmarks']:
            if r['joint'] in (15,16):marks[r['pts']][r['foot']]=r
        events=[]
        for stamp,row in marks.items():
            if row.keys()!={0,1}:continue
            target=np.asarray([row[j]['uv'] for j in (0,1)])
            radius=np.asarray([row[j]['radius_px'] for j in (0,1)])
            current=f['COCO_uv'][lookup[stamp],15:17]
            direct=np.linalg.norm(current-target,axis=1)
            crossed=np.linalg.norm(current[::-1]-target,axis=1)
            if np.all(direct-crossed>radius):
                events.append(dict(pts=stamp,event_id=row[0]['event_id'],direct_px=direct.tolist(),
                                   swapped_px=crossed.tolist(),roles=[row[0]['role'],row[1]['role']]))
        rotations=f['skate_relative_R_mid']
        def angular_rates(r):
            a=r[:-1].swapaxes(-1,-2)@r[1:]
            return np.arccos(np.clip((np.trace(a,axis1=-2,axis2=-1)-1.)/2.,-1.,1.))/(np.diff(pts)*tb[0]/tb[1])[:,None]
        speeds={k:angular_rates(a['skate_relative_R_mid']) for k,a in [('lsp',old),('s09',parent),('candidate',f)]}
        differences={}
        for e in evidence['events']:
            hit=(pts[1:]>=e['start_pts'])&(pts[1:]<=e['end_pts'])
            p=(pts>=e['start_pts'])&(pts<=e['end_pts'])
            dP=np.linalg.norm(f['P_world_mid'][p]-parent['P_world_mid'][p],axis=1)
            dG=np.linalg.norm(f['G_world_mid'][p]-parent['G_world_mid'][p],axis=1)
            differences[e['id']]=dict(relative_skate_rotation_rate_p95_rad_s={
                k:np.percentile(v[hit],95,axis=0).tolist() for k,v in speeds.items()},
                root_P_max_change_vs_s09_m=float(dP.max()),plane_G_max_change_vs_s09_m=float(dG.max()))
        depth=None
        if depth_reference is not None:
            with np.load(depth_reference,allow_pickle=False) as ref:
                axis=np.asarray(ref['camera_planar_axes'])[:,1]
            np.testing.assert_allclose(np.linalg.norm(axis),1.,atol=1.e-10,rtol=0.)
            window=(pts>=round(159.82*RATE))&(pts<=round(163.90*RATE))
            depth=dict(reference=binding(depth_reference),source_seconds=[159.82,163.90],
                pelvis_range_m={k:float(np.ptp(a['P_world_mid'][window,:2]@axis)) for k,a in [('lsp',old),('s09',parent),('candidate',f)]},
                COM_range_m={k:float(np.ptp(a['com_world_mid'][window,:2]@axis)) for k,a in [('lsp',old),('s09',parent),('candidate',f)]},
                meaning='Frozen camera, conditional metric model; not independently observed depth')
    return dict(exact_source_frames=len(pts),source_time_base=list(tb),clock_mismatches=0,contact_assignment_changes=0,
                two_ankles_observed_exposures=sum(len(r)==2 for r in marks.values()),possible_identity_conflicts=events,
                event_model_motion=differences,model_view_depth_diagnostic=depth,geometry_rates_are_diagnostics=True)


def failures(base,evidence,comparison,scan,temporal,acceptance):
    nav=deepcopy(base)
    old=nav['events'];new=[]
    reasons={'manual_native_leg':'膝踝','ordered_native_skate':'鞋刀','reviewed_native_support':'支撑带'}
    phase={q['id']:q for q in evidence['events']}
    phase.setdefault('whole_programme',dict(start_pts=min(e['start_pts'] for e in phase.values()),
                                         end_pts=max(e['end_pts'] for e in phase.values())))
    names={k:e.get('name_zh',k) for k,e in phase.items()}
    names['whole_programme']='全节目'
    names.update(steps_extension='S09 步法伸腿',lsp='S09 LSp')
    shoe_names=dict(boot_shell='可见鞋壳',heel_shell='鞋跟',toe_shell='鞋头',outsole='鞋底',rear_mount='后刀架',front_mount='前刀架',
                    runner_lower_edge='钢片下缘',runner_upper_edge='钢片上缘')
    events=defaultdict(list)
    for role,family,r in metrics_rows(comparison,'candidate'):
        value=r.get('distance_px',r.get('outside_candidate_band_px'))
        radius=r.get('radius_px',r.get('source_radius_px'))
        if value is None or radius is None:
            if r.get('status')=='no_visible_same_semantic_model_path':
                events[r['event_id'],family,role].append((float('inf'),r))
            continue
        value=float(np.percentile(np.atleast_1d(value),95))
        if value>radius:events[r['event_id'],family,role].append((value/radius,r))
    for (event,family,role),rows in events.items():
        value,r=max(rows,key=lambda v:v[0])
        e=phase[event]
        label=names.get(event,event)+' · '+reasons[family]+('（检查）' if role=='check' else '（拟合）')
        side=('左','右')[r['foot']]
        semantic=shoe_names.get(r.get('semantic'),reasons[family])
        reason=(f'{side}脚{semantic}：该可见描线没有可比较的同类模型边。' if np.isinf(value)
                else f'{side}脚{semantic}：剩余像素偏差约为标注半径的 {value:.1f} 倍。')
        new.append(dict(label=label,start_pts=e['start_pts'],end_pts=e['end_pts'],worst_pts=r['pts'],
                        reason_zh=reason,kind='source_residual',source_role=role,source_id=r['id']))
    for event,result in scan['results']['candidate'].items():
        if event not in phase:continue
        pairs=[(k,v) for k,v in result['pairs'].items() if v['witness_seconds'] is not None]
        if not pairs:continue
        pair,v=max(pairs,key=lambda t:(t[1]['samples'],t[1]['triangle_pairs']))
        e=phase[event];stamp=int(round(v['witness_seconds']*50)*256)
        new.append(dict(label=names.get(event,event)+' · 表面结构',start_pts=e['start_pts'],end_pts=e['end_pts'],
            worst_pts=int(np.clip(stamp,e['start_pts'],e['end_pts'])),kind='model_surface',
            reason_zh=f'模型表面配对 {pair} 有 {v["samples"]} 个采样相交；这项诊断不表示真实穿冰。'))
    anatomy=acceptance['final_guard']['pointwise_anatomical_comparison']
    for event,value in anatomy.items():
        if 'new_anatomical_excursion:'+event not in acceptance['final_guard']['failures']:continue
        e=phase[event]
        joints={'left_knee':'左膝','right_knee':'右膝','left_hip':'左髋','right_hip':'右髋'}
        new.append(dict(label=names.get(event,event)+' · 关节越界',start_pts=e['start_pts'],end_pts=e['end_pts'],
            worst_pts=value.get('new_excursion_worst_pts',value['worst_pts']),kind='joint_domain',
            reason_zh=joints.get(value.get('new_excursion_joint',value['joint']),value.get('new_excursion_joint',value['joint']))+'的同曝光关节域违例增大，已计入全片目标。'))
    identities=defaultdict(list)
    for r in temporal['possible_identity_conflicts']:identities[r['event_id']].append(r)
    for event,rows in identities.items():
        r=max(rows,key=lambda q:sum(q['direct_px'])-sum(q['swapped_px']))
        e=phase[event]
        new.append(dict(label=names.get(event,event)+' · 核对左右脚身份',start_pts=e['start_pts'],end_pts=e['end_pts'],
            worst_pts=r['pts'],kind='source_identity_review',
            reason_zh='原片两踝都有标注，交换左右预测后同时更接近。仅列为可能失配，未更改来源身份。'))
    nav['events']=new+old
    nav['temporal_clock_mismatches']=temporal['clock_mismatches']
    nav['description_zh']='历史摄影切镜与完整动作保留。新问题按实际源像素、可见范围及模型表面检查生成。'
    return nav


def projection(project,prepared,parameter,result,output,variant,fit,canonical,prior_review):
    registration=generate(project,prepared,parameter,result,output,variant=variant,source_evidence=canonical,
        review_evidence=prior_review,batch_frames=384,replay_fit_binding=fit)
    return Path(registration['output_asset']['path'])


def write_pack(target,files,parameter,result,receipt,evidence,earlier,previous):
    from io import BytesIO
    keys=('pts','time_base','P_world_mid','G_world_mid','Q_world_mid','skate_R_world_mid','skate_origin_world_mid',
          'skate_reference_world_mid','skate_witness_world_mid','native22_world_mid','native_body_local_R_mid',
          'COCO_uv','equipment_uv','source_shot_ids','sequence_contact_modes','sequence_contact_status','com_world_mid',
          'source_action_phase','source_action_phase_names','source_identity_warning')
    with ZipFile(target,'x',compression=ZIP_DEFLATED,compresslevel=3) as z:
        for p in files:z.write(p,'review/'+p.name)
        states={}
        for variant,path in [(NEW,result),(PARENT,previous),(EARLIER,earlier)]:
            with np.load(path,allow_pickle=False) as v:
                values={k:v[k] for k in keys if k in v}
            if any(k not in values for k in keys[:10]):raise ValueError('Missing required review state field')
            buffer=BytesIO();np.savez_compressed(buffer,**values)
            name='states/'+variant+'.npz';z.writestr(name,buffer.getvalue())
            states[variant]=dict(result_sha256=digest(path),member=name,missing_optional_fields=sorted(set(keys)-values.keys()))
        z.writestr('source/skate_evidence.json',json.dumps(evidence,ensure_ascii=False,indent=2))
        checkpoint=load(parameter)
        small={k:checkpoint[k] for k in ('camera_binding_json','programme_structure_json','anatomical_policy_json') if k in checkpoint}
        small['fixed_shared_parameters']=checkpoint['x'][:17]
        with np.load(result,allow_pickle=False) as state:
            for key in ('footwear_binding_json','skate_dimensions_json','template_version','binding_version','shared_betas','shared_body_scale'):
                if key in state:small[key]=state[key]
        buffer=BytesIO();np.savez_compressed(buffer,**small);z.writestr('states/fixed_binding.npz',buffer.getvalue())
        z.writestr('manifest.json',json.dumps(dict(schema='rinkspace.programme-skate-audit-pack.v1',
            states=states,parameter_sha256=digest(parameter),camera_sha256=receipt['same_camera_sha256'],
            same_source_frames=receipt['coverage']['native_exposures'],
            source_evidence_sha256=receipt['source_evidence_sha256'],accuracy_accepted=False),indent=2))
    with ZipFile(target) as z:
        if z.testzip() is not None:raise ValueError('Audit archive failed CRC verification')


def main():
    p=ArgumentParser(description=__doc__)
    for key in ('project','prepared','replay','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--parent-only',action='store_true')
    p.add_argument('--export-only',action='store_true')
    p.add_argument('--projection-only',action='store_true')
    p.add_argument('--stage-only',action='store_true',help='Verify full local projections and navigation without installing a player')
    p.add_argument('--parent-projection',type=Path,required=True)
    p.add_argument('--depth-reference',type=Path,help='Prior frozen-camera axis for a diagnostic COM comparison')
    p.add_argument('--other-shots',type=Path,help='Optional source scan receipt for the other eight shots')
    args=p.parse_args();project=args.project.resolve();programme=project/PROGRAMME
    old_config=programme/OLD_CONFIG;config=read(old_config);old_sha=digest(old_config)
    old_asset=Path(config['blade_review_asset']);base=read(old_asset)
    prior_review=Path(base['source_evidence_binding']['review_evidence']['path'])
    command=read(args.replay/'command.json')['inputs']
    parent_parameter=Path(command['latest']);parent_source=Path(command['source'])
    archive_record=programme/(PARENT+'.json');archived=read(archive_record)
    parent_result=Path(archived['result']['path'])
    if args.parent_only:
        historical=Path(archived['review']['refit_receipt']['path']);local=read(historical)
        if local['candidate_sha256']!=digest(parent_parameter) or digest(historical)!=archived['review']['refit_receipt']['sha256']:
            raise ValueError('Wrong historical proposal receipt')
        fit=dict(description_zh='上轮局部拟合的结果。新标记可用于同帧比较，未进入此版本的目标。',
                 parameter_sha256=digest(parent_parameter),source_evidence_sha256=digest(parent_source),
                 fit_receipt=binding(historical),feature_rows=[],support_rows=[],unassigned_source_sets=[])
        projection(project,args.prepared,parent_parameter,parent_result,args.parent_projection,PARENT,fit,parent_source,prior_review)
        return

    rr=read(args.replay/'acceptance.json');source=Path(rr['source_evidence']);parameter=args.replay/'candidate.npz'
    if (digest(parameter)!=rr['parameter_sha256'] or digest(source)!=rr['source_evidence_sha256']
            or digest(parent_parameter)!=rr['parent_sha256'] or rr['coverage']['native_exposures']!=8329
            or rr['coverage']['surface_checked_samples']!=16657):
        raise ValueError('Candidate lacks a complete, matching production replay')
    evidence=read(source)['programme_skate_evidence']
    comparison=read(args.replay/'source_comparison.json');scan=read(args.replay/'surface_comparison.json')
    export_receipt=args.output/'final/export_receipt.json'
    if export_receipt.is_file():
        exported=read(export_receipt)
        for k,file in [('parameter',parameter),('source',source),('result',Path(exported['result']['path']))]:
            if digest(file)!=exported[k]['sha256']:raise ValueError('Existing export has a different binding: '+k)
        if exported['exact_source_exposures']!=8329:raise ValueError('Existing export uses a different programme clock')
    else:exported=export(project,args.prepared,source,parameter,parent_parameter,parent_result,args.output/'final')
    result=Path(exported['result']['path'])
    if args.export_only:return
    new_projection=args.output/'projection/programme_blade_review.json'
    if new_projection.is_file():
        stored=read(new_projection)['prediction_binding']
        if (stored['parameter_sha256']!=digest(parameter) or stored['result_sha256']!=digest(result)
                or stored['camera_state_sha256']!=rr['same_camera_sha256']):
            raise ValueError('Existing full projection has a different binding')
    else:
        new_projection=projection(project,args.prepared,parameter,result,args.output/'projection',NEW,
                                  replay_fit(evidence,rr,comparison),source,prior_review)
    if args.projection_only:return
    previous_projection=args.parent_projection/'programme_blade_review.json'
    paths=[old_asset,previous_projection,new_projection]
    report=deepcopy(base)
    report.pop('projection_npz',None)
    report['projection_layers']=[]
    for file in paths:
        doc=read(file)
        report['projection_layers'].append(dict(path=str(file.parent/doc['projection_npz']),variants=list(doc['variants'])))
        if file==old_asset:continue
        for k in SOURCE_FIELDS:
            if doc.get(k,[])!=base.get(k,[]):raise ValueError('Prior source overlay differs: '+k)
        report['variants'].update(doc['variants'])
        report['source_structure_fit_variants'].update(doc['source_structure_fit_variants'])
        report['prediction_bindings'].update({k:doc['prediction_binding'] for k in doc['variants']})
    report['source_structures'].extend(source_shapes(evidence))
    report['source_overlay_revision_variant']=NEW
    report['source_evidence_binding']['programme_skate_ledger']=binding(source)
    report['reviewed_model_contours']={v:[{k:r[k] for k in ('id','pts','foot','semantic','role','model_uv')}
        for _,f,r in metrics_rows(comparison,label) if f=='ordered_native_skate' and 'model_uv' in r]
        for v,label in [(NEW,'candidate'),(PARENT,'s09_parent'),(EARLIER,'lsp_parent')]}
    asset=args.output/'review.json';register_json(asset,report)
    earlier_result=Path(config['variants'][-1]['result'])
    temporal=temporal_review(result,earlier_result,parent_result,evidence,args.depth_reference)
    temporal_path=args.output/'time_identity.json';register_json(temporal_path,temporal)
    summary_path=args.output/'source_by_part.json';register_json(summary_path,review_summary(evidence,comparison))
    nav=failures(read(config['source_navigation']),evidence,comparison,scan,temporal,rr)
    nav_path=args.output/'navigation.json';register_json(nav_path,nav)

    from rinkspace.storage import AssetStore
    from rinkspace.viewer3d.blade_review import BladeReview,overlay_primitives
    from rinkspace.viewer3d.source import SourceVideo
    store=AssetStore(project);review=BladeReview(asset,assets=store,time_base=(1,RATE))
    exact=np.asarray(list(review.index))
    assert len(exact)==8329 and all(len(review.models[v]['COCO_uv'])==len(exact) for v in (NEW,PARENT,EARLIER))
    lookup=set(map(int,exact))
    assert all(all(event[k] in lookup for k in ('start_pts','end_pts','worst_pts') if k in event) for event in nav['events'])
    stamp=round(157.24*RATE)
    source_video=Path(evidence['source_video'])
    if digest(source_video)!=evidence['source_video_sha256']:raise ValueError('Source video hash differs')
    reader=SourceVideo(source_video,(1,RATE))
    try:
        for t in (int(exact[0]),round(150.98*RATE),stamp,round(161.48*RATE),int(exact[-1])):
            rgb=reader.read(t)
            for v in (NEW,PARENT,EARLIER):
                state=review.at(v,t)
                assert state['pts']==t and rgb.shape==(1080,1920,3) and state['model'] is not None
                assert overlay_primitives(state,include_model_contours=True,include_model_body=True,include_trajectory=True)
    finally:reader.close()
    del review
    if args.stage_only:
        print(json.dumps(dict(event='same_frame_staging_verified',frames=len(exact),navigation=len(nav['events']))),flush=True)
        return

    # Validate all local artifacts before installing a distinct parameter, candidate and configuration.
    record_path=programme/(NEW+'.json');final_param=programme/(NEW+'.npz');new_config=programme/('viewer3d_'+NEW+'.json')
    if digest(old_config)!=old_sha:raise ValueError('Historical viewer configuration changed during export')
    record=dict(schema='rinkspace.programme-skate-evidence-review.v1',candidate_id=NEW,parameter=dict(path=str(final_param),sha256=digest(parameter)),
        result=binding(result),source=binding(source),camera_sha256=rr['same_camera_sha256'],whole_programme_exposures=8329,
        acceptance=rr['acceptance'],user_accepted=False,accuracy_accepted=False,review=binding(asset),navigation=binding(nav_path),
        replay=binding(args.replay/'acceptance.json'),source_comparison=binding(args.replay/'source_comparison.json'),
        surface=binding(args.replay/'surface_comparison.json'),source_by_part=binding(summary_path),time_identity=binding(temporal_path),
        other_shot_review=None if args.other_shots is None else binding(args.other_shots),export=binding(args.output/'final/export_receipt.json'))
    copy_checked(parameter,final_param)
    register_json(record_path,record,project_file=True)
    template=config['variants'][-1]
    for name,label,rec,recfile in [(PARENT,'S09 局部修复（旧版）',archived,archive_record),
                                   (NEW,'S09 鞋刀约束与全片重放（待复核）',record,record_path)]:
        v=deepcopy(template)
        v.update(id=name,candidate_id=name,label=label,parameter=rec['parameter']['path'],parameter_sha256=rec['parameter']['sha256'],
            result=rec['result']['path'],result_sha256=rec['result']['sha256'],
            diagnostic_record=str(recfile),diagnostic_record_sha256=digest(recfile),
            source_projection_origin='own full native five-node replay',body_reference_label=EARLIER if name==PARENT else PARENT,
            body_refinement_intervals_seconds=rec.get('scope_seconds',[[a/RATE,b/RATE] for a,b in exported['affected_intervals_pts']]))
        config['variants'].append(v)
    config.update(session_id='programme_skate_evidence_1109',latest_review_variant=NEW,
        comparison_variants=[EARLIER,PARENT,NEW],blade_review_asset=str(asset),source_navigation=str(nav_path),
        initial_source_pts=stamp,trajectory_reference='pelvis',source_review_model_trajectory_default=True)
    register_json(new_config,config,project_file=True)
    from rinkspace.viewer3d.data import ResultCatalog
    catalog=ResultCatalog(project=project,config=new_config)
    assert len(catalog.navigation_points('cuts'))==8 and len(catalog.navigation_points('events'))==len(nav['events'])
    assert catalog.config['default_variant']==read(old_config)['default_variant']
    files=[args.replay/f for f in ('acceptance.json','surface_comparison.json','source_comparison.json',
        'production_registry.json','directions.json','native_source_jacobian_check.json','diagnostic_rigid_task_rank.json',
        'ordered_associations.json','objective_weights.json','full_trials.json','command.json')]
    pack=args.output/'skate_audit.zip'
    files += [nav_path,summary_path,temporal_path,args.output/'final/export_receipt.json']
    if args.other_shots is not None:files.append(args.other_shots)
    panel=args.output/'same_frame/same_frame_comparison.jpg'
    if not panel.is_file():
        panel=render_comparison(project,asset,source_video,[EARLIER,PARENT,NEW],
            [round(t*RATE) for t in (150.98,157.24,161.48,163.12,164.30)],args.output/'same_frame')
    files += [panel,args.output/'same_frame/review_images.json']
    if not pack.exists():write_pack(pack,files,parameter,result,rr,evidence,earlier_result,parent_result)
    else:
        with ZipFile(pack) as z:
            if z.testzip() is not None or json.loads(z.read('manifest.json'))['parameter_sha256']!=digest(parameter):
                raise ValueError('Existing offline pack does not match the saved candidate')
    write(args.output/'delivery.json',dict(config=binding(new_config),record=binding(record_path),review=binding(asset),
        navigation=binding(nav_path),pack=binding(pack),comparison_panel=binding(panel),
        old_default_unchanged=True,source_frame_decode_verified=True))
    print(json.dumps(dict(config=str(new_config),candidate=str(result),audit=str(pack)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
