"""Frozen-q8 local RF4 response and all incident real motion consequences.

No optimizer, accepted state, source image, complete source graph or full J.
Linear inverses are diagnostic responses, never promoted camera parameters.
"""
from __future__ import annotations

from collections import OrderedDict, Counter
from copy import copy
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import time

import joint_absolute_source_context_20261011 as ctx
import joint_absolute_source_derivatives_20261011 as derivative
from calibration.solver_runtime.native_factor_dependency_fd import native_factor_dependency_jacobian
from calibration.solver_runtime.source_loss import source_irls_weights, weighted_source_cost

a, np = ctx.a, ctx.np
SIX = [229953536,229961216,230306816,230314496,230352896,230360576]
OUT = a.H/'LOW6_private_RF4_frozen_q8_absolute_and_incident_motion_20261011.json'


def emit(**record):
    print(json.dumps(record,ensure_ascii=False,default=a.plain),flush=True)


def group_values(c, value):
    values=[]
    for group in c.groups:
        if group['source_kind']=='original':
            r=ctx.original_value(c,group['packet_row_index'],value)
        else:
            assert group['source_kind']=='native'
            r=ctx.native_value(c,group['factor'],value,[group['row']])
        values.append(r)
    return values


def private_columns(c,at):
    block=c.state.bypts[at]
    start=block.start+4*block.bundle.index[at]
    return np.arange(start,start+4,dtype=int)


def native_private_J(c,group,multiplier):
    source=group['factor'];view=copy(source)
    view.rows=[group['row']];view.scales=group['scales'].copy()
    view.source_point_qualification=None;view._active_fit_view_cache=None;view._cache=OrderedDict()
    view.camera_steps={int(col):float(h)*multiplier for col,h in source.camera_steps.items()}
    columns=private_columns(c,group['actual_pts'])
    cams={group['actual_pts']:ctx.camera(c,c.q,group['actual_pts'])}
    associations=view._nominal_associations(c.q,cams)
    def evaluate(value,indices):
        now={group['actual_pts']:ctx.camera(c,value,group['actual_pts'])}
        return view._derivative_residual(value,cameras=now,associations=associations,
            apply_optical_support=True,row_indices=indices)
    nominal=evaluate(c.q,np.array([0],int))
    np.testing.assert_allclose(nominal,group['actual'],rtol=0.,atol=1e-10)
    J=native_factor_dependency_jacobian(view,c.q,view.rows,columns,np.empty(0,int),
        nominal,evaluate,np.empty(0,int))
    return J,dict(method='original_nominal_nearest_segment_clamp_and_optical_support',
        columns=columns.tolist(),FD_steps=[view.camera_steps.get(int(col),1e-6)for col in columns],
        scalar_rows=len(nominal),step_multiplier=multiplier)


def local_derivatives(c,multiplier):
    answer=[];receipts=[]
    for group in c.groups:
        if group['source_kind']=='native':
            J,receipt=native_private_J(c,group,multiplier)
        else:
            cam=ctx.camera(c,c.q,group['actual_pts']);row=c.provider.rebind(group['row'],c.world)
            r,J15=c.calls['low'](cam,[row],True)
            np.testing.assert_allclose(r,group['actual'],rtol=0.,atol=1e-10)
            cols,mapping=derivative.camera_map(c,group['actual_pts'])
            positions=[int(np.flatnonzero(cols==col)[0])for col in private_columns(c,group['actual_pts'])]
            J=(np.asarray(J15)@mapping)[:,positions]
            receipt=dict(method='original_analytic_camera_J15_times_exact_RF4_mapping',scalar_rows=len(r),
                columns=private_columns(c,group['actual_pts']).tolist(),step_multiplier=multiplier,
                identical_at_two_labels_by_analytic_construction=True)
        answer.append(J);receipts.append(dict(id=group['id'],actual_pts=group['actual_pts'],**receipt))
    return answer,receipts


def inverse_responses(c,first,half):
    responses=[]
    per_exposure=[]
    for at in SIX:
        ids=[i for i,g in enumerate(c.groups)if g['actual_pts']==at]
        r=np.concatenate([c.groups[i]['actual']for i in ids])
        w=np.concatenate([source_irls_weights(c.groups[i]['actual'],c.groups[i]['quadratic'],c.groups[i]['loss'])for i in ids])
        A=np.vstack([first[i]for i in ids])*np.sqrt(w)[:,None]
        B=np.vstack([half[i]for i in ids])*np.sqrt(w)[:,None]
        mean=(A+B)/2;change=A-B;U,s,VT=np.linalg.svd(mean,full_matrices=False)
        machine=float(np.finfo(float).eps*max(mean.shape)*s[0])
        observed=float(np.linalg.norm(change,2))
        rw=r*np.sqrt(w);columns=private_columns(c,at)
        entry=dict(actual_pts=at,source_rows=[dict(id=c.groups[i]['id'],family=c.groups[i]['family'],
            points=c.groups[i]['scalar_rows']//2)for i in ids],scalar_rows=len(r),
            RF4_evaluation_columns=columns.tolist(),RF4_canonical_columns=c.state.canonical_parameter_layout.evaluation_to_canonical[columns].tolist(),
            parameter_metric='literal current Rodrigues radians and native logf; no column unit normalization',
            singular_values_mean=s.tolist(),singular_values_first=np.linalg.svd(A,compute_uv=False).tolist(),
            singular_values_half=np.linalg.svd(B,compute_uv=False).tolist(),machine_rank_threshold=machine,
            observed_two_h_operator_difference_norm=observed,weighted_original_energy=float(rw@rw),responses={})
        # Every spectral prefix is reported, independent of residual reduction.
        # No physically attractive prefix is silently selected as a candidate.
        for rank in range(1,len(s)+1):
            name=f'PTS{at}_spectral_prefix_{rank}'
            mask=np.arange(len(s))<rank;step=-(VT[mask].T/s[mask])@(U[:,mask].T@rw)
            assert np.isfinite(step).all()
            delta=np.zeros_like(c.q);delta[columns]=step
            responses.append(dict(name=name,actual_pts=at,rank=rank,step=delta))
            remaining=rw+mean@step
            response_first=A@step;response_half=B@step
            import cv2
            original_R=cv2.Rodrigues(c.q[columns[:3]])[0]
            trial_R=cv2.Rodrigues(c.q[columns[:3]]+step[:3])[0]
            angle=float(np.degrees(np.arccos(np.clip((np.trace(trial_R@original_R.T)-1)/2,-1,1))))
            entry['responses'][name]=dict(rank=int(mask.sum()),linear_remaining_energy=float(remaining@remaining),
                nullity=4-int(mask.sum()),linear_solution_unique=bool(mask.sum()==4),
                minimum_norm_in_literal_RF4_metric=True,
                prefix_rank_not_selected_from_residual=True,
                machine_rank=int((s>machine).sum()),above_observed_two_h_change_rank=int((s>max(machine,observed)).sum()),
                Rodrigues_increment_rad=step[:3].tolist(),Rodrigues_increment_norm_rad=float(np.linalg.norm(step[:3])),
                equivalent_finite_orientation_change_degrees=angle,native_logf_increment=float(step[3]),
                focal_ratio=float(np.exp(step[3]))if abs(step[3])<700 else None,
                first_half_directional_response_difference_norm=float(np.linalg.norm(response_first-response_half)),
                infinitesimal_projection_does_not_assert_finite_physical_feasibility=True)
        per_exposure.append(entry)
    return responses,per_exposure


def guard_factors(c):
    from structured_programme_source_candidate.human_stroke_truth import WideStrokeTruth,UserConfirmedSourceTruth
    wide=ctx.wide_factory(c)
    chart=c.state.structured_programme_coordinate_chart
    wt=WideStrokeTruth(wide,c.options['--original-wide-human-truth-authority'],len(c.q),chart.high_indices)
    wt.bind_practical_location_envelope(c.options['--practical-human-location-envelope'])
    confirmed=UserConfirmedSourceTruth(c.state,c.options['--user-confirmed-source-truth-authority'],c.board,c.ice,
        parameter_count=len(c.q),circle_factor=c.circle,material_factor=c.material,placement_source=c.placement)
    return dict(native=c.human,wide=wt,confirmed=confirmed)


def guard_values(factors,value):
    human=factors['native'];primary=human.queries(value);full=human.queries(value,full_stroke=True)
    wide=factors['wide'].queries(value);_,_,_,details=factors['confirmed']._carriers(value)
    return dict(native=np.maximum([abs(r['query']['residual']).max()for r in primary],
                                 [abs(r['query']['residual']).max()for r in full]),
        wide=np.asarray([abs(r['query']['residual']).max()for r in wide]),
        confirmed=np.asarray([r['maximum_violation_px']for r in details])),dict(
        native=[r['id']for r in primary],wide=[r['id']for r in wide],confirmed=[r['id']for r in details])


def guarded_report(factors,q,caps,base):
    values,ids=guard_values(factors,q);rows=[]
    for name,v in values.items():
        saved=caps['families'][name]
        assert ids[name]==saved['group_ids'];limits=np.asarray(saved['limits'])
        rows.extend(dict(family=name,id=identity,original_q8_limit_px=float(limit),baseline_violation_px=float(old),
            trial_violation_px=float(new),value_change_px=float(new-old),exceeds_original_q8_limit=bool(new>limit+1e-8))
            for identity,limit,old,new in zip(ids[name],limits,base[name],v))
    assert len(rows)==86
    return dict(groups=86,all_original_ids_and_limits_matched=True,
        maximum_original_limit_excess_px=max(r['trial_violation_px']-r['original_q8_limit_px']for r in rows),
        maximum_changed_guard_value_px=max(abs(r['value_change_px'])for r in rows),
        passes_original_q8_caps=not any(r['exceeds_original_q8_limit']for r in rows),rows=rows)


def reconstruct_motion(c):
    coverage_path=a.H/'current_motion_role_whole_programme_exact_947_20261011.json'
    coverage=a.read(coverage_path);authority=coverage['actual_runtime_manifest']
    assert a.sha(authority['path'])==authority['sha256']
    previous=a.read(authority['path']);manifest=dict(c.manifest)
    for key in ('original_motion_far_board_role_correction','original_motion_edge_counts','original_motion_rows','actual_exposures','native_presentations'):
        assert manifest[key]==previous[key],key
    for key in ('schema','low_candidate','main_candidate','removed_motion_points','removed_motion_scalar_rows','original_retirement_scalar_remap'):
        assert manifest['original_motion_input_compaction'][key]==previous['original_motion_input_compaction'][key],key
    manifest['persistent_source_replay_identity']=previous['persistent_source_replay_identity']
    source_ref=coverage['reconstruction_module'];assert a.sha(source_ref['path'])==source_ref['sha256']
    spec=importlib.util.spec_from_file_location('rs_original_motion_reconstruction',source_ref['path'])
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.STATE=c.directory;old.RUN=a.RUN
    binding_path=Path(source_ref['path']).parent/'actual_pan_pairs_canonical_RF4_export_binding_27fa_699_readonly_20261011.json'
    bindings=next(r for r in a.read(binding_path)['models']if r['model']=='699')['pairs']
    blocks,checks,refs=old.reconstruct(manifest,a.read(a.RUN/'actual_original_motion_input_compaction.json'),bindings)
    source=blocks['low'];lengths=c.arrays['aux__block_lengths']//4;names=c.arrays['aux__block_names'].tolist()
    ordinal=names.index('low');begin=int(sum(lengths[:ordinal]));stamps=c.arrays['aux__canonical_pts'][begin:begin+lengths[ordinal]]
    expected=manifest['persistent_source_replay_identity']['actual_original_source_input_arrays']
    assert old.token(stamps)==expected['programme:low:stamps']['sha256']
    typed=a.R/'prepared_full8503_1791035775270492900/complete_original_sources/low_source/low_source_factory/finite_material_source_binding.json'
    original_board=np.asarray([r['global_bundle_index']for r in a.read(typed)if r['surface_kind']=='finite_board_face'],int)
    keep=source['active_to_original'];positions=np.searchsorted(keep,original_board)
    assert np.array_equal(keep[positions],np.sort(original_board))
    board=np.zeros(len(source['a']),bool);board[positions]=True
    role_indices=np.asarray(c.manifest['original_motion_far_board_role_correction']['original_global_bundle_indices_by_block']['low'],int)
    positions=np.searchsorted(keep,role_indices);assert np.array_equal(keep[positions],role_indices)
    board[positions]=True
    assert old.token(np.flatnonzero(board))==expected['programme:low:course_board_indices']['sha256']
    ice=source['finite_mask']&~board
    assert int(ice.sum())==expected['programme:low:static_noncourse_finite_point_indices']['shape'][0]
    pa,pb=stamps[source['ia']],stamps[source['ib']]
    selected=np.flatnonzero(np.isin(pa,SIX)|np.isin(pb,SIX))
    assert np.all((pa[selected]>=229945856)&(pa[selected]<230369024)&(pb[selected]>=229945856)&(pb[selected]<230369024))
    profile=c.arrays['profile__weights_'+str(ordinal)]
    assert profile.shape==(len(board),2)
    scene=a.read(c.directory/'scene_UNACCEPTED.json');saved_border=np.asarray(scene['ice_boundary']['points_xyz'],float)
    course,course_q,_=c.provider.print_course_binding(c.q[c.world_indices])
    border=np.asarray(course.sample_perimeter(course_q,with_jacobian=False)[0],float)
    np.testing.assert_array_equal(border,saved_border)
    contract=json.loads(str(c.arrays['aux__current_depth_and_material_domain_contract_json'].item()))
    floor=next(r['candidate_minimum_source_Z_m']for r in contract['blocks']if r['block']=='low')/contract['scene_unit_metres']
    return dict(old=old,source=source,stamps=stamps,pa=pa,pb=pb,selected=selected,board=board,ice=ice,
        profile=profile,border=border,floor=floor,coverage=a.ref(coverage_path),refs=refs,
        checks=checks['low'],typed=a.ref(typed),authority=authority,
        public_course_border_bitexact_with_saved_scene=True)


def motion_values(c,motion,q):
    from structured_programme_source_candidate.original_motion_far_board_roles import OriginalMotionFarBoardRoleBinding
    from shared_board_course_candidate.board_course_material import vertical_plane_value_jacobian
    import cv2
    src=motion['source'];ids=motion['selected'];stamps=motion['stamps'];border=motion['border'];old=motion['old']
    ca=np.asarray([ctx.camera(c,q,int(p))for p in stamps[src['ia'][ids]]])
    cb=np.asarray([ctx.camera(c,q,int(p))for p in stamps[src['ib'][ids]]])
    ra=np.asarray([cv2.Rodrigues(v[3:6])[0]for v in ca]);rb=np.asarray([cv2.Rodrigues(v[3:6])[0]for v in cb])
    planes=np.zeros((len(ids),4));planes[motion['ice'][ids]]=[0.,0.,1.,0.]
    for view in np.unique(src['ia'][ids[motion['board'][ids]]]):
        local=np.flatnonzero(motion['board'][ids]&(src['ia'][ids]==view));camera=ca[local[0]]
        course=SimpleNamespace(sample_perimeter=lambda value,with_jacobian=False:(border,))
        selector=SimpleNamespace(_facet_course_cache=None,records=[None]*len(local),source_row_groups={0:np.arange(len(local))},
            source_cameras={0:camera},source_uv=src['a'][ids[local]],exact_cache_statistics=dict(facet_group_hits=0,facet_group_misses=0))
        facets=OriginalMotionFarBoardRoleBinding._current_source_facets(selector,course,None,('saved_course',old.token(border)))
        edge=border[:-1]if np.array_equal(border[0],border[-1])else border;zero=np.zeros((3,0))
        planes[local]=np.asarray([vertical_plane_value_jacobian(edge[i],edge[(i+1)%len(edge)],zero,zero)[0]for i in facets])
    sigma=src['motion_sigma'].reshape(-1,2)[ids];weight=motion['profile'][ids]
    value,checks=old.native_motion(a=src['a'][ids],b=src['b'][ids],fa=np.exp(ca[:,6]),fb=np.exp(cb[:,6]),
        ra=ra,rb=rb,da=None,db=None,ca=ca[:,:3],cb=cb[:,:3],dca_shared=None,dcb_shared=None,dca_view=None,dcb_view=None,
        pp=ca[0,8:10],k1=ca[0,11],planes=planes,jacobian=False,sigma=sigma,profile_weight=weight,
        minimum_source_depth_scene_units=motion['floor'])
    return np.asarray(value).reshape(-1,2),checks


def motion_report(motion,base,trial,checks):
    ids=motion['selected'];sigma=motion['source']['motion_sigma'].reshape(-1,2)[ids]
    def cost(r,mask):
        z=r[mask]/sigma[mask];return float(np.where(abs(z)<=2,.5*z*z,2*abs(z)-2).sum())
    result=[]
    for name,mask in [('finite_board',motion['board'][ids]),('finite_ice',motion['ice'][ids]),
                      ('unknown_depth',~motion['source']['finite_mask'][ids]),('all_incident_motion',np.ones(len(ids),bool))]:
        result.append(dict(branch=name,points=int(mask.sum()),before_cost=cost(base,mask),trial_cost=cost(trial,mask),
            cost_change=cost(trial,mask)-cost(base,mask),before_raw_norm_px=a.stats(np.linalg.norm(base[mask],axis=1)),
            trial_raw_norm_px=a.stats(np.linalg.norm(trial[mask],axis=1))))
    pairs=[];pa=motion['pa'][ids];pb=motion['pb'][ids]
    for sa,sb in sorted(set(zip(pa.tolist(),pb.tolist()))):
        mask=(pa==sa)&(pb==sb)
        pairs.append(dict(source_pts=sa,target_pts=sb,points=int(mask.sum()),before_cost=cost(base,mask),
            trial_cost=cost(trial,mask),cost_change=cost(trial,mask)-cost(base,mask),
            finite_board_points=int((mask&motion['board'][ids]).sum()),finite_ice_points=int((mask&motion['ice'][ids]).sum())))
    return dict(branches=result,actual_pairs=pairs,native_checks=checks,
        isolated_camera_response_not_a_whole_shot_improvement=True)


def main():
    started=time.perf_counter();code_refs=[a.ref(__file__),a.ref(ctx.__file__),a.ref(derivative.__file__)]
    c=ctx.load_context();frozen_q=c.q.copy();baseline=ctx.baseline(c,set(SIX))
    ordinal=c.arrays['aux__block_names'].tolist().index('low')
    supplement=('profile__weights_'+str(ordinal),'aux__current_depth_and_material_domain_contract_json')
    with np.load(c.path,allow_pickle=False)as saved:
        c.arrays.update({key:saved[key].copy()for key in supplement})
    metadata=a.read(ctx.METADATA)
    wanted=[r for r in metadata['rows']if r['actual_pts']in SIX and r['role']=='FIT'and r['active_point_count']]
    actual=[(g['actual_pts'],g['id'],g['scalar_rows']//2)for g in c.groups]
    expected=[(r['actual_pts'],r['id'],r['active_point_count'])for r in wanted]
    assert Counter(actual)==Counter(expected),(actual,expected)
    assert len(c.groups)==7 and sum(g['scalar_rows']for g in c.groups)==40
    refs=code_refs+[a.ref(ctx.METADATA),a.ref(c.path),a.ref(a.CONFIG)]
    assert code_refs==[a.ref(record['path'])for record in code_refs]
    emit(stage='LOW6_source_scope_authenticated',operator_groups=len(c.groups),scalars=40,RF4_columns=24,seconds=time.perf_counter()-started)
    first,receipt1=local_derivatives(c,1.);half,receipt2=local_derivatives(c,.5)
    responses,exposures=inverse_responses(c,first,half)
    emit(stage='LOW6_private_RF4_conditioned_linear_responses',exposures=exposures)
    checkpoint=OUT.with_name(OUT.stem+'_linear_checkpoint.json')
    checkpoint.write_text(json.dumps(dict(frozen_complete_q_sha256=a.CHILD,references=refs,exposures=exposures,
        source_groups=7,source_scalars=40,derivative_receipts=receipt1+receipt2),ensure_ascii=False,
        indent=2,default=a.plain,allow_nan=False)+'\n',encoding='utf8')
    factors=guard_factors(c);caps=a.read(c.directory/'truth_restoration_state.json')
    base_guards,base_guard_ids=guard_values(factors,c.q)
    nominal_guards=guarded_report(factors,c.q,caps,base_guards)
    assert nominal_guards['passes_original_q8_caps']
    motion=reconstruct_motion(c);base_motion,base_checks=motion_values(c,motion,c.q)
    emit(stage='LOW6_all_actual_incident_motion_reconstructed',points=len(motion['selected']),
        actual_pairs=len(set(zip(motion['pa'][motion['selected']].tolist(),motion['pb'][motion['selected']].tolist()))),
        baseline_native_checks=base_checks,seconds=time.perf_counter()-started)
    before_values=group_values(c,c.q);before_cost=sum(weighted_source_cost(v,g['quadratic'],g['loss'])for v,g in zip(before_values,c.groups))
    evaluations=[]
    arrays=dict(actual_pts=np.asarray(SIX),RF4_columns=np.vstack([private_columns(c,p)for p in SIX]),
        source_group_offsets=np.r_[0,np.cumsum([g['scalar_rows']for g in c.groups])],
        nominal_residual=np.concatenate(before_values),J_first=np.vstack(first),J_half=np.vstack(half))
    for response in responses:
        name,step=response['name'],response['step']
        trial=c.q+step
        assert np.array_equal(trial[c.world_indices],c.q[c.world_indices])
        entry=dict(name=name,actual_pts=response['actual_pts'],spectral_prefix_rank=response['rank'],diagnostic_response_only=True)
        cols=private_columns(c,response['actual_pts']);logf=float(trial[cols[3]])
        focal_legal=bool(np.log(100.)<=logf<=np.log(50000.))
        entry['original_physical_domain_checks']=dict(
            all_changed_RF4_focal_log_values=[logf],original_focal_lower_px=100.,original_focal_upper_px=50000.,
            passes_original_changed_camera_focal_domain=focal_legal,
            shared_world_installation_and_other_cameras_bitexact=True,
            full_production_finite_extent_guard_reconstructed=False,
            full_physical_domain_pass_not_claimed=True)
        if not focal_legal:
            entry['actual_nonlinear_operator_failure']=dict(type='OriginalPhysicalFocalDomain',message='Original 100..50000 focal guard excludes the diagnostic response; nonlinear replay not treated as legal.')
            evaluations.append(entry)
            emit(stage='LOW6_original_nonlinear_consequences',name=name,domain=entry['original_physical_domain_checks'])
            continue
        try:
            values=group_values(c,trial);cost=sum(weighted_source_cost(v,g['quadratic'],g['loss'])for v,g in zip(values,c.groups))
            entry['absolute']=dict(before_original_cost=before_cost,trial_original_cost=cost,cost_change=cost-before_cost,
                groups=[dict(id=g['id'],actual_pts=g['actual_pts'],before_native_component_p95=float(np.quantile(abs(b*g['scales']),.95)),
                    trial_native_component_p95=float(np.quantile(abs(v*g['scales']),.95)),
                    original_cost_before=weighted_source_cost(b,g['quadratic'],g['loss']),
                    original_cost_trial=weighted_source_cost(v,g['quadratic'],g['loss']))for g,b,v in zip(c.groups,before_values,values)])
            arrays[name+'__source_residual']=np.concatenate(values)
        except Exception as exc:
            entry['actual_nonlinear_absolute_failure']=dict(type=type(exc).__name__,message=str(exc))
        try:
            entry['all86_original_dynamic_guards']=guarded_report(factors,trial,caps,base_guards)
        except Exception as exc:
            entry['actual_nonlinear_guard_failure']=dict(type=type(exc).__name__,message=str(exc))
        try:
            moved,checks=motion_values(c,motion,trial);entry['all_incident_motion']=motion_report(motion,base_motion,moved,checks)
            arrays[name+'__incident_motion_raw']=moved
        except Exception as exc:
            entry['actual_nonlinear_motion_failure']=dict(type=type(exc).__name__,message=str(exc))
        evaluations.append(entry)
        emit(stage='LOW6_original_nonlinear_consequences',name=name,
            absolute=entry.get('absolute'),guards={k:v for k,v in entry.get('all86_original_dynamic_guards',{}).items()if k!='rows'},
            motion=entry.get('all_incident_motion',{}).get('branches'),failure=entry.get('actual_nonlinear_operator_failure'))
    arrays.update(incident_motion_original_point_indices=motion['selected'],incident_motion_source_pts=motion['pa'][motion['selected']],
        incident_motion_target_pts=motion['pb'][motion['selected']],baseline_incident_motion_raw=base_motion)
    array_path=OUT.with_suffix('.npz');np.savez_compressed(array_path,**arrays)
    assert np.array_equal(c.q,frozen_q)
    assert refs==[a.ref(record['path'])for record in refs]
    report=dict(schema='rinkspace.LOW6-private-RF4-local-response-and-original-incident-motion.v1',script=refs[0],
        frozen_complete_q_sha256=a.CHILD,references=refs,metadata_exact_samePTS_all_absolute_FIT_crosscheck=True,
        source_rows=[{key:g[key]for key in ('family','actual_pts','id','scalar_rows','source_kind')}for g in c.groups],
        frozen_world_installation_shared_camera_wide_and_other_RF4=True,private_exposures=SIX,private_columns=24,
        derivation='All singular-spectrum prefix minimum-norm linear responses in the literal original RF4 chart, independent of residual reduction, without iterative fitting or accepted parameter state. Original machine and observed-two-h ranks are retained as references, not physical feasibility certificates.',
        derivative_receipts=receipt1+receipt2,exposures=exposures,nominal_all86_guard_values=nominal_guards,
        motion_reconstruction=dict(coverage=motion['coverage'],prior_array_authentication=motion['authority'],
            actual_LOW_input_hash_checks=motion['checks'],source_authorities=motion['refs'],original_typed_material=motion['typed'],
            total_original_LOW_motion_points=len(motion['source']['a']),all_incident_motion_points=len(motion['selected']),
            original_profiles_source_sigmas_and_roles_preserved=True,source_clock_shot=5,
            public_course_border_bitexact_with_saved_scene=motion['public_course_border_bitexact_with_saved_scene'],
            facet_selection='Public current source-ray facet selector and public current perimeter; perimeter is bitexact with immutable q8 saved scene.'),
        diagnostic_nonlinear_consequences=evaluations,arrays=dict(a.ref(array_path),bytes=array_path.stat().st_size,
            contains_small_RF4_J_and_values_only=True,no_complete_source_J_no_q_no_parameter_direction=True),
        no_fitted_or_promoted_q_no_source_graph_no_images_no_decoder_no_video=True,
        limitations=['A local camera response is not a legal full-source improvement or complete-shot accuracy result.',
            'Six isolated camera changes intentionally expose all incident motion consequences; no smoothing or neighbour optimization is performed.',
            'Full 86 saved caps are evaluated, including unchanged groups. No complete source QP or KKT certificate is constructed.',
            'Original changed-camera focal domain and real native incident-motion projection checks are explicit. Complete production finite-source extent guards are not reconstructed; full physical-domain passage is not claimed.',
            'Two h agreement cannot detect common derivative bias; the machine-reference response is explicitly untrusted near weak modes.',
            'No residual-derived weight, support, source role, physical geometry, mount or shared calibration is changed.'],
        seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=a.plain,allow_nan=False)+'\n',encoding='utf8')
    emit(stage='LOW6_causal_diagnostic_complete',path=str(OUT),sha256=a.sha(OUT),bytes=OUT.stat().st_size,seconds=report['seconds'])


if __name__=='__main__':
    main()
