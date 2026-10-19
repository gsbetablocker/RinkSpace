"""One fixed whole-shot RF3 covariation; original values, no fitting/J/images.

The finite rotation comes from the previously frozen LOW6 strongest mode.
Focal, world, installation, all other shots and all source inputs stay fixed.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import json
import time

import audit_low6_private_RF4_causal_response_20261011 as local
import joint_absolute_source_context_20261011 as ctx
from calibration.solver_runtime.source_loss import weighted_source_cost

a,np=ctx.a,ctx.np
LO,HI=229945856,230369024
SOURCE=local.OUT
OUT=a.H/'shot5_fixed_common_rotation_original_source_response_20261011.json'


def group_values(c,value):
    result=[]
    for group in c.groups:
        if group['source_kind']=='original':
            r=ctx.original_value(c,group['packet_row_index'],value)
        elif group['source_kind']=='native':
            r=ctx.native_value(c,group['factor'],value,[group['row']])
        elif group['source_kind']=='manual_centre':
            factor=group['factor'];row=group['row'];at=row['actual_pts']
            r=factor._residual(value,[row],cameras={at:ctx.camera(c,value,at)})/group['mean_scale']
        else:raise ValueError(group['source_kind'])
        result.append(np.asarray(r))
    return result


def fixed_rotation(c,shot_pts):
    import cv2
    receipt=a.read(SOURCE)
    assert receipt['frozen_complete_q_sha256']==a.CHILD
    exposure=next(row for row in receipt['exposures']if row['actual_pts']==230306816)
    definition=exposure['responses']['PTS230306816_spectral_prefix_1']
    columns=local.private_columns(c,230306816)
    base=cv2.Rodrigues(c.q[columns[:3]])[0]
    response=cv2.Rodrigues(c.q[columns[:3]]+np.asarray(definition['Rodrigues_increment_rad']))[0]
    W=response.T@base
    np.testing.assert_allclose(W.T@W,np.eye(3),rtol=0.,atol=2e-15)
    trial=c.q.copy();changed=[]
    errors=[];centres=[];pivots=[]
    for at in shot_pts:
        block=c.state.bypts[at];cal=block.bundle.calibration
        pivot=np.asarray(cal.pivot_xyz)
        before=ctx.camera(c,c.q,at);R=cv2.Rodrigues(before[3:6])[0]
        view=local.private_columns(c,at)
        trial[view[:3]]=cv2.Rodrigues(R@W.T)[0].ravel();changed.extend(view[:3])
        after=ctx.camera(c,trial,at);new_R=cv2.Rodrigues(after[3:6])[0]
        target_C=pivot+W@(before[:3]-pivot)
        errors.append(float(abs(new_R-R@W.T).max()))
        centres.append(float(abs(after[:3]-target_C).max()))
        pivots.append(pivot)
        np.testing.assert_array_equal(after[np.r_[6:25]],before[np.r_[6:25]])
    changed=np.unique(changed)
    unchanged=np.setdiff1d(np.arange(len(c.q)),changed)
    np.testing.assert_array_equal(trial[unchanged],c.q[unchanged])
    np.testing.assert_array_equal(pivots,np.broadcast_to(pivots[0],np.asarray(pivots).shape))
    assert max(errors)<1e-11 and max(centres)<1e-11,(max(errors),max(centres))
    return trial,dict(source_response=a.ref(SOURCE),source_actual_pts=230306816,
        source_fixed_spectral_prefix=1,W_world_rotation=W.tolist(),
        finite_rotation_degrees=definition['equivalent_finite_orientation_change_degrees'],
        all_original_logf_values_bitexact=True,discarded_source_local_logf_increment=definition['native_logf_increment'],
        no_direction_or_rotation_optimized=True,public_camera25_recomputed_not_directly_edited=True,
        camera_rotation_mapping_max_error=max(errors),optical_centre_mapping_max_error_scene_units=max(centres),
        only_shot5_existing_Rodrigues_RF3_changed=True,actual_exposures=len(shot_pts),changed_RF3_columns=len(changed))


def motion_values(c,motion,value):
    from structured_programme_source_candidate.original_motion_far_board_roles import OriginalMotionFarBoardRoleBinding
    from shared_board_course_candidate.board_course_material import vertical_plane_value_jacobian
    import cv2
    src=motion['source'];ids=motion['selected'];border=motion['border'];old=motion['old']
    cameras=np.asarray([ctx.camera(c,value,int(p))for p in motion['stamps']])
    rotations=np.asarray([cv2.Rodrigues(v[3:6])[0]for v in cameras])
    ca,cb=cameras[src['ia'][ids]],cameras[src['ib'][ids]]
    ra,rb=rotations[src['ia'][ids]],rotations[src['ib'][ids]]
    planes=np.zeros((len(ids),4));planes[motion['ice'][ids]]=[0.,0.,1.,0.]
    for view in np.unique(src['ia'][ids[motion['board'][ids]]]):
        positions=np.flatnonzero(motion['board'][ids]&(src['ia'][ids]==view))
        course=SimpleNamespace(sample_perimeter=lambda q,with_jacobian=False:(border,))
        selector=SimpleNamespace(_facet_course_cache=None,records=[None]*len(positions),
            source_row_groups={0:np.arange(len(positions))},source_cameras={0:cameras[view]},
            source_uv=src['a'][ids[positions]],exact_cache_statistics=dict(facet_group_hits=0,facet_group_misses=0))
        facets=OriginalMotionFarBoardRoleBinding._current_source_facets(selector,course,None,('saved_current_course',old.token(border)))
        edge=border[:-1]if np.array_equal(border[0],border[-1])else border;zero=np.zeros((3,0))
        planes[positions]=np.asarray([vertical_plane_value_jacobian(edge[i],edge[(i+1)%len(edge)],zero,zero)[0]for i in facets])
    sigma=src['motion_sigma'].reshape(-1,2)[ids]
    native,checks=old.native_motion(a=src['a'][ids],b=src['b'][ids],fa=np.exp(ca[:,6]),fb=np.exp(cb[:,6]),
        ra=ra,rb=rb,da=None,db=None,ca=ca[:,:3],cb=cb[:,:3],dca_shared=None,dcb_shared=None,
        dca_view=None,dcb_view=None,pp=ca[0,8:10],k1=ca[0,11],planes=planes,jacobian=False,
        sigma=sigma,profile_weight=motion['profile'][ids],minimum_source_depth_scene_units=motion['floor'])
    return np.asarray(native).reshape(-1,2),checks,dict(cameras=cameras,rotations=rotations,planes=planes)


def motion_summary(c,motion,before,after,base_camera,trial_camera,checks):
    ids=motion['selected'];src=motion['source'];pa=motion['pa'][ids];pb=motion['pb'][ids]
    coverage=a.read(motion['coverage']['path']);baseline_ref=coverage['baseline_saved_state']
    assert a.sha(baseline_ref['path'])==baseline_ref['sha256']
    baseline_identity=a.read(baseline_ref['path']);baseline_manifest_path=Path(baseline_ref['path']).parents[2]/'input_manifest.json'
    assert a.sha(baseline_manifest_path)==baseline_identity['input_manifest_sha256']
    old_role=a.read(baseline_manifest_path)['original_motion_far_board_role_correction']['original_global_bundle_indices_by_block']['low']
    current_role=c.manifest['original_motion_far_board_role_correction']['original_global_bundle_indices_by_block']['low']
    added=np.asarray(sorted(set(current_role)-set(old_role)),int)
    corrected=np.isin(src['active_to_original'][ids],added)
    assert int(corrected.sum())==704
    original_board=motion['board'][ids]&~corrected
    masks=dict(previous_authenticated_finite_board_roles=original_board,authenticated_new_947_correction=corrected,
        finite_ice=motion['ice'][ids],unknown_depth=~src['finite_mask'][ids])
    assert sum(np.asarray(list(masks.values()),int).sum(axis=0)==1)==len(ids)
    sigma=src['motion_sigma'].reshape(-1,2)[ids]
    retirement=c.manifest['static_motion_source_role_retirement']
    global_slots=np.asarray(retirement['global_scalar_slots'],int)
    local_slots=global_slots-retirement['programme_block']['original_start']
    assert np.all((local_slots>=0)&(local_slots<2*len(src['a'])))
    point=local_slots//2;positions=np.searchsorted(ids,point)
    inside=(positions<len(ids));inside[inside]&=ids[positions[inside]]==point[inside]
    active=np.ones((len(ids),2),bool);active[positions[inside],local_slots[inside]%2]=False
    def cost(r):
        z=np.where(active,r/sigma,0.)
        return np.where(abs(z)<=2,.5*z*z,2*abs(z)-2).sum(axis=1)
    old_cost,new_cost=cost(before),cost(after)
    branches=[]
    for name,mask in list(masks.items())+[('all_shot5_motion',np.ones(len(ids),bool))]:
        difference=after[mask]-before[mask]
        branches.append(dict(branch=name,points=int(mask.sum()),source_scalar_rows=int(mask.sum()*2),
            original_frozen_profile_Huber_cost=float(old_cost[mask].sum()),
            trial_frozen_profile_Huber_cost=float(new_cost[mask].sum()),
            cost_change=float(new_cost[mask].sum()-old_cost[mask].sum()),
            before_raw_norm_px=a.stats(np.linalg.norm(before[mask],axis=1)),
            trial_raw_norm_px=a.stats(np.linalg.norm(after[mask],axis=1)),
            maximum_component_change_px=float(abs(difference).max())if len(difference)else 0.,
            rms_component_change_px=float(np.sqrt(np.mean(difference*difference)))if len(difference)else 0.))
    unique,inverse=np.unique(np.c_[pa,pb],axis=0,return_inverse=True)
    original_pair_cost=np.bincount(inverse,weights=old_cost);trial_pair_cost=np.bincount(inverse,weights=new_cost)
    pair_counts=np.bincount(inverse)
    pair_affected=np.argsort(abs(trial_pair_cost-original_pair_cost))[::-1][:20]
    worst=[dict(source_actual_pts=int(unique[i,0]),target_actual_pts=int(unique[i,1]),points=int(pair_counts[i]),
        original_cost=float(original_pair_cost[i]),trial_cost=float(trial_pair_cost[i]),
        cost_change=float(trial_pair_cost[i]-original_pair_cost[i]))for i in pair_affected]
    ba=base_camera['rotations'][src['ia'][ids]];bb=base_camera['rotations'][src['ib'][ids]]
    ta=trial_camera['rotations'][src['ia'][ids]];tb=trial_camera['rotations'][src['ib'][ids]]
    relative_rotation_before=np.einsum('nij,nkj->nik',bb,ba)
    relative_rotation_after=np.einsum('nij,nkj->nik',tb,ta)
    def relative_translation(cameras,rotations):
        delta=cameras[src['ia'][ids],:3]-cameras[src['ib'][ids],:3]
        return np.einsum('nij,nj->ni',rotations[src['ib'][ids]],delta)
    relative_before=relative_translation(base_camera['cameras'],base_camera['rotations'])
    relative_after=relative_translation(trial_camera['cameras'],trial_camera['rotations'])
    relative_R_error=float(abs(relative_rotation_after-relative_rotation_before).max())
    relative_C_error=float(abs(relative_after-relative_before).max())
    assert relative_R_error<1e-11 and relative_C_error<1e-11
    unknown=masks['unknown_depth'];unknown_error=float(abs(after[unknown]-before[unknown]).max())
    unknown_point_error=np.max(abs(after[unknown]-before[unknown]),axis=1)
    unknown_numeric=dict(maximum_component_difference_px=unknown_error,
        p99_point_max_component_difference_px=float(np.quantile(unknown_point_error,.99)),
        counts_above_fixed_diagnostic_pixel_scales={str(scale):int((unknown_point_error>scale).sum())for scale in (1e-6,1e-4,1e-2)},
        strict_1e_minus7_diagnostic_check_passed=bool(unknown_error<1e-7),
        diagnostic_check_not_a_source_support_or_acceptance_tolerance=True)
    if unknown_error>=1e-7:
        point_delta=np.max(abs(after-before),axis=1)
        worst_ids=np.flatnonzero(unknown)[np.argsort(point_delta[unknown])[::-1][:20]]
        failure=dict(schema='rinkspace.original-motion-common-rotation-invariance-diagnostic.v1',
            original_q_sha256=a.CHILD,script=a.ref(__file__),maximum_unknown_component_difference_px=unknown_error,
            relative_R_max_error=relative_R_error,relative_C_max_error_scene_units=relative_C_error,
            original_native_unknown_values_not_certified_invariant=True,branches=branches,
            original_operator_difference_distribution_px=a.stats(point_delta[unknown]),
            worst_original_unknown_points=[dict(current_LOW_point_index=int(ids[i]),
                original_LOW_point_index=int(src['active_to_original'][ids[i]]),source_actual_pts=int(pa[i]),target_actual_pts=int(pb[i]),
                source_uv=src['a'][ids[i]].tolist(),target_uv=src['b'][ids[i]].tolist(),
                source_sigma=sigma[i].tolist(),frozen_profile_weights=motion['profile'][ids[i]].tolist(),
                original_raw=before[i].tolist(),trial_raw=after[i].tolist(),
                source_camera_original=base_camera['cameras'][src['ia'][ids[i]]].tolist(),
                target_camera_original=base_camera['cameras'][src['ib'][ids[i]]].tolist(),
                source_camera_trial=trial_camera['cameras'][src['ia'][ids[i]]].tolist(),
                target_camera_trial=trial_camera['cameras'][src['ib'][ids[i]]].tolist())for i in worst_ids])
        failure_path=OUT.with_name(OUT.stem+'_unknown_invariance_probe.json')
        failure_path.write_text(json.dumps(failure,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
        local.emit(stage='shot5_original_unknown_motion_invariance_failed',path=str(failure_path),
            maximum_unknown_component_difference_px=unknown_error,branches=branches)
        # Keep the observed strict-check failure. An arbitrary diagnostic
        # 1e-7 comparison must not alter the original source/cap acceptance.
        i=int(worst_ids[0]);j=int(ids[i]);sigma_one=src['motion_sigma'].reshape(-1,2)[j:j+1]
        single=[]
        for identity,data in [('original',base_camera),('counterfactual',trial_camera)]:
            ia,ib=int(src['ia'][j]),int(src['ib'][j]);ca=data['cameras'][ia:ia+1];cb=data['cameras'][ib:ib+1]
            value,check,trace=motion['old'].native_motion(a=src['a'][j:j+1],b=src['b'][j:j+1],
                fa=np.exp(ca[:,6]),fb=np.exp(cb[:,6]),ra=data['rotations'][ia:ia+1],rb=data['rotations'][ib:ib+1],
                da=None,db=None,ca=ca[:,:3],cb=cb[:,:3],dca_shared=None,dcb_shared=None,dca_view=None,dcb_view=None,
                pp=ca[0,8:10],k1=ca[0,11],planes=np.zeros((1,4)),jacobian=False,
                sigma=sigma_one,profile_weight=motion['profile'][j:j+1],minimum_source_depth_scene_units=motion['floor'],return_trace=True)
            single.append(dict(state=identity,raw_value=np.asarray(value).tolist(),trace=np.asarray(trace).tolist(),checks=check))
        unknown_numeric['worst_point_single_row_trace']=dict(active_LOW_point_index=j,replay=single,
            single_row_Brown_batch_stop_may_differ_from_full_batch=True)
    return dict(total_actual_pairs=len(unique),branches=branches,largest_pair_cost_changes=worst,
        new_947_subset_authority=dict(baseline_state=baseline_ref,baseline_manifest=a.ref(baseline_manifest_path),
            full_LOW_new_original_point_indices=len(added),shot5_current_points=int(corrected.sum())),
        original_outer_motion_retirement=dict(global_scalar_slots=global_slots.tolist(),same_shot_inactive_scalars=int((~active).sum()),
            original_structural_zero_cost_slots_applied=True,native_raw_values_preserved_for_invariance_only=True),
        maximum_relative_rotation_matrix_change=relative_R_error,
        maximum_target_frame_relative_translation_change_scene_units=relative_C_error,
        unknown_depth_original_numeric_covariance=unknown_numeric,
        original_native_checks=checks,unknown_depth_mathematical_relative_camera_invariance_verified=True),dict(
        motion_pairs=unique,motion_pair_counts=pair_counts,motion_pair_original_cost=original_pair_cost,
        motion_pair_trial_cost=trial_pair_cost)


def main():
    started=time.perf_counter()
    code_refs=[a.ref(__file__),a.ref(local.__file__),a.ref(ctx.__file__),
        a.ref(Path(__file__).with_name('low_shot_camera_dependent_tail_values_20261011.py'))]
    c=ctx.load_context();frozen=c.q.copy()
    ordinal=c.arrays['aux__block_names'].tolist().index('low')
    with np.load(c.path,allow_pickle=False)as saved:
        for key in ('profile__weights_'+str(ordinal),'aux__current_depth_and_material_domain_contract_json'):
            c.arrays[key]=saved[key].copy()
    shot_pts=sorted(int(p)for p in c.state.bypts if LO<=int(p)<HI)
    assert len(shot_pts)==1381
    baseline=ctx.baseline(c,set(shot_pts))
    assert len(c.groups)==187 and sum(g['scalar_rows']for g in c.groups)==3744
    local.emit(stage='shot5_complete_context_absolute_scope',exposures=len(shot_pts),groups=len(c.groups),
        scalar_rows=sum(g['scalar_rows']for g in c.groups),seconds=time.perf_counter()-started)
    trial,mapping=fixed_rotation(c,shot_pts)
    factors=local.guard_factors(c);caps=a.read(c.directory/'truth_restoration_state.json')
    base_guards,_=local.guard_values(factors,c.q)
    nominal_guard=local.guarded_report(factors,c.q,caps,base_guards)
    assert nominal_guard['passes_original_q8_caps']
    motion=local.reconstruct_motion(c);pa,pb=motion['pa'],motion['pb']
    inside_a=(pa>=LO)&(pa<HI);inside_b=(pb>=LO)&(pb<HI)
    cross=np.flatnonzero(inside_a^inside_b)
    assert len(cross)==0,'Existing cross-shot edges must not be silently dropped'
    motion['selected']=np.flatnonzero(inside_a|inside_b)
    assert len(motion['selected'])==264138
    before_motion,baseline_checks,before_camera=motion_values(c,motion,c.q)
    after_motion,trial_checks,after_camera=motion_values(c,motion,trial)
    motion_result,arrays=motion_summary(c,motion,before_motion,after_motion,before_camera,after_camera,trial_checks)
    local.emit(stage='shot5_common_rotation_original_motion',result=motion_result,seconds=time.perf_counter()-started)
    before_values=group_values(c,c.q);after_values=group_values(c,trial)
    absolute=[]
    for group,before,after in zip(c.groups,before_values,after_values):
        absolute.append(dict(family=group['family'],source_kind=group['source_kind'],id=group['id'],actual_pts=group['actual_pts'],
            physical_group=group['physical_group'],scalar_rows=group['scalar_rows'],
            original_cost=weighted_source_cost(before,group['quadratic'],group['loss']),
            trial_cost=weighted_source_cost(after,group['quadratic'],group['loss']),
            original_typed_component_p95_px=float(np.quantile(abs(before*group['scales']),.95)),
            trial_typed_component_p95_px=float(np.quantile(abs(after*group['scales']),.95))))
    trial_guards=local.guarded_report(factors,trial,caps,base_guards)
    local.emit(stage='shot5_common_rotation_original_absolute_and_guards',groups=len(absolute),
        original_cost=sum(r['original_cost']for r in absolute),trial_cost=sum(r['trial_cost']for r in absolute),
        guards={key:value for key,value in trial_guards.items()if key!='rows'},seconds=time.perf_counter()-started)
    import low_shot_camera_dependent_tail_values_20261011 as tail
    tail_context=tail.prepare_tail_context(c,shot_pts)
    tail_baseline=dict(reference=a.ref(a.H/'LOW_shot_private_camera_dependent_tail_q8_baseline_20261011.json'),
        receipts=[f['baseline_receipt']for f in tail_context.families])
    tail_values=tail.evaluate_tail_values(tail_context,trial)
    tail_domain=dict(all_FIT_families_valid=tail_values['all_families_valid'],
        persistent_FIT_and_HELD_original_domain_valid=tail_values['persistent_full_original_domain']['valid'])
    refs=code_refs+[a.ref(c.path),a.ref(SOURCE),a.ref(a.CONFIG)]
    assert refs==[a.ref(record['path'])for record in refs]
    assert np.array_equal(c.q,frozen)
    arrays.update(W_world_rotation=np.asarray(mapping['W_world_rotation']),absolute_actual_pts=np.asarray([r['actual_pts']for r in absolute]),
        absolute_original_cost=np.asarray([r['original_cost']for r in absolute]),absolute_trial_cost=np.asarray([r['trial_cost']for r in absolute]))
    array_path=OUT.with_suffix('.npz');np.savez_compressed(array_path,**arrays)
    report=dict(schema='rinkspace.frozen-q8-one-shot-common-world-rotation-value-response.v1',references=refs,
        frozen_complete_q_sha256=a.CHILD,frozen_profile_token=c.identity['frozen_profile_token'],
        profile_source_checkpoint=a.ref(c.path),latest_accepted_state_not_read=True,
        source_time_base='1/15360',source_shot=5,actual_pts_interval=[LO,HI],counterfactual=mapping,
        same_shot_original_absolute_scope_baseline=baseline,absolute_source_groups=absolute,
        motion_reconstruction=dict(coverage=motion['coverage'],authority=motion['authority'],actual_LOW_array_SHA_checks=motion['checks'],
            all_actual_exposures=len(shot_pts),all_actual_motion_points=len(motion['selected']),
            cross_shot_edges=int(len(cross)),all_incident_edges_covered_by_original_endpoints=True,
            public_perimeter_bitexact_with_saved_scene=motion['public_course_border_bitexact_with_saved_scene']),
        all_shot_motion=motion_result,baseline_all86_guard_values=nominal_guard,trial_all86_guard_values=trial_guards,
        camera_dependent_original_tail_baseline=tail_baseline,camera_dependent_original_tail_values=tail_values,
        independently_checked_tail_domain=tail_domain,
        domain=dict(original_focal_domain_bitexact_since_logf_unchanged=True,
            all_original_world_pivot_mount_lens_and_other_shot_RF4_bitexact=True,
            original_native_projection_and_positive_depth_checks_executed=True,
            complete_original_bundle_finite_extent_guard_not_yet_reconstructed=True,
            full_physical_domain_pass_not_claimed=True),
        arrays=dict(a.ref(array_path),bytes=array_path.stat().st_size,contains_pair_and_source_costs_and_W_only=True,no_q_no_dq_no_J=True),
        no_fit_no_candidate_promotion_no_decoder_no_images_no_video=True,
        limitations=['One fixed whole-shot rotation tests a mechanism; it is not a fitted model or an accuracy result.',
            'Only this specified finite W is tested; failure does not prove no other trajectory can satisfy all observations.',
            'Original source caps are distinct from complete physical ray/extents guards, and source values are not an all-source QP/KKT certificate.',
            'Original frozen profile weights are reused while unknown-depth nuisance is re-evaluated; this is not a new profile update.',
            'Engineering roll/optical-height priors are reported separately from measured material and finite-world sources.'],
        seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=a.plain,allow_nan=False)+'\n',encoding='utf8')
    local.emit(stage='shot5_common_rotation_diagnostic_complete',path=str(OUT),sha256=a.sha(OUT),seconds=report['seconds'])


if __name__=='__main__':main()
