"""Two saved states, original motion values and worst-pair parameter replay.

Only main4/main9 are evaluated. No J, optimizer, image or report is written.
The saved f300 profile is used once inside the original nuisance kernel.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import time
import cv2

import compare_complete_original_source_states_20261011 as comparison
import joint_absolute_source_context_20261011 as ctx
from structured_programme_source_candidate.original_motion_far_board_roles import OriginalMotionFarBoardRoleBinding
from shared_board_course_candidate.board_course_material import vertical_plane_value_jacobian

a,np=ctx.a,ctx.np
PARENT='8a15a4ed0c82beb490993db9c58294af592096e8159d9652b5bd5c50217c71eb'
AFTER='0884a8fb622a47f2f3742c5f7a33e102fd1a3a7e1e971310f5dfca641eabebb1'
RUN=a.R/'okada_review86_q8_trial_fraction_SOC_fix_20261011'
PROFILE='f3008d040fa4500af71b6d8ad0a4f13616613bf8f9963d7d178c7fd6241c0984'


def emit(**value):print(json.dumps(value,ensure_ascii=False,default=a.plain),flush=True)


def cost_rows(raw,sigma):
    r=raw/sigma
    return np.where(abs(r)<=2.,.5*r*r,2.*abs(r)-2.).sum(axis=1)


def current_border(c,value):
    course,q,_=c.provider.print_course_binding(value[c.world_indices])
    return np.asarray(course.sample_perimeter(q,with_jacobian=False)[0])


def public_camera_inventory(c,m,value):
    camera=np.asarray([ctx.camera(c,value,int(p))for p in m['stamps']])
    rotation=np.asarray([cv2.Rodrigues(v[3:6])[0]for v in camera])
    if not np.array_equal(camera[:,8:10],np.broadcast_to(camera[0,8:10],camera[:,8:10].shape)) or not np.all(camera[:,11]==camera[0,11]):
        raise ValueError('Original per-block intrinsics are not shared')
    return camera,rotation


def native_values(c,m,value,ids,*,inventory=None):
    ids=np.asarray(ids,int);src=m['source']
    if inventory is None:
        # A worst pair uses only its two exact exposure cameras.
        views=np.unique(np.r_[src['ia'][ids],src['ib'][ids]])
        camera={int(i):ctx.camera(c,value,int(m['stamps'][i]))for i in views}
        rotation={i:cv2.Rodrigues(cam[3:6])[0]for i,cam in camera.items()}
        ca=np.asarray([camera[int(i)]for i in src['ia'][ids]])
        cb=np.asarray([camera[int(i)]for i in src['ib'][ids]])
        ra=np.asarray([rotation[int(i)]for i in src['ia'][ids]])
        rb=np.asarray([rotation[int(i)]for i in src['ib'][ids]])
    else:
        camera,rotation=inventory;ca,cb=camera[src['ia'][ids]],camera[src['ib'][ids]]
        ra,rb=rotation[src['ia'][ids]],rotation[src['ib'][ids]]
    border=current_border(c,value);edge=border[:-1]if np.array_equal(border[0],border[-1])else border
    planes=m['base_planes'][ids].copy();zero=np.zeros((3,0))
    for view in np.unique(src['ia'][ids[m['board'][ids]]]):
        positions=np.flatnonzero(m['board'][ids]&(src['ia'][ids]==view));course=SimpleNamespace(sample_perimeter=lambda q,with_jacobian=False:(border,))
        selector=SimpleNamespace(_facet_course_cache=None,records=[None]*len(positions),source_row_groups={0:np.arange(len(positions))},
            source_cameras={0:ca[positions[0]]},source_uv=src['a'][ids[positions]],exact_cache_statistics=dict(facet_group_hits=0,facet_group_misses=0))
        facets=OriginalMotionFarBoardRoleBinding._current_source_facets(selector,course,None,('public_current_border',comparison.token(border)))
        planes[positions]=np.asarray([vertical_plane_value_jacobian(edge[i],edge[(i+1)%len(edge)],zero,zero)[0]for i in facets])
    sigma=src['motion_sigma'].reshape(-1,2)[ids]
    raw,check=m['old'].native_motion(a=src['a'][ids],b=src['b'][ids],fa=np.exp(ca[:,6]),fb=np.exp(cb[:,6]),
        ra=ra,rb=rb,ca=ca[:,:3],cb=cb[:,:3],da=None,db=None,dca_shared=None,dcb_shared=None,dca_view=None,dcb_view=None,
        pp=ca[0,8:10],k1=ca[0,11],planes=planes,sigma=sigma,profile_weight=m['profile'][ids],jacobian=False,
        minimum_source_depth_scene_units=m['floor'])
    raw=np.asarray(raw).reshape(-1,2)
    relative_rotation=np.einsum('nij,nkj->nik',rb,ra)
    relative_translation=np.einsum('nij,nj->ni',rb,ca[:,:3]-cb[:,:3])
    return raw,check,dict(relative_rotation=relative_rotation,relative_translation=relative_translation,
        source_camera=ca[0],target_camera=cb[0],planes=planes)


def full_values(c,m,value):
    inventory=public_camera_inventory(c,m,value);raw=np.empty_like(m['source']['a']);checks=[]
    for begin in range(0,len(raw),20000):
        ids=np.arange(begin,min(begin+20000,len(raw)));raw[ids],check,_=native_values(c,m,value,ids,inventory=inventory);checks.append(check)
    if not np.isfinite(raw).all():raise ValueError('Nonfinite original motion value')
    return raw,checks


def pair_replay(c,m,value,ids,before,after):
    sigma=m['source']['motion_sigma'].reshape(-1,2)[ids];dq=value-c.q;out=[];reference=None
    samples={}
    for fraction in (0.,.001,.5,1.):
        q=c.q+fraction*dq;raw,check,meta=native_values(c,m,q,ids);samples[fraction]=raw
        if fraction==0.:reference=meta
        delta_t=meta['relative_translation']-reference['relative_translation']
        rows=cost_rows(raw,sigma);unknown=~m['source']['finite_mask'][ids]
        out.append(dict(fraction=float(fraction),cost=float(rows.sum()),unknown_cost=float(rows[unknown].sum()),
            finite_board_cost=float(rows[m['board'][ids]].sum()),finite_other_cost=float(rows[m['source']['finite_mask'][ids]&~m['board'][ids]].sum()),
            native_norm_p50_p95_max=np.quantile(np.linalg.norm(raw,axis=1),[.5,.95,1]).tolist(),
            target_frame_relative_translation_scene_units=meta['relative_translation'][0].tolist(),
            relative_translation_norm_scene_units=float(np.linalg.norm(meta['relative_translation'][0])),
            relative_translation_change_norm_scene_units=float(np.linalg.norm(delta_t[0])),
            relative_rotation_matrix_change_max=float(abs(meta['relative_rotation']-reference['relative_rotation']).max()),
            source_focal_px=float(np.exp(meta['source_camera'][6])),target_focal_px=float(np.exp(meta['target_camera'][6])),native_checks=check))
    first_order=(samples[.001]-samples[0.])/.001
    emit(stage='worst_pair_real_complete_parameter_fraction_values',source_actual_pts=int(m['pa'][ids[0]]),target_actual_pts=int(m['pb'][ids[0]]),
        points=len(ids),profile_frozen=PROFILE,fractions=out,
        endpoint_vs_1e_minus3_secant_prediction_native_norm_p95_max=np.quantile(np.linalg.norm(samples[1.]-samples[0.]-first_order,axis=1),[.95,1]).tolist(),
        fresh_pair_batch_vs_full_batch_baseline_native_max=float(abs(samples[0.]-before[ids]).max()),
        fresh_pair_batch_vs_full_batch_after_native_max=float(abs(samples[1.]-after[ids]).max()),
        camera_from_complete_q_not_interpolated_external_R_C=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--after-state',type=Path,default=RUN/'accepted_states'/AFTER);args=ap.parse_args();started=time.perf_counter()
    directory=args.after_state;run=directory.parent.parent;parent=run/'initial_states'/PARENT
    identities=[a.read(d/'state_identity.json')for d in (parent,directory)]
    for d,i in zip((parent,directory),identities):
        if i['frozen_profile_token']!=PROFILE:raise ValueError('This causal replay requires the actual saved f300 frozen profile')
        if a.sha(d/'complete_state_UNACCEPTED.npz')!=i['state_npz_sha256']:raise ValueError('Saved state changed')
    arrays=[comparison.state_arrays(d/'complete_state_UNACCEPTED.npz')for d in (parent,directory)]
    for k in arrays[0]:
        if k!='q':np.testing.assert_array_equal(arrays[0][k],arrays[1][k],err_msg=k)
    c=ctx.load_context();np.testing.assert_array_equal(c.q,c.state.evaluation_q(arrays[0]['q']))
    value=c.state.evaluation_q(arrays[1]['q']);motions,receipt=comparison.reconstruct_all_motion(c,arrays[0])
    emit(stage='same_original_sources_and_f300_profile_ready',source_hash_checks_all_exact=all(
        item['exact']for block in receipt['source_hash_checks'].values()for item in block.values()),
        source_profile_not_taken_from_old_archived_430_checkpoint=True,seconds=time.perf_counter()-started)
    result=[]
    for name in ('main4','main9'):
        m=motions[name];src=m['source'];sigma=src['motion_sigma'].reshape(-1,2)
        if not np.isfinite(m['profile']).all()or not np.all(m['profile']>0):raise ValueError('Invalid actual saved source profile')
        emit(stage='original_motion_values_started',block=name,points=len(sigma),frozen_profile_range=[float(m['profile'].min()),float(m['profile'].max())])
        before,checks0=full_values(c,m,c.q);after,checks1=full_values(c,m,value);cost0,cost1=cost_rows(before,sigma),cost_rows(after,sigma)
        branch=[]
        for kind,mask in [('unknown_depth',~src['finite_mask']),('finite_board',m['board']),('finite_other',src['finite_mask']&~m['board'])]:
            if mask.any():branch.append(dict(kind=kind,points=int(mask.sum()),before=float(cost0[mask].sum()),after=float(cost1[mask].sum()),delta=float((cost1-cost0)[mask].sum()),
                native_norm_p95_before=float(np.quantile(np.linalg.norm(before[mask],axis=1),.95)),native_norm_p95_after=float(np.quantile(np.linalg.norm(after[mask],axis=1),.95))))
        pairs=[]
        for pa,pb in sorted(set(zip(m['pa'].tolist(),m['pb'].tolist()))):
            ids=np.flatnonzero((m['pa']==pa)&(m['pb']==pb));unknown=~src['finite_mask'][ids]
            pairs.append(dict(source_actual_pts=int(pa),target_actual_pts=int(pb),points=len(ids),before=float(cost0[ids].sum()),after=float(cost1[ids].sum()),
                delta=float((cost1-cost0)[ids].sum()),unknown_delta=float((cost1-cost0)[ids][unknown].sum()),finite_delta=float((cost1-cost0)[ids][~unknown].sum())))
        ordered=sorted(pairs,key=lambda r:r['delta'],reverse=True);positive=sum(max(r['delta'],0.)for r in pairs)
        summary=dict(block=name,points=len(sigma),before=float(cost0.sum()),after=float(cost1.sum()),delta=float((cost1-cost0).sum()),branches=branch,
            source_pairs=len(pairs),improved_pairs=sum(r['delta']<0 for r in pairs),worsened_pairs=sum(r['delta']>0 for r in pairs),
            positive_pair_delta=positive,top1_top3_top10_positive_delta_fractions=[sum(max(r['delta'],0.)for r in ordered[:n])/positive for n in (1,3,10)],
            worst_pairs=ordered[:5],best_pairs=ordered[-3:],structural_retirement_scalars=0,
            native_batches_before=len(checks0),native_batches_after=len(checks1),profile_finite_positive=True,source_sigma_once_and_profile_not_outer_weight=True)
        result.append(summary);emit(stage='motion_original_classification_partition',**summary)
        worst=ordered[0];ids=np.flatnonzero((m['pa']==worst['source_actual_pts'])&(m['pb']==worst['target_actual_pts']))
        pair_replay(c,m,value,ids,before,after)
    emit(stage='two_block_value_causal_partition_complete',blocks=[r['block']for r in result],seconds=time.perf_counter()-started,
        no_J_no_fit_no_report_no_images=True,source_profile_token=PROFILE)


if __name__=='__main__':main()
