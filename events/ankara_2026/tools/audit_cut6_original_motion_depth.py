"""Replay one original motion FIT pair with its frozen profile, no source graph.

Only data-only NPZ/JSON inputs and the existing authenticated native residual
kernel are loaded. Static-material qualification is a separate RGB review.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
PROGRAM = ROOT/'calibration/solver_runtime/legacy/snapshot/three_frame_refit/program_tracking'
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(PROGRAM)]
sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RINKSPACE_MOTION_THREADS'):
    os.environ[name] = '1'
import cv2
import numpy as np
from z190_motion_native import native_motion, metadata

BASE = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/calibration/runs/solver_acceleration_20261003')
RAYS = Path('D:/CodexData/.codex/visualizations/2026/10/01/private-thread-00010/RinkSpace_tmp/secondary/absolute_ray_graph_v6/source_ray_matches.npz')
RAW = Path('D:/CodexData/RinkSpace_program_tracking_tmp/motion/motion_shot_06.npz')
FINITE = BASE/'prepared_full8503_1791035775270492900/complete_original_sources/secondary_source/core_source_admission_ledger.json'
PEOPLE = Path('D:/CodexData/RinkSpace_program_tracking_tmp/people/boxes.json')
PA, PB = 230509824, 230510080
UNIT = .31353729994266055


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k:serial(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [serial(v) for v in value]
    return value


def stats(value):
    x=np.asarray(value,float)
    return dict(count=len(x),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(x.max())) if len(x) else dict(count=0)


def pair_slice(z, offset_name):
    pairs=z['pair_pts']
    hits=np.flatnonzero((pairs[:,0]==PA)&(pairs[:,1]==PB))
    if len(hits)!=1: raise ValueError('One unchanged original pair required')
    j=int(hits[0]);lo,hi=map(int,z[offset_name][j:j+2])
    return j,lo,hi


def people_flags(p,q):
    packet=json.loads(PEOPLE.read_text(encoding='utf8'))
    boxes={int(r['pts']):r['boxes'] for r in packet['frames']}
    clear=np.ones(len(p),bool)
    for pts,uv in ((PA,p),(PB,q)):
        for box in boxes.get(pts,[]):
            x0,y0,x1,y1=np.asarray(box[:4])*2
            dx=max(48.,.4*(x1-x0));dy=48.
            clear &= ~((uv[:,0]>=x0-dx)&(uv[:,0]<=x1+dx)&(uv[:,1]>=y0-dy)&(uv[:,1]<=y1+dy))
    return clear,{str(pts):boxes.get(pts,[]) for pts in (PA,PB)}


def board_replay(a):
    """Existing far-board operator on the immutable same-q exported course."""
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate.native_board_track_source import current_far_board_points
    from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown
    report_path=a.output/'cut6_original_FIT_current_far_board_check_v1.json'
    if report_path.exists(): raise FileExistsError(report_path)
    prior=json.loads((a.output/'cut6_original_motion_actual_profile_check_v1.json').read_text(encoding='utf8'))
    data=np.load(a.output/'cut6_original_motion_actual_profile_arrays_v1.npz',allow_pickle=False)
    scene_path=a.state.parent/'scene_UNACCEPTED.json'
    identity=json.loads((a.state.parent/'state_identity.json').read_text(encoding='utf8'))
    if sha(scene_path)!=identity['scene_sha256']: raise ValueError('Immutable current course scene changed')
    scene=json.loads(scene_path.read_text(encoding='utf8'))
    border=np.asarray(scene['ice_boundary']['points_xyz'],float)
    ca,cb=data['camera25'];p,q=data['source_uv'],data['target_uv']
    world=current_far_board_points(ca,p,border)
    pred,depth,valid=project_central_valid_full_brown(cb,world)
    if not np.all(valid): raise ValueError('Same-q current board projection outside central-forward domain')
    xy=border[:,:2]
    if np.array_equal(xy[0],xy[-1]): xy=xy[:-1]
    edge=np.roll(xy,-1,axis=0)-xy;length=np.linalg.norm(edge,axis=1)
    t=np.clip(np.einsum('nsi,si->ns',world[:,None,:2]-xy[None,:,:],edge)/length[None,:]**2,0.,1.)
    dist=np.linalg.norm(world[:,None,:2]-(xy[None,:,:]+t[:,:,None]*edge[None,:,:]),axis=2)
    segment=np.argmin(dist,axis=1)
    normals=np.c_[edge[segment,1],-edge[segment,0],np.zeros(len(p))]/length[segment,None]
    planes=np.c_[normals,-np.einsum('ni,ni->n',normals[:,:2],xy[segment])]
    sigma=data['sigma'];weights=data['profile_weights']
    k=dict(a=p,b=q,fa=np.exp(ca[6]),fb=np.exp(cb[6]),ra=cv2.Rodrigues(ca[3:6])[0],
        rb=cv2.Rodrigues(cb[3:6])[0],ca=ca[:3],cb=cb[:3],pp=ca[8:10],k1=ca[11],
        da=None,db=None,dca_shared=None,dcb_shared=None,dca_view=None,dcb_view=None,
        planes=planes,jacobian=False,sigma=sigma,profile_weight=weights)
    r,checks,trace=native_motion(**k,minimum_source_depth_scene_units=1./UNIT,schur=True,return_trace=True)
    r=r.reshape(-1,2);delta=pred-q
    agreement=float(np.max(np.abs(r-delta)))
    if agreement>1e-6: raise ValueError('Existing finite native kernel and current far-board projection disagree')
    right=data['right_print_ROI'];clear=data['people_clear'];confirmed=np.flatnonzero(right&clear)
    # Qualification below records the viewed points only. It does not adopt
    # source-plane assignments from model intersections or residual size.
    verified=[8,12,15,16,17,19,22,23,25,26,28,30,31,34,35]
    if confirmed.tolist()!=verified: raise ValueError('Viewed source-only people-clear print-point inventory changed')
    output=dict(schema='rinkspace.original-motion-same-q-current-far-board-kernel-comparison.v1',
        q_sha256=prior['q_sha256'],source_pair_pts=[PA,PB],time_base='1/15360',
        scene=dict(path=str(scene_path),sha256=sha(scene_path)),
        immutable_rendered_current_course_reused=True,full_source_graph_constructed=False,jacobian_requested=False,
        all36_far_board_error_native_px=stats(np.linalg.norm(delta,axis=1)),
        right21_far_board_error_native_px=stats(np.linalg.norm(delta[right],axis=1)),
        source_only_reviewed15_far_board_error_native_px=stats(np.linalg.norm(delta[verified],axis=1)),
        source_only_reviewed15_actual_unknown_depth_profile_error_native_px=stats(np.linalg.norm(data['native_profile_residual'][verified],axis=1)),
        finite_native_kernel_vs_existing_far_board_operator_max_px=agreement,finite_kernel_checks=checks,
        source_only_reviewed_local_indices=verified,source_only_people_excluded_local_indices=np.flatnonzero(~clear),
        source_only_qualification='Both actual original RGBs and all labelled actual FIT points viewed. Only the listed right blue-on-white printed strokes/underscript points outside conservative people boxes are proposed. The left partial red print and padded-near-person blue points are not assigned a finite role by this packet. No advertisement number is inferred from blurred brand lettering.',
        board_branch_or_side_index_from_model_is_diagnostic_not_source_fact=True,
        observed_depth_is_not_measured=True,current_model_board_intersection_Z_scene_minmax=[float(world[:,2].min()),float(world[:,2].max())],
        points=[dict(actual_fit_local_index=i,original_global_bundle_index=27468+i,source_uv=p[i],target_uv=q[i],
            native_unknown_profile_error_px=data['native_profile_residual'][i],current_far_board_error_px=delta[i],
            current_board_source_axial_depth_working_m=float((1.-trace[i,1])/trace[i,1]*UNIT),
            diagnostic_current_course_segment=int(segment[i]),diagnostic_plane=planes[i],
            diagnostic_world=world[i],source_only_material_reviewed=i in verified)for i in range(len(p))],
        source_r_roles_weights_or_solver_unchanged=True,role_revision_not_admitted=True,accuracy_accepted=False)
    report_path.write_text(json.dumps(serial(output),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    np.savez_compressed(a.output/'cut6_original_FIT_current_far_board_arrays_v1.npz',world=world,
        predicted_uv=pred,finite_native_residual=r,finite_native_trace=trace,diagnostic_planes=planes,diagnostic_segment=segment)
    print(json.dumps(serial({k:output[k]for k in ('all36_far_board_error_native_px','right21_far_board_error_native_px',
        'source_only_reviewed15_far_board_error_native_px','source_only_reviewed15_actual_unknown_depth_profile_error_native_px',
        'finite_native_kernel_vs_existing_far_board_operator_max_px')}),ensure_ascii=False))


def role_candidate(a):
    """Freeze an enumerated source-only role proposal; no plane or FIT mutation."""
    path=a.output/'cut6_original_FIT15_actual_RGB_far_board_role_candidate_v1.json'
    if path.exists(): raise FileExistsError(path)
    previous=a.output/'cut6_original_motion_actual_profile_check_v1.json'
    report=json.loads(previous.read_text(encoding='utf8'))
    data=np.load(a.output/'cut6_original_motion_actual_profile_arrays_v1.npz',allow_pickle=False)
    reviewed=[8,12,15,16,17,19,22,23,25,26,28,30,31,34,35]
    p,q=data['source_uv'],data['target_uv']
    if not np.all(data['people_clear'][reviewed]): raise ValueError('Reviewed point overlaps preserved conservative people box')
    raw_frames=[]
    for pts in (PA,PB):
        frame=a.rgb_dir/f'native_{pts}.png'
        raw_frames.append(dict(source_pts=pts,path=str(frame),sha256=sha(frame)))
    original=report['actual_fit_source'];rows=[]
    for i in reviewed:
        rows.append(dict(original_global_bundle_index=original['global_point_slice'][0]+i,
            original_pair_index=original['pair_index'],original_local_index=i,
            source_pts=PA,target_pts=PB,source_uv=p[i],target_uv=q[i],split='fit',
            status='source_qualified_far_inner_board_material_candidate_not_admitted',
            proposed_surface_kind='current_finite_far_inner_board_course',
            previous_surface_kind='untyped_positive_source_depth_profile',
            physical_material='same right blue-on-white printed stroke or grey underscript on visible inner board',
            advertisement_number=None,both_end_original_source_reviewed=True,
            source_only_people_clear=True,source_UV_unchanged=True,source_sigma_px=data['sigma'][i],
            no_new_source_point_or_scalar_row=True,model_projection_used_to_generate_or_select_uv=False))
    candidate=dict(schema='rinkspace.original-motion-explicit-source-material-role-candidate.v1',
        event_id='ankara_2026',programme_id='okada_mei_sp',status='independent_source_review_ready_for_parent_role_merge',
        source_video_sha256='f685d27882e78f2446c734ceee4321dcb275a5ad1ebf81ea90bca68c2fcec891',
        time_base='1/15360',original_motion_graph=dict(path=original['path'],sha256=original['sha256']),
        source_frames=raw_frames,source_inspection_images=report['RGB_point_overlays'],
        source_inspection='Both raw RGBs and labelled 36 actual FIT queries visually inspected before role admission; only the enumerated 15 points are proposed. Same static printed strokes remain visible at both endpoints. Severe horizontal blur limits exact pixel reading. Partial red print, people margins, blank-board ROIs and unclear brand number are not assigned new roles.',
        conservative_people_evidence=report['people_mask'],
        original_FIT_HELD_pts_UV_sigma_and_order_preserved=True,
        frozen_profile_weights_not_reweighted_by_candidate=True,
        proposed_operator_revision='Use source-qualified current finite board depth for these existing original motion rows; retain every other original motion row and unknown-depth branch.',
        new_source_scalar_rows=0,source_role_revision_adopted=False,
        observed_world_or_numeric_plane_supplied=False,
        plane_assignment_policy='Current named common-course side/plane is solver geometry. Do not turn the frozen q intersection or its distance into a measurement, and do not give all ROI points one guessed plane.',
        required_consumer='Existing original motion finite_planes/finite_records producer and current course plane/extent/J synchronizer, with original global-index/PTS/UV/hash binding.',
        excluded_actual_pair_local_indices=[i for i in range(len(p)) if i not in reviewed],records=rows,
        reference_diagnostic=dict(path=str(previous),sha256=sha(previous),residual_not_used_for_source_selection=True),
        accuracy_accepted=False)
    path.write_text(json.dumps(serial(candidate),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(path=str(path),sha256=sha(path),source_qualified_original_points=len(rows),new_scalar_rows=0),ensure_ascii=False))


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--state',required=True,type=Path)
    cli.add_argument('--rgb-dir',required=True,type=Path)
    cli.add_argument('--output',required=True,type=Path)
    cli.add_argument('--board-replay',action='store_true')
    cli.add_argument('--role-candidate',action='store_true')
    a=cli.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.board_replay:
        board_replay(a)
        return
    if a.role_candidate:
        role_candidate(a)
        return
    report_path=a.output/'cut6_original_motion_actual_profile_check_v1.json'
    if report_path.exists(): raise FileExistsError(report_path)
    identity=json.loads((a.state.parent/'state_identity.json').read_text(encoding='utf8'))
    if sha(a.state)!=identity['state_npz_sha256']: raise ValueError('Immutable state bytes changed')
    state=np.load(a.state,allow_pickle=False);fullq=state['q']
    if hashlib.sha256(np.ascontiguousarray(fullq).view(np.uint8)).hexdigest()!=identity['q_sha256']:
        raise ValueError('Immutable complete state identity changed')
    cam_path=Path(identity['camera_path'])
    if sha(cam_path)!=identity['camera_sha256']: raise ValueError('Immutable native camera bytes changed')
    cams=np.load(cam_path,allow_pickle=False)
    ca,cb=[cams['camera25'][np.flatnonzero(cams['pts']==s)[0]] for s in (PA,PB)]
    if (not np.array_equal(ca[8:10],cb[8:10]) or ca[11]!=cb[11]
            or np.any(ca[12:]!=0) or np.any(cb[12:]!=0)
            or ca[6]!=ca[7] or cb[6]!=cb[7] or ca[10]!=0 or cb[10]!=0):
        raise ValueError('Original constant-PP/k1 scalar-f motion operator required')
    rays=np.load(RAYS,allow_pickle=False);pair,lo,hi=pair_slice(rays,'offset')
    p,q=rays['source_uv'][lo:hi],rays['target_uv'][lo:hi]
    finite=json.loads(FINITE.read_text(encoding='utf8'))['source_plane_binding']
    typed={int(r['global_bundle_index']):r for r in finite}
    if any(i in typed for i in range(lo,hi)):
        raise ValueError('This pair now has finite source roles: explicit current-plane replay required')
    names=state['aux__block_names'].tolist();block=names.index('secondary')
    profiles=state[f'profile__weights_{block}'];present=bool(state[f'profile__present_{block}'])
    if len(profiles)!=len(rays['source_uv']): raise ValueError('Original profile/motion row inventory changed')
    weights=profiles[lo:hi] if present else None
    sigma=np.full((len(p),2),2.) # Original secondary producer's unchanged motion_sigma.
    domain=json.loads(str(state['aux__current_depth_and_material_domain_contract_json'].item()))
    record=next(r for r in domain['blocks'] if r['block']=='secondary')
    floor=float(record['candidate_minimum_source_Z_m'])/UNIT
    k=dict(a=p,b=q,fa=np.exp(ca[6]),fb=np.exp(cb[6]),ra=cv2.Rodrigues(ca[3:6])[0],
        rb=cv2.Rodrigues(cb[3:6])[0],ca=ca[:3],cb=cb[:3],pp=ca[8:10],k1=ca[11],
        da=None,db=None,dca_shared=None,dcb_shared=None,dca_view=None,dcb_view=None,
        planes=np.zeros((len(p),4)),jacobian=False,sigma=sigma,profile_weight=weights)
    residual,checks,trace=native_motion(**k,minimum_source_depth_scene_units=floor,schur=True,return_trace=True)
    native=residual.reshape(-1,2)
    rho=np.divide(trace[:,1],1.-trace[:,1],out=np.full(len(p),np.inf),where=trace[:,1]<1.)
    z=np.divide(1.,rho,out=np.full(len(p),np.inf),where=rho>0.)*UNIT
    # A labelled counterfactual isolates the depth contribution; it never enters FIT.
    kp=dict(k,cb=ca[:3])
    infinity,_,_=native_motion(**kp,minimum_source_depth_scene_units=floor,schur=True,return_trace=True)
    raw=np.load(RAW,allow_pickle=False);rawpair,rlo,rhi=pair_slice(raw,'offsets')
    rp,rq=raw['xy0'][rlo:rhi],raw['xy1'][rlo:rhi]
    matches=[]
    for x,y in zip(p,q):
        hit=np.flatnonzero(np.all(rp==x,axis=1)&np.all(rq==y,axis=1))
        if len(hit)>1: raise ValueError('Ambiguous identical original raw pair rows')
        # These are independently extracted source archives, not necessarily
        # a subset relation. Never relabel a nearest raw point as an exact FIT.
        matches.append(int(hit[0]) if len(hit) else -1)
    right=(p[:,0]>=1060)&(p[:,0]<1880)&(p[:,1]>=340)&(p[:,1]<575)&(q[:,0]>=1060)&(q[:,0]<1880)&(q[:,1]>=340)&(q[:,1]<575)
    clear,boxes=people_flags(p,q)
    points=[]
    for i in range(len(p)):
        points.append(dict(actual_fit_pair_local_index=i,original_global_bundle_index=lo+i,
            raw_pair_local_index=matches[i] if matches[i]>=0 else None,source_uv=p[i],target_uv=q[i],
            source_sigma_native_px=sigma[i],frozen_profile_weight=None if weights is None else weights[i],
            finite_plane_present=False,native_profile_residual_px=native[i],
            native_profile_residual_norm_px=float(np.linalg.norm(native[i])),
            normalized_actual_factor_residual=native[i]/sigma[i],
            profiled_source_axial_depth_working_m=float(z[i]) if np.isfinite(z[i]) else None,
            inverse_depth_scene=float(rho[i]),profile_interior=bool(trace[i,9]),profile_boundary=bool(trace[i,10]),
            source_only_right_print_ROI=bool(right[i]),conservative_people_clear=bool(clear[i]),
            physical_material_RGB_qualification='pending_independent_actual_RGB_point_review'))
    # Keep all actual FIT points visible. Colors identify source-only people masks.
    images=[]
    for pts,uv in ((PA,p),(PB,q)):
        path=a.rgb_dir/f'native_{pts}.png';image=cv2.imread(str(path))
        if image is None or image.shape!=(1080,1920,3): raise ValueError('Preserved actual RGB required')
        for i,xy in enumerate(uv):
            loc=tuple(np.rint(xy).astype(int));color=(0,255,0) if clear[i] else (0,128,255)
            cv2.circle(image,loc,5,color,1,cv2.LINE_AA)
            cv2.putText(image,str(i),(loc[0]+6,loc[1]-4),cv2.FONT_HERSHEY_SIMPLEX,.48,color,1,cv2.LINE_AA)
        dest=a.output/f'cut6_actual_FIT_points_{pts}.png';cv2.imwrite(str(dest),image)
        images.append(dict(path=str(dest),sha256=sha(dest),native_raw_path=str(path),native_raw_sha256=sha(path)))
    np.savez_compressed(a.output/'cut6_original_motion_actual_profile_arrays_v1.npz',
        source_uv=p,target_uv=q,native_profile_residual=native,native_infinite_depth_counterfactual=infinity.reshape(-1,2),
        native_trace=trace,sigma=sigma,profile_weights=np.ones_like(sigma) if weights is None else weights,
        raw_pair_local_indices=np.asarray(matches),people_clear=clear,right_print_ROI=right,camera25=np.asarray([ca,cb]))
    report=dict(schema='rinkspace.original-motion-one-pair-frozen-profile-kernel-replay.v1',
        source_pair_pts=[PA,PB],time_base='1/15360',q_sha256=identity['q_sha256'],frozen_profile_token=identity['frozen_profile_token'],
        state=dict(path=str(a.state),sha256=sha(a.state)),camera=dict(path=str(cam_path),sha256=sha(cam_path)),
        actual_fit_source=dict(path=str(RAYS),sha256=sha(RAYS),pair_index=pair,global_point_slice=[lo,hi],points=len(p)),
        preserved_raw_motion=dict(path=str(RAW),sha256=sha(RAW),pair_index=rawpair,global_point_slice=[rlo,rhi],points=len(rp)),
        exact_raw_pair_row_matches=int(np.sum(np.asarray(matches)>=0)),
        actual_FIT_is_not_claimed_a_subset_of_raw_tracking_archive=True,
        finite_role_inventory=dict(path=str(FINITE),sha256=sha(FINITE),all_finite_global_indices=sorted(typed),this_pair_finite_points=0),
        original_motion_sigma_px=2.,profile_present=present,minimum_source_axial_depth_working_m=floor*UNIT,
        native_kernel=metadata(),checks=checks,profile_residual_native_px=stats(np.linalg.norm(native,axis=1)),
        right_ROI_actual_FIT_points=int(right.sum()),right_ROI_profile_residual_native_px=stats(np.linalg.norm(native[right],axis=1)),
        infinite_depth_counterfactual_native_px=stats(np.linalg.norm(infinity.reshape(-1,2),axis=1)),
        counterfactual_scope='Optical-center displacement set to zero only to expose depth absorption; this camera is not adopted.',
        profile_depth_working_m=stats(z[np.isfinite(z)]),profile_floor_points=int(np.sum(np.isclose(z,floor*UNIT,rtol=1e-8))),
        people_mask=dict(path=str(PEOPLE),sha256=sha(PEOPLE),boxes=boxes,excluded_local_indices=np.flatnonzero(~clear)),
        RGB_point_overlays=images,points=points,source_UV_sigma_roles_FIT_HELD_weights_or_solver_not_modified=True,
        full_source_graph_constructed=False,jacobian_requested=False,video_generated=False,accuracy_accepted=False)
    report_path.write_text(json.dumps(serial(report),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(serial({k:report[k] for k in ('actual_fit_source','profile_residual_native_px','right_ROI_actual_FIT_points',
        'right_ROI_profile_residual_native_px','infinite_depth_counterfactual_native_px','profile_depth_working_m','profile_floor_points')}),ensure_ascii=False))


if __name__=='__main__':main()
