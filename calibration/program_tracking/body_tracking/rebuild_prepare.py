"""Bind original image/body evidence to a frozen rebuild shot; CPU only."""
from pathlib import Path
import json, hashlib, time
import numpy as np
import cv2
from scipy.spatial.transform import Rotation, Slerp
from rinkspace.core.native import NativeCore
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from .body_prior import original_observation_mask


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def native_metric_binding(scene_path, snapshot, state=None):
    """Use the same post-geometry uniform conversion as the native review grid.

    The source ParameterState, not the initial scene template or an old metric
    record in a different coordinate frame, supplies the actual rink spans.
    """
    from rinkspace.domain.parameters import SceneParameterModel, ParameterState
    from rinkspace.metric_scale import estimate_uniform_scale, SQUARED_DIMENSION_ERROR
    scene_path, snapshot = Path(scene_path), Path(snapshot)
    scene = json.loads(scene_path.read_text(encoding='utf8'))
    state = load(snapshot) if state is None else state
    model = SceneParameterModel(scene, scene_path, optimize_scene=True)
    current = ParameterState(model.initial_state.layout, state['scene_values']).apply_scene(scene)
    board = np.asarray(current['board'], float)
    np.testing.assert_allclose(board, state['board'], rtol=1e-12, atol=1e-14,
                               err_msg='Snapshot board differs from its ParameterState')
    span = 2 * board[:2]
    scale = estimate_uniform_scale(span)
    return dict(schema='rinkspace.native-body-uniform-metric-binding.v1',
                policy=SQUARED_DIMENSION_ERROR,
                formula='s=(60*L+30*W)/(L*L+W*W)',
                conversion='XYZ_m = s * XYZ_scene',
                metres_per_scene_unit=scale,
                source_span_xy_scene_units=span.tolist(),
                converted_span_xy_m=(scale*span).tolist(),
                snapshot=dict(path=str(snapshot.resolve()),sha256=digest(snapshot)),
                scene=dict(path=str(scene_path.resolve()),sha256=digest(scene_path)),
                implementation='rinkspace.metric_scale.estimate_uniform_scale',
                camera_scene_or_source_observations_modified=False)


def prepare(root, run_root, snapshot, ledger, *, shot_id='s01', metres_per_unit=None):
    root, run_root = Path(root), Path(run_root)
    manifest=json.loads((run_root/'joint_manifest.json').read_text(encoding='utf8'))
    state=load(snapshot); shot=next(s for s in manifest['shots'] if s['id']==shot_id)
    scale_binding=native_metric_binding(manifest['scene_path'],snapshot,state)
    bound_scale=scale_binding['metres_per_scene_unit']
    if metres_per_unit is not None and not np.isclose(float(metres_per_unit),bound_scale,rtol=1e-12,atol=0):
        raise ValueError('Requested scale differs from the bound native uniform conversion')
    metres_per_unit=bound_scale
    times_native=np.arange(shot['first_pts'],shot['last_pts']+256,256,dtype=np.int64)
    timeline=ShotMotionTimeline(manifest['shots'],manifest['control_dt'],manifest['first_control_time'])
    native_times=(times_native-shot['reference_pts'])/12800.+timeline.offset(shot)
    core=NativeCore(manifest['native_library'])
    rig=state['rig'][shot['camera_index']].copy();controls=state['controls'].copy()
    poses=core.evaluate_motion_camera(rig,controls,timeline.first,timeline.dt,native_times)
    if poses['status'].any():raise ValueError('Frozen camera contains a failed exposure')
    cache=root/'source_cache';obs=load(cache/'observations.npz');prior=load(cache/'body_prior.npz');raw=load(cache/'raw_shot_01.npz')
    lookup={int(p):i for i,p in enumerate(obs['pts'])}
    ledger_rows=[json.loads(l) for l in Path(ledger).read_text(encoding='utf8').splitlines()]
    candidates={};legacy_to_official={}
    for e in ledger_rows:
        p=e.get('accepted_official_pts');old=int(e['full_pts'])
        if p is None or e['shot_index']!=1:continue
        legacy_to_official[old]=int(p)
        if not times_native[0]<=p<=times_native[-1] or old not in lookup:continue
        i=lookup[old]
        score=(not bool(obs['valid'][i]),not bool(prior['prior_valid'][i]),float(e['best_roi_gray_mae']))
        if p not in candidates or score<candidates[p][0]:candidates[p]=(score,i,e)
    missing=[int(p) for p in times_native if int(p) not in candidates]
    if missing:raise ValueError('Unconfirmed source exposure mappings: '+str(missing[:10]))
    oi=np.array([candidates[int(p)][1] for p in times_native]);oldpts=obs['pts'][oi]
    mask,truncated=original_observation_mask(obs)
    # This camera-neutral local pose uses only confirmed raw image exposures.
    ri=np.array([i for i,p in enumerate(raw['source_pts']) if int(p) in legacy_to_official and raw['chosen_window'][i]>=0])
    rp=np.array([legacy_to_official[int(raw['source_pts'][i])] for i in ri]);order=np.argsort(rp);ri,rp=ri[order],rp[order]
    keep=np.r_[True,np.diff(rp)>0];ri,rp=ri[keep],rp[keep]
    pv=raw['body_pose'][ri].reshape(-1,21,3);qv=raw['global_orient'][ri]
    local_pose=np.empty((len(times_native),21,3));qprior=np.empty((len(times_native),3,3));brackets=[];inferred=[]
    raw_times=(rp-shot['reference_pts'])/12800.+timeline.offset(shot)
    raw_cameras=core.evaluate_motion_camera(rig,controls,timeline.first,timeline.dt,raw_times)
    world_q=raw_cameras['rotation'].transpose(0,2,1)@Rotation.from_rotvec(qv).as_matrix()
    for i,p in enumerate(times_native):
        right=min(int(np.searchsorted(rp,p)),len(rp)-1);left=max(0,right-1)
        if rp[right]==p:left=right
        if p<rp[0]:left=right=0
        if p>rp[-1]:left=right=len(rp)-1
        alpha=0. if left==right else float((p-rp[left])/(rp[right]-rp[left]))
        if left==right:
            local_pose[i]=pv[left];qprior[i]=world_q[left]
        else:
            for j in range(21):local_pose[i,j]=Slerp([0.,1.],Rotation.from_rotvec(pv[[left,right],j]))(alpha).as_rotvec()
            qprior[i]=Slerp([0.,1.],Rotation.from_matrix(world_q[[left,right]]))(alpha).as_matrix()
        brackets.append([int(rp[left]),int(rp[right])]);inferred.append(left!=right or rp[left]!=p)
    # Cropped opening priors lack independent lower-body geometry authority.
    # A same-shot donor is explicit; source pixels are never synthesized.
    portrait=(obs['bbox_native_xyxy'][oi,3]>=1056)&(np.diff(obs['bbox_native_xyxy'][oi][:,[1,3]],axis=1)[:,0]>540)
    visibility_path=root/'contact_evidence.json'
    evidence=json.loads(visibility_path.read_text(encoding='utf8'))
    reviewed_until=evidence.get('visibility_review',{}).get('lower_body_unreliable_before_pts')
    if reviewed_until is not None:portrait|=times_native<int(reviewed_until)
    source_mask=mask[oi].copy()
    source_mask[np.ix_(portrait,np.array([11,12,13,14,15,16]))]=False
    full=np.flatnonzero(~portrait & mask[oi][:,[5,6,11,12,13,14,15,16]].all(1))
    if not len(full):raise ValueError('No full-body source observation')
    donors=np.arange(len(times_native))
    for i in np.flatnonzero(portrait):
        donor=full[np.argmin(np.abs(full-i))];local_pose[i]=local_pose[donor];qprior[i]=qprior[donor];donors[i]=donor;inferred[i]=True
    audit=json.loads((cache/'rest_stature_audit.json').read_text(encoding='utf8'))
    body_scale=audit['scale_from_bio_stature']['nominal_body_scale']
    packet=dict(pts=times_native,fit_time_seconds=(times_native-times_native[0])/12800.,motion_times=native_times,
        source_legacy_pts=oldpts,source_uv=obs['native_coco17_xy'][oi],source_score=obs['keypoint_scores'][oi],
        source_qualified=source_mask,source_bbox=obs['bbox_native_xyxy'][oi],source_crop_affine=obs['crop_affine'][oi],
        source_clipped=truncated[oi],lower_body_inferred=portrait,pose_axis_angle=local_pose,raw_root_prior_Q_world=qprior,pose_bracket_official_pts=np.array(brackets),
        pose_interpolated=np.array(inferred),pose_donor_official_pts=times_native[donors],shared_betas=prior['shared_betas'],
        body_scale=np.asarray(body_scale),metres_per_scene_unit=np.asarray(metres_per_unit),rig=rig,controls=controls,
        first_control_time=np.asarray(timeline.first),control_dt=np.asarray(timeline.dt),camera_rotation=poses['rotation'],
        camera_centre=poses['centre'],fx=poses['fx'],fy=poses['fy'],cx=poses['cx'],cy=poses['cy'])
    np.savez_compressed(root/'input.npz',**packet)
    np.savez_compressed(root/'camera_snapshot.npz',rig=rig,controls=controls,scene_values=state['scene_values'],board=state['board'],layout=state['layout'])
    report=dict(schema='rinkspace.fixed-camera-body-input.v1',shot=shot_id,source_time_base='1/12800',frames=len(times_native),
        source_seconds=[float(times_native[0]/12800),float(times_native[-1]/12800)],source_mapping='confirmed_pixel_correspondence_ledger_only',
        source_UV_domain='legacy_native_1920x1080_pixels_reused_on_confirmed_official_exposure; no old camera remapping',
        source_crop_affine_preserved=True,prior='GVHMR image-conditioned local articulation; postproc=False; no network translation',
        global_orientation='world binding established once through frozen current native camera',
        raw_clipped_frames=int(truncated[oi].sum()),lower_body_inferred_frames=int(portrait.sum()),
        visibility_review=evidence.get('visibility_review'),visibility_evidence_sha256=digest(visibility_path),source_mapping_MAE_range=[min(v[2]['best_roi_gray_mae'] for v in candidates.values()),max(v[2]['best_roi_gray_mae'] for v in candidates.values())],
        metric_scale_status='project_uniform_conversion',metres_per_scene_unit=metres_per_unit,
        scale_basis='project uniform conversion evaluated on this exact native ParameterState',scale_binding=scale_binding,
        body_scale=body_scale,body_scale_basis='existing shared-beta stature audit; approximate reported stature, not source measurement',
        frozen_inputs={str(p):digest(p) for p in [Path(snapshot),run_root/'joint_manifest.json',Path(manifest['native_library']),Path(manifest['scene_path']),Path(ledger)]},
        native_library=manifest['native_library'],scene_path=manifest['scene_path'],snapshot=str(snapshot),run_root=str(run_root),
        camera_updates_allowed=False,scene_updates_allowed=False,old_display_trajectory_used=False,GPU_body_inference_started=False)
    (root/'input.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    return report
