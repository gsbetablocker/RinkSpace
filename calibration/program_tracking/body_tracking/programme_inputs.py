"""Prepare a self-contained official-clock programme from explicit saved inputs.

Legacy poses supply numerical local-pose seeds only when their old K differs.
No network world translation or per-shot XYZ alignment is imported.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,copy,json,hashlib,os
import numpy as np
from scipy.spatial.transform import Rotation,RotationSpline
from .rebuild_prepare import load,digest
from .programme_contract import required_clock
from .programme_camera import ProgrammeCamera
from .action_clip_input import map_observations,camera25_from_snapshot
from .camera_bridge import native_to_rectified,rectified_intrinsics
from .blade_current_reference import restore_current_reference
from .fullbody_state import matrices
from .fullbody_round import write_json

PROGRAMME=Path('events/ankara_2026/programmes/okada_mei_sp')

def rebind_source_slots(document,pts):
    """Reindex storage slots by exact PTS; source UV/world/time never changes."""
    lookup={int(p):i for i,p in enumerate(pts)};changes=[]
    def visit(value,stamp=None):
        if isinstance(value,dict):
            stamp=value.get('pts',stamp)
            if 'camera_row_time_seconds'in value and'frames'in value:
                if stamp not in lookup:raise ValueError('Source mapping outside programme clock')
                old=list(value['frames']);value['frames']=[lookup[stamp]]*len(old)
                changes.append(dict(pts=int(stamp),old_frame_slots=old,new_frame_slot=lookup[stamp]))
            for key,item in value.items():
                if key!='source_slot_rebinding':visit(item,stamp)
        elif isinstance(value,list):
            for item in value:visit(item,stamp)
    if'source_support_locations'in document:visit(document['source_support_locations'])
    document['source_slot_rebinding']=dict(changes=changes,raw_UV_and_world_mappings_and_row_times_unchanged=True)
    return document

def root_from_pixels(camera,local,Q,uv,use,score):
    R=camera.midpoint['rotation'];C=camera.midpoint['centre']*camera.scale
    relative=np.einsum('nij,nkj->nki',R@Q,local);cam25=camera25_from_snapshot(camera.midpoint)
    P=np.full((len(uv),3),np.nan);valid=np.zeros(len(uv),bool)
    for i in range(len(P)):
        keep=use[i]&np.isfinite(uv[i]).all(-1)
        if keep.sum()<4:continue
        xy,ok=native_to_rectified(uv[i,keep],cam25[i]);points=relative[i,keep][ok]
        if len(points)<4:continue
        K=rectified_intrinsics(cam25[i]);rays=(xy[ok]-K[:2,2])/np.diag(K)[:2];w=np.sqrt(np.clip(score[i,keep][ok],.01,1))
        A=np.zeros((len(points)*2,3));A[0::2,0]=1;A[1::2,1]=1;A[0::2,2]=-rays[:,0];A[1::2,2]=-rays[:,1]
        b=np.empty(len(A));b[0::2]=rays[:,0]*points[:,2]-points[:,0];b[1::2]=rays[:,1]*points[:,2]-points[:,1]
        x=np.linalg.lstsq(A*np.repeat(w,2)[:,None],b*np.repeat(w,2),rcond=None)[0]
        if x[2]>0 and np.isfinite(x).all():P[i]=C[i]+R[i].T@x;valid[i]=True
    if valid.sum()<4:raise ValueError('No positive-depth numerical root initialization')
    for j in range(3):P[:,j]=np.interp(np.arange(len(P)),np.flatnonzero(valid),P[valid,j])
    return P,valid,cam25

def prepare(project,base,root):
    project,base,root=map(Path,(project,base,root));out=root/'programme'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    scope_path=project/PROGRAMME/'rebuild_scope.json';scope=json.loads(scope_path.read_text(encoding='utf-8'));pts,shots,tb=required_clock(scope);times=(pts-pts[0])*tb[0]/tb[1];n=len(pts)
    parent_stage=base/'support_correction_1106/joint_grounded';parent_param=project/PROGRAMME/'spin_support_grounded_1106.npz'
    shared,_,shared_info,saved=restore_current_reference(project,parent_stage,parent_param)
    observed=load(base/'source_cache/observations.npz')
    ledger_path=Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/history/tracking_runs/source_cadence_comparison/O6j8L8TmfKo/full_programme_alignment/native60_to_official50.jsonl')
    ledger=[json.loads(v)for v in ledger_path.read_text(encoding='utf-8').splitlines()]
    native={key:[]for key in ('body_uv','body_score','body_use','bbox','source_mapping_missing','source_legacy_pts','source_mapping_mae')}
    for shot in dict.fromkeys(shots.tolist()):
        ix=np.flatnonzero(shots==shot);piece=map_observations(observed,ledger,pts[ix],shot)
        for key in native:native[key].append(piece[key])
    native={k:np.concatenate(v)for k,v in native.items()}
    p=dict(pts=pts,time_base=np.array(tb),source_time_base=np.array(tb),fit_time_seconds=times,shared_betas=shared.p['shared_betas'],body_scale=shared.p['body_scale'],metres_per_scene_unit=shared.p['metres_per_scene_unit'],source_uv=native['body_uv'],source_score=native['body_score'],source_qualified=native['body_use'],source_bbox=native['bbox'],lower_body_inferred=native['source_mapping_missing'],source_identity_suspect=native['source_mapping_missing'],shot_ids=shots,image_height=np.array(1080))
    camera=ProgrammeCamera(project,p,shots);p.update(camera.p);p['camera_rotation']=camera.midpoint['rotation'];p['camera_centre']=camera.midpoint['centre'];p['camera_values']=camera.midpoint['values']
    mapping={int(row['full_pts']):int(row['accepted_official_pts'])for row in ledger if row.get('accepted_official_pts')is not None}
    local=np.empty((n,21,3,3));Q=np.empty((n,3,3));prior_inferred=np.ones(n,bool);compatibility=[]
    for shot in dict.fromkeys(shots.tolist()):
        ix=np.flatnonzero(shots==shot);raw_path=root/'inputs/legacy_priors'/('raw_shot_'+shot[1:]+'.npz');raw=load(raw_path);stamps=[];indices=[]
        for j,stamp in enumerate(raw['source_pts']):
            target=mapping.get(int(stamp))
            if target is not None and pts[ix[0]]<=target<=pts[ix[-1]]and raw['chosen_window'][j]>=0:stamps.append(target);indices.append(j)
        order=np.argsort(stamps);stamps=np.asarray(stamps,dtype=np.int64)[order];indices=np.asarray(indices)[order];keep=np.r_[True,np.diff(stamps)>0];stamps,indices=stamps[keep],indices[keep]
        if len(stamps)<2:raise ValueError('No reusable same-cut local-pose seed '+shot)
        tt=(stamps-pts[0])/12800.;query=np.clip(times[ix],tt[0],tt[-1]);rot=matrices(raw['body_pose'][indices].reshape(-1,21,3))
        local[ix]=np.stack([RotationSpline(tt,Rotation.from_matrix(rot[:,j]))(query).as_matrix()for j in range(21)],axis=1)
        incam=RotationSpline(tt,Rotation.from_rotvec(raw['global_orient'][indices]))(query).as_matrix();Q[ix]=camera.midpoint['rotation'][ix].swapaxes(-1,-2)@incam
        prior_inferred[ix]=~np.isin(pts[ix],stamps)
        current_K=np.stack([rectified_intrinsics(row)for row in camera25_from_snapshot({k:v[ix[np.searchsorted(pts[ix],stamps)]] for k,v in camera.midpoint.items()}).reshape(-1,25)])
        kd=float(np.max(abs(current_K-raw['K_rect'][indices])))
        compatibility.append(dict(shot=shot,path=str(raw_path),sha256=digest(raw_path),same_source_mapping_slots=len(stamps),current_K_max_difference_px=kd,K_compatible=kd<1e-6,role='same-image local pose seed only; current camera and source pixels refit world; no network translation imported',prior_extrapolated_native_pts=pts[ix][(times[ix]<tt[0])|(times[ix]>tt[-1])].tolist()))
    # Current, already checked native caches replace their matching observations
    # and local poses; no detector UV is interpolated into absent source slots.
    caches=[base/'input.npz',base/'action_integration_1103/jump_s02/input.npz',parent_stage/'common/exmr_hand_nofoot/input.npz']
    for cache in caches:
        a=load(cache);inside=np.isin(a['pts'],pts);ix=np.searchsorted(pts,a['pts'][inside]);local[ix]=matrices(a['pose_axis_angle'][inside]);Q[ix]=a['raw_root_prior_Q_world'][inside]
        for key,target in [('source_uv','source_uv'),('source_score','source_score'),('source_qualified','source_qualified'),('source_bbox','source_bbox')]:p[target][ix]=a[key][inside]
    p['pose_axis_angle']=Rotation.from_matrix(local.reshape(-1,3,3)).as_rotvec().reshape(n,21,3);p['raw_root_prior_Q_world']=Q
    coco=shared.body.forward_rotations(local)[0];P,root_valid,cam25=root_from_pixels(camera,coco,Q,p['source_uv'],p['source_qualified'],p['source_score'])
    hands=np.broadcast_to(shared.body.default[3:].reshape(2,15,3,3),(n,2,15,3,3)).copy()if shared.body.default.shape==(33,3,3)else None
    # Saved current S05 supplies the middle initialization, not fixed boundaries.
    s05=load(parent_stage/'common/exmr_hand_nofoot/fitted/result.npz');ix=np.searchsorted(pts,s05['pts']);ratio=shared.binding.ratio
    P[ix]=s05['P_world_mid']/ratio;Q[ix]=s05['Q_world_mid'];local[ix]=s05['local_R_mid']
    if hands is not None and'hand_local_R_mid'in s05:hands[ix]=s05['hand_local_R_mid']
    obs=load(parent_stage/'common/exmr_hand_nofoot/observations.npz');use=p['source_qualified'].copy();uv=p['source_uv'].copy();sigma=np.maximum(2.,8.*(1-np.clip(np.nan_to_num(p['source_score']),0,1)));use[ix]=obs['use'];uv[ix]=obs['uv'];sigma[ix]=obs['sigma_px']
    # Preserve the user's accepted S01 as a separate exact initialization/control.
    from rinkspace.viewer3d.data import ResultCatalog
    s01_record=json.loads((project/PROGRAMME/'skate_s01_1103.json').read_text(encoding='utf-8'));config=s01_record['viewer_config'];config=config['path']if isinstance(config,dict)else config
    catalog=ResultCatalog(project=project,config=project/config);variant=catalog.variant(s01_record['variant_id']);a=load(catalog.result_path(variant));ii=np.searchsorted(pts,a['pts']);P[ii]=a['P_world_mid']/ratio;Q[ii]=a['Q_world_mid'];local[ii]=a['local_R_mid']
    p['pose_axis_angle']=Rotation.from_matrix(local.reshape(-1,3,3)).as_rotvec().reshape(n,21,3);p['raw_root_prior_Q_world']=Q
    relative=np.broadcast_to(np.eye(3),(n,2,3,3)).copy()
    for a in (a,s05):
        ix=np.searchsorted(pts,a['pts'])
        if'skate_relative_R_mid'in a:relative[ix]=a['skate_relative_R_mid'];continue
        _,_,_,frames=shared.body.forward_rotations(local[ix],with_frames=True);base_R=shared.binding.parameters()[0]
        for foot,j in enumerate((7,8)):relative[ix,foot]=base_R[foot].T@frames[:,j].swapaxes(-1,-2)@Q[ix].swapaxes(-1,-2)@a['skate_R_world_mid'][:,foot]
    np.savez_compressed(out/'input.npz',**p)
    np.savez_compressed(out/'initial.npz',pts=pts,P=P,Q=Q,local_R=local,skate_relative_R=relative,**({}if hands is None else dict(hand_local_R=hands)))
    np.savez_compressed(out/'observations.npz',pts=pts,uv=uv,use=use,sigma_px=sigma)
    np.savez_compressed(out/'export_source.npz',pts=pts,source_pts=pts,time_base=np.array(tb),source_lower_body_inferred=p['lower_body_inferred'],source_identity_suspect=p['source_identity_suspect'],window_mask=np.zeros(n,bool),source_observation_use=use,shared_geometry=shared.geometry,user_accepted=np.array(False))
    os.link(base/'compact_smplx.npz',out/'compact_smplx.npz')
    np.savez_compressed(out/'fixed_binding.npz',**shared.fixed_checkpoint_identity,x=shared.binding.x,shared_geometry=shared.geometry,collar_constraints_enabled=np.array(shared.binding.collar_constraints_enabled))
    document=copy.deepcopy(shared.source_document);document.update(generic_context=True,programme_scope=dict(path=str(scope_path),sha256=digest(scope_path)),root_prior_policy=dict(camera_ray_depth_confidence=.04),scene_contract=dict(ice_plane_known=True,plane_source='same frozen accepted Ankara scene Z=0',camera_optimized=False))
    rebind_source_slots(document,pts)
    write_json(out/'source_evidence.json',document);write_json(out/'grasp_relations.json',shared.grasp_relations);write_json(out/'palm_grip_geometry.json',shared.grasp_geometry)
    write_json(out/'model_identity.json',dict(path=shared.current_restore_info['model'],sha256=shared.current_restore_info['model_sha256'],generic_model_not_athlete_identity=True))
    write_json(out/'camera_context.json',camera.info)
    # Preserve the programme-selected finite policy as current prepared input;
    # solve/resume never rereads another programme's mutable configuration.
    ice=json.loads((project/PROGRAMME/'ice_soft_1103.json').read_text(encoding='utf-8'))
    write_json(out/'ice_soft_policy.json',ice)
    write_json(out/'preparation.json',dict(scope=str(scope_path),scope_sha256=digest(scope_path),source_sha256=scope['source']['sha256'],source_video=scope['source']['registered_path'],frames=n,by_shot={s:int((shots==s).sum())for s in np.unique(shots)},legacy_ledger=str(ledger_path),legacy_ledger_sha256=digest(ledger_path),missing_source_observations_pts=pts[native['source_mapping_missing']].tolist(),legacy_pose_compatibility=compatibility,prior_interpolated_or_extrapolated_frames=int(prior_inferred.sum()),numerical_pnp_valid=int(root_valid.sum()),old_S01_result=str(catalog.result_path(variant)),old_S01_result_sha256=digest(catalog.result_path(variant)),S05_parent=str(parent_param),S05_parameter_sha256=digest(parent_param),no_new_network=True,no_camera_or_scene_fit=True,no_per_shot_XYZ_alignment=True))
    np.savez_compressed(out/'camera_midpoints.npz',pts=pts,camera25=cam25,**camera.midpoint)
    print(json.dumps(dict(event='programme_prepared',path=str(out),frames=n,missing_source_frames=int(native['source_mapping_missing'].sum()))),flush=True)
    return out

def main():
    ap=argparse.ArgumentParser()
    for name in ('project','base','root'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();prepare(a.project,a.base,a.root)
if __name__=='__main__':main()
