"""Correct saved same-body interpolation on proven original exposure triples.

This only replays saved SMPL-X local articulation and root orientation. The
application API imports no Torch, infers no source pixels and fits no body.
Ambiguous times, official collisions and incomplete brackets retain the old
geometry. A reviewed whole-pose window must be applied afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def _binding(path):
    p=Path(path)
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def _json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def _slerp(a,b,alpha):
    qa,qb=Rotation.from_matrix(a).as_quat(),Rotation.from_matrix(b).as_quat()
    dot=np.sum(qa*qb,axis=1);qb[dot<0]*=-1
    angle=np.arccos(np.clip(np.abs(dot),0,1));den=np.sin(np.maximum(angle,1e-7))
    wa,wb=np.sin((1-alpha)*angle)/den,np.sin(alpha*angle)/den
    small=angle<1e-7;wa[small],wb[small]=1-alpha[small],alpha[small]
    q=qa*wa[:,None]+qb*wb[:,None]
    return Rotation.from_quat(q/np.linalg.norm(q,axis=1)[:,None]).as_matrix()


def apply_official_pose_clock_patch(packet, body_samples, patch_path, *, camera_path=None):
    """Return corrected copies plus provenance, before any reviewed pose overlay.

    Geometry/alpha follow the fresh packet's explicit pose donor. Root Q is
    recomposed from saved source incam orientations and the current camera at
    each exact native endpoint; no historical world rotation is imported.
    """
    p={k:np.asarray(v).copy() for k,v in packet.items()}
    s={k:np.asarray(v).copy() for k,v in body_samples.items()}
    z=_load(patch_path)
    for other in [s,z]:
        if not np.array_equal(p['actual_pts'],other['actual_pts']):
            raise ValueError('Official pose patch and fresh actual exposure clock differ')
    if 'body_local_pose_inference_rows' in p:
        raise ValueError('Official pose clock patch must precede the reviewed pose window')
    if not np.array_equal(s['mesh_original_vertex_index'],z['mesh_original_vertex_index']):
        raise ValueError('Original same-body surface sampling differs')
    for x in [p['body_scale'],s['body_scale']]:
        if not np.isclose(float(x),float(z['body_scale']),rtol=0,atol=1e-12):
            raise ValueError('One shared 153cm body scale must be retained')
    native=p['native_pts'];native_lookup={int(x):i for i,x in enumerate(native)}
    if camera_path is not None:
        cameras=_load(camera_path)
        if not np.array_equal(cameras['pts'],native):
            raise ValueError('Current full camera native clock differs')
        if not np.array_equal(cameras['exposure_representative_pts'],p['native_exposure_representative_pts']):
            raise ValueError('Current confirmed aliases differ')
        actual_rows=np.searchsorted(native,p['actual_pts'])
        if not np.array_equal(cameras['camera25'][actual_rows],p['camera25']):
            raise ValueError('Fresh packet uses a different current camera')
        full_cam=cameras['camera25']
        camera_mode='exact_full_native_camera25'
    else:
        full_cam=p['camera25'][p['native_actual_row']]
        camera_mode='explicit_confirmed_native_actual_row_aliases'
    # Rebind corrections through fresh donor identities, never replace a bad
    # opening/cropped canonical pose with its untrusted canonical geometry.
    donor_to_patch={int(x):j for j,x in enumerate(z['corrected_pose_donor_native_pts'])}
    apply_rows=np.asarray([i for i,x in enumerate(p['body_prior_native_pts']) if int(x) in donor_to_patch],np.int64)
    patch_ids=np.asarray([donor_to_patch[int(p['body_prior_native_pts'][i])] for i in apply_rows],np.int64)
    source_rows=z['corrected_actual_rows'][patch_ids]
    for key,old_key in [('raw_root_left_source_pts','original_raw_root_left_source_pts'),
                        ('raw_root_right_source_pts','original_raw_root_right_source_pts'),
                        ('raw_root_alpha','original_raw_root_alpha')]:
        if not np.array_equal(p[key][apply_rows],z[old_key][source_rows]):
            raise ValueError('Fresh saved pose donor/brackets/old alpha differs from qualified source')
    p['local_COCO17_m'][apply_rows]=z['corrected_local_COCO17_m'][patch_ids]
    p['local_native22_m'][apply_rows]=z['corrected_local_native22_m'][patch_ids]
    p['raw_root_alpha'][apply_rows]=z['corrected_raw_root_alpha'][source_rows]
    raw_orient={int(x):v for x,v in zip(z['raw_orientation_source_pts'],z['raw_incam_global_orient'])}
    world_a=[];world_b=[]
    for i in apply_rows:
        endpoints=[]
        for key in ['raw_root_left_source_pts','raw_root_right_source_pts']:
            source=int(p[key][i]);row=native_lookup[source]
            R=Rotation.from_rotvec(full_cam[row,3:6]).as_matrix()
            endpoints.append(R.T@Rotation.from_rotvec(raw_orient[source].astype(float)).as_matrix())
        world_a.append(endpoints[0]);world_b.append(endpoints[1])
    if len(apply_rows):
        p['raw_root_prior_Q_world'][apply_rows]=_slerp(np.asarray(world_a),np.asarray(world_b),p['raw_root_alpha'][apply_rows])
    p['body_local_pose_origin']=p['body_local_pose_origin'].astype('<U100')
    p['body_local_pose_origin'][apply_rows]='saved_local_pose_original_official50_triple_SLERP_FK'
    p['official_pose_clock_patch_applied_actual_rows']=apply_rows
    p['official_pose_clock_original_raw_root_alpha']=np.asarray(packet['raw_root_alpha']).copy()
    p['official_pose_clock_source_state']=z['source_clock_state'].copy()
    p['official_pose_clock_authority_donor_rows']=z['corrected_actual_rows'].copy()
    for key,patch_key in [('mesh_envelope_rootlocal_scaled_m','corrected_mesh_envelope_rootlocal_scaled_m'),
                          ('body_pose_axis_angle','corrected_body_pose_axis_angle')]:
        s[key][z['corrected_actual_rows']]=z[patch_key]
    s['local_pose_origin']=s['local_pose_origin'].astype('<U100')
    s['local_pose_origin'][z['corrected_actual_rows']]='saved_local_pose_original_official50_triple_SLERP_FK'
    s['official_pose_clock_patch_applied_actual_rows']=z['corrected_actual_rows'].copy()
    record=dict(schema='rinkspace.saved-body-official-clock-patch.application.v1',
        patch=_binding(patch_path),applied_actual_count=len(apply_rows),
        applied_source_donor_count=len(z['corrected_actual_rows']),
        current_camera_endpoint_binding=camera_mode,
        current_camera=_binding(camera_path) if camera_path else None,
        fresh_pose_donor_policy_retained=True,current_camera_source_clock_source_UV_unchanged=True,
        no_old_world_rotation_translation_imported=True,one153cm_scale=True,
        next_overlay_order='reviewed79 whole-pose packet and 81 endpoint-inclusive body samples')
    return p,s,record


def _authority(p, mapping):
    left,right,old=p['raw_root_left_source_pts'],p['raw_root_right_source_pts'],p['raw_root_alpha']
    target=p['body_prior_native_pts'];native=p['native_pts'];rep=p['native_exposure_representative_pts']
    aliases=dict(zip(map(int,native),map(int,rep)))
    new=old.copy();triples=np.full((len(old),3),-1,np.int64)
    states=np.full(len(old),'direct_or_same_endpoint_preserved',dtype='<U80')
    for i,(l,t,r) in enumerate(zip(left,target,right)):
        if l==r:continue
        rows=[mapping[int(x)] for x in (l,t,r)]
        if any(x['match_status']!='unique_pixel_match' or x['accepted_official_pts'] is None for x in rows):
            states[i]='unconfirmed_official_time_preserved';continue
        oo=np.asarray([x['accepted_official_pts'] for x in rows],np.int64);triples[i]=oo
        if not (oo[0]<oo[1]<oo[2]):
            states[i]='official_collision_or_inconsistent_time_preserved';continue
        if not np.array_equal(np.diff(oo),[256,256]):
            states[i]='non_complete_40ms_official_bracket_preserved';continue
        if aliases[int(l)]!=l or aliases[int(r)]!=r:
            states[i]='noncanonical_raw_endpoint_preserved';continue
        representatives=np.unique(rep[(native>=l)&(native<=r)])
        if len(representatives)!=3 or not np.array_equal(representatives,[l,t,r]):
            states[i]='incomplete_three_canonical_exposure_chain_preserved';continue
        proof=[mapping[int(x)] for x in representatives]
        if any(x['match_status']!='unique_pixel_match' or x['shot_index']!=int(p['shot_index'][i]) for x in proof):
            states[i]='inconsistent_source_role_or_time_preserved';continue
        states[i]='complete_original_official50_three_exposure_chain'
        new[i]=(int(oo[1])-int(oo[0]))/(int(oo[2])-int(oo[0]))
    changed=np.flatnonzero(np.abs(new-old)>1e-6)
    return new,triples,states,changed


def _forward(poses, beta, vertex_ids, upstream, model_path, scale):
    """Exact saved SMPL-X FK with the existing proper COCO vertex regressor."""
    sys.path.insert(0,str(upstream));sys.dont_write_bytecode=True
    import torch
    from pytorch3d.transforms import axis_angle_to_matrix
    from hmr4d.utils.body_model.smplx_lite import SmplxLite,batch_rigid_transform_v2
    torch.set_num_threads(6);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SmplxLite(model_path=Path(model_path),gender='neutral',num_betas=10).eval().requires_grad_(False).to(device)
    mapping=torch.load(Path(upstream)/'hmr4d/utils/body_model/smplx2smpl_sparse.pt',weights_only=False).to(device)
    regressor=torch.load(Path(upstream)/'hmr4d/utils/body_model/smpl_coco17_J_regressor.pt',weights_only=False).to(device)
    coco_reg=(regressor@mapping if mapping.layout==torch.strided else
              torch.sparse.mm(mapping.transpose(0,1),regressor.transpose(0,1)).transpose(0,1))
    beta=torch.from_numpy(np.asarray(beta,np.float32)).to(device)
    coco,native,surface=[],[],[]
    with torch.inference_mode():
        for start in range(0,len(poses),256):
            a=np.asarray(poses[start:start+256],np.float32);N=len(a)
            params=dict(body_pose=torch.from_numpy(a)[None].to(device),betas=beta.expand(1,N,-1),
                        global_orient=torch.zeros((1,N,3),device=device),transl=torch.zeros((1,N,3),device=device))
            vertices=model(**params)[0]
            other=model.other_default_pose.expand(1,N,-1)
            aa=torch.cat([params['global_orient'],params['body_pose'],other],dim=-1).reshape(1,N,55,3)
            joints,_=batch_rigid_transform_v2(axis_angle_to_matrix(aa),model.get_skeleton(params['betas']),model.parents)
            joints=joints[0];root=joints[:,0]
            c=torch.einsum('jv,nvc->njc',coco_reg,vertices)
            coco.append((c-root[:,None]).cpu().numpy()*scale)
            native.append((joints[:,:22]-root[:,None]).cpu().numpy()*scale)
            surface.append((vertices[:,vertex_ids]-root[:,None]).cpu().numpy()*scale)
            print(f'saved_FK {start+N}/{len(poses)} on {device}',flush=True)
    return np.concatenate(coco),np.concatenate(native),np.concatenate(surface),str(device)


def build(args):
    from .pelvis_footwear_geometry import reconstruct_local_poses
    started=time.perf_counter();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    p=_load(args.packet);prior=_load(Path(args.raw_directory)/'body_prior.npz');samples=_load(args.body_samples)
    if not np.array_equal(p['actual_pts'],samples['actual_pts']):
        raise ValueError('Original same-body surface samples and actual source clock differ')
    if not np.array_equal(prior['pts'][p['body_prior_native_row']],p['body_prior_native_pts']):
        raise ValueError('Original pose donor rows differ')
    for key,prior_key in [('raw_root_alpha','interpolation_alpha'),('raw_root_left_source_pts','interpolation_left_source_pts'),('raw_root_right_source_pts','interpolation_right_source_pts')]:
        if not np.array_equal(p[key],prior[prior_key][p['body_prior_native_row']].astype(p[key].dtype)):
            raise ValueError('Reference packet is not the original saved donor-bracket input')
    original_clock=json.loads(Path(args.clock_rows).read_text())
    mapping={int(x['full_pts']):x for x in map(json.loads,Path(args.mapping).read_text(encoding='utf-8-sig').splitlines())}
    if not all(mapping[int(x['full_pts'])]==x for x in original_clock):
        raise ValueError('Saved original clock rows differ from actual source mapping')
    new,triples,state,rows=_authority(p,mapping)
    raw_paths=[Path(args.raw_directory)/f'raw_shot_{i:02d}.npz' for i in range(1,10)]
    raws=[_load(path) for path in raw_paths]
    raw={k:np.concatenate([z[k] for z in raws]) for k in ['source_pts','body_pose','chosen_window','shot_slot','global_orient']}
    def poses(alpha):
        records=[dict(body_local_left_source_pts=int(p['raw_root_left_source_pts'][i]),
            body_local_right_source_pts=int(p['raw_root_right_source_pts'][i]),
            body_local_interpolation_alpha=float(alpha[i]),body_local_prior_valid=True) for i in rows]
        return reconstruct_local_poses(records,raw)[0]
    old_pose=poses(p['raw_root_alpha']);new_pose=poses(new)
    old_pose_error=float(np.max(np.abs(old_pose-samples['body_pose_axis_angle'][rows])))
    if old_pose_error>2e-6:
        raise ValueError('Original raw local SLERP no longer reproduces original saved surface pose')
    scale=float(p['body_scale']);vertex_ids=samples['mesh_original_vertex_index']
    old_c,old_n,old_m,device=_forward(old_pose,prior['shared_betas'],vertex_ids,args.upstream,args.model,scale)
    replay=dict(local_COCO17_m=float(np.max(np.abs(old_c-p['local_COCO17_m'][rows]))),
        local_native22_m=float(np.max(np.abs(old_n-p['local_native22_m'][rows]))),
        mesh460_m=float(np.max(np.abs(old_m-samples['mesh_envelope_rootlocal_scaled_m'][rows]))),
        original_local_pose_axis_angle=float(old_pose_error))
    _json(output/'original_alpha_FK_replay.json',dict(max_absolute_error=replay,
        numerical_replay_tolerance_m=3e-5,raw_model_pose_used=True,one153cm_scale=True))
    if max(replay[k] for k in ['local_COCO17_m','local_native22_m','mesh460_m'])>3e-5:
        raise ValueError('Original alpha FK/COCO mapping/scale replay failed; no corrected patch exported')
    corrected_c,corrected_n,corrected_m,_=_forward(new_pose,prior['shared_betas'],vertex_ids,args.upstream,args.model,scale)
    patch=dict(actual_pts=p['actual_pts'],original_pose_donor_native_pts=p['body_prior_native_pts'],
        original_raw_root_left_source_pts=p['raw_root_left_source_pts'],original_raw_root_right_source_pts=p['raw_root_right_source_pts'],
        original_raw_root_alpha=p['raw_root_alpha'],corrected_raw_root_alpha=new,
        source_clock_state=state,source_official_left_target_right_pts=triples,
        corrected_actual_rows=rows,corrected_pose_donor_native_pts=p['body_prior_native_pts'][rows],
        corrected_local_COCO17_m=corrected_c,corrected_local_native22_m=corrected_n,
        corrected_mesh_envelope_rootlocal_scaled_m=corrected_m,corrected_body_pose_axis_angle=new_pose,
        mesh_original_vertex_index=vertex_ids,shared_betas=prior['shared_betas'],body_scale=p['body_scale'],
        raw_orientation_source_pts=raw['source_pts'],raw_incam_global_orient=raw['global_orient'])
    patch_path=output/'official_body_pose_clock_patch.npz';np.savez_compressed(patch_path,**patch)
    patched_packet,patched_samples,application=apply_official_pose_clock_patch(p,samples,patch_path,camera_path=args.camera)
    packet_path=output/'fullprogramme_body_source_input.npz';sample_path=output/'saved_body_envelope_samples.npz'
    np.savez_compressed(packet_path,**patched_packet);np.savez_compressed(sample_path,**patched_samples)
    # The patch cannot alter real source evidence, camera, aliases or fit time.
    changed_keys={'local_COCO17_m','local_native22_m','raw_root_alpha','raw_root_prior_Q_world','body_local_pose_origin'}
    invariant_keys=[k for k in p if k not in changed_keys]
    for key in invariant_keys:
        if not np.array_equal(p[key],patched_packet[key],equal_nan=p[key].dtype.kind in 'fc'):
            raise ValueError('Clock patch unexpectedly changed '+key)
    report=dict(schema='rinkspace.saved-body-official-clock-patch.v1',
        inputs=[_binding(path) for path in [args.packet,args.body_samples,args.clock_rows,args.mapping,
            Path(args.raw_directory)/'body_prior.npz',args.model,Path(args.upstream)/'hmr4d/utils/body_model/smplx2smpl_sparse.pt',
            Path(args.upstream)/'hmr4d/utils/body_model/smpl_coco17_J_regressor.pt']],
        original_raw_model_outputs=[_binding(path) for path in raw_paths],
        output_patch=_binding(patch_path),output_reference_packet=_binding(packet_path),output_body_samples=_binding(sample_path),
        actual_exposures=len(p['actual_pts']),corrected_source_donors=len(rows),
        corrected_exposures_by_cut={str(k):int(np.sum(p['shot_index'][rows]==k)) for k in range(1,10)},
        source_clock_state_counts={str(k):int(v) for k,v in zip(*np.unique(state,return_counts=True))},
        corrected_alpha=.5,original_alpha_FK_replay=replay,
        one153cm_scale=scale,same_original_surface_vertex_indices=True,surface_vertex_count=len(vertex_ids),
        application=application,all_original_other_packet_arrays_unchanged=True,
        no_network_inference_or_body_fit=True,no_camera_scene_source_UV_changed=True,
        no_source_time_identity_created=True,incomplete_or_collision_times_preserved=True,
        reviewed79_pose_overlay_must_follow=True,accuracy_accepted=False,
        corrected_model_joint_displacement_m=dict(coco17_max=float(np.max(np.linalg.norm(corrected_c-old_c,axis=2))),
            native22_max=float(np.max(np.linalg.norm(corrected_n-old_n,axis=2)))),
        device=device,elapsed_s=time.perf_counter()-started,code=_binding(__file__))
    _json(output/'report.json',report)
    print(json.dumps(dict(patch=str(patch_path),corrected=len(rows),replay=replay,samples=str(sample_path),elapsed_s=report['elapsed_s']),ensure_ascii=False),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['packet','body-samples','raw-directory','clock-rows','mapping','upstream','model','camera','output']:
        ap.add_argument('--'+key,required=True,type=Path)
    build(ap.parse_args())


if __name__=='__main__':
    main()
