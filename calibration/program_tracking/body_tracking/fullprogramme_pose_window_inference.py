"""Infer one reviewed bad coarse-pose window from its complete-body ends.

This only SLERPs existing saved local SMPL-X articulation and performs FK.
There is no neural inference, joint optimization, historical world import,
source-pixel fill, foot condition, or position filtering.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .fullprogramme_body_sequence import BODY_SCALE, _load, _json, _digest


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    p=_load(args.packet);s=_load(args.body_samples)
    if not np.array_equal(p['actual_pts'],s['actual_pts']):
        raise ValueError('Saved complete-body articulation/source clock differs')
    actual=p['actual_pts']
    anchors=np.searchsorted(actual,[args.start_pts,args.end_pts])
    if np.any(anchors>=len(actual)) or not np.array_equal(actual[anchors],[args.start_pts,args.end_pts]):
        raise ValueError('Reviewed whole-body source ends must be exact actual exposures')
    first,last=anchors.tolist()
    if first>=last or len(np.unique(p['shot_index'][first:last+1]))!=1:
        raise ValueError('Complete-body inference cannot cross a cut or reverse time')
    if np.any(p['body_opening_temporal_inference_only'][first:last+1]):
        raise ValueError('Handled opening cannot enter this inference repair')
    rows=np.arange(first,last+1)
    middle=rows[1:-1]
    source_ends=s['body_pose_axis_angle'][anchors].reshape(2,21,3)
    alpha=(p['fit_time_seconds'][rows]-p['fit_time_seconds'][first])/(p['fit_time_seconds'][last]-p['fit_time_seconds'][first])
    pose=np.empty((len(rows),21,3),np.float32)
    for j in range(21):
        pose[:,j]=Slerp([0.,1.],Rotation.from_rotvec(source_ends[:,j]))(alpha).as_rotvec()
    pose=pose.reshape(len(rows),63)
    sys.dont_write_bytecode=True;sys.path.insert(0,str(args.upstream))
    import torch
    from pytorch3d.transforms import axis_angle_to_matrix
    from hmr4d.utils.body_model.smplx_lite import SmplxLite,batch_rigid_transform_v2
    torch.set_num_threads(6)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=SmplxLite(model_path=Path(args.model),gender='neutral',num_betas=10).eval().requires_grad_(False).to(device)
    beta=torch.from_numpy(s['shared_betas'].astype(np.float32)).to(device)
    N=len(rows)
    params=dict(body_pose=torch.from_numpy(pose)[None].to(device),
        betas=beta.expand(1,N,-1),global_orient=torch.zeros((1,N,3),device=device),
        transl=torch.zeros((1,N,3),device=device))
    with torch.inference_mode():
        vertices=model(**params)[0]
        other=model.other_default_pose.expand(1,N,-1)
        aa=torch.cat([params['global_orient'],params['body_pose'],other],dim=-1).reshape(1,N,55,3)
        skeleton=model.get_skeleton(params['betas'])
        native,_=batch_rigid_transform_v2(axis_angle_to_matrix(aa),skeleton,model.parents)
        native=native[0];root=native[:,0]
        mapping=torch.load(Path(args.upstream)/'hmr4d/utils/body_model/smplx2smpl_sparse.pt',weights_only=False).to(device)
        regressor=torch.load(Path(args.upstream)/'hmr4d/utils/body_model/smpl_coco17_J_regressor.pt',weights_only=False).to(device)
        if mapping.layout==torch.strided:
            smplx_to_coco=regressor@mapping
        else:
            smplx_to_coco=torch.sparse.mm(mapping.transpose(0,1),regressor.transpose(0,1)).transpose(0,1)
        coco=torch.einsum('jv,nvc->njc',smplx_to_coco,vertices)
        mesh_local=(vertices[:,s['mesh_original_vertex_index']]-root[:,None]).cpu().numpy()*BODY_SCALE
        native_local=(native[:,:22]-root[:,None]).cpu().numpy()*BODY_SCALE
        coco_local=(coco-root[:,None]).cpu().numpy()*BODY_SCALE
    endpoint_delta=dict(coco17_m=float(np.max(np.abs(coco_local[[0,-1]]-p['local_COCO17_m'][anchors]))),
        native22_m=float(np.max(np.abs(native_local[[0,-1]]-p['local_native22_m'][anchors]))),
        sampled_mesh_m=float(np.max(np.abs(mesh_local[[0,-1]]-s['mesh_envelope_rootlocal_scaled_m'][anchors]))))
    if max(endpoint_delta.values())>3e-5:
        raise ValueError('Complete-body endpoint FK does not reproduce saved geometry')
    p['local_COCO17_m'][rows]=coco_local
    p['local_native22_m'][rows]=native_local
    p['body_local_pose_inferred'][middle]=True
    p['body_local_pose_origin']=p['body_local_pose_origin'].astype('<U96')
    p['body_local_pose_origin'][middle]='whole_body_endpoint_source_conditioned_pose_SLERP'
    p['body_local_pose_donor_actual_row'][rows]=rows
    # The inferred head-local shape/direction is not a reliable face proxy in
    # this backbend. Preserve raw face UV/scores/qualification as diagnostics;
    # keep original shoulders/hips/knees/ankles in the whole-body source fit.
    p['body_source_point_fit_enabled']=np.ones((len(actual),17),bool)
    p['body_source_point_fit_enabled'][middle,:5]=False
    p['body_local_pose_inference_anchor_actual_pts']=np.asarray([args.start_pts,args.end_pts],np.int64)
    p['body_local_pose_inference_rows']=middle
    s['body_pose_axis_angle'][rows]=pose
    s['mesh_envelope_rootlocal_scaled_m'][rows]=mesh_local
    s['local_pose_origin']=s['local_pose_origin'].astype('<U96')
    s['local_pose_origin'][middle]='whole_body_endpoint_source_conditioned_pose_SLERP'
    packet_path=output/'fullprogramme_body_source_input.npz'
    samples_path=output/'saved_body_envelope_samples.npz'
    np.savez_compressed(packet_path,**p);np.savez_compressed(samples_path,**s)
    report=dict(schema='rinkspace.complete-body.reviewed-window-source-inference.v1',
        input_packet=dict(path=str(args.packet),sha256=_digest(args.packet)),
        input_body_samples=dict(path=str(args.body_samples),sha256=_digest(args.body_samples)),
        output_packet=dict(path=str(packet_path),sha256=_digest(packet_path)),
        output_body_samples=dict(path=str(samples_path),sha256=_digest(samples_path)),
        source_anchor_actual_pts=[args.start_pts,args.end_pts],
        source_anchor_original_native_pts=p['source_native_pts'][anchors].tolist(),
        inferred_actual_rows=middle.tolist(),inferred_actual_exposures=len(middle),
        original_full_source_review_window_native_pts=[231295488,231320064],
        original_full_source_review_continuously_decoded_native_frames=97,
        source_whole_body_fit_keeps_original_shoulders_hips_knees_ankles=True,
        original_face_pixels_qualification_scores_retained_diagnostic_only_in_inferred_window=True,
        original_cameras_sources_aliases_and_50_60_clock_unchanged=True,
        raw_root_world_prior_retains_original_exposure_spin_direction=True,
        all_points_share_one_body_R_T=True,
        no_local_joint_optimization_or_new_network_inference=True,
        no_floor_foot_contact_or_user_circle_factor=True,
        no_historical_world_translation_or_rotation_read=True,
        body153cm_scale_applied_once=BODY_SCALE,shape_shared=True,
        endpoint_complete_FK_minus_saved_geometry_max_absolute_m=endpoint_delta,
        model=dict(path=str(args.model),sha256=_digest(args.model)),
        code=dict(path=str(Path(__file__)),sha256=_digest(Path(__file__))),
        conditional_coarse_pose_not_true_articulation=True,accuracy_accepted=False)
    _json(output/'pose_inference_binding.json',report)
    print(dict(output_packet=str(packet_path),output_body_samples=str(samples_path),
               inferred_actual_exposures=len(middle),endpoint_delta=endpoint_delta),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['packet','body-samples','upstream','model','output']:
        ap.add_argument('--'+key,required=True)
    ap.add_argument('--start-pts',required=True,type=int)
    ap.add_argument('--end-pts',required=True,type=int)
    run(ap.parse_args())


if __name__=='__main__':main()
