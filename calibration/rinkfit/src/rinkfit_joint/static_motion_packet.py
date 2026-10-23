"""Compile immutable measured motion with one shared continuous-board latent per pair.

All selected correspondences are retained. The inverse ray is a numerical seed;
clipping an initial height is recorded and does not change any observed pixel.
"""
from pathlib import Path
from functools import lru_cache
from collections import Counter
import argparse, gzip, json
import numpy as np
import cv2
from .io_utils import read_json, read_jsonl, file_sha256
from .observed_primal_restart import reference, checked, write_new
from .dense_source_motion import _rays, _continuous_board_exit, _station_height, _inward_front
from .dense_motion_information import _independent
from .joint_model import as_event_view, graph_from_joint
from .camera_graph import CameraGraph
from rinkfit_start.native_scene import NativeScene


def compile_packet(selection_file, model_file, output, *, frame_roles):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    model = read_json(model_file)
    scene = NativeScene(as_event_view(model, 's01'))
    graph = CameraGraph(graph_from_joint(model))
    from .role_guard import FrameRoleGuard, DENY_TRAIN
    role_guard=FrameRoleGuard(frame_roles)
    role_ref=reference(frame_roles)
    if role_guard.document.get("schema")!="rinkfit.joint_source_frame_protection.v1" or role_guard.document["source_video_sha256"]!=model["source_clock"]["source_video_sha256"]:
        raise ValueError("Frame roles belong to another video or unsupported authority")
    selection = read_json(selection_file)
    records = list(read_jsonl(checked(selection['pair_index'])))
    ids = np.fromfile(checked(selection['selected_indices']), dtype='<u4')
    offsets = np.fromfile(checked(selection['pair_offsets']), dtype='<u8')
    if len(offsets) != len(records)+1 or offsets[0] != 0 or offsets[-1] != len(ids) or np.any(np.diff(offsets.astype('i8')) < 0):
        raise ValueError('Invalid complete selected-pair partition')
    evidence = {}
    def pin(ref):
        path = str(Path(ref['file']).resolve())
        if path in evidence:
            if evidence[path]['sha256'] != ref['sha256']:
                raise ValueError('Conflicting immutable source references')
            return Path(path)
        path = checked(ref)
        evidence[str(path)] = reference(path)
        return path
    @lru_cache(maxsize=6)
    def body(path, sha, sid, pts, rgb):
        doc = read_json(pin({'file':path, 'sha256':sha}))
        if (doc['shot_id'] != sid or doc['native_pts'] != pts or doc['decoded_native_RGB_sha256'] != rgb
            or doc['current_native_RGB_verified'] is not True or doc['domain_not_source_point_or_model_projection'] is not True):
            raise ValueError('Body domain is not this actual source exposure')
        mask = cv2.imread(str(pin(doc['body_material_mask_ref'])), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (1080,1920):
            raise ValueError('Native body mask size differs')
        for key in ('mask_ref','unknown_exclusion_ref','graphics_ref','parent_body_material_ref','near_board_frame_ref','board_body_visible_mask_ref','near_outside_exclusion_mask_ref'):
            if doc.get(key): pin(doc[key])
        # Exactly the original native 21px support convention, independently
        # recomputed from the stored body pixels, not a boolean report field.
        return cv2.erode((mask>0).astype('u1'),np.ones((21,21),'u1'),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    tables=[]; pair_records=[]; excluded=[]; counts=Counter(); seen=set()
    pin(role_ref)
    shot_order=list(model['shots'])
    for pi, entry in enumerate(records):
        raw=read_json(pin(entry)); sid=entry['shot_id']; p0=int(entry['previous_native_pts']);p1=int(entry['native_pts'])
        key=(sid,p0,p1)
        if key in seen or p1-p0 !=256 or raw['cross_cut'] or raw['source_world_projection_used']:
            raise ValueError('Duplicate, nonnative or projected pair')
        seen.add(key)
        for k in ('pixel_coordinates_not_model_prediction','source_or_current_UV_not_moved','complete_21x21_patch_required_in_both_real_exposure_domains'):
            if raw[k] is not True: raise ValueError('Missing real static pixel support')
        a=np.asarray(raw['source_uv'],float);b=np.asarray(raw['current_uv'],float);fb=np.asarray(raw['forward_backward_error_px'],float)
        if a.shape!=(entry['actual_2D_correspondence_pairs'],2) or b.shape!=a.shape or not np.isfinite(np.c_[a,b,fb]).all():
            raise ValueError('Raw motion denominator or finite coordinates differ')
        chosen=ids[int(offsets[pi]):int(offsets[pi+1])]
        if not np.array_equal(chosen,_independent(a,b,fb,21)):
            raise ValueError('Selected support differs from the complete original policy')
        for side,uv,pts,rgb in [('previous',a,p0,entry['previous_RGB_sha256']),('current',b,p1,entry['current_RGB_sha256'])]:
            if raw[side+'_source_RGB_sha256']!=rgb:raise ValueError('RGB identity changed')
            br=raw[side+'_body_material_ref'];valid=body(br['file'],br['sha256'],sid,pts,rgb)
            xy=np.rint(uv).astype(int)
            if ((xy<0).any() or (xy[:,0]>=1920).any() or (xy[:,1]>=1080).any() or not valid[xy[:,1],xy[:,0]].all()):
                raise ValueError('An original source patch is outside its static body domain')
            pin(raw[side+'_source_mask_ref'])
        roles=[role_guard.roles.get(p)for p in (p0,p1)]
        if any(r is None or r.get('shot_id')!=sid for r in roles):raise ValueError('Missing or cross-shot frame role')
        if any(r['role']in DENY_TRAIN for r in roles):
            excluded.append(dict(pair_index=pi,source=reference(entry['file']),selected_indices=chosen.tolist(),reason='protected_or_unresolved_frame_role',endpoint_roles=[r['role']for r in roles]))
            counts['excluded_selected_matches']+=len(chosen)
            continue
        role_guard.assert_training_track(p0,p1,shot_id=sid)
        a=a[chosen];b=b[chosen];n=len(chosen);camera=graph.native_camera(sid)
        t=graph.local_seconds(sid,np.full(n,p0));C,ray,brown,_=_rays(camera,t,a)
        X,hit,_=_continuous_board_exit(scene,C,ray)
        if not (brown & hit & _inward_front(scene,X,C)).all():
            raise ValueError('Numerical board seed failed; no observation silently removed')
        s,h=_station_height(scene,X);outside=(h<0)|(h>scene.height)
        counts['initial_height_projected_to_finite_board']+=int(outside.sum())
        h=np.clip(h,0,scene.height)
        if not np.isfinite(np.c_[s,h]).all():raise ValueError('Nonfinite shared surface seed')
        table=np.c_[np.full(n,shot_order.index(sid)),np.full(n,p0),np.full(n,p1),a,b,s,h/scene.height,np.full(n,pi),chosen,np.ones(n)]
        tables.append(table);pair_records.append(dict(pair_index=pi,source=reference(entry['file']),selected_indices=chosen.tolist()))
        counts[sid+'_raw']+=entry['actual_2D_correspondence_pairs'];counts[sid+'_selected']+=n
        if pi%200==0:print(json.dumps(dict(pair=pi,selected=sum(len(t) for t in tables))),flush=True)
    table=np.concatenate(tables).astype('<f8')
    if len(table)+counts['excluded_selected_matches']!=selection['selected_matches'] or len(records)!=selection['pairs']:
        raise ValueError('Complete input denominator changed')
    output.mkdir(parents=True)
    table.tofile(output/'rows.f64')
    manifest=dict(schema='rinkfit.static_surface_motion_packet.v2',surface='board_inner',latent_coordinates='board_arclength_and_height_fraction',
        selected_matches=len(table),raw_matches=selection['raw_matches'],pairs=pair_records,
        frame_role_authority=role_ref,original_selected_matches=selection['selected_matches'],excluded_selected_sources=excluded,
        complete_original_selection_retained_in_ledger=True,all_fitted_motion_exposures_pass_original_role_guard=True,
        selection=reference(selection_file),initialization_model=reference(model_file),
        table=reference(output/'rows.f64'),table_shape=list(table.shape),table_dtype='<f8',
        evidence_files=list(evidence.values()),counts=dict(counts),
        all_selected_sources_retained=True,world_truth_claimed=False,
        body_masks_decoded_and_complete_patches_rechecked=True,raw_RGB_redecoded=False,
        long_tracks_inferred=False,noise_covariance_calibrated=False,
        initial_height_policy='only the unmeasured numerical height seed is clipped to the finite board and stored as h/H; X uses h_fraction times the current scene height; original UV are unchanged')
    ref=write_new(output/'manifest.json',manifest)
    print(json.dumps(dict(manifest=ref,counts=dict(counts),checked_files=len(evidence))),flush=True)
    return ref


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);p.add_argument('--frame-roles',required=True)
    a=p.parse_args();compile_packet(a.selection,a.model,a.out,frame_roles=a.frame_roles)

if __name__=='__main__':main()
