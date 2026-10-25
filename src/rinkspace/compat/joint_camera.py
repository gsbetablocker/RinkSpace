"""One-time camera-only import of a complete joint numeric model.

Forward projection remains the public native motion evaluator. Other shots,
geometry, observations and optimizer state are not imported or rebuilt.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from .d78_state import decode_camera
from ..core.native import NativeCore


def decode_joint_camera(model,shot_id,core):
    shot=model['shots'][shot_id];rig=model['rigs'][shot['rig_id']];blocks=model['blocks']
    scene=blocks[model['scene_block']]['values']
    head=blocks[rig['head_block']]['values'];optics=blocks[rig['optics_block']]['values']
    if (len(scene),len(head),len(optics))!=(5,15,12):
        raise ValueError('joint camera import requires scene5/head15/optics12')
    ids=shot['control_blocks']
    if len(ids)!=shot['camera']['n_controls'] or len(set(ids))!=len(ids):
        raise ValueError('joint camera control identity/count mismatch')
    coordinates=shot.get('control_coordinates')
    if coordinates:
        # Packed ambient block values are the saved accepted controls; a
        # block's optimizer manifold is not needed to read that saved state.
        # Follow the explicit address of every triplet instead of stale alias
        # control_blocks, and never reconstruct controls from a motion fit.
        if len(coordinates)!=len(ids):
            raise ValueError('joint packed control coordinate/count mismatch')
        local_blocks=[];seen=set()
        for address in coordinates:
            block=address.get('block');offset=address.get('offset')
            if (type(block) is not int or type(offset) is not int or block<0
                    or block>=len(blocks) or offset<0 or offset+3>len(blocks[block]['values'])
                    or (block,offset) in seen):
                raise ValueError('invalid joint packed control coordinate')
            seen.add((block,offset))
            local_blocks.append({'values':deepcopy(blocks[block]['values'][offset:offset+3])})
    else:
        local_blocks=[deepcopy(blocks[i]) for i in ids]
    definition={**deepcopy(rig['camera']),**deepcopy(shot['camera']),'control_block_begin':1}
    # The recorded wide-angle hinge is identically zero on these shots when
    # every allocated cubic log-f control is at/above its activation boundary.
    # Prove this on the entire convex hull (including existing RS support),
    # never just the requested exposure. Active or unknown extensions still
    # fail instead of silently losing their coefficients.
    original_definition=deepcopy(definition)
    extension=definition.get('brown_wide_extension')
    inactive=None
    if extension is not None:
        if (extension.get('schema')!='rinkfit.source_supported_wide_brown_hinge.v1'
                or extension.get('power')!=2):
            raise ValueError('unsupported joint wide Brown extension')
        coefficients=np.asarray(extension.get('coefficients'),dtype=float)
        boundary=float(extension['activation_log_f_native'])
        local_controls=np.asarray([b['values'] for b in local_blocks],dtype=float)
        if (coefficients.shape!=(4,) or not np.isfinite(coefficients).all()
                or not np.isfinite(boundary) or local_controls.shape!=(len(ids),3)
                or not np.isfinite(local_controls).all()):
            raise ValueError('invalid joint wide Brown extension/control values')
        lower=float(local_controls[:,2].min())
        if lower<boundary:
            raise ValueError('active joint wide Brown extension is not supported by this native camera')
        inactive={'kind':'wide_brown_hinge','original_definition':deepcopy(extension),
            'reason':'identically_zero_on_entire_allocated_cubic_control_hull',
            'allocated_log_f_lower_bound':lower,'activation_log_f_native':boundary,
            'scope':'all valid native spline times, including allocated row-exposure support',
            'effective_coefficients_discarded':False}
        definition.pop('brown_wide_extension')
    local={'blocks':[{'values':[*scene,*head,*optics]}]+local_blocks,
           'camera':definition}
    camera,controls=decode_camera(local,core)
    if inactive is not None:
        camera['original_camera_definition']=original_definition
        camera['inactive_extension_proofs']=[inactive]
    angles=np.asarray(blocks[shot['reference_angles_block']]['values'],dtype=float)
    if angles.shape!=(2,) or not np.isfinite(angles).all():
        raise ValueError('joint camera shot reference angles must contain two finite values')
    # R_b(beta) R_b(beta_ref) R0 R_n(alpha_ref) R_n(alpha)
    # equals R_b(beta+beta_ref) R0 R_n(alpha+alpha_ref). This exact
    # chart translation avoids duplicating Rodrigues/physical-head math here.
    controls[:,:2]+=angles
    camera.update(shot_id=shot_id,rig_id=shot['rig_id'],reference_pts=int(shot['time_origin_pts']),
        first_pts=int(shot['scope']['first_pts']),stop_pts_exclusive=int(shot['scope']['stop_pts_exclusive']),
        shot_reference_angles=angles.tolist(),
        reference_angles_representation='constant angle offsets expanded into all native cubic controls',
        state_status='preserved_complete_joint_camera_initial_not_refitted')
    if coordinates:
        camera['control_storage_import']={'kind':'explicit_packed_ambient_triplets',
            'control_count':len(coordinates),'coordinates':deepcopy(coordinates),
            'legacy_alias_blocks_used':False,'trajectory_refitted':False}
    camera['reference_dependency']['absolute_controls_include_shot_reference_angles']=True
    camera['reference_dependency']['expanded_absolute_angles']=controls[definition['dependent_control'],:2].tolist()
    return camera,controls


def import_joint_camera(model_path,shot_id,scope_path,library_path,out_dir):
    model_path=Path(model_path).resolve(strict=True);scope_path=Path(scope_path).resolve(strict=True)
    out=Path(out_dir).resolve()
    if out.exists():raise FileExistsError(out)
    model=json.loads(model_path.read_text(encoding='utf-8-sig'))
    scope=json.loads(scope_path.read_text(encoding='utf-8-sig'))
    selected=next(s for s in scope['shots'] if s['id']==shot_id)
    camera,controls=decode_joint_camera(model,shot_id,NativeCore(library_path))
    interval=selected['processing']
    if camera['first_pts']!=interval['first_pts'] or camera['stop_pts_exclusive']!=interval['stop_pts_exclusive']:
        raise ValueError('joint camera scope differs from current programme shot')
    clock=scope['source_clock'];source=scope['source']
    camera.update(source_id=source['id'],time_base_num=clock['time_base_num'],time_base_den=clock['time_base_den'],
        time_base=f"{clock['time_base_num']}/{clock['time_base_den']}",
        image_width=source['width'],image_height=source['height'],
        origin={'kind':'one_time_complete_joint_camera_import','model_path':str(model_path),
            'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'source_sha256':source['sha256'],
            'scope_path':str(scope_path),'geometry_or_source_rows_imported':False,'fit_executed':False})
    out.mkdir(parents=True)
    (out/'camera.json').write_text(json.dumps(camera,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    np.save(out/'controls.npy',controls,allow_pickle=False)
    return {'camera':str(out/'camera.json'),'controls':str(out/'controls.npy'),'shot_id':shot_id,
        'rig_id':camera['rig_id'],'fit_executed':False,'control_count':len(controls)}
