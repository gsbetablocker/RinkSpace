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
    if shot.get('control_coordinates'):
        raise ValueError('joint camera import requires explicit complete local control blocks')
    scene=blocks[model['scene_block']]['values']
    head=blocks[rig['head_block']]['values'];optics=blocks[rig['optics_block']]['values']
    if (len(scene),len(head),len(optics))!=(5,15,12):
        raise ValueError('joint camera import requires scene5/head15/optics12')
    ids=shot['control_blocks']
    if len(ids)!=shot['camera']['n_controls'] or len(set(ids))!=len(ids):
        raise ValueError('joint camera control identity/count mismatch')
    definition={**deepcopy(rig['camera']),**deepcopy(shot['camera']),'control_block_begin':1}
    local={'blocks':[{'values':[*scene,*head,*optics]}]+[deepcopy(blocks[i]) for i in ids],
           'camera':definition}
    camera,controls=decode_camera(local,core)
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
