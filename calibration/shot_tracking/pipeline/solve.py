"""Bidirectional native-PTS local PTZ fit against measured static tracks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from camera import ROOT, anchor_camera, attach_to_surface, fast_project, from_state, read_json


def summary(errors):
    e = np.asarray(errors, float)
    return {'count':len(e), 'rms_px':float(np.sqrt(np.mean(e*e))) if len(e) else None,
            'p95_px':float(np.percentile(e,95)) if len(e) else None,
            'max_px':float(e.max()) if len(e) else None}


def run(run_dir, scene_path, anchor_pts=104693760):
    run_dir = Path(run_dir)
    scene = read_json(scene_path)
    v0 = anchor_camera(scene, anchor_pts)
    data = np.load(run_dir/'observations.npz', allow_pickle=False)
    pts = data['pts'].astype(np.int64)
    anchor_index = int(np.flatnonzero(pts==anchor_pts)[0])
    anchor_xy = data['anchor_xy']
    tracks, visible = data['tracks'], data['valid'].astype(bool)
    surface = data['surface'].astype(str)
    xyz, attached = attach_to_surface(scene, v0, anchor_xy, surface)
    attached &= np.linalg.norm(fast_project(v0, np.nan_to_num(xyz))-anchor_xy, axis=1)<.02
    visible &= attached[None,:]
    # Whole physical image patches, not alternating pixels, are excluded from
    # pose estimation. These test propagation only, not the anchor geometry.
    tilex=(anchor_xy[:,0]//120).astype(int)
    tiley=(anchor_xy[:,1]//90).astype(int)
    holdout=attached & ((tilex+3*tiley)%7==0)
    fit=attached & ~holdout
    if fit.sum()<15:
        raise ValueError('Too few surface-attached fit tracks')
    q = np.zeros((len(pts),3))
    offset=np.zeros(3)
    stats=[None]*len(pts)
    pairs=np.load(run_dir/'pair_tracks.npz',allow_pickle=False) if (run_dir/'pair_tracks.npz').exists() else None
    for direction in [-1,1]:
        for t in range(anchor_index+direction, len(pts) if direction>0 else -1,direction):
            initial=q[t-direction].copy()
            if pairs is not None:
                ix=(pairs['from_index']==t-direction)&(pairs['to_index']==t)
                local_xyz,local_ok=attach_to_surface(scene,from_state(v0,initial,offset),
                                                   pairs['xy_from'][ix],pairs['surface'][ix])
                local_labels=pairs['surface'][ix][local_ok]
                fit_xyz=local_xyz[local_ok]
                target=pairs['xy_to'][ix][local_ok]
            else:
                indices=np.flatnonzero(visible[t]&fit)
                fit_xyz=xyz[indices];target=tracks[t,indices];local_labels=surface[indices]
            if len(target)<12:
                raise ValueError(f'Insufficient measured tracks at PTS {pts[t]}: {len(target)}')
            weight=np.ones(len(target))
            for label in ['ice','board']:
                m=local_labels==label
                if m.any():weight[m]=np.sqrt(len(target)/(2*m.sum()))
            def residual(state):
                pred=fast_project(from_state(v0,state,offset),fit_xyz)
                return ((pred-target)*weight[:,None]).ravel()
            opt=least_squares(residual,initial,bounds=([-1.,-.4,-.7],[1.,.4,.7]),
                              loss='soft_l1',f_scale=1.5,xtol=1e-9,ftol=1e-9,gtol=1e-8,
                              max_nfev=100,x_scale=[.05,.03,.03])
            q[t]=opt.x
            delta=np.linalg.norm(fast_project(from_state(v0,opt.x,offset),fit_xyz)-target,axis=1)
            stats[t]={'optimizer_success':bool(opt.success),'nfev':opt.nfev,
                      'local_edge_residual':summary(delta),
                      'local_board':summary(delta[local_labels=='board']),
                      'local_ice':summary(delta[local_labels=='ice'])}
            if t%10==0:
                print(json.dumps({'pts':int(pts[t]),'tracks':len(target),'residual':summary(delta)}),flush=True)
    frames=[]
    for t,p in enumerate(pts):
        camera=from_state(v0,q[t],offset)
        ids=np.flatnonzero(visible[t])
        errors=np.full(len(xyz),np.nan)
        errors[ids]=np.linalg.norm(fast_project(camera,xyz[ids])-tracks[t,ids],axis=1)
        row={'pts':int(p),'camera_vector':camera.tolist(),'state':'observed',
             'pan_tilt_logzoom':q[t].tolist(),'solver':stats[t],
             'fit_tracks':summary(errors[visible[t]&fit]),
             'check_tracks':summary(errors[visible[t]&holdout]),
             'ice_tracks':summary(errors[visible[t]&(surface=='ice')]),
             'board_tracks':summary(errors[visible[t]&(surface=='board')])}
        if pairs is not None and 'athlete_bbox' in pairs:
            row['athlete_bbox']=pairs['athlete_bbox'][t].tolist()
        frames.append(row)
    report={'schema_version':'rinkspace.local_ptz_trial.v1','source_path':'data/input/oxXu1avhrvs.mp4',
            'time_base':'1/15360','source_size_wh':[1920,1080],'anchor_pts':anchor_pts,
            'scope':'anchor plus 60 native frames in each direction; inclusive endpoints',
            'source_model_path':str(scene_path),'source_model_sha256':hashlib.sha256(Path(scene_path).read_bytes()).hexdigest(),
            'camera_model':{'kind':'shared_pivot_pan_tilt_zoom','offset_camera_xyz':offset.tolist(),
                            'pan_axis_world_xyz':[0,0,1],'optics':'shared anchor Brown12; proportional focal and skew zoom',
                            'status':'initial zero-lever-arm hypothesis; not measured physical calibration'},
            'track_policy':{'attached_count':int(attached.sum()),'fit_count':int(fit.sum()),'holdout_count':int(holdout.sum()),
                            'holdout':'whole 120x90 anchor image tiles; complete temporal tracks withheld',
                            'geometry_independent':False},'frames':frames}
    (run_dir/'poses.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(run_dir/'surface_tracks.npz',pts=pts,xyz=xyz,attached=attached,fit=fit,holdout=holdout,
                        tracks=tracks,valid=visible,surface=surface,states=q)
    print(json.dumps({'wrote':str(run_dir/'poses.json'),'frames':len(frames),'attached':int(attached.sum())}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=ROOT/'calibration/shot_tracking/runs/low_end_pm1')
    p.add_argument('--scene',type=Path,default=ROOT/'calibration/deliverables/scene.json')
    a=p.parse_args();run(a.run,a.scene)
