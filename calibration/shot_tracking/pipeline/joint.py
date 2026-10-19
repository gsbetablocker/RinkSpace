"""Joint short-clip PTZ with one continuous zoom-dependent intrinsic profile.

The rink and the already fitted anchor are fixed. Source image point motion
and explicitly recorded development curves constrain the temporal states.
Any assessment readings reused for fitting are recorded as development data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree

from camera import ROOT, read_json, anchor_camera, from_state, attach_to_surface, fast_project, safe_project


def physical_curve(scene, identity):
    if identity in ['board_contact','board_top']:
        key='bottom_edge_xyz' if identity=='board_contact' else 'top_edge_xyz'
        points=np.asarray(scene['board_surface'][key],float)
        # Long straight sides of the exported polygon need dense samples too.
        return np.concatenate([np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/.3)+1))[:-1]
            for a,b in zip(points,np.roll(points,-1,axis=0))])
    if identity.rsplit('_',1)[-1] in ['outer','middle','inner','center']:
        fid,role=identity.rsplit('_',1);f=next(x for x in scene['ice_features'] if x['id']==fid)
        ring=next(x for x in f['rings'] if x['role']==role)
        c=np.asarray(ring['center_xyz']);th=np.linspace(0,2*np.pi,721)
        return c+np.c_[np.cos(th),np.sin(th),np.zeros(len(th))]*ring['radius']
    f=next(x for x in scene['ice_features'] if x['id']==identity)
    return np.concatenate([np.linspace(a,b,501) for a,b in f['segments_xyz']])


def curve_delta(observed, projected):
    finite=np.isfinite(projected).all(axis=1)
    good=np.flatnonzero(finite[:-1]&finite[1:] & (np.linalg.norm(np.diff(projected,axis=0),axis=1)<1000))
    if not len(good):return np.full_like(observed,1000.)
    mids=(projected[good]+projected[good+1])/2
    _,nearest=cKDTree(mids).query(observed,k=min(4,len(good)))
    if nearest.ndim==1:nearest=nearest[:,None]
    ids=good[nearest];a=projected[ids];v=projected[ids+1]-a
    t=np.clip(np.sum((observed[:,None]-a)*v,axis=2)/np.maximum(np.sum(v*v,axis=2),1e-15),0,1)
    delta=a+t[...,None]*v-observed[:,None]
    best=np.argmin(np.sum(delta*delta,axis=2),axis=1)
    return delta[np.arange(len(observed)),best]


def initialize(scene,v0,pts,pairs,profile,static_pixels=None):
    q=np.zeros((len(pts),3));anchor=int(np.flatnonzero(pts==104693760)[0])
    for direction in [-1,1]:
        for t in range(anchor+direction,len(pts) if direction>0 else -1,direction):
            prev=t-direction
            ix=np.flatnonzero((pairs['from_index']==prev)&(pairs['to_index']==t))
            if static_pixels is not None:
                ix=ix[static_pixels(prev,pairs['xy_from'][ix])&static_pixels(t,pairs['xy_to'][ix])]
            xyz,good=attach_to_surface(scene,from_state(v0,q[prev],lens_profile=profile),pairs['xy_from'][ix],pairs['surface'][ix])
            xyz=xyz[good];uv=pairs['xy_to'][ix][good]
            if len(xyz)<10:raise ValueError(f'No image support at {pts[t]}')
            r=least_squares(lambda x:(fast_project(from_state(v0,x,lens_profile=profile),xyz)-uv).ravel(),q[prev],
                            loss='soft_l1',f_scale=1.5,max_nfev=60)
            q[t]=r.x
    return q


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=ROOT/'calibration/shot_tracking/runs/low_end_pm1')
    ap.add_argument('--max-nfev',type=int,default=120)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--initial-checkpoint',type=Path,
                    help='Use a saved trial as the initializer while keeping its outputs unchanged')
    ap.add_argument('--long-links',type=Path,
                    help='Reviewed source-pixel long-range links; same fixed-centre ray convention')
    ap.add_argument('--extra-contours',type=Path,
                    help='Explicit additional development readings, never held-out checks')
    ap.add_argument('--fit-pan-axis',action='store_true',
                    help='Estimate one shared pan axis instead of assuming world +Z')
    ap.add_argument('--brown-zoom-order',type=int,choices=[0,1,2],default=0,
                    help='Shared Brown law in log zoom; zero keeps the frozen anchor distortion')
    ap.add_argument('--brown-indices',type=int,nargs='+',choices=range(12),default=[0,1,2,3],
                    help='OpenCV Brown12 coefficients participating in the shared zoom law')
    ap.add_argument('--long-link-weight',type=float,default=1.,
                    help='Residual multiplier for reviewed long links')
    ap.add_argument('--foreground-manifest',type=Path,
                    help='PTS-matched SAM2 person masks, used for exclusion and rendering')
    ap.add_argument('--lsmr-tolerance',type=float,default=1e-8,
                    help='Inner sparse solve accuracy; avoid premature outer convergence')
    ap.add_argument('--output-dir',type=Path,
                    help='Separate candidate directory; observations still come from --run')
    a=ap.parse_args();out=a.run
    savedir=a.output_dir or out
    savedir.mkdir(parents=True,exist_ok=True)
    extra_original=a.extra_contours
    if a.extra_contours:
        frozen_extra=a.extra_contours.read_bytes()
        a.extra_contours=savedir/'extra_contours_used.json'
        a.extra_contours.write_bytes(frozen_extra)
    if a.brown_zoom_order:
        from zoom_ray_motion import predict_pairs
    else:
        from ray_motion import predict_pairs
    scene_path=ROOT/'calibration/deliverables/scene.json';scene=read_json(scene_path);v0=anchor_camera(scene,104693760)
    obs=np.load(out/'observations.npz');pts=obs['pts'];pairs=np.load(out/'pair_tracks.npz')
    foreground=None;foreground_masks=None
    if a.foreground_manifest:
        foreground=read_json(a.foreground_manifest)
        if ([r['source_pts'] for r in foreground['frames']]!=pts.tolist()
            or foreground['source_clip_sha256']!=hashlib.sha256((out/'source.mp4').read_bytes()).hexdigest()):
            raise ValueError('Foreground masks do not match source clip/PTS')
        foreground_masks=[]
        for row in foreground['frames']:
            path=out/row['mask_path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['mask_sha256']:
                raise ValueError(f'Foreground mask hash mismatch: {path}')
            mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape!=(1080,1920):raise ValueError(f'Invalid mask: {path}')
            # LK features represent patches. Exclude support near the silhouette,
            # while using the original undilated alpha mask for video rendering.
            foreground_masks.append(cv2.dilate(mask,np.ones((33,33),np.uint8)))
    def static_pixels(frame_index,pixels):
        uv=np.rint(pixels).astype(int)
        good=(uv[:,0]>=0)&(uv[:,0]<1920)&(uv[:,1]>=0)&(uv[:,1]<1080)
        if foreground_masks is not None:
            ii=np.flatnonzero(good)
            good[ii]=foreground_masks[int(frame_index)][uv[ii,1],uv[ii,0]]==0
        return good
    profile=np.array([0.,0.,1.5,0.,.7,0.,0.,0.]);axis_xy=np.zeros(2)
    axis_count=2 if a.fit_pan_axis else 0
    brown_indices=np.asarray(a.brown_indices,int)
    if len(np.unique(brown_indices))!=len(brown_indices):raise ValueError('Duplicate Brown indices')
    brown_start=8+axis_count
    brown_count=len(brown_indices)*a.brown_zoom_order
    shared_count=brown_start+brown_count
    brown_profile=np.zeros((12,2))
    if not np.isfinite(a.long_link_weight) or a.long_link_weight<=0:
        raise ValueError('Long-link weight must be finite and positive')
    checkpoint=a.initial_checkpoint or savedir/'joint_checkpoint.npz'
    if a.initial_checkpoint and not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    if not checkpoint.exists():checkpoint=out/'joint_checkpoint.npz'
    if (a.resume or a.initial_checkpoint) and checkpoint.exists():
        saved=np.load(checkpoint)
        if not np.array_equal(saved['pts'],pts):raise ValueError('Checkpoint source PTS differ')
        q=saved['states'];profile=saved['profile']
        if a.fit_pan_axis and 'pan_axis_xy' in saved:axis_xy=saved['pan_axis_xy']
        if a.brown_zoom_order and 'brown_profile' in saved:brown_profile=saved['brown_profile']
    else:
        q=initialize(scene,v0,pts,pairs,profile,static_pixels if foreground_masks is not None else None)
        np.savez_compressed(savedir/'joint_initial.npz',states=q,profile=profile,pts=pts)
    chosen=[];mask_excluded_adjacent_tracks=0
    for t in range(len(pts)-1):
        for surface in ['board','ice']:
            ix=np.flatnonzero((pairs['from_index']==t)&(pairs['to_index']==t+1)&(pairs['surface']==surface))
            if foreground_masks is not None:
                keep_static=static_pixels(t,pairs['xy_from'][ix])&static_pixels(t+1,pairs['xy_to'][ix])
                mask_excluded_adjacent_tracks+=int((~keep_static).sum())
                ix=ix[keep_static]
            if len(ix)>8:
                _,m=cv2.findHomography(pairs['xy_from'][ix],pairs['xy_to'][ix],cv2.USAC_MAGSAC,2.5)
                if m is not None:ix=ix[m.ravel()>0]
            if len(ix)>55:ix=ix[np.linspace(0,len(ix)-1,55).astype(int)]
            chosen.extend(ix.tolist())
    chosen=np.array(chosen,int)
    fr=pairs['from_index'][chosen].astype(int);to=pairs['to_index'][chosen].astype(int)
    xy=pairs['xy_from'][chosen];target=pairs['xy_to'][chosen]
    adjacent_count=len(xy)
    link_provenance=None
    if a.long_links:
        links=np.load(a.long_links)
        lf=links['from_index'].astype(int);lt=links['to_index'].astype(int)
        lx=np.asarray(links['xy_from'],float);ly=np.asarray(links['xy_to'],float)
        if (lx.shape!=(len(lf),2) or ly.shape!=lx.shape or len(lt)!=len(lf)
            or not np.isfinite(lx).all() or not np.isfinite(ly).all()
            or np.any(lf<0) or np.any(lt<0) or np.any(lf>=len(pts)) or np.any(lt>=len(pts))
            or not np.array_equal(links['pts'],pts)):
            raise ValueError('Long-range links have invalid coordinates or source PTS')
        fr=np.r_[fr,lf];to=np.r_[to,lt];xy=np.vstack([xy,lx]);target=np.vstack([target,ly])
        link_provenance={'path':str(a.long_links),'sha256':hashlib.sha256(a.long_links.read_bytes()).hexdigest(),
                         'count':len(lf),'role':'development_constraints_not_independent_validation'}
    mask_excluded_long_links=0
    if foreground_masks is not None:
        keep=np.ones(len(xy),bool)
        for i in range(adjacent_count,len(xy)):
            keep[i]=static_pixels(fr[i],xy[i:i+1])[0]&static_pixels(to[i],target[i:i+1])[0]
        mask_excluded_long_links=int((~keep).sum())
        fr,to,xy,target=fr[keep],to[keep],xy[keep],target[keep]
    groups=[]
    if (out/'contours.json').exists():
        records=read_json(out/'contours.json')['observations']
        if a.extra_contours:
            extra=read_json(a.extra_contours)['observations']
            replace={(g['frame_index'],g['identity']) for g in extra}
            records=[g for g in records if (g['frame_index'],g['identity']) not in replace]+extra
        for g in records:
            if len(g['points_uv'])<3 or g['frame_index']==60:continue
            group={'frame':int(g['frame_index']),'identity':g['identity'],
                   'uv':np.asarray(g['points_uv'],float),'xyz':physical_curve(scene,g['identity'])}
            groups.append(group)
    if not groups:raise ValueError('Actual source-image contours required before joint fitting')
    active=np.array([i for i in range(len(pts)) if i!=60]);slot={t:i for i,t in enumerate(active)}
    x0=np.r_[profile,axis_xy if a.fit_pan_axis else [],
             brown_profile[brown_indices,:a.brown_zoom_order].ravel(),q[active].ravel()]
    lengths=[2*len(xy)]+[2*len(g['uv']) for g in groups]+[shared_count,3*(len(pts)-2)]
    sp=lil_matrix((sum(lengths),len(x0)),dtype=np.int8);sp[:, :shared_count]=1
    for i,(f,t) in enumerate(zip(fr,to)):
        for j in [f,t]:
            if j!=60:sp[2*i:2*i+2,shared_count+3*slot[j]:shared_count+3*slot[j]+3]=1
    at=lengths[0]
    for g in groups:
        n=2*len(g['uv']);j=g['frame']
        if j!=60:sp[at:at+n,shared_count+3*slot[j]:shared_count+3*slot[j]+3]=1
        at+=n
    at+=shared_count
    for t in range(len(pts)-2):
        for j in [t,t+1,t+2]:
            if j!=60:sp[at+3*t:at+3*t+3,shared_count+3*slot[j]:shared_count+3*slot[j]+3]=1
    sp=sp.tocsr();last=time.monotonic();calls=0;warm=None
    def decode_brown(x):
        value=np.zeros((12,2))
        if brown_count:value[brown_indices,:a.brown_zoom_order]=x[brown_start:shared_count].reshape(-1,a.brown_zoom_order)
        return value
    def decode(x):
        states=np.zeros((len(pts),3));states[active]=x[shared_count:].reshape(-1,3)
        axis=np.r_[x[8:10],1.] if a.fit_pan_axis else np.array([0.,0.,1.])
        brown=decode_brown(x)
        cams=np.array([from_state(v0,s,lens_profile=x[:8],axis=axis,brown_profile=brown) for s in states])
        return states,cams
    def residual(x):
        nonlocal last,calls,warm
        states,cams=decode(x)
        result=predict_pairs(cams,fr,to,xy,inverse_initial=warm,tolerance_px=1e-10)
        pred,valid,inv_error=result['predicted_uv'],result['valid'],result['inverse_error_px']
        warm=result['normalized_from']
        delta=np.nan_to_num(pred-target,nan=1000.,posinf=1000.,neginf=-1000.)
        delta[~valid]+=np.minimum(100.,np.nan_to_num(inv_error[~valid],nan=100.))[:,None]
        delta[adjacent_count:]*=a.long_link_weight
        parts=[delta.ravel()]
        for g in groups:
            uv,_,_=safe_project(cams[g['frame']],g['xyz'])
            d=curve_delta(g['uv'],uv)
            # Each whole physical curve gets comparable influence to a local
            # image edge. Extra dense samples do not manufacture confidence.
            parts.append((d*np.sqrt(35./len(d))).ravel())
        parts.append(np.r_[(x[:8]-profile)*.05,x[8:10]*2. if a.fit_pan_axis else [],
                           x[brown_start:shared_count]*.2])
        parts.append((np.diff(states,n=2,axis=0)*[120.,120.,80.]).ravel())
        res=np.concatenate(parts);calls+=1
        if time.monotonic()-last>30:
            print(json.dumps({'call':calls,'rms':float(np.sqrt(np.mean(res*res))),'profile':x[:8].tolist()}),flush=True)
            np.savez_compressed(savedir/'joint_checkpoint.npz',x=x,states=states,profile=x[:8],pts=pts,
                                pan_axis_xy=x[8:10] if a.fit_pan_axis else np.zeros(2),brown_profile=decode_brown(x))
            last=time.monotonic()
        return res
    print(json.dumps({'pair_points':len(xy),'curve_groups':len(groups),'parameters':len(x0),'residuals':sum(lengths),'initial_rms':float(np.sqrt(np.mean(residual(x0)**2)))}),flush=True)
    brown_bound=np.repeat([.8 if i in [2,3,8,9,10,11] else 4. for i in brown_indices],a.brown_zoom_order)
    lower=np.r_[[-4.]*8,[-.5,-.5] if a.fit_pan_axis else [],-brown_bound,np.tile([-1.,-.6,-1.],len(active))]
    upper=np.r_[[4.]*8,[.5,.5] if a.fit_pan_axis else [],brown_bound,np.tile([1.,.6,1.],len(active))]
    sol=least_squares(residual,x0,jac_sparsity=sp,bounds=(lower,upper),loss='soft_l1',f_scale=2.,
                      x_scale='jac',max_nfev=a.max_nfev,verbose=1,ftol=1e-8,xtol=1e-8,gtol=1e-6,
                      tr_options={'atol':a.lsmr_tolerance,'btol':a.lsmr_tolerance,'maxiter':1000})
    states,cams=decode(sol.x)
    axis=np.r_[sol.x[8:10],1.] if a.fit_pan_axis else np.array([0.,0.,1.])
    axis=axis/np.linalg.norm(axis)
    np.savez_compressed(savedir/'joint_checkpoint.npz',x=sol.x,states=states,profile=sol.x[:8],pts=pts,
                        pan_axis_xy=sol.x[8:10] if a.fit_pan_axis else np.zeros(2),brown_profile=decode_brown(sol.x))
    frames=[]
    for i,p in enumerate(pts):
        frames.append({'pts':int(p),'camera_vector':cams[i].tolist(),'state':'observed',
                       'pan_tilt_logzoom':states[i].tolist(),'athlete_bbox':pairs['athlete_bbox'][i].tolist()})
        if foreground is not None:
            frames[-1]['foreground_mask_path']=foreground['frames'][i]['mask_path']
    report={'schema_version':'rinkspace.local_ptz_trial.v2','source_path':'data/input/oxXu1avhrvs.mp4',
            'time_base':'1/15360','source_size_wh':[1920,1080],'anchor_pts':104693760,
            'source_model_path':str(scene_path),'source_model_sha256':hashlib.sha256(scene_path.read_bytes()).hexdigest(),
            'scope':'121 native source frames, inclusive ±1 second; offline jointly solved',
            'long_range_links':link_provenance,
            'foreground_masks':({'manifest_path':str(a.foreground_manifest),
                'sha256':hashlib.sha256(a.foreground_manifest.read_bytes()).hexdigest(),
                'frame_count':len(pts),'static_feature_exclusion_dilation_px':16,
                'excluded_adjacent_tracks':mask_excluded_adjacent_tracks,
                'excluded_long_links':mask_excluded_long_links} if foreground else None),
            'extra_contours':({'path':str(a.extra_contours),
                'original_source_path':str(extra_original),
                'sha256':hashlib.sha256(a.extra_contours.read_bytes()).hexdigest(),
                'role':'development_constraints_not_independent_validation'} if a.extra_contours else None),
            'manual_readings_used_for_fit':(read_json(a.extra_contours).get('manual_readings_used_for_fit',[])
                                           if a.extra_contours else []),
            'camera_model':{'kind':'fixed_pivot_pan_tilt_zoom','offset_camera_xyz':[0.,0.,0.],
                'pan_axis_world':axis.tolist(),'pan_axis_estimated':bool(a.fit_pan_axis),
                'shared_brown_logzoom_coefficients':decode_brown(sol.x).tolist(),
                'brown_zoom_order':a.brown_zoom_order,'brown_variable_indices':brown_indices.tolist() if brown_count else [],
                'shared_intrinsic_logzoom_quadratic':sol.x[:8].reshape(4,2).tolist(),
                'law_rows':['log_fy_extra','cx_shift_1000px','cy_shift_1000px','skew_extra_100px'],
                'anchor_preserved':True,'interpretation':'conditional local calibration; not measured unique optics'},
            'optimizer':{'success':bool(sol.success),'message':str(sol.message),'nfev':sol.nfev,
                         'optimality':float(sol.optimality),
                         'active_bound_parameter_indices':np.flatnonzero(sol.active_mask).tolist(),
                         'shared_parameter_count':shared_count,
                         'pair_count':len(xy),'adjacent_pair_count':adjacent_count,
                         'long_link_weight':a.long_link_weight,
                         'lsmr_tolerance':a.lsmr_tolerance,
                         'source_curve_group_count':len(groups),'cost':float(sol.cost)},
            'frames':frames}
    (savedir/'poses.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(savedir/'joint_parameters.npz',x=sol.x,pts=pts,fr=fr,to=to,xy_from=xy,xy_to=target,residual=residual(sol.x))
    print(json.dumps({'saved':str(savedir/'poses.json'),'success':bool(sol.success),'cost':float(sol.cost)}),flush=True)


if __name__=='__main__':main()
