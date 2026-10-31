"""Whole-shot photo composite for material identity, never for fitting UV."""
from pathlib import Path
import hashlib
import json
import time
import warnings

import cv2
import numpy as np
from scipy.spatial import Delaunay
from rinkspace.core.native import NativeCore
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline


def build(args):
    import run_native_programme as run
    cv2.setNumThreads(2)
    root=args.run_root;manifest=run.read(root/'joint_manifest.json')
    if args.state is None:raise ValueError('A frozen current geometry is required')
    state=dict(np.load(args.state,allow_pickle=False))
    core=NativeCore(manifest['native_library'])
    timeline=ShotMotionTimeline(manifest['shots'],manifest['control_dt'],manifest['first_control_time'])
    out=args.output or root/'registration/source_composites';out.mkdir(parents=True,exist_ok=True)
    print('MAIN LOG: '+str(out/'read.jsonl'),flush=True)
    started=time.monotonic()
    xx,yy=np.meshgrid(np.linspace(-.56,.56,161),np.linspace(-.30,.30,89))
    world=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
    with (out/'read.jsonl').open('a',encoding='utf-8',buffering=1) as log:
        for shot_id in args.shots:
            shot=next(s for s in manifest['shots'] if s['id']==shot_id)
            visibility,frames=run.source_frames(shot_id)
            reference_pts=getattr(args,'reference_pts',None) or frames[len(frames)//2]['pts']
            reference=next(f for f in frames if f['pts']==reference_pts)
            def project(pts):
                clock=(pts-shot['reference_pts'])/12800+timeline.offset(shot)
                return core.project_motion(state['rig'][shot['camera_index']],state['controls'],timeline.first,timeline.dt,1080,clock,world)
            p0=project(reference_pts)
            usable=(p0['status']==0)&np.isfinite(p0['uv']).all(1)
            indices=np.flatnonzero(usable)
            triang=Delaunay(p0['uv'][usable])
            image=cv2.imread(reference['image']);mask,_=run.surfaces(reference,image)
            rows,cols=np.where(mask['ice']);x0,x1=int(cols.min()),int(cols.max())+1;y0,y1=int(rows.min()),int(rows.max())+1
            px,py=np.meshgrid(np.arange(x0,x1),np.arange(y0,y1));xy=np.c_[px.ravel(),py.ravel()]
            simplex=triang.find_simplex(xy)
            inside=simplex>=0;simplex[~inside]=0
            delta=xy-triang.transform[simplex,2]
            bary=np.einsum('nij,nj->ni',triang.transform[simplex,:2],delta)
            weights=np.c_[bary,1.-bary.sum(1)]
            vertices=indices[triang.simplices[simplex]]
            shape=px.shape
            pixels=np.empty((len(frames),*shape,3),np.uint8)
            validity=np.empty((len(frames),*shape),bool)
            for fi,frame in enumerate(frames):
                raw=cv2.imread(frame['image'])
                if hashlib.sha256(raw.tobytes()).hexdigest()!=frame['source_BGR_sha256']:
                    raise ValueError('Original RGB changed')
                domains,_=run.surfaces(frame,raw);proj=project(frame['pts'])
                mapped=np.einsum('nki,nk->ni',proj['uv'][vertices],weights).astype(np.float32)
                good=inside&(proj['status'][vertices]==0).all(1)&np.isfinite(mapped).all(1)
                mapped[~good]=-1000
                mx,my=mapped[:,0].reshape(shape),mapped[:,1].reshape(shape)
                warped=cv2.remap(raw,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                valid=cv2.remap(domains['ice'].astype(np.uint8),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT).astype(bool)&good.reshape(shape)
                pixels[fi]=warped;validity[fi]=valid
                if fi%25==0 or fi==len(frames)-1:
                    record=dict(event='source_material_composite',shot=shot_id,frames=fi+1,total=len(frames),elapsed_s=time.monotonic()-started)
                    line=json.dumps(record);log.write(line+'\n');print(line,flush=True)
            count=validity.sum(0)
            median=np.empty((*shape,3),np.float32);mean=np.empty_like(median)
            # Tiles bound temporary float storage without limiting source frames.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                for begin in range(0,shape[0],32):
                    end=min(shape[0],begin+32);values=pixels[:,begin:end].astype(np.float32)
                    values[~validity[:,begin:end]]=np.nan
                    median[begin:end]=np.nanmedian(values,axis=0);mean[begin:end]=np.nanmean(values,axis=0)
            folder=out/shot_id;folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/'photo_arrays.npz',median_BGR=median,mean_BGR=mean,source_count=count,
                crop_xyxy=np.array([x0,y0,x1,y1]),source_pts=np.array([f['pts'] for f in frames]))
            median_picture=np.clip(np.nan_to_num(median),0,255).astype(np.uint8)
            comparison=np.vstack([image[y0:y1,x0:x1],median_picture])
            cv2.imwrite(str(folder/'source_and_median.png'),comparison)
            complete=np.zeros_like(image);complete[y0:y1,x0:x1]=median_picture
            cv2.imwrite(str(folder/'identity_navigation_composite.png'),complete)
            receipt=dict(shot=shot_id,reference_pts=reference_pts,source_frames=len(frames),source_sha256=visibility['source_video_sha256'],
                state=str(args.state),state_sha256=run.digest(args.state),native_library_sha256=manifest['native_library_sha256'],
                source_pixel_bindings=[dict(pts=f['pts'],source_BGR_sha256=f['source_BGR_sha256']) for f in frames],
                mapping='native_plane_projection_piecewise_image_map',crop_xyxy=[x0,y0,x1,y1],
                source_count_min=int(count.min()),source_count_max=int(count.max()),
                purpose='material_identity_navigation_only',not_a_native_source_exposure=True,
                fitting_UV_created=False,observations_changed=False,uncertain_current_geometry_is_not_ground_truth=True)
            run.write_json(folder/'receipt.json',receipt)
            print(json.dumps({k:v for k,v in receipt.items() if k!='source_pixel_bindings'}),flush=True)
