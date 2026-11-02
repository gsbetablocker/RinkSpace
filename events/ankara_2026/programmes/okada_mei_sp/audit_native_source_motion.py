"""Read-only consecutive-source-frame check of an explicit native camera state.

Source optical flow is measured on stationary board texture. A source keypoint
is intersected with the current native board once, then the same 3D point is
projected at the next native exposure. No optimizer or source admission runs.
This is a local temporal consistency diagnostic, not independent calibration.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import cv2
import av
from rinkspace.core.native import NativeCore
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.path_policy import guard_project_write
import prepare_native_multicamera as prepared
from export_native_six_review import _verify_native_library_binding


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def stats(values):
    values=np.asarray(values)
    if not len(values):return None
    return dict(median=float(np.median(values)),p95=float(np.quantile(values,.95)),max=float(values.max()))


def cross(a,b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def board_intersections(core,rig,controls,first,dt,frame_time,uv,board):
    row_times=frame_time+rig[37]*(uv[:,1]/1079-.5)
    camera=core.evaluate_motion_camera(rig,controls,first,dt,row_times)
    if np.any(camera['status']):raise ValueError('Invalid row camera')
    rays=[]
    for j,pixel in enumerate(uv):
        K=np.array([[camera['fx'][j],0,camera['cx'][j]],[0,camera['fy'][j],camera['cy'][j]],[0,0,1.]])
        distortion=np.array([camera[k][j] for k in ('k1','k2','p1','p2')]+[0.])
        undistort=getattr(cv2,'undistortPointsIter',cv2.undistortPoints)
        point=undistort(pixel.reshape(1,1,2),K,distortion,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,80,1e-12))[0,0]
        rays.append([*point,1.])
    direction=np.einsum('nij,ni->nj',camera['rotation'],np.asarray(rays))
    centre=camera['centre']
    perimeter=core.board_perimeter(board)
    ss=np.linspace(0,perimeter,8193)
    boundary=np.asarray(core.board_points(board,ss,np.zeros(len(ss))))
    start=boundary[:-1,:2];edge=np.diff(boundary[:,:2],axis=0)
    offset=start[None,:,:]-centre[:,None,:2]
    denominator=cross(direction[:,None,:2],edge[None,:,:])
    with np.errstate(divide='ignore',invalid='ignore'):
        distance=cross(offset,edge[None,:,:])/denominator
        station=cross(offset,direction[:,None,:2])/denominator
        height=centre[:,None,2]+distance*direction[:,None,2]
    valid=(abs(denominator)>1e-14)&(distance>0)&(station>=0)&(station<=1)&(height>0)&(height<float(board[5]))
    candidates=np.where(valid,distance,np.inf)
    distance=np.min(candidates,axis=1);good=np.isfinite(distance)
    xyz=centre[good]+distance[good,None]*direction[good]
    return good,xyz


def main(args):
    started=time.monotonic();out=guard_project_write(args.output)
    if out.exists() or out.with_suffix('.npz').exists():raise FileExistsError(out)
    m=prepared.read(args.run_root/'joint_manifest.json');v=prepared.npz(args.state)
    _verify_native_library_binding(m,v)
    watched=[args.state,args.source,args.run_root/'point_observations.npz',args.run_root/'joint_manifest.json',Path(m['native_library'])]
    hashes={str(p):digest(p) for p in watched}
    if hashes[str(args.run_root/'point_observations.npz')]!=m['point_input_sha256']:raise ValueError('Point identity differs')
    if hashes[m['native_library']]!=m['native_library_sha256']:raise ValueError('Native library changed')
    dependency=json.loads(str(v['joint_numeric_dependencies_json']))
    if dependency['shots']!=m['shots']:raise ValueError('State and source exposure clock differ')
    z=prepared.npz(args.run_root/'point_observations.npz');mapping=prepared.read(args.run_root/'mapping.json')
    names=np.asarray(mapping['path_ids'])[z['path_indices']]
    timeline=ShotMotionTimeline(m['shots'],m['control_dt'],m['first_control_time']);core=NativeCore(m['native_library'])
    records=[];arrays={}
    for number,window in enumerate(args.window):
        sid,aa,bb=window.split(':');begin,end=int(aa),int(bb)
        shot=next(s for s in m['shots'] if s['id']==sid)
        if begin<shot['first_pts'] or end>shot['last_pts'] or (begin-shot['first_pts'])%256 or (end-begin)%256:
            raise ValueError('Window must lie on one registered source shot')
        frames={}
        with av.open(str(args.source)) as container:
            stream=container.streams.video[0]
            if stream.time_base.numerator!=1 or stream.time_base.denominator!=12800:raise ValueError('Wrong source clock')
            container.seek(begin,stream=stream,backward=True)
            for frame in container.decode(stream):
                if frame.pts>end:break
                if frame.pts>=begin:frames[int(frame.pts)]=frame.to_ndarray(format='gray')
        for pts in range(begin,end,256):
            if pts not in frames or pts+256 not in frames:raise ValueError('Missing exact source exposure')
            source,next_source=frames[pts],frames[pts+256];mask=np.zeros_like(source)
            edge_pairs=[('board/continuous/top','board/continuous/bottom')]
            present=set(names[z['pts']==pts])
            for path in sorted(present):
                if path.startswith('print_instance/') and path.endswith('.top_edge/path0'):
                    other=path.replace('.top_edge/path0','.bottom_edge/path0')
                    if other in present:edge_pairs.append((path,other))
            support=[]
            for pair in edge_pairs:
                edges=[]
                for path,offset in zip(pair,(8,-12)):
                    uv=z['uv'][(z['pts']==pts)&(names==path)]
                    if len(uv)<2:break
                    uv=uv[np.argsort(uv[:,0])]
                    edges.append(np.interp(np.arange(1920),uv[:,0],uv[:,1],left=np.nan,right=np.nan)+offset)
                if len(edges)!=2:continue
                support.append(pair)
                for x in range(16,1904):
                    if np.isfinite([edges[0][x],edges[1][x]]).all():
                        lo=max(0,int(edges[0][x]));hi=min(1080,int(edges[1][x]))
                        if hi>lo:mask[lo:hi,x]=255
            if args.board_roi:
                x,y,w,h=args.board_roi
                if min(x,y)<0 or min(w,h)<=0 or x+w>1920 or y+h>1080:raise ValueError('Source-board ROI outside original image')
                mask[:]=0;mask[y:y+h,x:x+w]=255;support=[['explicit_original_source_board_interior_ROI',list(args.board_roi)]]
            if args.person_masks:
                for exposure in (pts,pts+256):
                    mp=args.person_masks/sid/(str(exposure)+'.png')
                    if not mp.is_file():raise ValueError('Missing same-exposure source person mask')
                    hashes[str(mp)]=digest(mp)
                person=cv2.imread(str(args.person_masks/sid/(str(pts)+'.png')),0)
                mask[person>0]=0
            record=dict(shot=sid,pts=pts,next_pts=pts+256,source_seconds=pts/12800,mask_source_paths=support)
            if not support:
                records.append(dict(**record,status='missing_source_board_or_ad_edges'));continue
            mask[:,500:1240]=0
            points=cv2.goodFeaturesToTrack(source,1000,.005,5,mask=mask)
            if points is None:
                records.append(dict(**record,status='insufficient_source_texture'));continue
            nxt,ok,_=cv2.calcOpticalFlowPyrLK(source,next_source,points,None,winSize=(31,31),maxLevel=5,
                criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,60,.001))
            back,back_ok,_=cv2.calcOpticalFlowPyrLK(next_source,source,nxt,None,winSize=(31,31),maxLevel=5,
                criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,60,.001))
            fb=np.linalg.norm(back[:,0]-points[:,0],axis=1)
            valid=ok[:,0].astype(bool)&back_ok[:,0].astype(bool)&(fb<1.)
            old=points[valid,0].astype(float);new=nxt[valid,0].astype(float)
            if args.person_masks:
                next_person=cv2.imread(str(args.person_masks/sid/(str(pts+256)+'.png')),0);inside=(new[:,0]>=0)&(new[:,0]<1919)&(new[:,1]>=0)&(new[:,1]<1079);keep=inside.copy();ii=np.flatnonzero(inside);keep[ii]&=next_person[np.rint(new[ii,1]).astype(int),np.rint(new[ii,0]).astype(int)]==0;old,new=old[keep],new[keep]
            if len(old)<12:
                records.append(dict(**record,status='insufficient_consistent_source_tracks',tracks=len(old)));continue
            rig=v['rig'][shot['camera_index']]
            tt=timeline.exposure_time(shot,pts)
            valid,xyz=board_intersections(core,rig,v['controls'],timeline.first,timeline.dt,tt,old,v['board'])
            old,new=old[valid],new[valid]
            if len(old)<12:
                records.append(dict(**record,status='insufficient_board_intersections',tracks=len(old)));continue
            at=[core.project_motion(rig,v['controls'],timeline.first,timeline.dt,1080,np.full(len(xyz),t),xyz) for t in (tt,timeline.exposure_time(shot,pts+256))]
            seed_error=np.linalg.norm(at[0]['uv']-old,axis=1)
            valid=(at[0]['status']==0)&(at[1]['status']==0)&(seed_error<1e-4)
            old,new=old[valid],new[valid];predicted=at[1]['uv'][valid]
            if len(old)<12:
                records.append(dict(**record,status='insufficient_valid_inverse_rays',tracks=len(old)));continue
            error=np.linalg.norm(predicted-new,axis=1)
            records.append(dict(**record,status='compared',tracks=len(old),
                source_dx_median_px=float(np.median(new[:,0]-old[:,0])),native_dx_median_px=float(np.median(predicted[:,0]-old[:,0])),
                source_dy_median_px=float(np.median(new[:,1]-old[:,1])),native_dy_median_px=float(np.median(predicted[:,1]-old[:,1])),
                source_displacement_px=stats(np.linalg.norm(new-old,axis=1)),native_vs_source_displacement_px=stats(error),
                inverse_source_reprojection_max_px=float(seed_error[valid].max())))
            for key,array in dict(source_uv=old,next_source_uv=new,native_next_uv=predicted).items():arrays[f'{sid}_{pts}_{key}']=array
        print(json.dumps(dict(window=window,completed_pairs=sum(r['shot']==sid for r in records)),ensure_ascii=True),flush=True)
    for path,sha in hashes.items():
        if digest(path)!=sha:raise ValueError('Diagnostic input changed: '+path)
    result=dict(schema='rinkspace.source_camera_temporal_check.v1',state=str(args.state),identities=hashes,records=records,
        optimizer_started=False,state_modified=False,source_uv_modified=False,observations_added_to_fit=False,explicit_source_board_roi=args.board_roi,source_person_masks=None if args.person_masks is None else str(args.person_masks),measurement_implementation_sha256=digest(__file__),
        method='Forward/backward raw-source LK in source-measured board strips, central moving-person exclusion; inverse each initial source ray onto a dense native-board polygon, then project the same static 3D point at the next exposure through full native rolling shutter.',
        limits=['Local temporal check on development footage, not independent physical calibration.',
                'Board polygon is a numerical intersection aid; no board or camera variable is fitted.',
                'Optical flow under motion blur and repeated print may be unreliable even after the one-pixel forward/backward check.',
                'Mask excludes x=500..1239. Default support uses source board/ad edges; an explicit ROI is a separately declared source-image selection, not invented edge observations. Missing support is reported.'],
        elapsed_seconds=time.monotonic()-started)
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out.with_suffix('.npz'),**arrays)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+chr(10),encoding='utf-8')
    print(str(out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','state','source','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--window',action='append',required=True,help='shot:first_integer_PTS:last_integer_PTS')
    parser.add_argument('--board-roi',type=int,nargs=4,metavar=('X','Y','W','H'),help='Explicit actual-source board-interior region for auxiliary motion inspection')
    parser.add_argument('--person-masks',type=Path,help='Same-exposure foreground mask root')
    main(parser.parse_args())
