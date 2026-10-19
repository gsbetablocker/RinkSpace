"""Small physical-camera diagnostic driven by real secondary source motion.

Use a separately audited absolute source resection as a seed. All later views
share its optical centre and zero-skew, centred, square-pixel pinhole lens.
Static image correspondences estimate only world-to-camera rotation and zoom.
This is an explicit conditional model, not a production release or a substitute
for independent ice/board checks. Decode actual native frames sequentially;
confirmed duplicate exposure groups share one candidate camera exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import av
import cv2
import numpy as np
from scipy.optimize import least_squares
from ice_refine import _curve_error, _signals, detect_contact

ROOT=Path('D:/OneDrive/codex_projs/RinkSpace')
B = ROOT/'outputs/three_camera_joint_20260930/run11_joint_C_LM_diagnostic/B_field/programme_review'
GROUPS = Path('D:/CodexData/RinkSpace_program_tracking_tmp/source_cadence_comparison/O6j8L8TmfKo/full_programme_alignment/exposure_groups.json')
PEOPLE = Path('D:/CodexData/RinkSpace_program_tracking_tmp/people/boxes.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_mask(pts, people):
    mask = np.full((540,960),255,np.uint8)
    mask[:135,:305]=0
    mask[450:,835:]=0
    for box in people.get(int(pts),[]):
        x0,y0,x1,y1=np.rint(np.asarray(box[:4])).astype(int)
        mask[max(0,y0-15):min(540,y1+15),max(0,x0-15):min(960,x1+15)]=0
    return mask


def tracks(first, second, first_pts, second_pts, people):
    xy=cv2.goodFeaturesToTrack(first,450,.007,9,mask=source_mask(first_pts,people),blockSize=7)
    if xy is None:
        raise ValueError('No real static source features')
    nxt,good,_=cv2.calcOpticalFlowPyrLK(first,second,xy,None,winSize=(29,29),maxLevel=3,
                                      criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,25,.01))
    back,backgood,_=cv2.calcOpticalFlowPyrLK(second,first,nxt,None,winSize=(29,29),maxLevel=3,
                                          criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,25,.01))
    a,b=xy.reshape(-1,2),nxt.reshape(-1,2)
    keep=good.ravel().astype(bool)&backgood.ravel().astype(bool)&(np.linalg.norm(back.reshape(-1,2)-a,axis=1)<.8)
    keep&=(b[:,0]>=0)&(b[:,0]<960)&(b[:,1]>=0)&(b[:,1]<540)
    valid_ids=np.flatnonzero(keep);target_mask=source_mask(second_pts,people)
    if len(valid_ids):
        pix=np.rint(b[valid_ids]).astype(int);pix[:,0]=np.clip(pix[:,0],0,959);pix[:,1]=np.clip(pix[:,1],0,539)
        keep[valid_ids]&=target_mask[pix[:,1],pix[:,0]]>0
    a,b=a[keep],b[keep]
    if len(a)<12:
        raise ValueError('Insufficient source-only LK support')
    _,inlier=cv2.findHomography(a,b,cv2.RANSAC,1.1,maxIters=1000,confidence=.999)
    if inlier is None or int(inlier.sum())<12:
        raise ValueError('Source static motion has no supported homography')
    return 2.*a[inlier.ravel().astype(bool)],2.*b[inlier.ravel().astype(bool)]


def source_board_pixels(image, pts, people):
    """Read native blue-cap top and yellow kick-strip bottom, without a camera."""
    small=cv2.resize(image,(960,540));hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    allowed=source_mask(pts,people)>0
    output={}
    for name,low,high,edge in [('board_top',90,128,'first'),('ice_board_contact',17,40,'last')]:
        mask=((hsv[:,:,0]>=low)&(hsv[:,:,0]<=high)&(hsv[:,:,1]>45)&
              (hsv[:,:,2]>45)&allowed).astype('uint8')
        _,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        ids=[i for i in range(1,len(stats)) if stats[i,cv2.CC_STAT_WIDTH]>180 and
             stats[i,cv2.CC_STAT_WIDTH]>10*stats[i,cv2.CC_STAT_HEIGHT]]
        if not ids:
            output[name]=np.empty((0,2));continue
        maximum=max(stats[i,cv2.CC_STAT_WIDTH] for i in ids)
        selected=np.isin(labels,[i for i in ids if stats[i,cv2.CC_STAT_WIDTH]>.3*maximum])
        pixels=[]
        for x in range(15,946,24):
            ys=np.flatnonzero(np.count_nonzero(selected[:,x-2:x+3],axis=1)>=3)
            if len(ys):
                pixels.append([2*x,2*(ys[0] if edge=='first' else ys[-1])])
        output[name]=np.array(pixels,float).reshape(-1,2)
    # A pale kick strip often has less than 45 HSV saturation. Read the
    # actual source chroma profile, including its falling ice-side edge.
    exclusion=cv2.resize((~allowed).astype('uint8'),(1920,1080),interpolation=cv2.INTER_NEAREST)>0
    _,contact=detect_contact(image,_signals(image),exclusion)
    if len(contact):
        output['ice_board_contact']=contact
    return output


def camera_from_matches(previous, a, b, *, guess=None, board_pixels=None, board_curves=None, board_weight=1/3):
    rotation=cv2.Rodrigues(previous[3:6])[0]
    focal=np.exp(previous[6])
    ray=np.c_[(a-previous[8:10])/focal,np.ones(len(a))]@rotation
    initial=np.r_[previous[3:6],previous[6]] if guess is None else np.r_[guess[3:6],np.mean(guess[6:8])]

    def projected(q):
        local=ray@cv2.Rodrigues(q[:3])[0].T
        return np.exp(q[3])*local[:,:2]/local[:,2:]+previous[8:10]

    def jacobian(q):
        focal=np.exp(q[3]);k=np.array([[focal,0,previous[8]],[0,focal,previous[9]],[0,0,1.]])
        _,raw=cv2.projectPoints(np.ascontiguousarray(ray),q[:3],np.zeros(3),k,np.zeros(5))
        parts=[np.column_stack([raw[:,:3],focal*(raw[:,6]+raw[:,7])])]
        if board_pixels is not None:
            camera=previous.copy();camera[3:6]=q[:3];camera[6:8]=q[3]
            for name,pixels in board_pixels.items():
                if len(pixels):
                    _,raw_board=_curve_error(camera,board_curves[name],pixels,True)
                    parts.append(np.column_stack([raw_board[:,3:6],raw_board[:,6]+raw_board[:,7]])*board_weight)
        return np.vstack(parts)

    def residual(q):
        parts=[(projected(q)-b).ravel()]
        if board_pixels is not None:
            camera=previous.copy();camera[3:6]=q[:3];camera[6:8]=q[3]
            for name,pixels in board_pixels.items():
                if len(pixels):
                    parts.append(_curve_error(camera,board_curves[name],pixels)*board_weight)
        return np.concatenate(parts)

    # Source-only RANSAC has already removed motion outliers. Huber from a
    # large pan's initial 50px residual can terminate with a huge gradient;
    # first resect the supported correspondences, then robustly polish them.
    linear=least_squares(residual,initial,jac=jacobian,loss='linear',max_nfev=40,
                         x_scale='jac',ftol=1e-7,xtol=1e-9,gtol=1e-6)
    result=least_squares(residual,linear.x,jac=jacobian,loss='huber',f_scale=2.,max_nfev=15,
                         x_scale='jac',ftol=1e-7,xtol=1e-9,gtol=1e-6)
    camera=previous.copy();camera[3:6]=result.x[:3];camera[6:8]=result.x[3]
    errors=np.linalg.norm(projected(result.x)-b,axis=1)
    return camera,dict(matches=len(a),median_px=float(np.median(errors)),
                        p95_px=float(np.percentile(errors,95)),nfev=linear.nfev+result.nfev,
                        optimality=float(result.optimality))


def sift_bridge(first, second, first_pts, second_pts, people):
    detector=cv2.SIFT_create(nfeatures=2500,contrastThreshold=.012)
    a,da=detector.detectAndCompute(first,source_mask(first_pts,people))
    b,db=detector.detectAndCompute(second,source_mask(second_pts,people))
    if da is None or db is None:
        raise ValueError('No real source bridge descriptors')
    matching=cv2.BFMatcher().knnMatch(da,db,k=2)
    good=[m for m,n in matching if m.distance<.72*n.distance]
    if len(good)<12:
        raise ValueError('No real source SIFT bridge support')
    xy0=np.array([a[m.queryIdx].pt for m in good]);xy1=np.array([b[m.trainIdx].pt for m in good])
    _,inlier=cv2.findHomography(xy0,xy1,cv2.RANSAC,2.,maxIters=3000,confidence=.999)
    if inlier is None or int(inlier.sum())<12:
        raise ValueError('Bridge static source homography is unsupported')
    keep=inlier.ravel().astype(bool)
    return 2.*xy0[keep],2.*xy1[keep]


def run(output, seed_file, cached_source=None, absolute_boards=False, far_board_only=False, board_weight=1/3):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if not str(output).startswith('D:\\CodexData\\'):
        raise ValueError('Large intermediate output must be in D:/CodexData')
    seed=json.loads(Path(seed_file).read_text(encoding='utf8'))
    first_pts=int(seed['pts']);seed_camera=np.asarray(seed['candidate_camera'],float)
    if np.any(seed_camera[11:]!=0) or seed_camera[10]!=0 or not np.all(seed_camera[8:10]==[960,540]) or seed_camera[6]!=seed_camera[7]:
        raise ValueError('This diagnostic requires the declared centred square-pixel zero-skew pinhole seed')
    source=json.loads((ROOT/'three_frame_refit/program_tracking/source_shots.json').read_text(encoding='utf8'))
    groups=json.loads(GROUPS.read_text(encoding='utf8'))
    people={int(row['pts']):row['boxes'] for row in json.loads(PEOPLE.read_text(encoding='utf8'))['frames']}
    by_shot={shot:[g for g in groups['groups'] if g['shot_index']==shot] for shot in (2,6)}
    canonical={int(p):int(g['full_pts'][0]) for gs in by_shot.values() for g in gs for p in g['full_pts']}
    wanted=sorted(set(canonical.values()));positions={p:i for i,p in enumerate(wanted)}
    frames=(np.load(cached_source,mmap_mode='r') if cached_source else
            np.lib.format.open_memmap(output/'source_gray.npy',mode='w+',dtype='uint8',shape=(len(wanted),540,960)))
    if frames.shape!=(len(wanted),540,960) or frames.dtype!=np.uint8:
        raise ValueError('Real source grayscale cache differs')
    started=time.perf_counter();decoded=set();boards={}
    scene=json.loads((B/'candidate_scene.json').read_text(encoding='utf8'))
    board_curves={name:np.asarray(scene[geom]['points_xyz'] if geom=='ice_boundary' else scene[geom]['top_edge_xyz'],float)
                  for name,geom in [('ice_board_contact','ice_boundary'),('board_top','board_surface')]}
    board_curves={name:np.vstack([q,q[0]]) if not np.array_equal(q[0],q[-1]) else q for name,q in board_curves.items()}
    if far_board_only:
        y=scene['parameters']['bounds'][3];height=scene['parameters']['board_height']
        board_curves={name:np.c_[np.linspace(-65,65,240),np.full(240,y),np.full(240,z)]
                      for name,z in [('board_top',height),('ice_board_contact',0.)]}
    with av.open(source['source_path']) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        for shot in (2,6):
            start=source['shots'][shot]['start_pts'];end=source['shots'][shot]['end_pts']
            container.seek(start-15360,stream=stream,backward=True)
            for frame in container.decode(stream):
                if frame.pts is None or frame.pts<start:
                    continue
                if frame.pts>=end:
                    break
                if frame.pts in positions:
                    if not cached_source:
                        frames[positions[frame.pts]]=cv2.resize(frame.to_ndarray(format='gray'),(960,540))
                    decoded.add(int(frame.pts))
                    if absolute_boards:
                        boards[int(frame.pts)]=source_board_pixels(frame.to_ndarray(format='bgr24'),int(frame.pts),people)
    if decoded!=set(wanted):
        raise ValueError('Canonical real source decode coverage differs')
    print(json.dumps(dict(stage='decoded',canonical_frames=len(wanted),seconds=time.perf_counter()-started)),flush=True)
    camera_by_pts={canonical[first_pts]:seed_camera.copy()};rows=[];all_matches=[]

    def propagate(sequence):
        for a,b in zip(sequence[:-1],sequence[1:]):
            xy0,xy1=tracks(frames[positions[a]],frames[positions[b]],a,b,people)
            camera,metrics=camera_from_matches(camera_by_pts[a],xy0,xy1,
                                               board_pixels=boards.get(b),board_curves=board_curves,board_weight=board_weight)
            camera_by_pts[b]=camera
            rows.append(dict(source_pts=a,target_pts=b,source_kind='real static image LK + source-only RANSAC',**metrics))
            all_matches.append((a,b,xy0,xy1))

    shot2=[int(g['full_pts'][0]) for g in by_shot[2]];anchor=shot2.index(canonical[first_pts])
    propagate(shot2[anchor:]);propagate(shot2[:anchor+1][::-1])
    # A cut creates a new localization. It does not transfer another camera's
    # matrix. Here the verified source pixels explicitly bridge the same
    # classified physical installation, and the conditional lens is refitted.
    with np.load(B/'candidate_poses.npz') as z:
        old={int(p):c.copy() for p,c in zip(z['pts'],z['camera25'])}
    shot6=[int(g['full_pts'][0]) for g in by_shot[6]]
    np.savez_compressed(output/'shot2_partial_candidate.npz',pts=np.array(shot2),camera25=np.array([camera_by_pts[p] for p in shot2]))
    (output/'shot2_partial_source_diagnostics.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
    bridge_candidates=[]
    bridge_failures=[]
    for bridge_target in shot6[::45]:
        for bridge_source in shot2[::45]:
            try:
                a,b=sift_bridge(frames[positions[bridge_source]],frames[positions[bridge_target]],bridge_source,bridge_target,people)
                c,metrics=camera_from_matches(camera_by_pts[bridge_source],a,b,guess=old[bridge_target],
                                               board_pixels=boards.get(bridge_target),board_curves=board_curves,board_weight=board_weight)
                bridge_failures.append(dict(source_pts=bridge_source,target_pts=bridge_target,**metrics))
                if metrics['median_px']<8:
                    bridge_candidates.append((metrics['median_px']+30./len(a),bridge_source,bridge_target,c,metrics,a,b))
            except (ValueError,cv2.error) as error:
                bridge_failures.append(dict(source_pts=bridge_source,target_pts=bridge_target,error=str(error)))
    (output/'cross_shot_bridge_all_diagnostics.json').write_text(json.dumps(bridge_failures,indent=2),encoding='utf8')
    if not bridge_candidates:
        raise ValueError('No cross-shot source bridge meets diagnostic support; shot6 remains unknown')
    _,bridge_source,bridge_target,c,metrics,a,b=min(bridge_candidates,key=lambda row:row[0]);camera_by_pts[bridge_target]=c
    rows.append(dict(source_pts=bridge_source,target_pts=bridge_target,source_kind='real same-installation cross-shot SIFT bridge',**metrics));all_matches.append((bridge_source,bridge_target,a,b))
    print(json.dumps(dict(stage='bridged_shot6',source_pts=bridge_source,target_pts=bridge_target,**metrics)),flush=True)
    anchor6=shot6.index(bridge_target);propagate(shot6[anchor6:]);propagate(shot6[:anchor6+1][::-1])
    native_pts=np.array(sorted(canonical),np.int64)
    camera25=np.array([camera_by_pts[canonical[int(p)]] for p in native_pts])
    np.savez_compressed(output/'candidate_cameras.npz',pts=native_pts,camera25=camera25,
                        exposure_representative_pts=np.array([canonical[int(p)] for p in native_pts]),
                        source_sha256=source['source_sha256_from_timeline'],model_sha256=sha(B/'candidate_scene.json'),
                        status=np.array(['source_ray_camera_candidate_unvalidated']*len(native_pts)))
    np.savez_compressed(output/'source_motion_matches.npz',pair_pts=np.array([[a,b] for a,b,_,_ in all_matches]),
                        offset=np.r_[0,np.cumsum([len(x) for _,_,x,_ in all_matches])],
                        source_uv=np.vstack([x for _,_,x,_ in all_matches]),target_uv=np.vstack([x for _,_,_,x in all_matches]))
    report=dict(schema='rinkspace.secondary-source-ray-camera-diagnostic.v1',
                method='fixed conditional optical centre, centred square-pixel zero-skew pinhole; per-exposure R/logf from real static rays',
                source_path=source['source_path'],source_sha256=source['source_sha256_from_timeline'],
                model_path=str(B/'candidate_scene.json'),model_sha256=sha(B/'candidate_scene.json'),
                seed_path=str(Path(seed_file).resolve()),seed_sha256=sha(seed_file),
                exposure_groups_sha256=sha(GROUPS),native_frames=len(native_pts),canonical_frames=len(wanted),
                confirmed_aliases=len(native_pts)-len(wanted),scope_shots=[2,6],
                rows=rows,seconds=time.perf_counter()-started,scene_geometry_changed=False,
                source_pixels_changed=False,camera25_sha256=sha(output/'candidate_cameras.npz'),
                static_source_support_is_not_absolute_ice_validation=True,accuracy_accepted=False)
    report['absolute_blue_top_yellow_contact_source_enabled']=absolute_boards
    report['far_board_straight_branch_only']=far_board_only
    report['absolute_board_residual_weight']=board_weight
    if absolute_boards:
        pts_board=sorted(boards)
        for name in board_curves:
            np.savez_compressed(output/(name+'_source_pixels.npz'),pts=np.array(pts_board),
                offset=np.r_[0,np.cumsum([len(boards[p][name]) for p in pts_board])],
                uv=np.vstack([boards[p][name] for p in pts_board]))
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(stage='complete',native_frames=len(native_pts),seconds=report['seconds'],
                          max_step_median_px=max(row['median_px'] for row in rows))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=Path,required=True)
    parser.add_argument('--cached-source',type=Path)
    parser.add_argument('--absolute-boards',action='store_true')
    parser.add_argument('--far-board-only',action='store_true')
    parser.add_argument('--board-weight',type=float,default=1/3)
    args=parser.parse_args();run(args.output,args.seed,args.cached_source,args.absolute_boards,args.far_board_only,args.board_weight)
