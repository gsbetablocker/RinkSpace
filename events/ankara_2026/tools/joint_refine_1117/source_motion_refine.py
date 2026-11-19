"""Bidirectional registration of visible, stationary board material at source PTS."""
from argparse import ArgumentParser
from collections import deque
import cv2
import numpy as np
from scipy.spatial import cKDTree

from lane_end_evidence import BASE, WORK, foreground
from review import camera_state, ji, native_for, source_frames
from shared import PREPARED, arrays, read, save, write
from source_paint import matte
from flow_depths import bind


def board_grid(native,shot,base,name):
    board=native.scene['board'];core=native.core
    st=np.linspace(0,core.board_perimeter(board),385)
    line=np.asarray(core.board_points(board,st,np.zeros(len(st))))
    tangent=np.gradient(line[:,:2],axis=0)
    inward=np.column_stack((tangent[:,1],-tangent[:,0]))
    inward*=np.where(np.sum(-line[:,:2]*inward,axis=1)>0,1.,-1.)[:,None]
    world=np.stack([line+[0,0,z*board[5]] for z in (0.,.5,1.)],axis=1).reshape(-1,3)
    times=base[name+'_times'];rigs=base[name+'_rigs'];c=base[name+'_controls']
    centres=np.empty((len(times),3))
    unique,inv=np.unique(rigs,axis=0,return_inverse=True)
    for j,rig in enumerate(unique):
        centres[inv==j]=core.evaluate_motion_camera(rig,c,shot['first'],shot['dt'],times[inv==j])['centre']
    uv=[];valid=[]
    for lo in range(0,len(times),64):
        jobs=[(i,'wall',times[i]) for i in range(lo,min(lo+64,len(times)))]
        for (p,status),(i,_,_) in zip(ji.project_groups(native,rigs,c,shot['first'],shot['dt'],jobs,{'wall':world}),jobs):
            p=p.reshape(-1,3,2)
            ok=(status.reshape(-1,3)==0).all(1)&(np.sum((centres[i,:2]-line[:,:2])*inward,axis=1)>0)
            ok&=np.all(np.abs(p)<25000,axis=(1,2))
            uv.append(np.nan_to_num(p,nan=0.).astype(np.float32));valid.append(ok)
    return np.array(uv),np.array(valid)


def image_features(raw,mask,grid,valid):
    # Require patches to lie inside the board on the source image, away from
    # the foot, cap, occlusions and broadcast graphics on both exposures.
    shape=(540,960)
    gray=cv2.resize(cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY),shape[::-1],interpolation=cv2.INTER_AREA)
    safe=np.zeros(shape,np.uint8)
    q=grid*.5
    use=(valid[:-1]&valid[1:]&(q[:-1].max(axis=1)>[-80,-80]).all(1)&
         (q[:-1].min(axis=1)<[1040,620]).all(1))
    for j in np.flatnonzero(use):
        poly=q[[j,j+1,j+1,j],[0,0,2,2]]
        cv2.fillConvexPoly(safe,np.rint(poly*16).astype(np.int32),255,shift=4)
    if mask is None:safe.fill(0)
    else:safe[cv2.resize(mask,shape[::-1],interpolation=cv2.INTER_NEAREST)!=0]=0
    safe[:80,:296]=0;safe[:147,:227]=0;safe[464:,809:]=0
    safe=cv2.erode(safe,np.ones((19,19),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    corners=cv2.goodFeaturesToTrack(gray,maxCorners=220,qualityLevel=.013,minDistance=13,mask=safe,blockSize=5)
    points=np.empty((0,2),np.float32)
    if corners is not None:
        coords=corners.reshape(-1,2)
        bins=(coords[:,0]//50).astype(int)+20*(coords[:,1]//48).astype(int)
        _,ix=np.unique(bins,return_index=True);points=coords[np.sort(ix)].copy()
        cv2.cornerSubPix(gray,points,(3,3),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,12,.025))
    return gray,safe,points


def texture_registration(left,right,grid0,grid1,valid):
    src,_,points=left;dst,safe,_=right
    if len(points)<2 or not valid.any():return None
    a=grid0[valid].reshape(-1,2)*.5;b=grid1[valid].reshape(-1,2)*.5
    dist,idx=cKDTree(a).query(points,k=min(3,len(a)))
    w=1/(dist+1.)**2
    initial=points+np.sum((b-a)[idx]*w[...,None],axis=1)/w.sum(1)[:,None]
    ok=(initial>=[11,11]).all(1)&(initial<[949,529]).all(1)
    if not ok.any():return None
    x=points[ok].copy();y=initial[ok].astype(np.float32)
    opts=dict(winSize=(19,19),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,24,.025),
              flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    y,fwd,_=cv2.calcOpticalFlowPyrLK(src,dst,x,y,**opts)
    back,rev,_=cv2.calcOpticalFlowPyrLK(dst,src,y,x.copy(),**opts)
    good=(fwd.ravel()!=0)&(rev.ravel()!=0)&(np.linalg.norm(back-x,axis=1)<.42)
    good&=(y>=[10,10]).all(1)&(y<[950,530]).all(1)
    ix=np.flatnonzero(good);coord=np.rint(y[ix]).astype(int)
    ix=ix[safe[coord[:,1],coord[:,0]]!=0]
    retained=[];quality=[]
    for j in ix:
        u=cv2.getRectSubPix(src,(17,17),tuple(x[j])).astype(np.float32)
        v=cv2.getRectSubPix(dst,(17,17),tuple(y[j])).astype(np.float32)
        u-=u.mean();v-=v.mean()
        norm=np.linalg.norm(u)*np.linalg.norm(v)
        corr=float(np.sum(u*v)/max(norm,1e-6))
        if corr>=.87 and min(u.std(),v.std())>4.:
            retained.append(j);quality.append(corr)
    if not retained:return None
    # centre of a 2x2 area-downsampled source pixel is (2u+.5,2v+.5).
    return 2*x[retained]+.5,2*y[retained]+.5,quality


def scan(programme):
    cv2.setNumThreads(1)
    folder=WORK/'texture';folder.mkdir(exist_ok=True)
    setup=read(PREPARED/'setup.json')['shots'];base=camera_state(BASE);native=native_for(BASE,programme)
    for name,shot in setup.items():
        if shot['program']!=programme:continue
        dest=folder/(name+'.npz')
        if dest.exists():continue
        pts=base[name+'_pts'];grids,valid=board_grid(native,shot,base,name)
        masks=matte(shot,pts);history=deque(maxlen=8);indices=[];lags=[];old=[];new=[];quality=[]
        for i,(pt,raw) in enumerate(source_frames(shot['source'],pts)):
            mask=foreground(shot,masks,i,pt)
            item=image_features(raw,mask,grids[i],valid[i])
            for lag in (1,8):
                if len(history)<lag:continue
                obs=texture_registration(history[-lag],item,grids[i-lag],grids[i],valid[i-lag]&valid[i])
                if obs is None:continue
                left,right,confidence=obs
                indices.extend([i]*len(left));lags.extend([lag]*len(left));old.extend(left);new.extend(right);quality.extend(confidence)
            history.append(item)
        index=np.array(indices,int);lag=np.array(lags,int)
        left_uv=np.array(old).reshape(-1,2);uv=np.array(new).reshape(-1,2)
        world,res=bind(native,shot,base[name+'_rigs'],base[name+'_controls'],base[name+'_times'],
            index-lag,left_uv,np.zeros(len(index),int))
        good=np.isfinite(world).all(1)
        # Hold out a separate horizontal band of source texture in each shot.
        # Original fitted observations and previous comparisons are development data.
        held=((np.floor(left_uv[:,0]/120).astype(int)+int(shot['shot'][1:]))%5)==0
        save(dest,pts=pts,index=index,lag=lag,uv=uv,left_uv=left_uv,world=world,quality=np.array(quality),
            valid=good,held=held,source_sha256=shot['source_sha256'],inverse_error=res)
        print(name,len(index),'pairs',int((good&held).sum()),'held',flush=True)


def verify_matches():
    """Reject repeated typography or a moving patch using only source correspondences."""
    folder=WORK/'texture';report={}
    for path in sorted(folder.glob('*.npz')):
        z=arrays(path);ix=np.flatnonzero(z['valid']);index=z['index'];lag=z['lag']
        k=index[ix]*16+lag[ix];order=ix[np.argsort(k,kind='stable')];codes=index[order]*16+lag[order]
        good=np.zeros(len(index),bool)
        for rows in np.split(order,np.flatnonzero(np.diff(codes))+1):
            if len(rows)<5:continue
            left=z['left_uv'][rows];right=z['uv'][rows]
            if np.ptp(left,axis=0).max()<100:continue
            # A local similarity handles PTZ image speed and magnification. A
            # full homography is supported for larger non-collinear groups.
            threshold=1.5 if lag[rows[0]]==1 else 2.8
            _,mask=cv2.estimateAffinePartial2D(left,right,method=cv2.RANSAC,ransacReprojThreshold=threshold)
            if mask is None:continue
            valid=mask.ravel()!=0
            if len(rows)>=12 and np.ptp(left[:,1])>=65:
                H,mask=cv2.findHomography(left,right,cv2.RANSAC,threshold)
                if H is not None and mask is not None and mask.sum()>valid.sum()+2:
                    valid=mask.ravel()!=0
            if valid.sum()>=5 and valid.mean()>=.63:good[rows[valid]]=True
        audit=arrays(WORK/'audit_texture'/path.name)
        loc=audit['source_row']
        delta=audit['error'];select=good[loc]&~z['held'][loc]&(lag[loc]==1)
        transient=[];pts=z['pts']
        for i in np.unique(index[loc[select]]):
            ids=select&(index[loc]==i)
            if ids.sum()<4:continue
            e=np.median(delta[ids],axis=0)
            scatter=np.median(np.linalg.norm(delta[ids]-e,axis=1))
            if np.linalg.norm(e)>max(1.6,2.8*scatter):transient.append(int(i))
        save(folder/'quality'/path.name,keep=good,transient=np.array(transient,int),pts=pts)
        report[path.stem]=dict(pairs=len(good),admitted=int(good.sum()),new_check=int((good&z['held']).sum()),
                               locally_resolved_exposures=len(transient))
    write(folder/'quality/summary.json',report)


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__);ap.add_argument('programme',choices=['jin','okada','chen','nayeon','verify'])
    name=ap.parse_args().programme
    if name=='verify':verify_matches()
    else:scan(name)
