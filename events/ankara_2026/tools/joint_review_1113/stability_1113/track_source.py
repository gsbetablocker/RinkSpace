"""Track real board print and qualified ice texture on the full native source clock."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from time import monotonic
import threading

from camera_audit import HERE, ji, sj, vj, setup, native_scene, PREP
import av
import cv2
import numpy as np

LOCK = threading.Lock()
SIZE=(960,540)
LAGS=(1,8)
TB=Fraction(1,12800)
# The left side of the centre overlay and the timer are broadcast graphics.
HUD = ((0,0,249,137),(833,463,960,540))


def source_gray(shot):
    with av.open(shot.source) as reader:
        stream=reader.streams.video[0]
        if stream.time_base!=TB or (stream.width,stream.height)!=(1920,1080):
            raise ValueError('Unregistered source clock or dimensions')
        stream.codec_context.thread_count=2
        reader.seek(int(shot.pts[0]),stream=stream,backward=True)
        n=0
        for frame in reader.decode(stream):
            if frame.pts<shot.pts[0]:
                continue
            if n==len(shot.pts):
                break
            if frame.pts!=shot.pts[n] or frame.time_base!=TB:
                raise ValueError((shot.name,n,frame.pts,shot.pts[n]))
            yield cv2.resize(frame.to_ndarray(format='gray'),SIZE,interpolation=cv2.INTER_AREA)
            n+=1
        if n!=len(shot.pts):
            raise ValueError('Incomplete source '+shot.name)


def rail_points(native):
    board=native.scene['board']
    u=np.linspace(0.,native.core.board_perimeter(board),385)
    cap=np.array(native.core.board_points(board,u,np.full(len(u),float(board[5]))))
    foot=np.array(native.core.board_points(board,u,np.zeros(len(u))))
    return np.vstack((cap,foot)),foot


def board_mask(pixels,status,n):
    from audit_tail_source import y_curve

    x=np.arange(0.,1920.,12.)
    top=y_curve(pixels[:n],status[:n],x)
    bottom=y_curve(pixels[n:],status[n:],x)
    valid=np.isfinite(top)&np.isfinite(bottom)
    depth=np.zeros(len(x))
    depth[valid]=bottom[valid]-top[valid]
    # Deep enough for a local board-only patch. Exclude the cap, toe and ice.
    valid&=depth>=27.
    margin=np.maximum(7.,depth*.12)
    low=np.interp(np.arange(960)*2,x,np.where(valid,top+margin,1e5))*.5
    high=np.interp(np.arange(960)*2,x,np.where(valid,bottom-margin,-1e5))*.5
    y=np.arange(540)[:,None]
    m=((y>low[None,:])&(y<high[None,:])).astype('u1')*255
    return cv2.erode(m,np.ones((5,5),np.uint8))


def measured_ice(name):
    a=vj.load_checkpoint(PREP/(name+'.npz'))
    take=np.asarray(a['fresh'],bool)&~np.char.startswith(a['family'],'Ad')&~np.char.startswith(a['family'],'board')
    return {i:a['uv'][(a['idx']==i)&take] for i in np.unique(a['idx'][take])}


def point_ice_mask(i, measured, rail, gray):
    m=np.zeros(gray.shape,np.uint8)
    if i not in measured:
        return m
    for uv in measured[i]:
        cv2.circle(m,tuple(np.rint(uv*.5).astype(int)),7,255,-1)
    m[rail!=0]=0
    m[gray>229]=0
    return m


def safe_mask(mask,points):
    ix=np.rint(points).astype(int)
    x,y=ix.T
    valid=(x>=8)&(x<952)&(y>=8)&(y<532)
    return valid&(mask[y.clip(0,539),x.clip(0,959)]>0)


def pick(gray,mask,max_corners):
    q=cv2.goodFeaturesToTrack(gray,max_corners,.015,12.,mask=mask,blockSize=5)
    if q is None:
        return np.empty((0,2),np.float32)
    q=q[:,0]
    return np.concatenate([q[(q[:,0]>=j)&(q[:,0]<j+160)][:max(3,max_corners//8)] for j in range(0,960,160)])


def bind(core, shot, c, r, i, u, family, foot, height):
    from tail_materials import rays, intersect

    uv=u.astype(float)*2+.5
    time=shot.times[i]+r[37]*(uv[:,1]/1079.-.5)
    cam=core.evaluate_motion_camera(r,c,shot.first,shot.dt,time)
    direction=rays(uv,cam)
    w=np.full((len(u),3),np.nan)
    b=family==0
    w[b]=intersect(cam['centre'][b],direction[b],foot,height)
    with np.errstate(divide='ignore',invalid='ignore'):
        distance=-cam['centre'][~b,2]/direction[~b,2]
        w[~b]=cam['centre'][~b]+distance[:,None]*direction[~b]
    keep=(cam['status']==0)&np.isfinite(w).all(1)
    keep[~b]&=(distance>0)
    keep[~b]&=(abs(w[~b,0])<=np.max(abs(foot[:,0]))+.003)&(abs(w[~b,1])<=np.max(abs(foot[:,1]))+.003)
    ix=np.flatnonzero(keep)
    check=core.project_motion(r,c,shot.first,shot.dt,1080,shot.times[i],w[ix])
    ix=ix[(check['status']==0)&(np.linalg.norm(check['uv']-uv[ix],axis=1)<.10)]
    return ix,w[ix]


def lk(gray,old_gray,points):
    p=np.ascontiguousarray(points[:,None],dtype='f4')
    spec=dict(winSize=(19,19),maxLevel=3,minEigThreshold=.00008,
              criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,19,.025))
    q,st,e=cv2.calcOpticalFlowPyrLK(old_gray,gray,p,None,**spec)
    if q is None:
        return None
    rev,valid,_=cv2.calcOpticalFlowPyrLK(gray,old_gray,q,p.copy(),flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**spec)
    ok=(st[:,0]!=0)&(valid[:,0]!=0)&(e[:,0]<22.)&(np.linalg.norm(rev[:,0]-p[:,0],axis=1)<.46)
    return q[:,0],ok,e[:,0]


def cells(indices,pts):
    return (indices//24+int(pts[0]//256)%7)%4


def tracks(shot,core,native,c,rig,world,foot,masks,out):
    from read_current_material import mask_frame

    n=len(shot.pts)
    height=float(native.scene['board'][5])
    use_ice=measured_ice(shot.name)
    start=monotonic()
    previous=None
    active=[]
    result=[]
    counts=Counter()
    src=source_gray(shot)
    # Mark texture inside the board face. Prior projections only delimit the ROI.
    for lo in range(0,n,40):
        hi=min(lo+40,n)
        with LOCK:
            rails=ji.project_groups(native,rig,c,shot.first,shot.dt,
                [(i,'rails',shot.times[i]) for i in range(lo,hi)],{'rails':world})
        for i,(proj,st) in enumerate(rails,lo):
            raw=next(src)
            mask=board_mask(proj,st,len(foot))
            ice=point_ice_mask(i,use_ice,mask,raw)
            if shot.program=='jin' and int(shot.pts[i]) in masks:
                person=cv2.resize(mask_frame(masks,int(shot.pts[i])).astype('u1'),SIZE,interpolation=cv2.INTER_AREA)
                invalid=cv2.dilate(person,np.ones((11,11),np.uint8))!=0
                mask[invalid]=0
                ice[invalid]=0
            for x0,y0,x1,y1 in HUD:
                mask[y0:y1,x0:x1]=0
                ice[y0:y1,x0:x1]=0
            if active and previous is not None:
                sizes=[len(g['uv']) for g in active]
                flow=lk(raw,previous,np.vstack([g['uv'] for g in active]))
                next_active=[]
                if flow is not None:
                    for g,ix in zip(active,np.split(np.arange(sum(sizes)),np.cumsum(sizes)[:-1]),strict=True):
                        q,good,quality=(u[ix] for u in flow)
                        good &= safe_mask(mask,q)|(g['family']==1)
                        good &= (q[:,0]>7)&(q[:,0]<952)&(q[:,1]>7)&(q[:,1]<532)
                        # Check athlete/HUD exclusion even for ice tracks between marked frames.
                        if shot.program=='jin' and int(shot.pts[i]) in masks:
                            good &= ~invalid[np.rint(q[:,1]).astype(int).clip(0,539),np.rint(q[:,0]).astype(int).clip(0,959)]
                        if not good.any():
                            continue
                        d=dict(uv=q[good],world=g['world'][good],family=g['family'][good],
                               id=g['id'][good],start=g['start'],start_uv=g['start_uv'][good],
                               prev=g['uv'][good],prev2=g['prev'][good])
                        age=i-g['start']
                        # Adjacent coverage uses only one tracklet age at a time.
                        # Held folds are consecutive frame blocks with disjoint support.
                        if age<=4 or age==8:
                            fold=cells(np.full(good.sum(),i),shot.pts)
                            def record(lag,left,older=None):
                                same=cells(np.full(len(fold),i-lag),shot.pts)==fold
                                if not same.any():
                                    return
                                count=int(same.sum())
                                result.append(dict(
                                    index=np.full(count,i,np.int32), lag=np.full(count,lag,np.int8),
                                    start=np.full(count,g['start'],np.int32),fold=fold[same].astype('i1'),
                                    uv=d['uv'][same]*2+.5,left_uv=left[same]*2+.5,
                                    prev2_uv=(older[same]*2+.5) if older is not None else np.full((count,2),np.nan),
                                    world=d['world'][same],family=d['family'][same],track_id=d['id'][same],
                                    quality=quality[good][same],anchor_uv=d['start_uv'][same]*2+.5))
                                counts[lag]+=count
                            if age<=4:
                                record(1,d['prev'],d['prev2'] if age>=2 else None)
                            if age==8:
                                record(8,d['start_uv'])
                        if age<max(LAGS):
                            next_active.append(d)
                active=next_active
            if i%4==0:
                board_p=pick(raw,mask,50)
                ice_p=pick(raw,ice,29)
                q=np.vstack((board_p,ice_p))
                fam=np.r_[np.zeros(len(board_p),np.uint8),np.ones(len(ice_p),np.uint8)]
                if len(q):
                    j,xyz=bind(core,shot,c,rig[i],i,q,fam,foot,height)
                    if len(j):
                        active.append(dict(uv=q[j],world=xyz,family=fam[j],start=i,
                                           start_uv=q[j],prev=q[j],id=i*100+j))
            previous=raw
            if i%300==299:
                print(shot.name,i+1,n,dict(counts),round(monotonic()-start),flush=True)
    if next(src,None) is not None:
        raise ValueError('Source iterator longer than expected')
    if not result:
        raise ValueError('No source tracks '+shot.name)
    arrays={k:np.concatenate([r[k] for r in result]) for k in result[0]}
    sj.atomic_arrays(out/(shot.name+'.npz'),pts=shot.pts,**arrays)
    print(shot.name,n,dict(counts),round(monotonic()-start,1),flush=True)


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('programme',choices=('jin','okada'))
    p.add_argument('--shots',nargs='+')
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    ji.enable_historical_source_readers()
    cv2.setNumThreads(1)
    m,core,hist,state=setup()
    native=native_scene(m,core,state)
    world,foot=rail_points(native)
    masks=dict(ji.source_reader().mask_index()) if a.programme=='jin' else {}
    jobs=[k for k in m['shots'] if k.startswith(a.programme) and (not a.shots or k.split('_')[1] in a.shots)]
    jobs.sort(key=lambda k:(k not in {'jin_s13','jin_s14','jin_s22','jin_s24','okada_s05','okada_s08'},k))
    out=HERE/'source_tracks'
    out.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        tasks={}
        for name in jobs:
            if (out/(name+'.npz')).is_file():
                continue
            d=vj.shot_from_inputs(name,m,hist)
            z=hist if d.program=='okada' else state
            f=pool.submit(tracks,d,core,native,z[name+'_controls'],z[name+'_rig'],world,foot,masks,out)
            tasks[f]=name
        for f in as_completed(tasks):
            f.result()


if __name__=='__main__':
    main()
