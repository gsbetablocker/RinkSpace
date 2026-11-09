from argparse import ArgumentParser
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import json

import av
import cv2
import numpy as np

PARTS={
    'steel': ('runner_lower_edge','rear_to_front',4.),
    'boot': ('boot_shell','image_trace',5.),
    'outsole': ('outsole','rear_to_front',4.),
    'rear': ('rear_mount','boot_to_steel',4.),
    'front': ('front_mount','boot_to_steel',4.),
}


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def put(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def decimate(p,distance=2.):
    result=[p[0]]
    for q in p[1:]:
        if np.linalg.norm(q-result[-1])>=distance:result.append(q)
    return np.array(result)


def arclength(p,count=9):
    d=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    t=np.linspace(0,d[-1],count)
    return np.array([np.interp(t,d,p[:,i]) for i in range(2)]).T


def native_video(source,pts,needed):
    scale=Fraction(1,12800)
    with av.open(str(source)) as v:
        stream=v.streams.video[0]
        v.seek(int(int(pts[0])*scale/stream.time_base),stream=stream,backward=True)
        stream.thread_type='AUTO'
        i=0
        for f in v.decode(stream):
            if f.pts is None:continue
            t=f.pts*f.time_base/scale
            if t<pts[0]:continue
            if t!=pts[i]:raise ValueError('Native programme clock changed')
            if i in needed:yield i,f.to_ndarray(format='rgb24')
            i+=1
            if i==len(pts):return
    raise ValueError('Source programme incomplete')


def normal_samples(img,line):
    p=np.asarray(line)
    tangent=np.gradient(p,axis=0)
    normal=np.c_[-tangent[:,1],tangent[:,0]]
    normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-6)
    off=np.array([-3,-1.5,1.5,3.])
    coordinates=p[:,None]+off[None,:,None]*normal[:,None]
    values=cv2.remap(img.astype(np.float32),coordinates[:,:,0].astype(np.float32),
        coordinates[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    return values[:,2:].mean(1)-values[:,:2].mean(1),normal


def follow(img,new,points):
    lo=np.maximum(np.floor(points.min(0)-108).astype(int),0)
    hi=np.minimum(np.ceil(points.max(0)+108).astype(int),[1920,1080])
    x,y=lo;u,v=hi
    left=cv2.cvtColor(img[y:v,x:u],cv2.COLOR_RGB2GRAY)
    right=cv2.cvtColor(new[y:v,x:u],cv2.COLOR_RGB2GRAY)
    trace=arclength(points-lo)
    k=cv2.goodFeaturesToTrack(left,40,.010,3)
    if k is None:return None
    # Frame-to-frame flow must be supported by the shoe and near its curve.
    near=np.min(np.linalg.norm(k[:,0,None,:]-trace[None,:,:],axis=2),axis=1)<26
    k=k[near]
    if len(k)<4:return None
    a=np.r_[trace[:,None],k].astype(np.float32)
    b,s,_=cv2.calcOpticalFlowPyrLK(left,right,a,None,winSize=(19,19),maxLevel=3)
    if b is None:return None
    c,q,_=cv2.calcOpticalFlowPyrLK(right,left,b,None,winSize=(19,19),maxLevel=3)
    if c is None:return None
    valid=(s[:,0]>0)&(q[:,0]>0)&(np.linalg.norm(c[:,0]-a[:,0],axis=1)<.80)
    if valid[:len(trace)].sum()<6 or valid.sum()<9:return None
    T,inliers=cv2.estimateAffinePartial2D(a[valid,0],b[valid,0],method=cv2.RANSAC,
                                          ransacReprojThreshold=1.9,maxIters=400)
    if T is None or inliers.mean()<.74:return None
    if not .90<=np.linalg.norm(T[0,:2])<=1.10 or abs(T[1,0])>.34:return None
    predicted=trace@T[:,:2].T+T[:,2]
    if np.quantile(np.linalg.norm(predicted[valid[:len(trace)]]-b[:len(trace),0][valid[:len(trace)]],axis=1),.75)>2.5:return None
    initial,_=normal_samples(left,trace)
    # Require contrast across the same physical line in the target image.
    # Select one normal offset for the observed arc, never nearest model pixels.
    _,normal=normal_samples(right,predicted)
    best=None
    signal=abs(initial)>=6.
    if signal.sum()<4:return None
    for step in (-2.,-1.,0.,1.,2.):
        current=predicted+step*normal
        measured,_=normal_samples(right,current)
        aligned=(measured*np.sign(initial)>=7.) & signal
        cost=np.sum(np.minimum(np.abs(measured[aligned]),26.))-.7*abs(step)
        if best is None or cost>best[0]:best=cost,current,aligned
    _,current,aligned=best
    runs=np.split(np.flatnonzero(aligned),np.flatnonzero(np.diff(np.flatnonzero(aligned))>1)+1)
    keep=max(runs,key=len)
    if len(keep)<4:return None
    fit=decimate(current[keep]+lo)
    if len(fit)<3 or np.linalg.norm(fit[-1]-fit[0])<9:return None
    return fit


def main():
    p=ArgumentParser()
    for key in ('source','prepared','scan','marks','parent','old','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir()
    cv2.setNumThreads(2);cv2.setRNGSeed(3921)
    doc=read(a.parent)['programme_skate_evidence']
    with np.load(a.prepared/'input.npz') as z:pts,shots=(z[k] for k in ('pts','shot_ids'))
    ix={int(p):i for i,p in enumerate(pts)}
    ledger=read(a.scan/'atlas/index.json')
    raw_scan={(r['pts'],r['foot']):r for r in read(a.scan/'candidates.json')['rows']}
    old_doc=[read(a.old/'admission.json'),read(a.old/'check_admission.json')]
    row_counts=Counter()
    events=[dict(id='blade_'+str(s),start_pts=int(pts[shots==s][0]),end_pts=int(pts[shots==s][-1]),
        active_feet=[0,1],description_zh='原片鞋壳与可见钢片，精确源曝光') for s in dict.fromkeys(shots.tolist())]
    rows=[];sources=set();by_seed={}
    marks=read(a.marks)
    check={'s01-5','s02-3','s03-1','s04-0','s05-4','s06-7','s07-5','s08-4','s09-0'}
    used={(r['pts'],r['foot'],r['semantic']) for r in doc['ordered_contours'] if r['role']!='unknown'}

    def add(shot,t,foot,semantic,order,radius,role,points,source,tag,strength):
        gid=f'blade:{tag}:{t}:{foot}:{semantic}'
        if gid in sources:raise ValueError('Repeated source ID '+gid)
        sources.add(gid)
        rows.append(dict(raw_group_id=gid,track_id='blade:'+shot+':'+str(foot)+':'+semantic,event_id='blade_'+shot,
             pts=int(t),foot=int(foot),semantic=semantic,role=role,phase='whole_programme',
             polyline_uv=np.asarray(points).tolist(),source_order=order,complete_visible=False,
             visible_range='only observed polyline segments',visibility='partial',radius_px=radius,
             relative_weight=strength,provenance=source))
        row_counts[shot,semantic,role,tag.split('_')[0]]+=1
        return gid

    for archive in old_doc:
        for r in archive['records']:
            if not r['status'].startswith('admitted'):continue
            t=int(r['pts']);foot=int(r['foot']);role=r['source_role']
            if role not in ('fit','check'):raise ValueError('Unknown archived role')
            p=np.array(r['source_boundary_native_uv'],float)
            segments=np.split(p,np.flatnonzero(np.linalg.norm(np.diff(p,axis=0),axis=1)>6)+1)
            usable=sorted((decimate(q) for q in segments if len(q)>=4 and np.linalg.norm(q[-1]-q[0])>=9),
                          key=lambda q:-len(q))[:2]
            if not usable:row_counts['s01','boot_shell',role,'too_short']+=1
            for j,q in enumerate(usable):
                add('s01',t,foot,'boot_shell','image_trace',float(r['sigma_px']),role,q,
                    dict(kind='existing source-RGB-qualified upper boot; visible disconnected arcs retained',
                         source_crop_sha256=r['source_RGB_crop_sha256'],
                         path=str(a.old/('admission.json' if role=='fit' else 'check_admission.json'))),f'archive_{j}',.85)
                used.add((t,foot,'boot_shell'))

    stamp_sets=defaultdict(set)
    for key,parts in marks.items():
        r=ledger[key];t,foot,shot=r['pts'],r['foot'],r['shot']
        row=raw_scan[t,foot]; f=ix[t]
        by_seed[key]=[]
        for part,locations in parts.items():
            semantic,order,radius=PARTS[part]
            if (t,foot,semantic) in used:continue
            p=np.asarray(locations)*r['scale']+r['origin']
            role='check' if key in check else 'fit'
            if any(abs(t-j)<=256*11 and shot==s for j,s in [(ledger[q]['pts'],ledger[q]['shot']) for q in check]):
                role='check'
            gid=add(shot,t,foot,semantic,order,radius,role,p,
                dict(kind='direct source visual read',frame_RGB_sha256=row['frame_rgb_sha256'],
                     crop=row['qa_crop'],atlas_key=key),key,1.)
            used.add((t,foot,semantic))
            if part not in ('steel','boot'):continue
            group=range(max(0,f-5),min(len(pts),f+6))
            group=[i for i in group if shots[i]==shot]
            for i in group:stamp_sets[i].add(key)
            by_seed[key].append((p,semantic,order,radius,gid,role))
    # Decode the native clock once. Only the handful of crops around active
    # source anchors are retained concurrently.
    gathered=defaultdict(dict)
    last={key:max(i for i,keys in stamp_sets.items() if key in keys) for key in by_seed if by_seed[key]}
    emitted=[]
    for i,image in native_video(a.source,pts,stamp_sets):
        for key in stamp_sets[i]:
            gathered[key][i]=image
        for key in tuple(stamp_sets[i]):
            if i!=last[key]:continue
            sequence=gathered.pop(key)
            r=ledger[key];t,foot,shot=r['pts'],r['foot'],r['shot'];f=ix[t]
            h=sha256(np.ascontiguousarray(sequence[f]).tobytes()).hexdigest()
            assert h==raw_scan[t,foot]['frame_rgb_sha256'],(key,'incorrect native crop')
            for p,semantic,order,radius,gid,role in by_seed[key]:
                for direction in (-1,1):
                    n=f;current=p
                    for step in range(1,6):
                        j=f+step*direction
                        if j not in sequence:break
                        current=follow(sequence[n],sequence[j],current)
                        if current is None:break
                        n=j
                        ti=int(pts[j])
                        if step%2==0 and (ti,foot,semantic) not in used:
                            add(shot,ti,foot,semantic,order,radius+2,role,current,
                                dict(kind='native image edge with bidirectional motion and signed contrast',
                                     depends_on_track_anchor=gid,frame_RGB_sha256=sha256(
                                         np.ascontiguousarray(sequence[j]).tobytes()).hexdigest()),
                                'flow_'+key,.52)
                            used.add((ti,foot,semantic))
            emitted.append(key)
    assert len(emitted)==len(last)
    result=dict(schema='rinkspace.programme-skate-amendment.v1',time_base=doc['time_base'],
        source_video_sha256=doc['source_video_sha256'],events=events,ordered_contours=rows,manual_landmarks=[],
        exposure_review=[],method_zh='九镜头原片鞋壳和可见钢片短弧。人工确认的局部边和原片正反光流、对比度共证的邻帧入账，区分支撑与钢片。S01 既有 fit/check 保持原角色。')
    put(a.output/'amendment.json',result)
    counts={'/'.join(k):v for k,v in sorted(row_counts.items())}
    put(a.output/'admission_summary.json',dict(by_shot_and_semantic=counts,rows=len(rows),
        fit_exposures=len({r['pts'] for r in rows if r['role']=='fit'}),
        check_exposures=len({r['pts'] for r in rows if r['role']=='check'}),
        independent_human_steel_anchors=sum(1 for r in rows if r['semantic']=='runner_lower_edge'
            and r['provenance']['kind']=='direct source visual read'),
        auto_unreviewed_rows_excluded=True))
    print(json.dumps(dict(rows=len(rows),steel=sum(r['semantic']=='runner_lower_edge' for r in rows),
        by_shot={s:dict(Counter(r['role'] for r in rows if r['event_id']=='blade_'+s)) for s in dict.fromkeys(shots.tolist())})),flush=True)


if __name__=='__main__':main()
