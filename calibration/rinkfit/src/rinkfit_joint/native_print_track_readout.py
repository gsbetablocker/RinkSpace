"""Source-only, per-exposure named print readings with compact numeric evidence.

Tracks navigate actual RGB. Every emitted reading is measured again in the
current exposure; neither the fitted camera nor its projected grid is consulted.
Outputs remain candidates until their physical identities have been reviewed.
"""
from pathlib import Path
import json,hashlib,collections,time
import numpy as np
import cv2,av
from .source_worker import _channels,_sample


def batch_interfaces(uv,normal,channels):
    uv=np.asarray(uv,float).reshape(-1,2);normal=np.asarray(normal,float).reshape(-1,2)
    norm=np.linalg.norm(normal,axis=1);normal=normal/np.maximum(norm[:,None],1e-12)
    if not len(uv):return dict(uv=uv,normal=normal,valid=np.zeros(0,bool),profile=np.zeros((0,5)))
    off=np.arange(-3.,3.0001,.25);query=uv[:,None,:]+off[None,:,None]*normal[:,None,:]
    values=_sample(channels,query.reshape(-1,2)).reshape(len(uv),len(off),3)
    gradient=np.gradient(values,.25,axis=1);channel=np.max(abs(gradient),axis=1).argmax(1);i=np.arange(len(uv));g=gradient[i,:,channel];v=abs(g);k=v.argmax(1);peak=v[i,k];contrast=np.ptp(values[i,:,channel],axis=1);shift=off[k].copy();left=v[i,np.maximum(0,k-1)];right=v[i,np.minimum(24,k+1)];den=left-2*peak+right;refine=(k>0)&(k<24)&(abs(den)>1e-9);shift[refine]+=np.clip(.5*(left[refine]-right[refine])/den[refine],-1,1)*.25
    valid=np.isfinite(uv).all(1)&np.isfinite(normal).all(1)&(norm>1e-8)&(peak>=.75)&(contrast>=4)&(k>0)&(k<24)
    return dict(uv=uv+shift[:,None]*normal,normal=normal,valid=valid,profile=np.c_[peak,contrast,shift,channel,np.sign(g[i,k])])


def clear_profile(uv,normal,mask):
    uv=np.asarray(uv,float).reshape(-1,2);n=np.asarray(normal,float).reshape(-1,2)
    q=uv[:,None,:]+np.arange(-3,4)[None,:,None]*n[:,None,:];i=np.rint(np.nan_to_num(q)).astype(np.int32);inside=np.isfinite(q).all((1,2))&(q[:,:,0]>=4).all(1)&(q[:,:,0]<1916).all(1)&(q[:,:,1]>=4).all(1)&(q[:,:,1]<1076).all(1)
    return inside&~(mask[np.clip(i[:,:,1],0,1079),np.clip(i[:,:,0],0,1919)]>0).any(1)


def track_step(previous,current,points,alive,mask):
    if not len(points):return points,alive,np.zeros(0)
    p=points.reshape(-1,1,2).astype(np.float32);q,ok,err=cv2.calcOpticalFlowPyrLK(previous,current,p,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01));back,bo,_=cv2.calcOpticalFlowPyrLK(current,previous,q,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01));fb=np.linalg.norm(back.reshape(-1,2)-p.reshape(-1,2),axis=1).reshape(-1,3);ok=((ok.ravel()!=0)&(bo.ravel()!=0)&(err.ravel()<35)).reshape(-1,3).all(1)&(fb<=.8).all(1);points=q.reshape(-1,3,2);t=points[:,2]-points[:,1];length=np.linalg.norm(t,axis=1);n=np.c_[-t[:,1],t[:,0]]/np.maximum(length[:,None],1e-12);ok&=clear_profile(points[:,0],n,mask)&(length>.1)
    return points,alive&ok,fb.max(1)


def read_interval(images,masks,anchors,left,right,writer,already_written,endpoint_channels):
    for p in (left,right):
        if p not in endpoint_channels:endpoint_channels[p]=_channels(images[p][:,:,::-1])
    pts=sorted(images);gray={p:cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)for p,im in images.items()};nav=collections.defaultdict(list)
    for apt,order in ((left,pts),(right,list(reversed(pts)))):
        seeds=anchors.get(apt,[])
        if not seeds:continue
        uv=np.array([x['uv']for x in seeds]);n=np.array([x['normal']for x in seeds]);r=batch_interfaces(uv,n,endpoint_channels[apt]);good=r['valid']&clear_profile(r['uv'],r['normal'],masks[apt]);ids=np.flatnonzero(good)
        if not len(ids):continue
        seeds=[seeds[i]for i in ids];uv=r['uv'][ids];n=r['normal'][ids];t=np.c_[n[:,1],-n[:,0]];span=np.asarray([s.get('tangent_halfspan_px',2.)for s in seeds])[:,None];points=np.stack((uv,uv-t*span,uv+t*span),axis=1).astype(np.float32);alive=np.ones(len(ids),bool);reference=r['profile'][ids];prev=apt;fb=np.zeros(len(ids));
        for pt in order:
            if pt!=apt:points,alive,fb=track_step(gray[prev],gray[pt],points,alive,masks[pt])
            nav[pt].append((apt,seeds,points.copy(),alive.copy(),reference,fb.copy()));prev=pt
            if not alive.any():break
    for pt in pts:
        if pt in already_written:continue
        candidates=nav.get(pt,[]);per=[];seen=collections.defaultdict(set);channels=endpoint_channels[pt]if pt in endpoint_channels else _channels(images[pt][:,:,::-1])
        for apt,seeds,points,alive,reference,fb in sorted(candidates,key=lambda x:abs(x[0]-pt)):
            ix=np.flatnonzero(alive)
            if not len(ix):continue
            pp=points[ix];t=pp[:,2]-pp[:,1];normal=np.c_[-t[:,1],t[:,0]];r=batch_interfaces(pp[:,0],normal,channels);rr=reference[ix];good=r['valid']&clear_profile(r['uv'],r['normal'],masks[pt])&(r['profile'][:,3]==rr[:,3])&(r['profile'][:,4]==rr[:,4])
            for j in np.flatnonzero(good):
                seed=seeds[ix[j]];label=seed['label'];cell=tuple(np.rint(r['uv'][j]).astype(int))
                if cell in seen[label]:continue
                seen[label].add(cell);per.append((seed['seed_index'],r['uv'][j],r['normal'][j],r['profile'][j],float(fb[ix[j]])))
        writer(pt,images[pt],per)


def run(config):
    """Decode each native exposure once, retaining only adjacent anchor intervals."""
    cfg=json.loads(Path(config).read_text(encoding='utf8'));out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);(out/'frames').mkdir();cv2.setNumThreads(2);anchors=collections.defaultdict(list);seeds=cfg['seeds']
    for i,s in enumerate(seeds):s=dict(s,seed_index=i);anchors[int(s['native_pts'])].append(s)
    from calibration.program_tracking.extract_first_shot_ice import current_broadcast_exclusion
    import gzip
    with gzip.open(cfg['source_coverage'],'rt',encoding='utf8')as f:coverage={int(r['native_pts']):r for r in map(json.loads,f)}
    wrote=set();rows_total=0;start=time.monotonic();records=[];source_sha=cfg['source_video_sha256']
    def writer(pt,im,rows):
        nonlocal rows_total
        if pt in wrote:return
        wrote.add(pt);seed=np.array([r[0]for r in rows],np.uint32);uv=np.array([r[1]for r in rows],np.float64).reshape(-1,2);normal=np.array([r[2]for r in rows],np.float64).reshape(-1,2);profile=np.array([r[3]for r in rows],np.float64).reshape(-1,5);fb=np.array([r[4]for r in rows],np.float32);dest=out/'frames'/f'{pt}.npz';np.savez_compressed(dest,seed_index=seed,uv=uv,normal=normal,profile=profile,FB_error_px=fb);by_ad=collections.Counter(seeds[int(i)]['label'].split('.')[0]for i in seed);r=dict(native_pts=pt,source_BGR_sha256=coverage[pt]['source_BGR_sha256'],rows=len(rows),by_ad=dict(by_ad),file=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),person_mask=coverage[pt]['person_mask'],unknown_mask=coverage[pt]['unknown_mask'],source_identity_review_completed=False);records.append(r);rows_total+=len(rows)
    for shot in cfg['shots']:
        first,last=int(shot['first_pts']),int(shot['last_pts']);cuts=sorted({first,last,*[p for p in anchors if first<=p<=last]});interval=0;images={};masks={};endpoint_channels={}
        with av.open(cfg['source_video'])as con:
            stream=con.streams.video[0];assert str(stream.time_base)=='1/12800';con.seek(first,stream=stream,backward=True)
            for fr in con.decode(stream):
                pt=int(fr.pts)
                if pt<first:continue
                if pt>last:break
                im=fr.to_ndarray(format='bgr24');assert hashlib.sha256(im.tobytes()).hexdigest()==coverage[pt]['source_BGR_sha256'],('native_frame_changed',pt);mask=cv2.imread(coverage[pt]['person_mask']['file'],0);ur=coverage[pt]['unknown_mask']
                if ur:mask=cv2.bitwise_or(mask,cv2.imread(ur['file'],0))
                graphics,_=current_broadcast_exclusion(im);mask[graphics]=255;images[pt]=im;masks[pt]=mask
                if pt==cuts[interval+1]:
                    left,right=cuts[interval:interval+2];read_interval(images,masks,anchors,left,right,writer,wrote,endpoint_channels);endpoint_channels={right:endpoint_channels[right]};interval+=1;images={pt:im};masks={pt:mask};progress=dict(shot=shot['id'],native_pts=pt,frames=len(wrote),rows=rows_total,elapsed_seconds=time.monotonic()-start,physical_solver_started=False);(out/'progress.json').write_text(json.dumps(progress),encoding='utf8');print(json.dumps(progress),flush=True)
                    if interval==len(cuts)-1:break
        assert all(p in wrote for p in range(first,last+1,256)),('missing_native_frame',shot['id'])
    (out/'frame_index.json').write_text(json.dumps(sorted(records,key=lambda r:r['native_pts']),ensure_ascii=False),encoding='utf8');manifest=dict(schema='rinkfit.compact_native_named_print_candidates.v1',config_file=str(Path(config)),config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),reader_file=str(Path(__file__).resolve()),reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_video_sha256=source_sha,source_time_base='1/12800',frames=len(wrote),rows=rows_total,elapsed_seconds=time.monotonic()-start,source_model_projection_used=False,source_identity_review_completed=False,fit_ready=False,physical_solver_started=False);(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(manifest),flush=True);return manifest
