"""Source-pixel blade candidates and continuous two-instance review.

No body projection, world trajectory, camera fit or ice-plane snapping is used.
Foot keypoints navigate a search region only. Outputs remain machine candidates.
"""
from pathlib import Path
import argparse,json,time,hashlib
import av,cv2,numpy as np
from .rebuild_prepare import load,digest


def sample_image(image,xy):
    q=np.asarray(xy,np.float32);return cv2.remap(image,q[...,0],q[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)


def make_cache(root,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=False);folder=output/'source_crops';folder.mkdir();p=load(root/'input.npz');wb=load(root/'next_round_20261029/wholebody/wholebody_coco23.npz');np.testing.assert_array_equal(p['pts'],wb['pts']);rows=[];start=time.perf_counter();by={int(t):i for i,t in enumerate(p['pts'])}
    with (output/'tracking_progress.jsonl').open('w',encoding='utf8',buffering=1) as log,av.open(str(root/'offline_review/source_s01.mp4')) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=2
        for f in container.decode(stream):
            if f.pts not in by:continue
            i=by[f.pts];im=f.to_ndarray(format='bgr24');height=float(p['source_bbox'][i,3]-p['source_bbox'][i,1]);height=height if np.isfinite(height) and height>1 else float(wb['source_bbox_model'][i,3]-wb['source_bbox_model'][i,1]);points=np.r_[p['source_uv'][i,[13,14,15,16]],wb['kp2d_coco23'][i,15:23,:2]];points=points[np.isfinite(points).all(1)];lo=np.maximum(np.floor(points.min(0)-[height*.16,height*.09]),[0,0]).astype(int);hi=np.minimum(np.ceil(points.max(0)+[height*.16,height*.15]),[1920,1080]).astype(int)
            if hi[0]<=lo[0] or hi[1]<=lo[1]:lo=np.array([0,800]);hi=np.array([1920,1080])
            crop=im[lo[1]:hi[1],lo[0]:hi[0]];file=folder/f'{f.pts}.png';cv2.imwrite(str(file),crop,[cv2.IMWRITE_PNG_COMPRESSION,2]);rows.append(dict(pts=int(f.pts),source_seconds=f.pts/12800,frame=i,crop_xyxy=[*lo.tolist(),*hi.tolist()],file=str(file),lower_body_inferred=bool(p['lower_body_inferred'][i])))
            if len(rows)%100==0:log.write(json.dumps(dict(event='source_crops',frames=len(rows),seconds=time.perf_counter()-start))+'\n')
        if [r['pts'] for r in rows]!=p['pts'].tolist():raise ValueError('Full native exposure coverage required')
        result=dict(schema='rinkspace.s01-native-blade-source-cache.v1',source_input_sha256=digest(root/'input.npz'),wholebody_sha256=digest(root/'next_round_20261029/wholebody/wholebody_coco23.npz'),source_time_base=[1,12800],source_size=[1920,1080],crop_resized=False,rows=rows,source_qualified_preserved=True);(output/'source_index.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');log.write(json.dumps(dict(event='source_cache_complete',frames=len(rows),seconds=time.perf_counter()-start))+'\n')
    print(json.dumps(dict(source_exposures=len(rows),seconds=time.perf_counter()-start)),flush=True)


def trace_edge(gray,lab,centre,u,v,length,height):
    """Trace an actual outward intensity edge; return only supported segments."""
    count=max(17,int(length/2)+1);s=np.linspace(-length/2,length/2,count);r=np.arange(-7,8,dtype=float);grid=centre+s[:,None,None]*u+r[None,:,None]*v
    inside=sample_image(gray,grid-v*1.5);outside=sample_image(gray,grid+v*1.5);gradient=abs(outside-inside);neutral=np.linalg.norm(lab[:,:,1:]-128,axis=2)<=14;rim=np.zeros(gray.shape,bool);rim[-max(3,gray.shape[0]//8):]=True;rim[:,:max(3,gray.shape[1]//10)]=True;rim[:,-max(3,gray.shape[1]//10):]=True;ice=lab[:,:,0][rim&neutral];threshold=max(165.,float(np.median(ice))+7) if len(ice) else 165.;mask=((lab[:,:,0]>=threshold)&neutral).astype(np.uint8);n,labels,areas,_=cv2.connectedComponentsWithStats(mask);white=np.zeros_like(gray)
    for j in range(1,n):
        if 15<=areas[j,cv2.CC_STAT_AREA]<=height*height*.05 and areas[j,cv2.CC_STAT_WIDTH]<height*.35 and areas[j,cv2.CC_STAT_HEIGHT]<height*.35:white[labels==j]=1.
    above=np.mean([sample_image(white,grid-v*distance) for distance in [5.,10.,18.]],axis=0);below=np.mean([sample_image(white,grid+v*distance) for distance in [7.,14.]],axis=0)
    score=np.clip(gradient,0,60)/12+above*.5-below*.25-.025*r[None]**2
    # Viterbi along the source edge, with bounded local slope. This never fills
    # a source segment whose edge support is absent.
    dp=score[0].copy();parents=[];ri=np.arange(len(r));penalty=.18*abs(ri[:,None]-ri[None,:]);penalty[abs(ri[:,None]-ri[None,:])>3]=1e3
    for j in range(1,count):
        choices=dp[:,None]-penalty;best=np.argmax(choices,axis=0);parents.append(best);dp=score[j]+choices[best,ri]
    path=[int(np.argmax(dp))]
    for parent in parents[::-1]:path.append(int(parent[path[-1]]))
    path=np.asarray(path[::-1]);xy=grid[np.arange(count),path];contrast=gradient[np.arange(count),path];support=contrast>=6
    # A small break in sampling may be retained, but not a long invisible edge.
    filled=support.copy()
    for j in range(1,count-1):
        if support[j-1] and support[j+1]:filled[j]=True
    starts=np.flatnonzero(filled&~np.r_[False,filled[:-1]]);ends=np.flatnonzero(filled&~np.r_[filled[1:],False])+1
    if not len(starts):return None
    k=int(np.argmax(ends-starts));a,b=int(starts[k]),int(ends[k]);xy=xy[a:b];contrast=contrast[a:b];support=support[a:b]
    if len(xy)<6 or np.linalg.norm(xy[-1]-xy[0])<8:return None
    take=np.linspace(0,len(xy)-1,17);curve=np.stack([np.interp(take,np.arange(len(xy)),xy[:,axis]) for axis in range(2)],1)
    outside_distances=np.array([5.,8.,12.,17.,23.]);exterior=sample_image(gray,curve[:,None]+v*outside_distances[None,:,None]);flatness=float(np.median(np.std(exterior,axis=1)));far_gradient=sample_image(gray,curve[:,None]+v*(outside_distances+1.5)[None,:,None])-sample_image(gray,curve[:,None]+v*(outside_distances-1.5)[None,:,None]);outward_edge=float(np.median(np.maximum(far_gradient,0).max(axis=1)))
    return dict(curve=curve,contrast=float(np.median(contrast[support])),support_fraction=float(support.mean()),white_above=float(above[np.arange(count),path][a:b].mean()),length=float(np.linalg.norm(curve[-1]-curve[0])),exterior_variation=flatness,further_outward_edge=outward_edge)


def candidates(image,offset,kp,body_height,foot):
    ankle=kp[15+foot,:2]-offset;ids=[17,18,19] if foot==0 else [20,21,22];toe=kp[ids[:2],:2].mean(0)-offset;heel=kp[ids[2],:2]-offset;axis=toe-heel;predicted_length=float(np.linalg.norm(axis));axis/=max(predicted_length,1e-8);centre=(toe+heel)/2;radius=max(35.,body_height*.14,predicted_length);x0,y0=np.maximum(np.floor(np.minimum.reduce([ankle,toe,heel])-radius*.6),[0,0]).astype(int);x1,y1=np.minimum(np.ceil(np.maximum.reduce([ankle,toe,heel])+radius*.65),[image.shape[1],image.shape[0]]).astype(int)
    if x1-x0<12 or y1-y0<12:return []
    crop=image[y0:y1,x0:x1];gray=cv2.GaussianBlur(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),.6);lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB).astype(np.float32);shift=np.array([x0,y0]);a=ankle-shift;c=centre-shift
    lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD);lines=lsd.detect(np.clip(gray,0,255).astype(np.uint8))[0]
    if lines is None:return []
    proposals=[]
    for line in np.asarray(lines).reshape(-1,4):
        p,q=line[:2].astype(float),line[2:].astype(float);length=float(np.linalg.norm(q-p))
        if length<max(5.,body_height*.009) or length>body_height*.32:continue
        u=(q-p)/length;middle=(p+q)/2;v=np.array([-u[1],u[0]]);lower_leg=ankle-(kp[13+foot,:2]-offset)
        if np.dot(v,lower_leg)<0:v=-v
        orientation=abs(float(u@axis)) if predicted_length>=8 else .65
        if orientation<.10:continue
        distance=float(np.linalg.norm(middle-c));search=max(predicted_length*.9,body_height*.11)
        if distance>search:continue
        normal_from_ankle=float((middle-a)@v)
        if normal_from_ankle<body_height*.015 or normal_from_ankle>body_height*.20:continue
        sample=np.linspace(p,q,13);contrast=float(np.median(abs(sample_image(gray,(sample+v*1.5)[:,None])[:,0]-sample_image(gray,(sample-v*1.5)[:,None])[:,0])))
        if contrast<2:continue
        score=contrast/15+min(length/max(predicted_length,20),1.2)+orientation-distance/search+min(normal_from_ankle/(body_height*.08),1.)*1.5
        proposals.append((score,middle,u,v,length))
    proposals.sort(key=lambda x:x[0],reverse=True);out=[]
    for _,middle,u,v,length in proposals[:40]:
        if any(abs((middle-(o['curve'].mean(0)-offset-shift))@v)<3 and np.linalg.norm(middle-(o['curve'].mean(0)-offset-shift))<length*.6 for o in out):continue
        traced=trace_edge(gray,lab,middle,u,v,min(max(length*1.3,predicted_length*.8),body_height*.25),body_height)
        if traced is None:continue
        direction=abs(float((traced['curve'][-1]-traced['curve'][0])@axis))/max(traced['length'],1e-9) if predicted_length>=8 else .65;distance=float(np.linalg.norm(traced['curve'].mean(0)-c));extent=min(traced['length']/max(predicted_length,20),1.4);depth=float((traced['curve'].mean(0)-a)@v);quality=traced['contrast']/18+traced['white_above']*1.5+extent+direction*.5-distance/max(radius,1)+min(depth/(body_height*.08),1.)*1.5
        quality+=2./(1+traced['exterior_variation']/5)-traced['exterior_variation']/8-traced['further_outward_edge']/20
        traced['curve']+=shift+offset;traced.update(quality=float(quality),foot=foot,source='native RGB signed edge; keypoints navigate only',human_reviewed=False)
        if quality<1.8 or traced['support_fraction']<.6:continue
        out.append(traced)
        if len(out)==8:break
    return sorted(out,key=lambda x:x['quality'],reverse=True)


def detect(root,output):
    root,output=Path(root),Path(output);idx=json.loads((output/'source_index.json').read_text(encoding='utf8'));p=load(root/'input.npz');wb=load(root/'next_round_20261029/wholebody/wholebody_coco23.npz');cv2.setNumThreads(2);records=[];start=time.perf_counter()
    with (output/'candidate_progress.jsonl').open('w',encoding='utf8',buffering=1) as log:
        for row in idx['rows']:
            i=row['frame'];height=float(p['source_bbox'][i,3]-p['source_bbox'][i,1]);height=height if np.isfinite(height) and height>1 else float(wb['source_bbox_model'][i,3]-wb['source_bbox_model'][i,1]);im=cv2.imread(row['file']);offset=np.asarray(row['crop_xyxy'][:2]);pair=[]
            for foot in [0,1]:pair.append([] if row['lower_body_inferred'] else candidates(im,offset,wb['kp2d_coco23'][i],height,foot))
            records.append(dict(pts=row['pts'],frame=i,source_seconds=row['source_seconds'],candidates=pair,lower_body_inferred=row['lower_body_inferred']))
            if len(records)%100==0:log.write(json.dumps(dict(event='native_edge_candidates',frames=len(records),candidate_counts=[sum(len(r['candidates'][f])>0 for r in records) for f in [0,1]],seconds=time.perf_counter()-start))+'\n')
        def serial(x):return x.tolist() if isinstance(x,np.ndarray) else x
        result=dict(schema='rinkspace.source-only-blade-candidates.v1',camera_or_body_projection_used=False,wholebody_is_search_only=True,human_reviewed=False,contact_state_inferred=False,records=records,seconds=time.perf_counter()-start);(output/'candidates.json').write_text(json.dumps(result,ensure_ascii=False,default=serial),encoding='utf8');log.write(json.dumps(dict(event='complete',frames=len(records),seconds=time.perf_counter()-start))+'\n')
    print(json.dumps(dict(frames=len(records),candidate_counts=[sum(len(r['candidates'][f])>0 for r in records) for f in [0,1]],seconds=time.perf_counter()-start)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['cache','detect']);p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();(make_cache if a.action=='cache' else detect)(a.root,a.output)
