"""Measure missing Ad7 glyph contours directly from each authentic source exposure."""
from pathlib import Path
import sys,json,gzip,hashlib,collections,time
import numpy as np
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_replacement_s01 as b
import ad_increment_s01 as delta
from rinkspace.vision_source_shapes import read_source_material_transition_profiles
OUT=delta.OUT
WORDS={'top':'INTERNATIONAL','bottom':'SKATINGUNION'}

def candidate_rows(im,raw_uv):
    B,G,R=cv2.split(im.astype(float));blue=((B-R>15)&(B-G>8)).astype(np.uint8)
    blue[:180]=0
    blue=cv2.morphologyEx(blue,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    cc,_=cv2.findContours(blue,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cc=[c for c in cc if cv2.contourArea(c)>4000]
    if not cc:return None
    score=lambda c:sum(cv2.pointPolygonTest(c,tuple(map(float,p)),False)>=0 for p in raw_uv)
    panelcontour=max(cc,key=score)
    if score(panelcontour)<4:return None
    bx,by,bw,bh=cv2.boundingRect(panelcontour)
    panel=np.zeros(im.shape[:2],np.uint8);cv2.drawContours(panel,[panelcontour],-1,1,-1)
    panel=cv2.erode(panel,np.ones((5,5),np.uint8))
    val=im.min(axis=2);threshold,_=cv2.threshold(val[panel>0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ink=((val>threshold)&(panel>0)).astype(np.uint8)
    _,labels,stats,centers=cv2.connectedComponentsWithStats(ink,8)
    parts=[]
    for i,(x,y,w,h,area) in enumerate(stats[1:],1):
        if 7<=h<.3*bh and area>=12 and w<3*h and .06*bh<y-by<.8*bh:
            parts.append(dict(id=i,x=int(x),y=int(y),w=int(w),h=int(h),area=int(area),cy=y+h/2))
    clusters=[]
    for p in sorted(parts,key=lambda p:p['cy']):
        matches=[g for g in clusters if abs(np.median([q['cy'] for q in g])-p['cy'])<.42*np.median([q['h'] for q in g])]
        if matches:matches[0].append(p)
        else:clusters.append([p])
    options={}
    for name,word in WORDS.items():
        found=[]
        for g in clusters:
            g=sorted(g,key=lambda p:p['x'])
            for start in range(len(g)-len(word)+1):
                v=g[start:start+len(word)];h=np.median([p['h'] for p in v]);xs=np.array([p['x'] for p in v]);ys=np.array([p['cy'] for p in v])
                if np.max(np.abs(np.array([p['h'] for p in v])/h-1))>.30:continue
                coeff=np.polyfit(xs,ys,1)
                if np.max(abs(ys-np.polyval(coeff,xs)))>.22*h:continue
                widths=np.array([p['w']/h for p in v]);narrow=np.array([c=='I' for c in word])
                if np.any(widths[narrow]>.39) or np.any(widths[~narrow]<.40):continue
                if np.any(widths>1.35):continue
                gaps=np.array([v[k+1]['x']-v[k]['x']-v[k]['w'] for k in range(len(v)-1)])/h
                if np.any(gaps<.10) or np.any(gaps>1.70):continue
                if name=='bottom' and not (gaps[6]>.8 and np.max(np.delete(gaps,6))<.90):continue
                if name=='top' and np.max(gaps)>.95:continue
                found.append(v)
        options[name]=found
    pairs=[]
    for top in options['top']:
        for bottom in options['bottom']:
            ht=np.median([p['h'] for p in top]);hb=np.median([p['h'] for p in bottom]);h=(ht+hb)/2
            dy=np.median([p['cy'] for p in bottom])-np.median([p['cy'] for p in top])
            leftdiff=abs(top[0]['x']-bottom[0]['x']);rightdiff=abs(top[-1]['x']+top[-1]['w']-bottom[-1]['x']-bottom[-1]['w'])
            if .7<ht/hb<1.3 and 1.1<dy/h<2.3 and leftdiff<.55*h and rightdiff<.55*h:
                pairs.append((leftdiff+rightdiff,dict(top=top,bottom=bottom)))
    if len(pairs)!=1:return None
    return dict(rows=pairs[0][1],labels=labels,threshold=float(threshold),panel_bbox=[bx,by,bw,bh])

def contour_samples(contour,step=6.):
    p=contour[:,0,:].astype(float);p=np.vstack([p,p[:1]]);dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    if dist[-1]<5:return np.empty((0,2))
    targets=np.arange(.5,dist[-1],step)
    return np.column_stack([np.interp(targets,dist,p[:,i]) for i in range(2)])

def measure():
    if (OUT/'increment_manifest.json').exists():raise ValueError('Frozen increment cannot be rewritten; prepare a new increment')
    cv2.setNumThreads(2);data=b.load();baseline=b.read(b.OUT/'incomplete_baseline_manifest.json')
    z=dict(np.load(OUT/'additions.npz'));existing=np.load(b.OUT/'ads_only/point_observations.npz');mapping=b.read(OUT/'mapping.json');lookup={p:i for i,p in enumerate(mapping['path_ids'])}
    if any(str(s).startswith('s01_ad7_source_') for s in z['observation_ids']):raise ValueError('Supplement already appended')
    frames={f['pts']:f for f in b.read(b.OUT/'visibility.json')['frames']};pts=sorted(map(int,np.unique(data['pts'][data['ad']==7])));accepted=[];summaries=[];reasons=collections.Counter();start=time.monotonic()
    with gzip.open(OUT/'new_glyph_measurements.jsonl.gz','wt',encoding='utf-8') as log:
        for fi,pt in enumerate(pts):
            f=frames[pt];im=cv2.imread(f['image']);mask=cv2.imread(f['mask_path'],0)
            if hashlib.sha256(im.tobytes()).hexdigest()!=f['source_BGR_sha256']:raise ValueError('Source pixels changed')
            raw=data['uv'][(data['pts']==pt)&(data['ad']==7)]
            found=candidate_rows(im,raw)
            if found is None:reasons['complete_word_identity_not_resolved']+=1;continue
            labels=found['labels'];gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(float);gray=cv2.GaussianBlur(gray,(3,3),.55)
            dx=cv2.Scharr(gray,cv2.CV_64F,1,0)/32;dy=cv2.Scharr(gray,cv2.CV_64F,0,1)/32
            safe=cv2.distanceTransform((mask==0).astype(np.uint8),cv2.DIST_L2,3)
            qpoints=[];normals=[];identities=[];glyphproof=[]
            for name,parts in found['rows'].items():
                for gi,(letter,part) in enumerate(zip(WORDS[name],parts)):
                    x,y,w,h=[part[k] for k in ('x','y','w','h')];roi=(labels[y-1:y+h+1,x-1:x+w+1]==part['id']).astype(np.uint8)
                    contours,hierarchy=cv2.findContours(roi,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
                    if hierarchy is None:continue
                    parents=hierarchy[0,:,3];outer=np.flatnonzero(parents<0)
                    if len(outer)!=1:reasons['fragmented_glyph']+=1;continue
                    holes=np.flatnonzero(parents==outer[0]);prefix=f'Ad7.{name}_{gi:02d}_{letter}_part'
                    expect_hole=f'print_instance/Ad7::{prefix}01' in lookup
                    glyphproof.append(dict(word=name,index=gi,letter=letter,source_bbox=[x,y,w,h],threshold=found['threshold'],holes_observed=len(holes),holes_expected=int(expect_hole)))
                    contour_defs=[(int(outer[0]),0)]
                    if expect_hole and len(holes)==1:contour_defs.append((int(holes[0]),1))
                    elif expect_hole:reasons['unresolved_inner_hole']+=1
                    for ci,pi in contour_defs:
                        pid=f'print_instance/Ad7::{prefix}{pi:02d}'
                        if pid not in lookup:continue
                        c=contours[ci].copy();c[:,0,:]+=np.array([x-1,y-1]);samples=contour_samples(c)
                        near=np.concatenate([a['uv'][(a['pts']==pt)&(a['path_indices']==lookup[pid])] for a in (z,existing)],axis=0)
                        for uv in samples:
                            xx,yy=np.rint(uv).astype(int)
                            if not(4<=xx<1916 and 4<=yy<1076) or safe[yy,xx]<4:continue
                            if len(near) and np.min(np.linalg.norm(near-uv,axis=1))<3:continue
                            g=np.array([dx[yy,xx],dy[yy,xx]]);norm=np.linalg.norm(g)
                            if norm<1:continue
                            qpoints.append(uv);normals.append(g/norm);identities.append(dict(path=pid,word=name,glyph_index=gi,letter=letter,part=pi,source_bbox=[x,y,w,h]))
            if not qpoints:continue
            points=np.array(qpoints);normals=np.array(normals)
            binding=dict(source_sha256=f['source_video_sha256'],source_pts=pt,time_base='1/12800',shot_id='s01',event_id='ankara_2026',rig_id='main',installation_id='main',pixel_domain='native_distorted',width=1920,height=1080)
            result=read_source_material_transition_profiles(np.ascontiguousarray(im[:,:,::-1]),binding=binding,search_points_uv=points,normals=normals,search_intervals_px=np.tile([-3.,3.],(len(points),1)),transition='dark_to_light',tangent_half_width_px=1,smoothing_sigma_px=.55,engineering_support_margin_px=.1)
            new=[]
            for qi,profile in enumerate(result['readings']):
                if any(profile['source_flags'].values()) or np.asarray(profile['actual_source_uv']).shape!=(2,):reasons['local_transition_unresolved']+=1;continue
                uv=np.asarray(profile['actual_source_uv']);identity=identities[qi];xx,yy=np.rint(uv).astype(int)
                if safe[yy,xx]<4:continue
                sid=f's01_ad7_source_{pt}_{identity["word"]}_{identity["glyph_index"]:02d}_p{identity["part"]}_{qi:04d}'
                r=dict(stable_id=sid,pts=pt,uv=uv.tolist(),path_id=identity['path'],source_row=len(data['pts'])+len(accepted),source_image=f['image'],source_BGR_sha256=f['source_BGR_sha256'],source_video_sha256=f['source_video_sha256'],time_base='1/12800',native_pixel_domain=True,method='actual_source_material_transition_on_semantically_named_glyph_contour',identity_evidence=identity,query_uv=points[qi].tolist(),normal=normals[qi].tolist(),profile_offsets=np.asarray(profile['source_q10_q50_q90_normal_offsets_px']).tolist(),native_geometry_used_for_identity_or_uv=False)
                accepted.append(r);new.append(r);log.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
            summaries.append(dict(pts=pt,added=len(new),panel_bbox=found['panel_bbox'],glyphs=glyphproof))
            if fi%20==0:print(json.dumps(dict(frame=fi+1,of=len(pts),new=len(accepted),seconds=round(time.monotonic()-start,1))),flush=True)
    n=len(accepted)
    a=dict(source_rows=np.array([r['source_row'] for r in accepted],np.int64),path_indices=np.array([lookup[r['path_id']] for r in accepted],np.int32),stations=np.full(n,.5),pts=np.array([r['pts'] for r in accepted],np.int64),uv=np.array([r['uv'] for r in accepted]).reshape(-1,2),kind=np.ones(n,np.uint8),observation_ids=np.array([r['stable_id'] for r in accepted]))
    np.savez_compressed(OUT/'additions.npz',**{k:np.concatenate([z[k],a[k]]) for k in z})
    report=dict(new_source_measured_rows=n,restored_source_rows=len(z['pts']),total_increment_rows=len(z['pts'])+n,source_frames=len(set(r['pts'] for r in accepted)),by_path=dict(collections.Counter(r['path_id'] for r in accepted)),unresolved=dict(reasons),per_frame=summaries,baseline_id=baseline['baseline_id'],baseline_rewritten=False,all_native_parameters_unchanged=True,non_model_source_measurement=True)
    b.write(OUT/'glyph_measurements.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('per_frame','by_path')}),flush=True)

if __name__=='__main__':measure()
