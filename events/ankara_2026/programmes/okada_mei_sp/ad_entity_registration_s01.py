"""Current source identity charts: full-pattern seeds and adjacent RGB feature tracks.

No native projection, residual, or imported point label is an identity authority.
"""
from pathlib import Path
import sys,json,collections,hashlib,functools,time
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_replacement_s01 as b
import ad_replace_package_s01 as package
OUT=b.OUT/'native_correspondence_recheck'
# Separate original source charts preserve words / panels that have independent layouts.
ATLAS_IDS=[i for i in range(78) if i != 74]

def rgba_read(path):
    im=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if im.ndim==2:return cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    if im.shape[2]==4:
        alpha=im[:,:,3:4].astype(float)/255;return np.rint(im[:,:,:3]*alpha+255*(1-alpha)).astype('uint8')
    return im

@functools.lru_cache(maxsize=18)
def current(pt):
    f=VIS[pt];im=cv2.imread(f['image'])
    if hashlib.sha256(im.tobytes()).hexdigest()!=f['source_BGR_sha256']:raise ValueError('Source image identity changed')
    return im,cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),cv2.imread(f['mask_path'],0)

def sample(im,q):
    q=np.asarray(q,np.float32);return cv2.remap(im,q[...,0],q[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).astype(float)

def reference(atlas):
    if b.sha(atlas['image'])!=atlas['image_sha256']:raise ValueError('Source reference file changed')
    color=rgba_read(atlas['image']);gray=cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
    if atlas['ad']==7 and atlas['id']==76:gray=255-gray
    if atlas['id']==73:
        raw=cv2.imread(atlas['image'],cv2.IMREAD_UNCHANGED);gray=255-raw[:,:,3];color=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    shapes=[];labels=[];fractions=[];starts=[]
    for pi,path in enumerate(atlas['paths']):
        points=np.asarray(path['pixels'],float);length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        if length[-1]<1:continue
        # Probe density belongs to source-reference pixels, not native geometry.
        tt=np.linspace(0,length[-1],min(300,max(8,int(length[-1]/2)+1)))
        q=np.column_stack([np.interp(tt,length,points[:,k]) for k in range(2)]);shapes.append(q);labels.extend([pi]*len(q));fractions.extend((tt/length[-1]).tolist())
    pts=np.concatenate(shapes);lo=pts.min(0);hi=pts.max(0)
    probe=np.unique(np.linspace(0,len(pts)-1,min(len(pts),220)).astype(int))
    return dict(atlas=atlas,color=color,gray=gray,points=pts,labels=np.array(labels),fractions=np.array(fractions),lo=lo,hi=hi,probe=probe)

def geometry(H,ref):
    H=np.asarray(H,float)
    if H.shape!=(3,3) or not np.isfinite(H).all():return None
    lo,hi=ref['lo'],ref['hi'];cx=(lo+hi)/2;q=np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]],cx],float)
    jac=[]
    for p in q:
        h=np.r_[p,1.];den=H[2]@h
        if abs(den)<1e-7:return None
        t=H@h;t=t[:2]/den;J=(H[:2,:2]-np.outer(t,H[2,:2]))/den
        sv=np.linalg.svd(J,compute_uv=False)
        if np.linalg.det(J)<=0 or J[0,0]<=0 or J[1,1]<=0 or sv[0]/sv[1]>4 or sv[1]<.015 or sv[0]>14:return None
        jac.append(sv.tolist())
    target=cv2.perspectiveTransform(q[None],H)[0]
    if np.any(~np.isfinite(target)) or np.max(abs(target))>100000:return None
    return dict(projected_bbox=[target[:4].min(0).tolist(),target[:4].max(0).tolist()],local_singular_values=jac)

def full_pattern_score(ref,H,pt,source_image=None,allow_partial=False):
    im,gray,mask=current(pt) if source_image is None else source_image;q0=ref['points'][ref['probe']];q=cv2.perspectiveTransform(q0[None],np.asarray(H,float))[0];ix=np.rint(q).astype(int)
    valid=(ix[:,0]>10)&(ix[:,0]<1909)&(ix[:,1]>190)&(ix[:,1]<1069);inside=np.flatnonzero(valid);valid[inside]&=cv2.dilate(mask,np.ones((5,5),np.uint8))[ix[inside,1],ix[inside,0]]==0
    if valid.sum()<18:return None
    q=q[valid];path_indices=ref['labels'][ref['probe']][valid];off=np.mgrid[-9:10:3,-9:10:3].reshape(2,-1).T[:,::-1];dst=q[:,None,:]+off;src=cv2.perspectiveTransform(dst.reshape(1,-1,2),np.linalg.inv(H))[0].reshape(dst.shape)
    a=sample(ref['gray'],src);c=sample(gray,dst);a-=a.mean(1,keepdims=True);c-=c.mean(1,keepdims=True);den=np.linalg.norm(a,axis=1)*np.linalg.norm(c,axis=1);ncc=np.divide((a*c).sum(1),den,out=np.full(len(a),-1.),where=den>1)
    informative=(np.std(a,axis=1)>3)&(np.std(c,axis=1)>3);chosen_blur=1
    # A source-pattern identity may remain observable under directional motion
    # spread. Match a single shared image-space blur for the entire pattern;
    # this never creates UV or resolves individual material edges.
    best_median=float(np.median(ncc[informative])) if informative.any() else -1.
    if best_median<.72 and source_image is None:
        low=np.maximum(np.floor(q.min(0)-30).astype(int),[0,0]);high=np.minimum(np.ceil(q.max(0)+31).astype(int),[gray.shape[1],gray.shape[0]])
        shift=np.eye(3);shift[:2,2]=-low
        warped=cv2.warpPerspective(ref['gray'],shift@H,tuple(high-low),flags=cv2.INTER_LINEAR)
        local=dst-low
        for width in [5,9,17,25]:
            ap=sample(cv2.blur(warped,(width,1)),local);ap-=ap.mean(1,keepdims=True);dn=np.linalg.norm(ap,axis=1)*np.linalg.norm(c,axis=1)
            nc=np.divide((ap*c).sum(1),dn,out=np.full(len(ap),-1.),where=dn>1);info=(np.std(ap,axis=1)>3)&(np.std(c,axis=1)>3)
            med=float(np.median(nc[info])) if info.any() else -1.
            if med>best_median:best_median=med;ncc=nc;informative=info;chosen_blur=width

    src_rgb=sample(ref['color'],src);dst_rgb=sample(im,dst)
    if ref['atlas']['id'] in (71,73):
        color_ok=np.ones(len(q),bool) # Original artwork palette differs from the photographed print; physical palette is checked below.
    elif ref['atlas']['id']==76:
        blue=(dst_rgb[:,:,0]-dst_rgb[:,:,2]>25)&(dst_rgb[:,:,0]-dst_rgb[:,:,1]>10);color_ok=blue.mean(1)>.12
    else:
        sa=np.ptp(src_rgb,axis=2)>20;sc=np.ptp(dst_rgb,axis=2)>20;fa=sa.mean(1);fc=sc.mean(1)
        da=np.stack([src_rgb[:,:,0]-src_rgb[:,:,1],src_rgb[:,:,2]-src_rgb[:,:,1]],axis=-1);dc=np.stack([dst_rgb[:,:,0]-dst_rgb[:,:,1],dst_rgb[:,:,2]-dst_rgb[:,:,1]],axis=-1)
        va=(da*sa[:,:,None]).sum(1)/np.maximum(sa.sum(1)[:,None],1);vc=(dc*sc[:,:,None]).sum(1)/np.maximum(sc.sum(1)[:,None],1)
        den_color=np.linalg.norm(va,axis=1)*np.linalg.norm(vc,axis=1);cos=np.divide((va*vc).sum(1),den_color,out=np.ones(len(va)),where=den_color>1)
        color_ok=~(((fa>.10)&(fc>.10)&(cos<.766))|((fa>.3)&(fc<.03))|((fc>.3)&(fa<.03)))
    # Typed photographed regions separate otherwise similar glyphs. This is source RGB,
    # never a comparison with unfitted native positions or dimensions.
    B,G,R=dst_rgb[:,:,0],dst_rgb[:,:,1],dst_rgb[:,:,2];purple=((R-G>15)&(B-G>15)&(R-B>-55)).mean(1);blue=((B-R>25)&(B-G>10)).mean(1);red=((R-B>25)&(R-G>25)).mean(1)
    ad=ref['atlas']['ad'];entity_names=np.array([ref['atlas']['paths'][int(k)]['path_id'] for k in path_indices])
    if ad==2:color_ok&=purple>.025
    elif ad==4:
        ankara=np.array(['ANKARA' in p for p in entity_names]);words=np.array(['shot4_' in p for p in entity_names]);color_ok[ankara]&=purple[ankara]>.025;color_ok[words]&=(blue[words]>.025)&(purple[words]<.45)
    elif ad==6:color_ok&=red>.025
    elif ad==7:color_ok&=blue>.025
    elif ad==8:color_ok&=blue>.025
    elif ad==9:color_ok&=red>.025
    passing=informative&(ncc>.45)&color_ok
    groups=[int(k) for k in np.unique(path_indices) if np.count_nonzero(passing&(path_indices==k))>=2]
    npaths=len(set(ref['labels']));needed=min(3,npaths)
    if allow_partial:needed=min(needed,len(set(path_indices)))
    med=float(np.median(ncc[informative])) if informative.any() else -1.
    if len(groups)<needed or passing.sum()<15 or passing.sum()/max(1,informative.sum())<.50 or med<.43:return None
    support=q[passing];span=np.ptp(support,axis=0)
    if min(span)<5 or cv2.contourArea(cv2.convexHull(support.astype(np.float32)))<70:return None
    return dict(pattern_correlation_median=med,shared_source_navigation_blur_px=chosen_blur,locally_supported_probes=int(passing.sum()),informative_probes=int(informative.sum()),supported_native_entities=[ref['atlas']['paths'][i]['path_id'] for i in groups],source_pattern_coverage_bbox=[support.min(0).tolist(),support.max(0).tolist()],complete_pattern_not_single_patch=True,actual_RGB_material_context_checked=True,color_context_pass_count=int(color_ok.sum()))

def authority(ad_numbers=range(2,13)):
    assets=b.assets();core,model,native=package.current_native();paths=[];ats=b.read(b.OUT/'atlases.json')['atlases'];by_path=collections.defaultdict(list)
    for a in ats:
        for p in a['paths']:by_path[p['path_id']].append(dict(atlas_id=a['id'],source_image=a['image'],source_image_sha256=a['image_sha256'],source_contour_sha256=p['contour_sha256'],proof=p['proof']))
    for n,a in assets.items():
        if n not in ad_numbers:continue
        for name,path in a['paths'].items():
            native_id=f'print_instance/Ad{n}::{name}'
            if native_id not in native.paths:raise ValueError('No current native consumer for '+native_id)
            paths.append(dict(ad=n,native_id=native_id,asset=str(a['path']),asset_sha256=b.sha(a['path']),template_id=a['template']['id'],role=path.get('role'),closed=path.get('closed',False),shape_provenance=path.get('shape_provenance',path.get('shape_evidence')),parameter_dependencies=list(model.dependencies(native_id)),source_references=by_path[name]))
    OUT.mkdir(parents=True,exist_ok=True);b.write(OUT/'native_entity_registry.json',dict(scene=str(b.SCENE),scene_sha256=b.sha(b.SCENE),scene_revision=native.scene_revision,paths=paths,native_geometry_modified=False,missing_source_reference=[p['native_id'] for p in paths if not p['source_references']]))
    print(json.dumps(dict(stage='native_source_registry',entities=len(paths),without_source_reference=sum(not p['source_references'] for p in paths))),flush=True)

def registrations():
    cv2.setNumThreads(2);OUT.mkdir(parents=True,exist_ok=True);ats=b.read(b.OUT/'atlases.json')['atlases'];data=b.load();refs={i:reference(ats[i]) for i in ATLAS_IDS};candidates=collections.defaultdict(list);exact={str(Path(v['image']).resolve()).lower():pt for pt,v in VIS.items()}
    for p in (b.OUT/'verified_final').glob('*.json'):
        d=b.read(p)
        for r in d['registrations']:
            ai=r['atlas_id'];H=np.asarray(r['H'])
            if ai in (78,79,80):ai=76
            if ai not in refs:continue
            shape=geometry(H,refs[ai])
            if shape is None:continue
            if r.get('inliers',0)<8:continue
            candidates[(ai,d['pts'])].append(dict(H=r['H'],origin='old_source_registration_rechecked_as_hypothesis',source_file=str(p),source_sha256=b.sha(p),prior_inliers=r.get('inliers',0),prior_p95=r.get('p95'),source_shape=shape))
    accepted=collections.defaultdict(dict);rejected=collections.Counter();start=time.monotonic()
    for ai,ref in refs.items():
        exact_pt=exact.get(str(Path(ref['atlas']['image']).resolve()).lower())
        if exact_pt is not None:
            accepted[ai][exact_pt]=dict(H=np.eye(3).tolist(),atlas_id=ai,pts=exact_pt,method='same_authentic_source_exposure_identity',source_BGR_sha256=VIS[exact_pt]['source_BGR_sha256'],source_contours_from_current_native_definition=True)
    keys=sorted(candidates,key=lambda x:(x[1],x[0]))
    for ni,(ai,pt) in enumerate(keys):
        if pt in accepted[ai]:continue
        hypotheses=sorted(candidates[(ai,pt)],key=lambda r:(-r['prior_inliers'],r['prior_p95'] or 0))[:3];best=None
        for h in hypotheses:
            score=full_pattern_score(refs[ai],np.asarray(h['H']),pt)
            if score is not None and (best is None or score['pattern_correlation_median']>best['full_pattern_score']['pattern_correlation_median']):
                best=dict(h,atlas_id=ai,pts=pt,method='source_pattern_registered_and_semantic_neighborhood_rechecked',full_pattern_score=score,source_BGR_sha256=VIS[pt]['source_BGR_sha256'])
        if best:accepted[ai][pt]=best
        else:rejected['old_hypothesis_not_confirmed_as_pattern']+=1
        if ni%150==0:print(json.dumps(dict(stage='pattern_anchors',groups=ni+1,total=len(keys),confirmed=sum(map(len,accepted.values())),seconds=round(time.monotonic()-start,1))),flush=True)
    b.write(OUT/'source_pattern_anchors.json',dict(atlases=ATLAS_IDS,records={str(ai):{str(pt):r for pt,r in pts.items()} for ai,pts in accepted.items()},rejections=dict(rejected),native_projection_used=False,rows_excluded_by_this_stage=0))
    print(json.dumps(dict(stage='pattern_anchors_done',by_atlas={str(ai):len(pts) for ai,pts in accepted.items()},seconds=round(time.monotonic()-start,1))),flush=True)

VIS={f['pts']:f for f in b.read(b.OUT/'visibility.json')['frames']}
if __name__=='__main__':{'authority':authority,'anchors':registrations}[sys.argv[1]]()
