"""Rebind immutable S01 advertisement pixels to a frozen current native model.

Preparation only: image registrations identify material. They are not camera or
scene parameters and never enter the physical model. No solver is launched.
"""
from pathlib import Path
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import argparse, collections, hashlib, json, sys, time
import numpy as np
import cv2
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[4]
DATA=Path(r'D:\CodexData\RinkSpace_Data')
OUT=DATA/'runs/native_ad_replacement_s01_20261026'
FULL=DATA/'runs/solve2_s01/full_original/data_current/input01'
SCENE=ROOT/'events/ankara_2026/scene/field/model.json'
CURRENT=DATA/'runs/native_joint_s01_20261026'
SOURCE=DATA/'runs/solve2_s01'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def emit(event,**kw):
    line=json.dumps(dict(event=event,**kw),ensure_ascii=True,allow_nan=False)
    print(line,flush=True)
    with (OUT/'prepare.jsonl').open('a',encoding='utf-8') as f:f.write(line+'\n')
def array(d):return np.fromfile(d['file'],dtype=d['dtype']).reshape(d['shape'])
def load():
    pk=read(FULL/'packet/manifest.json');se=read(FULL/'semantic/manifest.json');st=read(pk['string_tables']['file'])
    uv=array(pk['arrays']['native_uv']);pts=array(pk['arrays']['native_pts']);mi=array(se['resolved_material_index'])
    labels=np.asarray(st['material_labels']);rowlabels=labels[mi]
    ad=np.zeros(len(uv),np.int16)
    for n in range(2,13):ad[np.char.startswith(rowlabels,f'Ad{n}.')]=n
    return dict(uv=uv,pts=pts,ids=np.asarray(st['stable_observation_ids']),ad=ad,
        labels=rowlabels,datasets=np.asarray(st['original_source_dataset_ids']),kind=array(se['arrays']['kind']),
        old_active=array(se['active_source_mask']),raw_prepared_ids=np.asarray(st['raw_prepared_curve_ids']))
def assets():
    scene=read(SCENE);out={}
    for ref in scene['advertisement_assets']:
        n=ref['advertisement_number'];p=(SCENE.parent/ref['path']).resolve();a=read(p)
        t=next(x for x in a['templates'] if x['id']==ref['template_id'])
        aliases=collections.defaultdict(list)
        for q in t['paths']:
            for key in [q['id'],q.get('original_path_id'),q.get('historical_curve_id'),q.get('historical_material_id'),*q.get('logical_labels',[])]:
                if key:aliases[key].append(q['id'])
        out[n]=dict(path=p,asset=a,template=t,aliases=aliases,paths={x['id']:x for x in t['paths']})
    return out

def build_atlases():
    OUT.mkdir(exist_ok=True);aa=assets();atlases={};missing=[]
    def add(n,image,path,poly,closed=False,proof=None):
        if path not in aa[n]['paths']:return
        image=Path(image)
        if not image.exists():missing.append(str(image));return
        q=np.asarray(poly,float)
        if q.ndim!=2 or q.shape[1]!=2 or len(q)<2 or not np.isfinite(q).all():return
        if closed and not np.allclose(q[0],q[-1]):q=np.vstack([q,q[0]])
        key=(n,str(image));entry=atlases.setdefault(key,dict(ad=n,image=str(image),image_sha256=sha(image),paths=[]))
        digest=hashlib.sha256(q.tobytes()).hexdigest()
        if any(x['path_id']==path and x['contour_sha256']==digest for x in entry['paths']):return
        entry['paths'].append(dict(path_id=path,pixels=q.tolist(),contour_sha256=digest,proof=proof))
    for n in range(2,13):
        d=aa[n]['asset']
        for ob in d.get('observations',[]):
            feature=ob.get('feature_id');sources=d.get('sources',{});source=sources.get(ob.get('source_id'),{})
            image=ob.get('source_frame_path') or source.get('frame_path')
            if image and feature in aa[n]['paths']:
                add(n,image,feature,ob.get('source_native_uv',[]),ob.get('closed',False),dict(kind='current_asset_source_observation',id=ob['id'],source_pts=ob.get('pts')))
        for path in aa[n]['paths'].values():
            pr=path.get('shape_provenance',{})
            if 'source_native_boundary' in pr:
                add(n,pr['source_frame']['path'],path['id'],pr['source_native_boundary'],path.get('closed',False),dict(kind='current_native_shape_source',asset=str(aa[n]['path'])))
    # Retained source-image contours, joined by explicit physical glyph IDs.
    gs=read(SOURCE/'full_original/ad45_identity/source_groups12.json')
    for group in gs:
        n=group['ad']
        for p in group['paths']:
            ids=aa[n]['aliases'].get(p['label'],[])
            if len(ids)==1:add(n,group['source_image'],ids[0],p['source_native_uv'],True,dict(kind='actual_high_resolution_glyph',group=group['name'],source_pts=group['source_pts']))
    atlas=read(SOURCE/'full_original/data_current/ad23_repair/ad2_atlas_320000.json')
    for p in atlas['paths']:
        ids=aa[2]['aliases'].get(p['label'],[])
        if len(ids)==1:add(2,atlas['source_image']['file'],ids[0],p['source_native_uv'],True,dict(kind='actual_frame_320000_glyph',source_pts=320000))
    src=read(SOURCE/'ad3_binding/typed_source_composition.json');layout=read(src['frozen_layout_reference']);H=np.asarray(layout['evidence']['Japanese_source_pixel_to_native_matrix'],float)
    for p in src['contours']:
        path='Ad3.print.'+p['id']
        if path in aa[3]['paths']:
            q=cv2.perspectiveTransform(np.asarray(p['xy_composition'],float)[None],H)[0]
            add(3,layout['evidence']['raw_source_image'],path,q,True,dict(kind='existing_official_artwork_to_actual_source_chart',composition_sha256=sha(SOURCE/'ad3_binding/typed_source_composition.json')))
    # Full-resolution original artwork is an identity reference, never a new observation.
    for n in (8,9,10):
        a=aa[n]
        for path in a['paths'].values():
            evidence=path.get('shape_evidence',{})
            if not evidence.get('path'):continue
            art=read(evidence['path']);r=art.get('raster',{});image=r.get('path')
            if not image:continue
            for component in art.get('components',[]):
                for g in component.get('geometry',[]):
                    gid=g.get('id','');key=path.get('original_shape_key','')
                    if not (gid==key or gid.endswith('.'+key)):continue
                    points=g.get('points_uv',g.get('uv',g.get('points')))
                    if points is None:continue
                    q=np.asarray(points,float)*[r['width'],r['height']]
                    add(n,image,path['id'],q,True,dict(kind='original_artwork_component',geometry_id=gid,artwork=str(evidence['path'])))
    # Exact source contour indices behind the current Ad5 40-mark identities.
    mature=DATA/'runs/okada_joint6_1021b/ad_shape_repair'
    original=read(mature/'native_print_model.json');mark=read(mature/'ad5_40_contours.json')
    photo5=DATA/'events/ankara_2026/programmes/okada_mei_sp/results/shot5_review_20261014/frames/003976.png'
    for material in original['materials']:
        ids=aa[5]['aliases'].get(material['label'],[])
        if len(ids)!=1:continue
        for oldpath in material['paths']:
            provenance=oldpath.get('provenance',{})
            for _ in range(8):
                if isinstance(provenance,str):provenance=json.loads(provenance)
                if 'native_contour' in provenance:break
                provenance=provenance.get('previous_source_provenance',{})
                if not provenance:break
            if 'native_contour' in provenance:
                contour=next(x for x in mark[str(provenance['native_threshold'])] if x['id']==provenance['native_contour'])
                add(5,photo5,ids[0],contour['uv'],True,dict(kind='original_40_mark_native_contour',threshold=provenance['native_threshold'],contour_id=contour['id']))

    # The blue and green rectangles are actual coloured MEIWA material edges,
    # not the white advertisement panel perimeter.
    im5=cv2.imread(str(photo5))
    for label,box in [('blue_bar',[1235,355,1475,407]),('green_bar',[1235,458,1475,502])]:
        x0,y0,x1,y1=box
        cs,_=cv2.findContours(np.uint8(cv2.cvtColor(im5[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)<180)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        q=max(cs,key=cv2.contourArea)[:,0,:]+[x0,y0]
        ids=aa[5]['aliases']['Ad5.print.native3976_'+label]
        if len(ids)!=1:raise ValueError('Missing exact native colour-bar identity')
        add(5,photo5,ids[0],q,True,dict(kind='actual_coloured_bar_source_boundary',source_pts=1273856,source_crop=box,threshold=180,recipe=str(mature/'restore_meiwa_v14.py')))
    # Ad6's real source-pixel contours: no current fitted/material coordinates.
    six=DATA/'runs/ad_glyph_fixes_20261025/ad6';clear=read(six/'clear550_contours.json');cc={x['i']:x for x in clear['contours']}
    headmask=cv2.imread(str(six/'material_half_coverage_mask.png'),0)
    hc,_=cv2.findContours(headmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for path in aa[6]['paths'].values():
        pr=path.get('shape_provenance',{}).get('glyph_revision_r2',{})
        if 'source_raster_contour' in pr and not pr.get('repeat_inference_not_direct_first_glyph_observation'):
            q=np.asarray(cc[pr['source_raster_contour']]['sourcecrop_uv'],float)+[0,436]
            add(6,pr['reference']['frame']['path'],path['id'],q,True,dict(kind='clear_native_glyph_material',source_contour_id=pr['source_raster_contour'],source_pts=396800))
        elif 'raster_contour' in pr:
            add(6,pr['source_plate']['path'],path['id'],hc[pr['raster_contour']][:,0,:],True,dict(kind='actual_native_material_plate',source_contour_id=pr['raster_contour']))
    # Official Ad7 vector curves are a pattern atlas, separate from unfitted
    # current internal/layout values. No current material_uv is read here.
    canvas=np.full((350,2470),255,np.uint8);official=[]
    for path in aa[7]['paths'].values():
        q=path.get('shape_provenance',{}).get('canonical_print_uv')
        if q is None:continue
        pixels=np.asarray(q,float)*[2470,350];official.append((path,pixels))
    for path,pixels in official:
        if path.get('closed',False):
            hole=bool(path.get('parent_contour_key')) or 'inner' in path.get('role','')
            cv2.fillPoly(canvas,[np.rint(pixels).astype(np.int32)],255 if hole else 0)
    official_image=OUT/'reference_ad7_official.png';cv2.imwrite(str(official_image),canvas)
    for path,pixels in official:
        add(7,official_image,path['id'],pixels,path.get('closed',False),dict(kind='original_official_vector_pattern',source_definition=path['shape_provenance']['source'],native_internal_geometry_used=False))
    # Ad3's boundary references are actual same-frame edge samples.
    raw=load()
    for suffix in ('top_edge','bottom_edge','left_vertical','right_vertical'):
        target='Ad3.'+suffix
        ix=np.flatnonzero((raw['pts']==332800)&(raw['labels']==target))
        if len(ix)>=2:add(3,DATA/'events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/first_shot_remaining_50fps/frames/000300.png',target,raw['uv'][ix],False,dict(kind='original_same_frame_material_edge',source_pts=332800))
    items=list(atlases.values())
    # A word row is a separate entity chart. The actual print may have a
    # different word-row placement from the official artwork; this does not
    # authorize changing the native layout.
    import copy
    official_entry=next(a for a in items if a['image']==str(official_image))
    for group,prefix in [('logo','Ad7.logo'),('top_word','Ad7.top_'),('bottom_word','Ad7.bottom_')]:
        a=copy.deepcopy(official_entry);a['paths']=[p for p in a['paths'] if p['path_id'].startswith(prefix)]
        a['entity_group']=group;a['photometric_transform']='invert_original_artwork_for_white_print_on_dark_field'
        items.append(a)
    for i,x in enumerate(items):x['id']=i
    identity=read(OUT/'authority_inventory.json')
    if sha(SCENE)!=identity['scene']['sha256']:raise ValueError('Native scene changed during preparation')
    for a in identity['assets']:
        if sha(a['path'])!=a['sha256']:raise ValueError('Native template changed during preparation')
    write(OUT/'atlases.json',dict(atlases=items,missing=sorted(set(missing)),scene=identity['scene']))
    emit('atlases_created',atlases=len(items),by_ad={n:dict(atlases=sum(x['ad']==n for x in items),paths=len({p['path_id'] for x in items if x['ad']==n for p in x['paths']}),native_paths=len(aa[n]['paths'])) for n in range(2,13)},missing_files=sorted(set(missing)))


def dense_poly(poly):
    p=np.asarray(poly,float);delta=np.diff(p,axis=0);length=np.linalg.norm(delta,axis=1)
    result=[];fraction=[]
    for j,(a,d,le) in enumerate(zip(p[:-1],delta,length)):
        t=np.linspace(0,1,max(2,int(np.ceil(le/.65))+1))
        result.append(a+t[:,None]*d);fraction.extend(((j+t)/(len(p)-1)).tolist())
    return np.vstack(result),np.asarray(fraction)

def gray_read(path):
    im=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if im is None:raise ValueError('Missing source pixels: '+str(path))
    if im.ndim==2:return im
    if im.shape[2]==4:
        alpha=im[:,:,3:4].astype(float)/255;im=np.rint(im[:,:,:3]*alpha+255*(1-alpha)).astype('uint8')
    return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

def prepare_reference(a,sift):
    gray=gray_read(a['image'])
    if a.get('photometric_transform'):gray=255-gray
    mask=np.zeros(gray.shape,np.uint8);polys=[];labels=[];fractions=[]
    for j,p in enumerate(a['paths']):
        q,f=dense_poly(p['pixels']);polys.append(q);labels.extend([j]*len(q));fractions.append(f)
        v=np.asarray(p['pixels'],float);lo=np.floor(v.min(0)-14).astype(int);hi=np.ceil(v.max(0)+15).astype(int)
        lo=np.maximum(lo,0);hi=np.minimum(hi,[gray.shape[1],gray.shape[0]])
        if np.all(hi>lo):mask[lo[1]:hi[1],lo[0]:hi[0]]=255
    kp,des=sift.detectAndCompute(gray,mask)
    return dict(metadata=a,gray=gray,mask=mask,points=np.vstack(polys),labels=np.asarray(labels),fractions=np.concatenate(fractions),
        keypoints=np.asarray([k.pt for k in kp],float),descriptors=des)

def registration(reference,keypoints,descriptors,matcher):
    des=reference['descriptors']
    if des is None or descriptors is None or len(des)<6 or len(descriptors)<6:return None
    pairs=matcher.knnMatch(des,descriptors,k=2)
    good=[m for q in pairs if len(q)==2 for m,n in [q] if m.distance<.72*n.distance]
    if len(good)<8:return None
    src=reference['keypoints'][[x.queryIdx for x in good]];dst=keypoints[[x.trainIdx for x in good]]
    H,inlier=cv2.findHomography(src,dst,cv2.RANSAC,2.5,maxIters=2500,confidence=.999)
    if H is None or inlier is None:return None
    ok=inlier.ravel().astype(bool)
    if ok.sum()<8 or ok.mean()<.28:return None
    pred=cv2.perspectiveTransform(src[ok][None],H)[0];errors=np.linalg.norm(pred-dst[ok],axis=1)
    p95=float(np.percentile(errors,95))
    if p95>2.5 or abs(np.linalg.det(H))<1e-10:return None
    if cv2.contourArea(cv2.convexHull(dst[ok].astype('float32')))<25:return None
    return dict(H=H,p95=p95,inliers=int(ok.sum()),matches=len(good),source_matches=src[ok],target_matches=dst[ok])

def samples(image,xy):
    return cv2.remap(image.astype(np.float32),xy[...,0].astype(np.float32),xy[...,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)

def patch_correlation(reference,current,H,uv):
    off=np.mgrid[-5:6,-5:6].reshape(2,-1).T[:,::-1]
    query=uv[:,None,:]+off[None,:,:]
    source=cv2.perspectiveTransform(query.reshape(1,-1,2),np.linalg.inv(H))[0].reshape(query.shape)
    a=samples(reference,source);b=samples(current,query)
    a-=a.mean(axis=1,keepdims=True);b-=b.mean(axis=1,keepdims=True)
    den=np.sqrt((a*a).sum(1)*(b*b).sum(1));corr=np.divide((a*b).sum(1),den,out=np.full(len(uv),-1.,float),where=den>1.)
    return corr

def assess(reference,reg,gray,uv):
    H=reg['H'];projected=cv2.perspectiveTransform(reference['points'][None],H)[0]
    finite=np.isfinite(projected).all(1)&(np.abs(projected).max(1)<10000)
    if finite.sum()<10:return []
    xy=projected[finite];lab=reference['labels'][finite];frac=reference['fractions'][finite]
    dist,idx=cKDTree(xy).query(uv,k=min(24,len(xy)))
    if dist.ndim==1:return []
    nearest=idx[:,0];bestlab=lab[nearest];best=dist[:,0]
    # Only actual source-image contours are compared here; no native projection
    # or current native dimensions appear in this function.
    different=lab[idx]!=bestlab[:,None];other=np.min(np.where(different,dist,np.inf),axis=1)
    close=(best<=3.0)&((other-best)>=max(1.,reg['p95']))
    rows=np.flatnonzero(close)
    if not len(rows):return []
    corr=patch_correlation(reference['gray'],gray,H,uv[rows])
    gradx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)/8;grady=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)/8
    gx=samples(gradx,uv[rows,None,:]).ravel();gy=samples(grady,uv[rows,None,:]).ravel();gradient=np.hypot(gx,gy)
    accepted=[]
    for k,row in enumerate(rows):
        if corr[k]<.70 or gradient[k]<1.0:continue
        j=int(bestlab[row]);record=reference['metadata']['paths'][j]
        accepted.append(dict(local_row=int(row),path_id=record['path_id'],atlas_id=reference['metadata']['id'],
            image_match_correlation=float(corr[k]),material_gradient=float(gradient[k]),
            source_contour_distance=float(best[row]),other_contour_distance=float(other[row]) if np.isfinite(other[row]) else None,
            registration_p95=reg['p95'],registration_inliers=reg['inliers'],source_curve_fraction=float(frac[nearest[row]]),
            identity_method='named_actual_source_entity_with_image_registration_and_local_appearance',
            native_geometry_distance_used=False))
    return accepted

def associate(ad_filter=None):
    import av
    cv2.setNumThreads(2);data=load();atlasdoc=read(OUT/'atlases.json');sift=cv2.SIFT_create(nfeatures=3000,contrastThreshold=.018)
    refs=collections.defaultdict(list)
    for a in atlasdoc['atlases']:
        if ad_filter is None or a['ad'] in ad_filter:refs[a['ad']].append(prepare_reference(a,sift))
    rows=np.flatnonzero(np.isin(data['ad'],list(refs)));wanted=sorted(set(map(int,data['pts'][rows])))
    frame_rows={pt:rows[data['pts'][rows]==pt] for pt in wanted};matcher=cv2.BFMatcher(cv2.NORM_L2)
    source=read(ROOT/'events/ankara_2026/programmes/okada_mei_sp/primary_source_selection.json')
    video=next(p for p in (DATA/'events/ankara_2026/programmes/okada_mei_sp').glob('*.mp4') if 'Short Program' in p.name)
    start=time.monotonic();done=0;accepted_count=0;total=0
    emit('source_entity_matching_started',rows=len(rows),frames=len(wanted),ads=sorted(refs),native_geometry_fixed=True,source_video=str(video))
    with av.open(str(video)) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO';container.seek(wanted[0],stream=stream,backward=True)
        for frame in container.decode(stream):
            pt=int(frame.pts)
            if pt>wanted[-1]:break
            if pt not in frame_rows:continue
            gray=frame.to_ndarray(format='gray');rr=frame_rows[pt]
            for ad in sorted(set(map(int,data['ad'][rr]))):
                sub=rr[data['ad'][rr]==ad];destination=OUT/'image_checks'/f'ad{ad}_{pt}.json'
                if destination.exists():
                    prior=read(destination)
                    if prior['rows']!=sub.tolist():raise ValueError('Source rows changed in resume')
                    accepted_count+=len(prior['confirmed']);total+=len(sub);continue
                q=data['uv'][sub];mask=np.zeros_like(gray);lo=np.maximum(np.floor(q.min(0)-35).astype(int),0);hi=np.minimum(np.ceil(q.max(0)+36).astype(int),[gray.shape[1],gray.shape[0]])
                if np.all(hi>lo):mask[lo[1]:hi[1],lo[0]:hi[0]]=255
                kp,des=sift.detectAndCompute(gray,mask);kp=np.asarray([x.pt for x in kp],float)
                votes=collections.defaultdict(list);regs=[]
                for ref in refs[ad]:
                    reg=registration(ref,kp,des,matcher)
                    if reg is None:continue
                    regs.append(dict(atlas_id=ref['metadata']['id'],H=reg['H'].tolist(),p95=reg['p95'],inliers=reg['inliers'],matches=reg['matches']))
                    for x in assess(ref,reg,gray,q):votes[int(sub[x.pop('local_row')])].append(x)
                confirmed=[];conflicts=[]
                for row,vv in votes.items():
                    choices={v['path_id'] for v in vv}
                    if len(choices)!=1:
                        conflicts.append(dict(row=row,choices=sorted(choices)));continue
                    v=max(vv,key=lambda x:x['image_match_correlation']);confirmed.append(dict(row=row,stable_id=str(data['ids'][row]),**v))
                write(destination,dict(schema='rinkspace.source_entity_image_check.v1',ad=ad,pts=pt,decoded_gray_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),rows=sub.tolist(),registrations=regs,confirmed=confirmed,conflicts=conflicts,
                    unconfirmed=[int(r) for r in sub if int(r) not in {x['row'] for x in confirmed}],model_projection_used_for_identity=False))
                accepted_count+=len(confirmed);total+=len(sub)
            done+=1
            if done%40==0:emit('image_match_progress',frames=done,total_frames=len(wanted),checked_rows=total,confirmed=accepted_count,seconds=round(time.monotonic()-start,2))
    emit('image_matching_complete',frames=done,checked_rows=total,confirmed=accepted_count,seconds=round(time.monotonic()-start,2))


def visibility_index():
    base=DATA/'events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013'
    packets=ROOT/'events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013'
    rows={}
    for folder in ('opening_100_50fps','continuation_101_200_50fps'):
        for p in sorted((packets/folder/'frames').glob('*.json')):
            d=read(p);r=d['source'];pt=int(r['native_pts'])
            rows[pt]=dict(pts=pt,image=r['image_path'],mask_path=r['mask_path'],mask_sha256=r['mask_sha256'],source_BGR_sha256=r.get('decoded_BGR_sha256'),source_video_sha256=r['source_video_sha256'],authority=str(p),authority_sha256=sha(p),timer_rectangle=r.get('broadcast_timer_exclusion',{}).get('actual_native_rectangle_xyxy_inclusive'))
    manifest=base/'first_shot_remaining_50fps/repair_20261014/frame_manifest.json'
    d=read(manifest)
    for f in d['frames']:
        r=f['source'];pt=int(r['native_pts'])
        rows[pt]=dict(pts=pt,image=r['image_path'],mask_path=f.get('mask_path',r['mask_path']),mask_sha256=f.get('mask_sha256',r['mask_sha256']),source_BGR_sha256=r.get('decoded_BGR_sha256'),source_video_sha256=r['source_video_sha256'],authority=str(manifest),authority_sha256=sha(manifest))
    correction=base/'first_shot_remaining_50fps/repair_20261014/mask_corrections.json'
    for c in read(correction)['rows']:
        r=rows[256000+256*c['index']]
        r.update(mask_path=c['new_mask_path'],mask_sha256=c['new_mask_sha256'],mask_revision=str(correction),mask_revision_sha256=sha(correction))
    source=load();hits=[];bad=[]
    for pt,r in sorted(rows.items()):
        p=Path(r['mask_path'])
        if not p.exists() or sha(p)!=r['mask_sha256']:raise ValueError('Mask authority changed: '+str(p))
        mask=cv2.imread(str(p),0)
        if mask.shape!=(1080,1920):raise ValueError('Source mask resolution changed')
        r['foreground_pixels']=int(np.count_nonzero(mask))
        ix=np.flatnonzero((source['pts']==pt)&(source['ad']>0))
        xy=np.rint(source['uv'][ix]).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<1920)&(xy[:,1]>=0)&(xy[:,1]<1080)
        rr=ix[inside];q=xy[inside]
        hit=rr[mask[q[:,1],q[:,0]]>0]
        hits.extend(hit.tolist());bad.extend(ix[~inside].tolist())
    if set(rows)!=set(range(256256,575489,256)):raise ValueError('Incomplete 1248-frame mask coverage')
    write(OUT/'visibility.json',dict(schema='rinkspace.s01_existing_source_sam2_exclusion.v1',frames=list(rows.values()),mask_frames=len(rows),mask_contains_existing_conservative_margin=True,extra_dilation_applied=False,inside_person_mask_rows=hits,outside_image_rows=bad,source_images_unchanged=True))
    emit('source_sam2_coverage',frames=len(rows),person_mask_rows=len(hits),outside_image=len(bad),known_occluded_row_28179_is_masked=28179 in hits)
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['atlases','associate','visibility']);ap.add_argument('--ads',type=int,nargs='+');args=ap.parse_args()
    if args.stage=='visibility':visibility_index()
    elif args.stage=='atlases':build_atlases()
    elif args.stage=='associate':associate(args.ads)

if __name__=='__main__':main()
