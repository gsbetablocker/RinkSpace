"""S04 source-only advertisement review, reusing the S01 material/identity audit."""
from pathlib import Path
import sys,json,gzip,hashlib,base64,collections,time,shutil,re
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ad_replacement_s01 as b
ROOT=b.ROOT;DATA=b.DATA;PROG=Path(__file__).resolve().parent
OUT=DATA/'runs/native_ad_points_s04_20261026';WORK=OUT/'source_review'
PREVIOUS=DATA/'runs/native_ad_replacement_s01_20261026'
REC=DATA/'runs/okada_rec_1019/recover_v2';PRINT=REC/'dense_print_v1'
FIRST=745984;LAST=948736;ADNUMBERS=range(1,6);PREFIX='s04_native_source_v1_'

def emit(stage,**kw):print(json.dumps(dict(stage=stage,**kw),ensure_ascii=False),flush=True)
def write(p,x):b.write(p,x)
def sha(p):return b.sha(p)

def prepare():
    import av
    cv2.setNumThreads(2);OUT.mkdir(exist_ok=True);WORK.mkdir(exist_ok=True);frames=OUT/'source_frames';frames.mkdir(exist_ok=True)
    scope=b.read(PROG/'rebuild_scope.json');selected=next(s for s in scope['shots'] if s['id']=='s04')['processing']
    assert selected['first_pts']==FIRST and selected['last_pts']==LAST and selected['native_frame_count']==793
    video=next((DATA/'events/ankara_2026/programmes/okada_mei_sp').glob('Mei OKADA*.mp4'))
    if sha(video)!=scope['source']['sha256']:raise ValueError('Source video differs from programme authority')
    index={}
    with gzip.open(PRINT/'final_index_anchor.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            c=json.loads(line)
            if c['shot_id']=='s04':index[c['native_pts']]=c
    vis=[];start=time.monotonic()
    with av.open(str(video)) as c:
        stream=c.streams.video[0];c.seek(FIRST,stream=stream)
        for frame in c.decode(stream):
            pt=frame.pts
            if pt<FIRST:continue
            if pt>LAST:break
            im=frame.to_ndarray(format='bgr24');digest=hashlib.sha256(im.tobytes()).hexdigest();prior=index[pt]
            if digest!=prior['decoded_native_BGR_sha256']:raise ValueError('Current source pixels disagree with prior source readout')
            mask=prior['mask_ref']
            if sha(mask['file'])!=mask['sha256']:raise ValueError('Prior source actor mask changed')
            dst=frames/f'{pt}.png'
            if not dst.exists():cv2.imwrite(str(dst),im,[cv2.IMWRITE_PNG_COMPRESSION,1])
            elif hashlib.sha256(cv2.imread(str(dst)).tobytes()).hexdigest()!=digest:raise ValueError('Cached source RGB changed')
            vis.append(dict(pts=pt,shot_id='s04',image=str(dst),source_BGR_sha256=digest,source_video_sha256=scope['source']['sha256'],mask_path=mask['file'],mask_sha256=mask['sha256'],historical_frame_record=prior['file'],historical_frame_record_sha256=prior['sha256']))
            if len(vis)%150==0:emit('source_decode',frames=len(vis),seconds=round(time.monotonic()-start,1))
    if [x['pts'] for x in vis]!=list(range(FIRST,LAST+1,256)):raise ValueError('Incomplete original exposure range')
    write(OUT/'visibility.json',dict(source_video=str(video),source_video_sha256=scope['source']['sha256'],source_time_base='1/12800',shot_id='s04',frames=vis))
    # Existing atlas indices stay stable because the shared checks refer to exact
    # independent artwork charts. Unused advertisements are never registered.
    atlases=b.read(PREVIOUS/'atlases.json')['atlases'];assets=b.assets();new={}
    def add(image,name,points,proof,partial=False,closed=False):
        image=Path(image);q=np.asarray(points,float)
        if name not in assets[1]['paths'] or q.ndim!=2 or len(q)<2 or not image.exists():return
        if closed and not np.allclose(q[0],q[-1]):q=np.vstack([q,q[0]])
        key=(str(image),('ALL_SKATING' if any(x in name for x in ['.ALL_','.SKATING_']) else 'ONE_APP') if proof['kind']=='independent_original_artwork_contour' else 'source');a=new.setdefault(key,dict(ad=1,image=str(image),image_sha256=sha(image),paths=[],partial_source_reference=partial))
        digest=hashlib.sha256(q.tobytes()).hexdigest()
        if any(p['path_id']==name and p['contour_sha256']==digest for p in a['paths']):return
        a['paths'].append(dict(path_id=name,pixels=q.tolist(),contour_sha256=digest,proof=proof))
    a=assets[1]['asset']
    for ob in a['observations']:
        src=ob.get('source_image',{}).get('file')
        if src:add(src,ob['feature_id'],ob['source_native_uv'],dict(kind='current_native_asset_actual_source_observation',observation_id=ob['id'],pts=ob.get('pts'),asset=str(assets[1]['path'])),partial=True)
    for ev in a['shape_evidence']:
        src=a['sources'][ev['source_id']]
        add(src['path'],ev['feature_id'],ev['source_native_uv'],dict(kind='independent_original_artwork_contour',source_sha256=src['sha256'],asset=str(assets[1]['path'])),closed=True)
    for v in new.values():v['id']=len(atlases);atlases.append(v)
    write(OUT/'atlases.json',dict(atlases=atlases,inherited_reference_manifest=str(PREVIOUS/'atlases.json'),inherited_reference_sha256=sha(PREVIOUS/'atlases.json'),new_Ad1_source_charts=len(new)))
    write(WORK/'additional_source_atlases.json',dict(atlases=[]))
    authority=dict(scene=dict(path=str(b.SCENE),sha256=sha(b.SCENE)),camera=dict(path=str(PROG/'ad_cameras/s04.json'),sha256=sha(PROG/'ad_cameras/s04.json')),assets=[dict(ad=n,path=str(assets[n]['path']),sha256=sha(assets[n]['path'])) for n in ADNUMBERS],source_scope=selected,source_video=dict(path=str(video),sha256=scope['source']['sha256']))
    write(OUT/'authority_inventory.json',authority)
    emit('source_preparation_complete',frames=len(vis),new_Ad1_charts=len(new),seconds=round(time.monotonic()-start,1))

def inventory():
    OUT.mkdir(exist_ok=True);WORK.mkdir(exist_ok=True);start=time.monotonic();alluv=[];allpts=[];allids=[];labels=[];ads=[];kinds=[];provenance=[];counts=collections.Counter();seen=set();framecount=0
    with gzip.open(OUT/'candidate_provenance.jsonl.gz','wt',encoding='utf8') as evidence:
        for pt in range(FIRST,LAST+1,256):
            path=PRINT/'final_frames_anchor'/f'{pt}.json';d=b.read(path);eligible=np.frombuffer(base64.b64decode(d['eligible_uint8_base64']),np.uint8);row=0;local=[]
            for mi,member in enumerate(d['source_members']):
                with open(member['file'],'rb') as f:f.seek(member['offset_bytes']);blob=f.read(member['length_bytes'])
                if hashlib.sha256(blob).hexdigest()!=member['sha256']:raise ValueError('Source compressed member changed')
                rows=[json.loads(x) for x in gzip.decompress(blob).decode('utf8').splitlines()]
                if len(rows)!=member['row_count']:raise ValueError('Original member row count changed')
                for ri,c in enumerate(rows):
                    counts['all_original_print_rows']+=1
                    if not eligible[row]:counts['previously_unqualified_retained_in_source']+=1;row+=1;continue
                    row+=1;sid=c['point_id'];key=(pt,sid)
                    if key in seen:counts['duplicate_stable_source_ID']+=1;continue
                    seen.add(key);uv=c['uv'];label=c.get('feature_id','unassigned_print');match=re.match(r'Ad(\d+)\.',label);ad=int(match.group(1)) if match else 99
                    idx=len(alluv);alluv.append(uv);allpts.append(pt);allids.append(sid);labels.append(label if match else 'unassigned_current_print');ads.append(ad);kinds.append(0)
                    evidence.write(json.dumps(dict(source_row=idx,stable_id=sid,pts=pt,uv=uv,origin='latest_complete_dense_print',source_frame=str(path),member_index=mi,member_row=ri,source_kind=c.get('source_kind'),previous_eligibility=True,previous_named_native_correspondence=bool(match)),ensure_ascii=False)+'\n');counts['latest_dense_candidate_rows']+=1
            if row!=len(eligible):raise ValueError('Eligibility mask does not address exact source member sequence')
            framecount+=1
            if framecount%150==0:emit('source_inventory',frames=framecount,rows=len(alluv),seconds=round(time.monotonic()-start,1))
        # The earlier reviewed packets hold explicit names and manual/corner
        # provenance. Only actual readout coordinates are collected as candidates.
        packets=sorted((PROG/'source_datasets/shot4_20261014/frames').glob('*.json'))
        for path in packets:
            d=b.read(path);packetsha=sha(path);pt=int(d['source_pts'])
            for feature in d.get('features',[]):
                label=feature['id'];match=re.match(r'Ad(\d+)\.',label)
                if not match:continue
                for si,seg in enumerate(feature.get('segments',[])):
                    points=seg.get('source_native_readout_uv')
                    if points is None and seg.get('pixels_measured_in_this_frame'):points=seg.get('uv')
                    if points is None:counts['historical_curve_estimates_not_observations']+=len(seg.get('uv',[]));continue
                    for j,uv in enumerate(points):
                        sid=f's04_reviewed_raw_{pt}_{label}_{si}_{j}';idx=len(alluv);alluv.append(uv);allpts.append(pt);allids.append(sid);labels.append(label);ads.append(int(match.group(1)));kinds.append(0)
                        evidence.write(json.dumps(dict(source_row=idx,stable_id=sid,pts=pt,uv=uv,origin='earlier_named_actual_source_readout',source_packet=str(path),source_packet_sha256=packetsha,feature_id=label,segment=si,raw_point=j,previous_allowed_for_fit=feature.get('allowed_for_fit'),no_previous_fit_status_implicitly_promoted=True),ensure_ascii=False)+'\n');counts['earlier_named_raw_candidates']+=1
    z=dict(uv=np.asarray(alluv,float),pts=np.asarray(allpts,np.int64),ids=np.asarray(allids),labels=np.asarray(labels),ad=np.asarray(ads,np.int16),kind=np.asarray(kinds,np.uint8));np.savez_compressed(OUT/'original_ad_candidates.npz',**z)
    sourcefiles=[PRINT/'final_manifest_anchor.json',PRINT/'final_index_anchor.jsonl.gz',REC/'dense234_v1/generation_v2/manifest.json',PROG/'source_datasets/dense234_20261020/result.json',PROG/'source_datasets/shot4_20261014/batch_manifest.json']
    write(OUT/'source_inventory.json',dict(shot_id='s04',frames=framecount,counts=dict(counts),candidate_rows=len(alluv),candidate_file_sha256=sha(OUT/'original_ad_candidates.npz'),sources=[dict(path=str(p),sha256=sha(p)) for p in sourcefiles],latest_full_pool_not_limited_to_500_point_display_selection=True,source_UV_and_PTS_unchanged=True,old_source_files_changed=False,candidates_not_automatically_fit_observations=True))
    emit('source_inventory_complete',rows=len(alluv),counts=dict(counts),seconds=round(time.monotonic()-start,1))

def complete_references():
    doc=b.read(OUT/'atlases.json');ats=doc['atlases'];assets=b.assets();added=[];groups={}
    def add(ad,image,name,q,proof):
        q=np.asarray(q,float)
        if q.ndim!=2 or len(q)<2:return
        image=Path(image)
        if not image.is_file():return
        key=(ad,str(image));a=groups.setdefault(key,dict(ad=ad,image=str(image),image_sha256=sha(image),paths=[],partial_source_reference=True))
        digest=hashlib.sha256(q.tobytes()).hexdigest()
        if any(p['path_id']==name and p['contour_sha256']==digest for old in ats if old['ad']==ad and old['image']==str(image) for p in old['paths']):return
        if any(p['path_id']==name and p['contour_sha256']==digest for p in a['paths']):return
        a['paths'].append(dict(path_id=name,pixels=q.tolist(),contour_sha256=digest,proof=proof))
    for ob in assets[1]['asset']['observations']:
        name=ob['feature_id']
        if ob.get('source_image',{}).get('file'):continue
        ann=ob.get('source_annotation');ann=ann if isinstance(ann,str) else ann.get('path',ann.get('file'))
        d=b.read(ann);image=d['source']['image_path']
        add(1,image,name,ob['source_native_uv'],dict(kind='current_native_asset_source_observation',asset=str(assets[1]['path']),observation_id=ob['id'],source_annotation=ann,source_pts=ob['pts']))
    for file in sorted((PROG/'source_datasets/shot4_20261014/frames').glob('*.json')):
        d=b.read(file);image=d['source']['image_path']
        for f in d['features']:
            match=re.match(r'Ad(\d+)\.',f['id'])
            if not match or int(match.group(1)) not in ADNUMBERS:continue
            ad=int(match.group(1));matches=assets[ad]['aliases'].get(f['id'],[])
            if len(matches)!=1:continue
            for si,seg in enumerate(f.get('segments',[])):
                q=seg.get('source_native_readout_uv')
                if q is None and seg.get('pixels_measured_in_this_frame'):q=seg.get('uv')
                if q is not None:add(ad,image,matches[0],q,dict(kind='earlier_actual_source_readout_with_unique_current_asset_alias',source_packet=str(file),source_packet_sha256=sha(file),original_feature=f['id'],current_native_path=matches[0],source_pts=d['source_pts'],segment=si))
    for a in groups.values():
        if not a['paths']:continue
        a['id']=len(ats);ats.append(a);added.append(dict(id=a['id'],ad=a['ad'],paths=len(a['paths']),image=a['image']))
    write(OUT/'atlases.json',doc);write(WORK/'reference_completion.json',dict(additions=added,native_parameters_used_for_registration=False,actual_source_readouts_not_renderer_geometry=True))
    emit('references_completed',added=added)

def configure():
    import ad_entity_registration_s01 as r,ad_entity_tracks_s01 as t,ad_entity_measure_s01 as m,ad_entity_observability_s01 as o,ad_entity_select_s01 as s
    b.OUT=OUT;r.OUT=WORK;r.VIS={f['pts']:f for f in b.read(OUT/'visibility.json')['frames']};r.current.cache_clear()
    for module in (t,m,o,s):module.OUT=WORK
    ats=b.read(OUT/'atlases.json')['atlases'];chosen=[a['id'] for a in ats if a['ad'] in ADNUMBERS and a['id'] not in (0,2,5,8)]
    t.PRIMARY=chosen;r.ATLAS_IDS=chosen
    o.CANONICAL_BY_AD={1:[81,82,83,84],2:[1,3],3:[70,77],4:[4,6,69],5:[7]}
    o.LOCAL_PRINT_SHARPNESS_ADS={5}
    o.BEST_INDEPENDENT_SHAPE_ADS={1}
    extra=WORK/'additional_source_atlases.json'
    if extra.exists():
        o.CANONICAL_BY_PATH={p['path_id']:[a['id']] for a in b.read(extra)['atlases'] for p in a['paths'] if a.get('individually_registered_official_glyph')}
    return r,t,m,o,s,ats,chosen

def anchors():
    import ad_source_verify_s01 as old
    r,t,m,o,s,ats,chosen=configure();cv2.setNumThreads(2);sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.008);matcher=cv2.BFMatcher(cv2.NORM_L2);refs={};features={};result=collections.defaultdict(dict);start=time.monotonic();hashpts={f['source_BGR_sha256']:pt for pt,f in r.VIS.items()}
    for ai in chosen:
        ref=r.reference(ats[ai]);refs[ai]=ref;roi=np.zeros_like(ref['gray'])
        for path in ats[ai]['paths']:cv2.polylines(roi,[np.rint(path['pixels']).astype(np.int32)],False,255,25)
        kp,ds=sift.detectAndCompute(ref['gray'],roi);features[ai]=dict(keypoints=np.array([k.pt for k in kp]),descriptors=ds)
        digest=hashlib.sha256(ref['color'].tobytes()).hexdigest()
        if digest in hashpts:
            pt=hashpts[digest];result[ai][pt]=dict(H=np.eye(3).tolist(),atlas_id=ai,pts=pt,method='same_authentic_source_exposure_identity',source_BGR_sha256=digest,source_contours_from_current_native_definition=True)
    pts=sorted(r.VIS);sample=sorted(set(pts[::10]+[pts[-1]]))
    for fi,pt in enumerate(sample):
        im,gray,mask=r.current(pt);roi=np.ones_like(gray,np.uint8)*255;roi[:185]=0;roi[cv2.dilate(mask,np.ones((9,9),np.uint8))>0]=0;kp,ds=sift.detectAndCompute(gray,roi);xy=np.array([k.pt for k in kp])
        for ai in chosen:
            if pt in result[ai]:continue
            best=None
            for c in old.image_candidates(features[ai],xy,ds,matcher):
                if c['inliers']<8 or r.geometry(np.asarray(c['H']),refs[ai]) is None:continue
                score=r.full_pattern_score(refs[ai],np.asarray(c['H']),pt,allow_partial=True)
                if score is None:continue
                rank=score['pattern_correlation_median']+min(c['inliers'],100)*.001
                if best is None or rank>best['rank']:best=dict(c,rank=rank,atlas_id=ai,pts=pt,full_pattern_score=score,source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],method='actual_source_pattern_correspondence',native_geometry_used=False)
            if best:result[ai][pt]=best
        if fi%15==0:emit('source_pattern_anchors',frames=fi+1,total=len(sample),anchors=sum(map(len,result.values())),seconds=round(time.monotonic()-start,1))
    write(WORK/'source_pattern_anchors.json',dict(atlases=chosen,records={str(ai):{str(pt):v for pt,v in records.items()} for ai,records in result.items()},native_projection_used=False,rows_excluded_by_this_stage=0))
    r.authority(ad_numbers=ADNUMBERS);emit('anchors_done',by_atlas={ai:len(v) for ai,v in result.items()},seconds=round(time.monotonic()-start,1))

def recover_patterns():
    import ad_entity_ecc_s01 as ecc,ad_source_verify_s01 as old
    r,t,m,o,s,ats,chosen=configure();cv2.setNumThreads(2);anchors=b.read(WORK/'source_pattern_anchors.json');result=collections.defaultdict(dict);start=time.monotonic();proof=[]
    # Independent official artwork may use a different two-row layout. Each
    # row is registered to actual named source glyphs, not to native geometry.
    for ai in (82,83):
        reference=r.reference(ats[ai]);a=[];c=[]
        for path in ats[ai]['paths']:
            if '_inner' in path['path_id']:continue
            matching=[x for x in ats[81]['paths'] if x['path_id']==path['path_id']]
            if not matching:continue
            x=np.asarray(path['pixels']);y=np.concatenate([np.asarray(x['pixels']) for x in matching]);lo=x.min(0);hi=x.max(0);ll=y.min(0);hh=y.max(0)
            a.extend([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]]);c.extend([[ll[0],ll[1]],[hh[0],ll[1]],[hh[0],hh[1]],[ll[0],hh[1]]])
        A,_=cv2.estimateAffine2D(np.asarray(a,float),np.asarray(c,float),method=cv2.RANSAC,ransacReprojThreshold=5,maxIters=6000)
        if A is None:continue
        initial=np.vstack([A,[0,0,1]])
        for pt,row in anchors['records'].get('81',{}).items():
            pt=int(pt);H=np.asarray(row['H'])@initial;score=r.full_pattern_score(reference,H,pt,allow_partial=True)
            best=dict(H=H.tolist(),rank=score['pattern_correlation_median'],full_pattern_score=score) if score else ecc.refine(reference,H,pt,allow_partial=True)
            if best:result[ai][pt]=dict(best,atlas_id=ai,pts=pt,method='independent_artwork_row_matched_to_actual_current_pixels',source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],source_corresponding_glyph_boxes_are_navigation_only=True,native_geometry_used=False)
    # The last S04 Ad5 is cropped and moving quickly. The visible whole word
    # supplies only a navigation hypothesis; current material checks decide dots.
    ref=r.reference(ats[7]);source_box=np.array([[1485.,380.],[1850.,477.]])
    target_box=np.array([[151.,568.],[779.,710.]])
    scale=(target_box[1]-target_box[0])/(source_box[1]-source_box[0]);H=np.diag([*scale,1.]);H[:2,2]=target_box[0]-scale*source_box[0]
    score=r.full_pattern_score(ref,H,LAST,allow_partial=True)
    best=dict(H=H.tolist(),rank=score['pattern_correlation_median'],full_pattern_score=score) if score else ecc.refine(ref,H,LAST,allow_partial=True)
    if best:result[7][LAST]=dict(best,atlas_id=7,pts=LAST,method='visually_identified_current_Ad5_word_then_actual_RGB_registration',source_BGR_sha256=r.VIS[LAST]['source_BGR_sha256'],approximate_word_boxes_are_navigation_only=True,native_geometry_used=False)
    proof.append(dict(atlas_id=7,pts=LAST,source_navigation_box=source_box.tolist(),current_navigation_box=target_box.tolist(),accepted_pattern=best is not None))
    # Panel interiors contain identifying lettering even when the modeled
    # outer-edge atlas has no distinctive SIFT features on its perimeter.
    sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.008);matcher=cv2.BFMatcher(cv2.NORM_L2);refs={};features={}
    for ai in (77,84,85,86,87):
        ref=r.reference(ats[ai]);refs[ai]=ref;roi=np.zeros_like(ref['gray']);q=np.concatenate([np.asarray(p['pixels']) for p in ats[ai]['paths']]);cv2.fillConvexPoly(roi,cv2.convexHull(np.rint(q).astype(np.int32)),255);key,desc=sift.detectAndCompute(ref['gray'],roi);features[ai]=dict(keypoints=np.array([k.pt for k in key]),descriptors=desc)
    for pt in sorted(r.VIS)[::20]:
        im,gray,mask=r.current(pt);roi=np.ones_like(gray,np.uint8)*255;roi[:185]=0;roi[cv2.dilate(mask,np.ones((9,9),np.uint8))>0]=0;kp,ds=sift.detectAndCompute(gray,roi);xy=np.array([k.pt for k in kp])
        for ai in refs:
            best=None
            for candidate in old.image_candidates(features[ai],xy,ds,matcher):
                HH=np.asarray(candidate['H'])
                if candidate['inliers']<8 or r.geometry(HH,refs[ai]) is None:continue
                score=r.full_pattern_score(refs[ai],HH,pt,allow_partial=True)
                if score is None:continue
                rank=score['pattern_correlation_median']+min(candidate['inliers'],100)*.001
                if best is None or rank>best['rank']:best=dict(candidate,rank=rank,full_pattern_score=score,atlas_id=ai,pts=pt,method='actual_panel_interior_identity_with_measured_outer_contour',source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],native_geometry_used=False)
            if best:result[ai][pt]=best
    write(WORK/'additional_pattern_anchors.json',dict(records={str(ai):{str(pt):v for pt,v in rows.items()} for ai,rows in result.items()},proof=proof,native_geometry_used=False))
    emit('additional_pattern_anchors',by_atlas={ai:len(rows) for ai,rows in result.items()},proof=proof,seconds=round(time.monotonic()-start,1))

def tracks():
    r,t,m,o,s,ats,chosen=configure();t.run()

def extra_tracks():
    r,t,m,o,s,ats,chosen=configure()
    t.PRIMARY=list(map(int,b.read(WORK/'additional_pattern_anchors.json')['records']))
    t.run(WORK/'additional_pattern_anchors.json',WORK/'additional_identity_charts.json')

def refine_ad1_references():
    r,t,m,o,s,ats,chosen=configure();cv2.setNumThreads(2);doc=b.read(WORK/'source_identity_charts.json');extras=[];proof=[]
    for ai in (82,83):
        groups=collections.defaultdict(list)
        for p in ats[ai]['paths']:groups[o.group_name(p['path_id'])].append(p)
        for group,paths in groups.items():
            old=[p for p in ats[81]['paths'] if o.group_name(p['path_id'])==group]
            if not old:continue
            aa=np.concatenate([np.asarray(p['pixels']) for p in paths]);cc=np.concatenate([np.asarray(p['pixels']) for p in old]);al,ah=aa.min(0),aa.max(0);cl,ch=cc.min(0),cc.max(0)
            scale=(ch-cl)/(ah-al);A=np.diag([*scale,1.]);A[:2,2]=cl-scale*al
            atlas=dict(ats[ai],id=len(ats)+len(extras),paths=paths,individually_registered_official_glyph=True,registration_uses_current_native_geometry=False,registration_navigation=dict(official_group=group,corresponding_actual_source_atlas=81,actual_source_group_bounds=[cl.tolist(),ch.tolist()],official_bounds=[al.tolist(),ah.tolist()]))
            extras.append(atlas);ref=r.reference(atlas);rows={}
            for pt,base in doc['records']['81'].items():
                H=np.asarray(base['H'])@A;score=r.full_pattern_score(ref,H,int(pt),allow_partial=True)
                if score is None:continue
                rows[pt]=dict(base,H=H.tolist(),atlas_id=atlas['id'],method='named_actual_source_glyph_to_independent_official_shape',full_pattern_score=score,independent_glyph_alignment_not_native_shape=True,parent_source_chart=dict(atlas_id=81,pts=int(pt)))
            doc['records'][str(atlas['id'])]=rows;proof.append(dict(atlas_id=atlas['id'],group=group,charted_frames=len(rows)))
    write(WORK/'additional_source_atlases.json',dict(atlases=extras));write(WORK/'source_identity_charts.json',doc);write(WORK/'ad1_glyph_registration.json',dict(native_parameters_changed=False,groups=proof));emit('ad1_individual_glyph_references',groups=proof)

def merge_tracks():
    original=WORK/'primary_identity_charts.json'
    if not original.exists():original.write_bytes((WORK/'source_identity_charts.json').read_bytes())
    doc=b.read(original);extra=b.read(WORK/'additional_identity_charts.json')
    for ai,rows in extra['records'].items():
        target=doc['records'].setdefault(ai,{})
        for pt,row in rows.items():
            if pt not in target or row.get('source_track_steps',999)<target[pt].get('source_track_steps',999):target[pt]=row
    doc['additional_charts_sha256']=b.sha(WORK/'additional_identity_charts.json')
    doc['tracking_failures']+=extra['tracking_failures']
    write(WORK/'source_identity_charts.json',doc)
    emit('merged_identity_charts',by_atlas={ai:len(rows) for ai,rows in doc['records'].items()})

def refine_ad1_pixels():
    import ad_entity_ecc_s01 as ecc
    r,t,m,o,s,ats,_=configure();cv2.setNumThreads(2);doc=b.read(WORK/'source_identity_charts.json');extra=b.read(WORK/'additional_source_atlases.json')['atlases'];seed=872448;seedH=np.asarray(doc['records']['81'][str(seed)]['H']);proof=[]
    for a in extra:
        nav=a['registration_navigation'];al,ah=np.asarray(nav['official_bounds']);cl,ch=np.asarray(nav['actual_source_group_bounds']);scale=(ch-cl)/(ah-al);A=np.diag([*scale,1.]);A[:2,2]=cl-scale*al;H=seedH@A;ref=r.reference(a)
        initial=o.group_score(ref,H,nav['official_group'],*r.current(seed)[1:]);best=ecc.refine(ref,H,seed,allow_partial=True)
        if best is None:continue
        candidate=np.asarray(best['H']);ss=o.group_score(ref,candidate,nav['official_group'],*r.current(seed)[1:]);span=np.ptp(cv2.perspectiveTransform(ref['points'][None],H)[0],axis=0)
        if ss.get('unblurred_shape_correlation',-1)<=initial.get('unblurred_shape_correlation',-1) or best['navigation_refinement_median_px']>min(10.,.35*max(10.,span[0])):continue
        relative=np.linalg.inv(seedH)@candidate;rows={}
        for pt,base in doc['records']['81'].items():
            HH=np.asarray(base['H'])@relative;score=r.full_pattern_score(ref,HH,int(pt),allow_partial=True)
            if score is None:continue
            rows[pt]=dict(base,H=HH.tolist(),atlas_id=a['id'],method='individual_official_glyph_registered_to_identified_actual_RGB',full_pattern_score=score,independent_current_glyph_seed_pts=seed,source_identity_seed=nav,registration_navigation_only=True)
        doc['records'][str(a['id'])]=rows;proof.append(dict(atlas_id=a['id'],group=nav['official_group'],before=initial,after=ss,ecc=best,frames=len(rows)))
    write(WORK/'source_identity_charts.json',doc);write(WORK/'ad1_pixel_refinement.json',dict(proof=proof,native_geometry_used=False));emit('ad1_pixel_refinement',groups=[dict(id=p['atlas_id'],name=p['group'],frames=p['frames'],ncc=p['after'].get('unblurred_shape_correlation')) for p in proof])

def measure():
    r,t,m,o,s,ats,chosen=configure();source=dict(np.load(OUT/'original_ad_candidates.npz'));m.run(source=source,banned=set(),first_pts=FIRST,source_prefix=PREFIX)

def extend_partial_panels():
    r,t,m,o,s,ats,_=configure();cv2.setNumThreads(2);doc=b.read(WORK/'source_identity_charts.json');changed=set();proof=[]
    for ai in [3,77,4,84,7]:
        rows=doc['records'].get(str(ai),{});ref=r.reference(ats[ai]);original=set(map(int,rows))
        for seed in sorted(original):
            for step in [-256,256]:
                previous=seed;pt=seed+step;H=np.asarray(rows[str(seed)]['H'])
                while FIRST<=pt<=LAST and str(pt) not in rows:
                    _,a,ma=r.current(previous);_,c,mc=r.current(pt)
                    q=cv2.perspectiveTransform(ref['points'][None],H)[0];lo=np.maximum(np.floor(q.min(0)-[220,35]).astype(int),[5,185]);hi=np.minimum(np.ceil(q.max(0)+[220,35]).astype(int),[1914,1074])
                    if min(hi-lo)<10:break
                    roi=np.zeros_like(a);roi[lo[1]:hi[1],lo[0]:hi[0]]=255;roi[cv2.dilate(ma,np.ones((9,9),np.uint8))>0]=0
                    x=cv2.goodFeaturesToTrack(a,650,.015,5,mask=roi,blockSize=5)
                    if x is None or len(x)<12:break
                    y,st,_=cv2.calcOpticalFlowPyrLK(a,c,x,None,winSize=(21,21),maxLevel=3);xx,sb,_=cv2.calcOpticalFlowPyrLK(c,a,y,None,winSize=(21,21),maxLevel=3)
                    if y is None or xx is None:break
                    old=x[:,0];new=y[:,0];ij=np.rint(new).astype(int);ok=(st[:,0]>0)&(sb[:,0]>0)&(np.linalg.norm(xx[:,0]-old,axis=1)<.8)&(ij[:,0]>5)&(ij[:,0]<1914)&(ij[:,1]>185)&(ij[:,1]<1074);ii=np.flatnonzero(ok);ok[ii]&=cv2.dilate(mc,np.ones((9,9),np.uint8))[ij[ii,1],ij[ii,0]]==0
                    if ok.sum()<12:break
                    A,inlier=cv2.estimateAffine2D(old[ok],new[ok],method=cv2.RANSAC,ransacReprojThreshold=1.25,maxIters=2500,confidence=.999)
                    if A is None or inlier.sum()<10 or inlier.mean()<.65:break
                    HH=np.vstack([A,[0,0,1]])@H
                    if r.geometry(HH,ref) is None:break
                    score=r.full_pattern_score(ref,HH,pt,allow_partial=True)
                    if score is None:break
                    rows[str(pt)]=dict(H=HH.tolist(),atlas_id=ai,pts=pt,method='adjacent_actual_board_RGB_navigation_and_current_panel_pattern',anchor_pts=seed,previous_pts=previous,source_track_steps=abs(pt-seed)//256,source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],expanded_reference_context_px=[220,35],feature_inliers=int(inlier.sum()),full_pattern_score=score,native_geometry_used=False,navigation_not_observed_UV=True)
                    changed.add(pt);proof.append(dict(atlas_id=ai,pts=pt,previous_pts=previous));previous=pt;pt+=step;H=HH
        doc['records'][str(ai)]=rows
    write(WORK/'source_identity_charts.json',doc);write(WORK/'partial_panel_extension.json',dict(new_charts=proof,native_geometry_used=False))
    patch=sorted(changed|set(map(int,doc['records']['81'])))
    write(WORK/'remeasure_frames.json',dict(pts=patch,reason='source glyph placement refined and partial panels recovered',native_geometry_used=False));emit('partial_panel_extension',new_charts=len(proof),new_chart_frames=len(changed),remeasure_frames=len(patch))

def remeasure():
    r,t,m,o,s,ats,_=configure();pts=set(b.read(WORK/'remeasure_frames.json')['pts']);r.VIS={pt:f for pt,f in r.VIS.items() if pt in pts};patch=WORK/'measurement_patch';patch.mkdir(exist_ok=True)
    for name in ['source_identity_charts.json','additional_source_atlases.json','native_entity_registry.json']:(patch/name).write_bytes((WORK/name).read_bytes())
    source=dict(np.load(OUT/'original_ad_candidates.npz'));source['source_rows']=np.arange(len(source['pts']));keep=np.isin(source['pts'],list(pts));source={k:v[keep] for k,v in source.items()};m.OUT=patch
    m.run(source=source,banned=set(),first_pts=FIRST,source_prefix=PREFIX)

def audit():
    r,t,m,o,s,ats,chosen=configure();prefix='sample_' if '--sample' in sys.argv else ''
    o.run(WORK/(prefix+'pointwise_checks.jsonl.gz'),WORK/(prefix+'verified_mapping.json'))

def recover_corners():
    r,t,m,o,s,ats,_=configure();source=dict(np.load(OUT/'original_ad_candidates.npz'));source['source_rows']=np.arange(len(source['pts']));keep=source['kind']==1;source={k:v[keep] for k,v in source.items()};pts=set(map(int,source['pts']));r.VIS={pt:f for pt,f in r.VIS.items() if pt in pts};patch=WORK/'corner_patch';patch.mkdir(exist_ok=True)
    for name in ['source_identity_charts.json','additional_source_atlases.json','native_entity_registry.json']:(patch/name).write_bytes((WORK/name).read_bytes())
    m.OUT=patch;m.run(source=source,banned=set(),first_pts=FIRST,source_prefix=PREFIX,query_new=False)

def merge_measurements():
    import heapq
    patch=WORK/'measurement_patch';corners=WORK/'corner_patch';history=WORK/'before_reference_repair'
    if history.exists():raise ValueError('Reference repair already merged')
    base=dict(np.load(WORK/'verified_ads.npz'));add=dict(np.load(patch/'verified_ads.npz'));cc=dict(np.load(corners/'verified_ads.npz'));source=dict(np.load(OUT/'original_ad_candidates.npz'));N=len(source['pts']);corner_rows=set(map(int,np.flatnonzero(source['kind']==1)));pts=set(b.read(WORK/'remeasure_frames.json')['pts']);nextrow=int(base['source_rows'].max())+1
    remap={int(old):nextrow+i for i,old in enumerate(add['source_rows'][add['source_rows']>=N])}
    add['source_rows']=np.array([remap.get(int(v),int(v)) for v in add['source_rows']],np.int64)
    bb=(~np.isin(base['pts'],list(pts)))&(~np.isin(base['source_rows'],list(corner_rows)));aa=~np.isin(add['source_rows'],list(corner_rows));z={k:np.concatenate([base[k][bb],add[k][aa],cc[k]]) for k in base};order=np.argsort(z['pts'],kind='stable');z={k:v[order] for k,v in z.items()}
    z['observation_ids']=np.array([str(sid) if str(sid).startswith(PREFIX) else str(sid)+'@pts='+str(int(pt)) for sid,pt in zip(z['observation_ids'],z['pts'])])
    if len(np.unique(z['source_rows']))!=len(z['pts']) or len(np.unique(z['observation_ids']))!=len(z['pts']):raise ValueError('Duplicate merged source identity')
    history.mkdir()
    for name in ['verified_ads.npz','measurement_summary.json','pointwise_checks.jsonl.gz','coverage_intervals.jsonl.gz']:
        target=(history/name).resolve()
        if not target.is_relative_to(OUT.resolve()):raise ValueError('Invalid history target')
        (WORK/name).rename(target)
    np.savez_compressed(WORK/'verified_ads.npz',**z);mapping=b.read(WORK/'verified_mapping.json');paths=mapping['path_ids'];decisions=collections.Counter()
    def lines(file,which):
        with gzip.open(file,'rt',encoding='utf8') as f:
            for line in f:
                c=json.loads(line)
                if which==0 and c['pts'] in pts:continue
                if which!=2 and c['source_row'] in corner_rows:continue
                if which==1:c['source_row']=remap.get(c['source_row'],c['source_row'])
                if not c['stable_id'].startswith(PREFIX):
                    c['original_point_id']=c['stable_id'];c['stable_id']+='@pts='+str(c['pts']);c['observation_key_includes_source_pts']=True
                yield c
    sources=[lines(history/'pointwise_checks.jsonl.gz',0),lines(patch/'pointwise_checks.jsonl.gz',1),lines(corners/'pointwise_checks.jsonl.gz',2)]
    with gzip.open(WORK/'pointwise_checks.jsonl.gz','wt',encoding='utf8') as f:
        for c in heapq.merge(*sources,key=lambda c:c['pts']):
            decisions[c.get('reason',c['decision'])]+=1;f.write(json.dumps(c,ensure_ascii=False)+'\n')
    counts=collections.Counter((int(pt),paths[int(pi)]) for pt,pi in zip(z['pts'],z['path_indices']))
    def coverage(file,which):
        with gzip.open(file,'rt',encoding='utf8') as f:
            for line in f:
                c=json.loads(line)
                if which==0 and c['pts'] in pts:continue
                c['included_points']=counts[(c['pts'],c['path_id'])];yield c
    with gzip.open(WORK/'coverage_intervals.jsonl.gz','wt',encoding='utf8') as f:
        for c in heapq.merge(coverage(history/'coverage_intervals.jsonl.gz',0),coverage(patch/'coverage_intervals.jsonl.gz',1),key=lambda c:c['pts']):f.write(json.dumps(c,ensure_ascii=False)+'\n')
    frames=[];fresh=np.char.startswith(z['observation_ids'],PREFIX)
    for pt in range(FIRST,LAST+1,256):
        ix=np.flatnonzero(z['pts']==pt);frames.append(dict(pts=pt,source_seconds=pt/12800,ad_points=len(ix),verified_original=int((~fresh[ix]).sum()),new_source_measurements=int(fresh[ix].sum()),by_ad=dict(collections.Counter(paths[k].split('::')[0].split('/')[-1] for k in z['path_indices'][ix]))))
    summary=dict(decisions=dict(decisions),ad_rows=len(z['pts']),source_frames=793,frames=frames,by_ad=dict(collections.Counter(paths[k].split('::')[0].split('/')[-1] for k in z['path_indices'])),by_path=dict(collections.Counter(paths[k] for k in z['path_indices'])),native_parameters_unchanged=True,old_rows_not_overwritten=True,human_visual_acceptance=False)
    write(WORK/'measurement_summary.json',summary);mapping.update(identity_charts_sha256=sha(WORK/'source_identity_charts.json'),point_input_sha256=sha(WORK/'verified_ads.npz'));write(WORK/'verified_mapping.json',mapping)
    write(WORK/'reference_repair_merge.json',dict(base_rows=len(base['pts']),rechecked_frames=len(pts),corner_candidates=len(corner_rows),checked_fixed_corners=len(cc['pts']),result_rows=len(z['pts']),new_row_ids_remapped_to_avoid_collisions=True,original_source_ids_UV_PTS_unchanged=True));emit('reference_repair_merged',rows=len(z['pts']),fixed_corners=int((z['kind']==0).sum()))

def promote_audit():
    history=WORK/'before_local_observability';history.mkdir(exist_ok=True);raw=dict(np.load(WORK/'verified_ads.npz'));z=dict(np.load(WORK/'observability_ads.npz'));mapping=b.read(WORK/'verified_mapping.json');paths=mapping['path_ids'];index={int(v):i for i,v in enumerate(raw['source_rows'])};ix=np.array([index[int(v)] for v in z['source_rows']])
    if any(not np.array_equal(raw[k][ix],z[k]) for k in raw):raise ValueError('Local audit changed measured source values')
    counts=collections.Counter();reasons=collections.defaultdict(collections.Counter)
    with gzip.open(WORK/'observability_checked_points.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            c=json.loads(line)
            if 'path_id' not in c:continue
            key=c['pts'],c['path_id']
            if c['decision']!='not_admitted':counts[key]+=1
            else:reasons[key][c['reason']]+=1
    if sum(counts.values())!=len(z['pts']):raise ValueError('Audit rows differ from point input')
    for name in ['verified_ads.npz','measurement_summary.json','pointwise_checks.jsonl.gz','coverage_intervals.jsonl.gz']:
        target=(history/name).resolve()
        if target.exists() or not target.is_relative_to(OUT.resolve()):raise ValueError('Unsafe repeated audit publication')
        (WORK/name).rename(target)
    (WORK/'observability_ads.npz').rename(WORK/'verified_ads.npz');(WORK/'observability_checked_points.jsonl.gz').rename(WORK/'pointwise_checks.jsonl.gz')
    summary=b.read(WORK/'observability_summary.json');summary.update(candidate_rows_before_local_visibility=len(raw['pts']),source_scope=dict(first_pts=FIRST,last_pts=LAST,time_base='1/12800'),old_source_UV_PTS_ID_preserved=True,stable_observation_key='original point id + source PTS, because dense point ids repeat across exposures');write(WORK/'measurement_summary.json',summary)
    with gzip.open(history/'coverage_intervals.jsonl.gz','rt',encoding='utf8') as f,gzip.open(WORK/'coverage_intervals.jsonl.gz','wt',encoding='utf8') as dest:
        for line in f:
            c=json.loads(line);key=c['pts'],c['path_id'];c['candidate_included_points']=c['included_points'];c['included_points']=counts[key];c['local_visibility_exclusions']=dict(reasons[key]);dest.write(json.dumps(c,ensure_ascii=False)+'\n')
    mapping.update(actual_local_observability_checked=True,point_input_sha256=sha(WORK/'verified_ads.npz'));write(WORK/'verified_mapping.json',mapping);emit('local_observability_published',rows=len(z['pts']),candidate_rows=len(raw['pts']))

def select():
    r,t,m,o,s,ats,_=configure();s.run();s.publish(source_prefix=PREFIX)

def authority_check():
    a=b.read(OUT/'authority_inventory.json')
    for row in [a['scene'],a['camera'],*a['assets']]:
        if sha(row['path'])!=row['sha256']:raise ValueError('Reviewed native authority changed: '+row['path'])
    return a

def fill_ad4_visible_borders():
    import heapq
    r,t,m,o,s,ats,_=configure();cv2.setNumThreads(2);z=dict(np.load(WORK/'verified_ads.npz'));mapping=b.read(WORK/'verified_mapping.json');paths=mapping['path_ids'];names=['Ad4.top_edge/path0','Ad4.bottom_edge/path0'];guides={name:[] for name in names}
    # This source image was visually inspected. Approximate guides only locate
    # the real colored perimeter; all admitted UV below are reread from RGB.
    manual={'Ad4.top_edge/path0':np.array([[6.,466.],[720.,462.]]),'Ad4.bottom_edge/path0':np.array([[6.,773.],[741.,764.]])}
    for name,q in manual.items():guides[name].append(dict(pts=913920,q=q,proof=dict(kind='assistant_visual_panel_identity_and_navigation_only',source_pts=913920,source_BGR_sha256=r.VIS[913920]['source_BGR_sha256'],not_observed_UV=True)))
    for ai in [129,130,131,132]:
        for name in names:
            pp=[p for p in ats[ai]['paths'] if p['path_id']==name]
            if pp:guides[name].append(dict(pts=pp[0]['proof']['source_pts'],q=np.concatenate([np.asarray(p['pixels']) for p in pp]),proof=pp[0]['proof']))
    serialized=[]
    for name,gg in guides.items():
        for g in gg:
            q=g['q'];g['line']=np.polynomial.Polynomial.fit(q[:,0],q[:,1],1).convert().coef;g['xrange']=np.array([q[:,0].min(),q[:,0].max()]);serialized.append(dict(path=name,pts=g['pts'],line=g['line'].tolist(),xrange=g['xrange'].tolist(),proof=g['proof']))
    added=[];nextrow=int(z['source_rows'].max())+1;reasons=collections.Counter()
    for pt in range(913920,LAST+1,256):
        im,gray,mask=r.current(pt)
        for name,gg in guides.items():
            ts=np.array([g['pts'] for g in gg]);j=int(np.clip(np.searchsorted(ts,pt)-1,0,len(ts)-2));a,c=gg[j:j+2];alpha=(pt-a['pts'])/(c['pts']-a['pts']);line=a['line']*(1-alpha)+c['line']*alpha;xr=a['xrange']*(1-alpha)+c['xrange']*alpha;xx=np.arange(max(6.,xr[0]),min(1914.,xr[1]),6.)
            if len(xx)<2:continue
            q=np.column_stack([xx,line[0]+line[1]*xx]);normal=np.array([-line[1],1.]);normal/=np.linalg.norm(normal)
            if 'top_' in name:normal*=-1
            offsets=np.arange(-24.,24.5,.5);v=r.sample(gray,q[:,None,:]+offsets[None,:,None]*normal);v=cv2.GaussianBlur(v.astype(np.float32),(5,1),.8);grad=np.diff(v,axis=1);peak=grad.argmax(1);search=q+((offsets[peak]+offsets[peak+1])*.5)[:,None]*normal;normals=np.tile(normal,(len(q),1));readings=m.read_profiles(im,r.VIS[pt],search,normals)
            for k,p in enumerate(readings):
                reason=m.profile_reason(p)
                if reason:reasons[reason]+=1;continue
                uv=np.asarray(p['actual_source_uv']);off=np.asarray(p['source_q10_q50_q90_normal_offsets_px']);probe=np.array([off[0]-3,off[0]-1,off[2]+1,off[2]+3]);rgb=r.sample(im,(search[k]+probe[:,None]*normal)[None])[0];sides=np.stack([rgb[:2].mean(0),rgb[2:].mean(0)]);chroma=np.ptp(sides,axis=1)
                if not (chroma.max()>20 and chroma.min()<max(15.,chroma.max()*.60)):reasons['no_observed_colored_panel_to_background_interface']+=1;continue
                pid=next(i for i,pid in enumerate(paths) if pid.endswith('::'+name));sid=f'{PREFIX}border_{pt}_{pid}_{k}'
                rec=dict(source_row=nextrow,stable_id=sid,pts=pt,uv=uv.tolist(),path_id=paths[pid],kind=1,station=.5,decision='new_actual_source_boundary',identity=dict(method='same_Ad4_named_outer_material_edge_from_actual_source_guides_and_current_RGB',native_entity_id=paths[pid],source_guide_pts=[a['pts'],c['pts']],source_guide_fraction=float(alpha),guides_are_navigation_only=True,unique_native_entity=True),source_profile=dict(search_uv=search[k].tolist(),normal=normal.tolist(),offsets=off.tolist(),search_radius_px=p['source_search_radius_px'],flags=p['source_flags']),current_observability=dict(actual_colored_panel_to_background_transition=True,structure='large_outer_material_border_partial_arc',whole_text_recognition_not_required=True),source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],native_geometry_used_for_identity=False)
                added.append(rec);nextrow+=1
    targetnames={next(p for p in paths if p.endswith('::'+name)) for name in names};existing=[]
    with gzip.open(WORK/'pointwise_checks.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            c=json.loads(line)
            if c['decision']!='not_admitted' and c['pts']>=913920 and c['path_id'] in targetnames:existing.append(c)
    grouped=collections.defaultdict(list)
    for c in existing+added:grouped[c['pts'],c['path_id']].append(c)
    kept=set()
    for key,records in grouped.items():
        ix,why=s.select_path(key[0],records,{},ats)
        for i in ix:kept.add(records[i]['source_row']);records[i]['fit_sampling']=dict(selected=True,reasons=sorted(why[i]),observed_UV_unchanged=True)
    newgood=[c for c in added if c['source_row'] in kept];oldremove={c['source_row'] for c in existing if c['source_row'] not in kept};keep=~np.isin(z['source_rows'],list(oldremove));a=dict(source_rows=np.array([c['source_row'] for c in newgood],np.int64),observation_ids=np.array([c['stable_id'] for c in newgood]),uv=np.array([c['uv'] for c in newgood]).reshape(-1,2),pts=np.array([c['pts'] for c in newgood],np.int64),kind=np.ones(len(newgood),np.uint8),stations=np.full(len(newgood),.5),path_indices=np.array([paths.index(c['path_id']) for c in newgood],np.int32));final={k:np.concatenate([z[k][keep],a[k]]) for k in z};order=np.argsort(final['pts'],kind='stable');final={k:v[order] for k,v in final.items()}
    history=WORK/'before_partial_border_completion';history.mkdir()
    for name in ['verified_ads.npz','measurement_summary.json','pointwise_checks.jsonl.gz','coverage_intervals.jsonl.gz','sampling_report.json']:
        target=(history/name).resolve()
        if not target.is_relative_to(OUT.resolve()):raise ValueError('Invalid history target')
        (WORK/name).rename(target)
    np.savez_compressed(WORK/'verified_ads.npz',**final)
    def oldrows():
        with gzip.open(history/'pointwise_checks.jsonl.gz','rt',encoding='utf8') as f:
            for line in f:
                c=json.loads(line)
                if c['source_row'] in oldremove:c.update(previous_decision=c['decision'],decision='not_admitted',reason='redundant_same_visible_contour_after_perimeter_completion',source_observation_valid=True)
                yield c
    with gzip.open(WORK/'perimeter_dense_source_readings.jsonl.gz','wt',encoding='utf8') as f:
        for c in added:f.write(json.dumps(c,ensure_ascii=False)+'\n')
    for c in added:
        if c['source_row'] not in kept:c.update(previous_decision=c['decision'],decision='not_admitted',reason='redundant_same_visible_contour',source_observation_valid=True)
    with gzip.open(WORK/'pointwise_checks.jsonl.gz','wt',encoding='utf8') as f:
        for c in heapq.merge(oldrows(),iter(added),key=lambda c:c['pts']):f.write(json.dumps(c,ensure_ascii=False)+'\n')
    count=collections.Counter((int(pt),paths[i]) for pt,i in zip(final['pts'],final['path_indices']));newpairs=set((c['pts'],c['path_id']) for c in added);oldpairs=set((int(pt),paths[i]) for pt,i in zip(z['pts'],z['path_indices']));summary=b.read(history/'measurement_summary.json');fresh=np.char.startswith(final['observation_ids'],PREFIX)
    summary.update(ad_rows=len(final['pts']),by_path=dict(collections.Counter(paths[i] for i in final['path_indices'])),by_ad=dict(collections.Counter(paths[i].split('::')[0].split('/')[-1] for i in final['path_indices'])),decisions=dict(include_original_pixel=int((~fresh).sum()),new_actual_source_boundary=int(fresh.sum())))
    for frame in summary['frames']:
        ix=np.flatnonzero(final['pts']==frame['pts']);frame.update(ad_points=len(ix),verified_original=int((~fresh[ix]).sum()),new_source_measurements=int(fresh[ix].sum()),by_ad=dict(collections.Counter(paths[i].split('::')[0].split('/')[-1] for i in final['path_indices'][ix])))
    write(WORK/'measurement_summary.json',summary)
    with gzip.open(history/'coverage_intervals.jsonl.gz','rt',encoding='utf8') as f,gzip.open(WORK/'coverage_intervals.jsonl.gz','wt',encoding='utf8') as out:
        for line in f:
            c=json.loads(line);key=c['pts'],c['path_id'];c['included_points']=count[key]
            if key in newpairs:c['direct_current_material_perimeter_readout']=True
            out.write(json.dumps(c,ensure_ascii=False)+'\n')
    report=b.read(history/'sampling_report.json');report.update(dense_verified_rows=report['dense_verified_rows']+len(added),selected_rows=len(final['pts']),source_frames=len(np.unique(final['pts'])),native_entity_frame_pairs_before=len(oldpairs|newpairs),native_entity_frame_pairs_after=len(count),perimeter_completion_added_representatives=len(newgood),perimeter_duplicates_removed=len(oldremove));write(WORK/'sampling_report.json',report);mapping['point_input_sha256']=sha(WORK/'verified_ads.npz');write(WORK/'verified_mapping.json',mapping)
    write(WORK/'perimeter_completion.json',dict(guides=serialized,source_readings=len(added),selected_new=len(newgood),removed_redundant_existing=len(oldremove),rejection_reasons=dict(reasons),new_native_entity_frame_pairs=len(newpairs-oldpairs),no_native_model_or_GPU_evaluation=True));emit('perimeter_completion',rows=len(final['pts']),frames=len(np.unique(final['pts'])),new_source_readings=len(added),new_representatives=len(newgood))

def merge_completion(added,targetnames,selected_pts,history_label,metadata):
    import heapq
    r,t,m,o,s,ats,_=configure();ats+=b.read(WORK/'additional_source_atlases.json')['atlases'];doc=b.read(WORK/'source_identity_charts.json');z=dict(np.load(WORK/'verified_ads.npz'));mapping=b.read(WORK/'verified_mapping.json');paths=mapping['path_ids']
    existing=[]
    with gzip.open(WORK/'pointwise_checks.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            c=json.loads(line)
            if c['decision']!='not_admitted' and c['pts'] in selected_pts and c['path_id'] in targetnames:existing.append(c)
    grouped=collections.defaultdict(list)
    for c in existing+added:grouped[c['pts'],c['path_id']].append(c)
    kept=set()
    for key,records in grouped.items():
        ix,why=s.select_path(key[0],records,doc,ats)
        for i in ix:kept.add(records[i]['source_row']);records[i]['fit_sampling']=dict(selected=True,reasons=sorted(why[i]),observed_UV_unchanged=True)
    newgood=[c for c in added if c['source_row'] in kept];oldremove={c['source_row'] for c in existing if c['source_row'] not in kept};keep=~np.isin(z['source_rows'],list(oldremove));a=dict(source_rows=np.array([c['source_row'] for c in newgood],np.int64),observation_ids=np.array([c['stable_id'] for c in newgood]),uv=np.array([c['uv'] for c in newgood]).reshape(-1,2),pts=np.array([c['pts'] for c in newgood],np.int64),kind=np.ones(len(newgood),np.uint8),stations=np.full(len(newgood),.5),path_indices=np.array([paths.index(c['path_id']) for c in newgood],np.int32));final={k:np.concatenate([z[k][keep],a[k]]) for k in z};order=np.argsort(final['pts'],kind='stable');final={k:v[order] for k,v in final.items()}
    history=WORK/history_label;history.mkdir()
    for name in ['verified_ads.npz','measurement_summary.json','pointwise_checks.jsonl.gz','coverage_intervals.jsonl.gz','sampling_report.json']:
        target=(history/name).resolve()
        if not target.is_relative_to(OUT.resolve()):raise ValueError('Invalid history target')
        (WORK/name).rename(target)
    np.savez_compressed(WORK/'verified_ads.npz',**final)
    def oldrows():
        with gzip.open(history/'pointwise_checks.jsonl.gz','rt',encoding='utf8') as f:
            for line in f:
                c=json.loads(line)
                if c['source_row'] in oldremove:c.update(previous_decision=c['decision'],decision='not_admitted',reason='redundant_same_visible_contour_after_perimeter_completion',source_observation_valid=True)
                yield c
    with gzip.open(WORK/(history_label+'_dense_source_readings.jsonl.gz'),'wt',encoding='utf8') as f:
        for c in added:f.write(json.dumps(c,ensure_ascii=False)+'\n')
    for c in added:
        if c['source_row'] not in kept:c.update(previous_decision=c['decision'],decision='not_admitted',reason='redundant_same_visible_contour',source_observation_valid=True)
    with gzip.open(WORK/'pointwise_checks.jsonl.gz','wt',encoding='utf8') as f:
        for c in heapq.merge(oldrows(),iter(added),key=lambda c:c['pts']):f.write(json.dumps(c,ensure_ascii=False)+'\n')
    count=collections.Counter((int(pt),paths[i]) for pt,i in zip(final['pts'],final['path_indices']));newpairs=set((c['pts'],c['path_id']) for c in added);oldpairs=set((int(pt),paths[i]) for pt,i in zip(z['pts'],z['path_indices']));summary=b.read(history/'measurement_summary.json');fresh=np.char.startswith(final['observation_ids'],PREFIX)
    summary.update(ad_rows=len(final['pts']),by_path=dict(collections.Counter(paths[i] for i in final['path_indices'])),by_ad=dict(collections.Counter(paths[i].split('::')[0].split('/')[-1] for i in final['path_indices'])),decisions=dict(include_original_pixel=int((~fresh).sum()),new_actual_source_boundary=int(fresh.sum())))
    for frame in summary['frames']:
        ix=np.flatnonzero(final['pts']==frame['pts']);frame.update(ad_points=len(ix),verified_original=int((~fresh[ix]).sum()),new_source_measurements=int(fresh[ix].sum()),by_ad=dict(collections.Counter(paths[i].split('::')[0].split('/')[-1] for i in final['path_indices'][ix])))
    write(WORK/'measurement_summary.json',summary)
    with gzip.open(history/'coverage_intervals.jsonl.gz','rt',encoding='utf8') as f,gzip.open(WORK/'coverage_intervals.jsonl.gz','wt',encoding='utf8') as out:
        for line in f:
            c=json.loads(line);key=c['pts'],c['path_id'];c['included_points']=count[key]
            if key in newpairs:c['direct_current_material_perimeter_readout']=True
            out.write(json.dumps(c,ensure_ascii=False)+'\n')
    report=b.read(history/'sampling_report.json');report.update(dense_verified_rows=report['dense_verified_rows']+len(added),selected_rows=len(final['pts']),source_frames=len(np.unique(final['pts'])),native_entity_frame_pairs_before=len(oldpairs|newpairs),native_entity_frame_pairs_after=len(count),latest_completion_added_representatives=len(newgood),latest_completion_duplicates_removed=len(oldremove));write(WORK/'sampling_report.json',report);mapping['point_input_sha256']=sha(WORK/'verified_ads.npz');write(WORK/'verified_mapping.json',mapping)
    write(WORK/(history_label+'_completion.json'),dict(metadata=metadata,source_readings=len(added),selected_new=len(newgood),removed_redundant_existing=len(oldremove),new_native_entity_frame_pairs=len(newpairs-oldpairs),no_native_model_or_GPU_evaluation=True));emit('source_completion',kind=history_label,rows=len(final['pts']),frames=len(np.unique(final['pts'])),new_source_readings=len(added),new_representatives=len(newgood))

def prepare_delivery():
    auth=authority_check();ads=OUT/'ads_only';ads.mkdir(exist_ok=True);z=dict(np.load(WORK/'verified_ads.npz'));mapping=b.read(WORK/'verified_mapping.json');camera=PROG/'ad_cameras/s04.json';cam=b.read(camera)
    if np.any((z['pts']<FIRST)|(z['pts']>LAST)):raise ValueError('Observation outside the requested shot')
    shutil.copy2(WORK/'verified_ads.npz',ads/'point_observations.npz')
    mapping.update(schema='rinkspace.native_s04_ads_only.v1',scene_path=str(b.SCENE),camera_path=str(camera),camera_sha256=sha(camera),control_sha256=sha(camera.parent/cam['controls_ref']),selected_rows=len(z['pts']),frames=793,source_scope=auth['source_scope'],source_video_sha256=auth['source_video']['sha256'],pixel_domain='native_distorted',source_dimensions=[1920,1080],time_base='1/12800',native_parameters_replaced_by_dataset=False,fit_started=False,observation_key='original point ID plus PTS; original_point_id retained in source evidence')
    write(ads/'mapping.json',mapping);shutil.copy2(WORK/'native_entity_registry.json',ads/'native_entities.json');before=sha(ads/'point_observations.npz')
    write(ads/'initialization_pending.json',dict(native_initialization_executed=False,native_GPU_validation_executed=False,reason='User instructed not to run validation while neighboring fit occupies GPU',curve_station_values_are_uninitialized_placeholders=True,fixed_material_corner_stations_are_semantically_bound=True,required_on_integration='Initialize only curve stations on their already identified native paths using the receiving current numerical state; preserve UV, PTS, source IDs and fixed stations'))
    write(ads/'point_input.json',dict(schema='rinkspace.native_ads_only_source_input.v1',shot_id='s04',sha256=before,rows=len(z['pts']),frames=793,fields=sorted(z),source_scope=auth['source_scope'],current_native_parameters_from_files_only=True,source_observations_do_not_supply_weights_or_geometry=True));authority_check();emit('ads_only_input_prepared',rows=len(z['pts']),frames=793)

def complete_ad1_actual_glyphs():
    r,t,m,o,s,ats,_=configure();cv2.setNumThreads(2);seed=872448;im,gray,mask=r.current(seed);doc=b.read(WORK/'source_identity_charts.json');extra=b.read(WORK/'additional_source_atlases.json');aid=len(ats)+len(extra['atlases']);binary=(gray[450:501,1245:1643]>155).astype(np.uint8)*255;n,labels,stats,_=cv2.connectedComponentsWithStats(binary)
    # The actual pixels and word order were inspected at full resolution.
    # Threshold contours navigate profiles; thresholded pixels are not data UV.
    named={1:'SKATING_G',2:'SKATING_S',3:'SKATING_KA',4:'SKATING_T',5:'SKATING_I',6:'SKATING_N',7:'ALL_A',8:'ALL_L1',9:'ALL_L2'};pp=[]
    for component,name in named.items():
        contours,_=cv2.findContours((labels==component).astype(np.uint8)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);q=max(contours,key=cv2.contourArea)[:,0].astype(float)+[1245,450];q=np.vstack([q,q[0]]);pid='Ad1.print.'+name
        if pid not in {p['path_id'] for p in ats[82]['paths']}:raise ValueError('Actual glyph has no current retained native identity')
        pp.append(dict(path_id=pid,pixels=q.tolist(),contour_sha256=hashlib.sha256(q.tobytes()).hexdigest(),proof=dict(kind='visually_identified_actual_source_glyph_by_word_order',source_pts=seed,source_BGR_sha256=r.VIS[seed]['source_BGR_sha256'],component=int(component),threshold_navigation_only=155,whole_word='ALL SKATING',current_native_path=pid)))
    atlas=dict(id=aid,ad=1,image=r.VIS[seed]['image'],image_sha256=sha(r.VIS[seed]['image']),paths=pp,partial_source_reference=False,actual_current_source_letter_reference=True);extra['atlases'].append(atlas);ref=r.reference(atlas);seedH=np.asarray(doc['records']['81'][str(seed)]['H']);rows={}
    for pt,row in doc['records']['81'].items():
        H=np.asarray(row['H'])@np.linalg.inv(seedH);score=r.full_pattern_score(ref,H,int(pt),allow_partial=True)
        if score is not None:rows[pt]=dict(row,H=H.tolist(),atlas_id=aid,method='actual_named_current_glyphs_with_adjacent_source_identity',reference_pts=seed,full_pattern_score=score,native_geometry_used=False)
    doc['records'][str(aid)]=rows;write(WORK/'source_identity_charts.json',doc);write(WORK/'additional_source_atlases.json',extra)
    z=dict(np.load(WORK/'verified_ads.npz'));paths=b.read(WORK/'verified_mapping.json')['path_ids'];targets={'Ad1.print.SKATING_G','Ad1.print.SKATING_N','Ad1.print.SKATING_I'};added=[];nextrow=int(z['source_rows'].max())+1;reasons=collections.Counter();small_doc=dict(records={str(aid):rows});o.CANONICAL_BY_PATH={name:[aid] for name in targets}
    for pt,row in sorted(rows.items(),key=lambda x:int(x[0])):
        pt=int(pt);im,gray,mask=r.current(pt);mm=cv2.dilate(mask,np.ones((3,3),np.uint8));candidates=[]
        for v in m.chart_paths(ref,row,im,gray,mm):
            name=v['path']
            if name not in targets:continue
            pid=next(i for i,p in enumerate(paths) if p.endswith('::'+name));old=z['uv'][(z['pts']==pt)&(z['path_indices']==pid)];ii=np.arange(1,len(v['q'])-1,2);ii=ii[v['valid'][ii]&(v['ncc'][ii]>.05)]
            if len(old):ii=np.array([i for i in ii if np.min(np.linalg.norm(old-v['q'][i],axis=1))>=2.8],int)
            if not len(ii):continue
            profiles=m.read_profiles(im,r.VIS[pt],v['q'][ii],v['n'][ii])
            for qi,p in zip(ii,profiles):
                reason=m.profile_reason(p)
                if reason:reasons[reason]+=1;continue
                uv=np.asarray(p['actual_source_uv']);sid=f'{PREFIX}ad1_detail_{pt}_{pid}_{qi}'
                candidates.append(dict(source_row=nextrow,stable_id=sid,pts=pt,uv=uv.tolist(),path_id=paths[pid],kind=1,station=.5,decision='new_actual_source_boundary',identity=dict(atlas_id=aid,source_path_index=v['source_path_index'],source_contour_sha256=v['source_contour_sha256'],source_chart_pts=pt,reference_arc_fraction=float(v['s'][qi]),current_neighborhood_correlation=float(v['ncc'][qi]),unique_native_entity=True),source_profile=dict(search_uv=v['q'][qi].tolist(),normal=v['n'][qi].tolist(),offsets=np.asarray(p['source_q10_q50_q90_normal_offsets_px']).tolist(),search_radius_px=p['source_search_radius_px'],flags=p['source_flags']),source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],native_geometry_used_for_identity=False));nextrow+=1
        checked,_=o.frame_check(pt,candidates,small_doc,{aid:ref}) if candidates else ([],{})
        added.extend(c for c in checked if c['decision']!='not_admitted');reasons.update(c['reason'] for c in checked if c['decision']=='not_admitted')
    merge_completion(added,{p for p in paths if p.split('::')[1] in targets},set(map(int,rows)),'before_ad1_detail_completion',dict(reference_atlas=aid,source_pts=seed,visually_inspected_actual_source_word_order=True,source_contours_are_navigation_only=True,rejection_reasons=dict(reasons),native_GPU_used=False))

def render():
    import av
    from fractions import Fraction
    sys.path.insert(0,str(ROOT/'src'))
    from rinkspace.review.render import source_frames,validate_video
    r,t,m,o,s,ats,_=configure();ads=OUT/'ads_only';review=OUT/'review';review.mkdir(exist_ok=True);z=dict(np.load(ads/'point_observations.npz'));mapping=b.read(ads/'mapping.json');video=next((DATA/'events/ankara_2026/programmes/okada_mei_sp').glob('Mei OKADA*.mp4'));expected=np.arange(FIRST,LAST+1,256);tb=Fraction(1,12800)
    if sha(video)!=mapping['source_video_sha256']:raise ValueError('Source video changed')
    target=review/'S04_ads_checked.mp4';container=av.open(str(target),'w');stream=container.add_stream('libx264',rate=50);stream.width=1920;stream.height=1160;stream.pix_fmt='yuv420p';stream.time_base=tb;stream.options={'crf':'17','preset':'fast'};start=time.monotonic();allcounts=[];sample=set(map(int,expected[::25]))|{LAST};sheets=[];sheetpts=[]
    with (review/'frame_index.jsonl').open('w',encoding='utf8') as ledger:
        for fi,(pt,im) in enumerate(source_frames(video,dict(time_base_num=1,time_base_den=12800,width=1920,height=1080),expected)):
            digest=hashlib.sha256(im.tobytes()).hexdigest()
            if digest!=r.VIS[pt]['source_BGR_sha256']:raise ValueError('Decoded review frame does not match measurement pixels')
            ix=np.flatnonzero(z['pts']==pt);canvas=np.zeros((1160,1920,3),np.uint8);canvas[:1080]=im
            for k in ix:
                xy=tuple(np.rint(z['uv'][k]).astype(int));cv2.circle(canvas,xy,1,(0,255,0),-1,cv2.LINE_8)
                if z['kind'][k]==0:cv2.rectangle(canvas,(xy[0]-3,xy[1]-3),(xy[0]+3,xy[1]+3),(0,255,0),1)
            ads_count=collections.Counter(mapping['path_ids'][j].split('::')[0].split('/')[-1] for j in z['path_indices'][ix]);label=f'S04  |  SOURCE {pt/12800:.3f} s  |  PTS {pt}  |  {len(ix)} advertisement observations'
            cv2.putText(canvas,label,(18,1112),cv2.FONT_HERSHEY_SIMPLEX,.75,(240,240,240),1,cv2.LINE_AA);cv2.putText(canvas,'   '.join(f'Ad{n}: {ads_count.get("Ad"+str(n),0)}' for n in ADNUMBERS),(18,1144),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),1,cv2.LINE_AA)
            frame=av.VideoFrame.from_ndarray(canvas,format='bgr24');frame.pts=pt-FIRST;frame.time_base=tb
            for packet in stream.encode(frame):container.mux(packet)
            record=dict(frame_index=fi,source_pts=pt,source_seconds=pt/12800,source_BGR_sha256=digest,ad_points=len(ix),by_ad=dict(ads_count),observation_rows_sha256=hashlib.sha256(z['source_rows'][ix].tobytes()).hexdigest());ledger.write(json.dumps(record)+'\n');allcounts.append(record)
            if pt in sample:
                cv2.imwrite(str(review/f'checked_{pt}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,96]);crop=canvas[330:840].copy();crop=cv2.resize(crop,(960,255));cv2.putText(crop,f'{pt/12800:.3f}s / {len(ix)}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),1);sheets.append(crop);sheetpts.append(pt)
                if len(sheets)==4 or pt==LAST:
                    name=review/f'overview_checked_{sheetpts[0]}.jpg';cv2.imwrite(str(name),np.vstack(sheets),[cv2.IMWRITE_JPEG_QUALITY,95]);sheets=[];sheetpts=[]
            if fi%200==0:emit('review_video',frames=fi+1,total=793,seconds=round(time.monotonic()-start,1))
    for packet in stream.encode():container.mux(packet)
    container.close();check=validate_video(target,expected,tb);write(review/'video.json',dict(path=str(target),sha256=sha(target),source_video=str(video),source_video_sha256=mapping['source_video_sha256'],source_scope=mapping['source_scope'],point_input_sha256=sha(ads/'point_observations.npz'),width=1920,height=1160,original_pixels_width=1920,original_pixels_height=1080,frames=793,frame_index_sha256=sha(review/'frame_index.jsonl'),native_projection_displayed=False,only_current_exposure_advertisement_points=True,**check));emit('review_video_complete',path=str(target),frames=793)

def collect_handoff():
    r,t,m,o,s,ats,_=configure();authority_check();ev=OUT/'evidence';ev.mkdir(exist_ok=True);ads=OUT/'ads_only';z=dict(np.load(ads/'point_observations.npz'));source=dict(np.load(OUT/'original_ad_candidates.npz'));mapping=b.read(ads/'mapping.json');paths=mapping['path_ids'];registry=b.read(WORK/'native_entity_registry.json');allowed={p['native_id'] for p in registry['paths']};rows={int(v):i for i,v in enumerate(z['source_rows'])};seen=set();original=0;fresh=0;by=collections.Counter();frames=collections.Counter()
    if len(rows)!=len(z['pts']) or len(np.unique(z['observation_ids']))!=len(z['pts']):raise ValueError('Duplicate observation key')
    if set(z)!=set(['source_rows','observation_ids','uv','pts','kind','stations','path_indices']):raise ValueError('Unexpected numerical input field')
    with gzip.open(WORK/'pointwise_checks.jsonl.gz','rt',encoding='utf8') as f,gzip.open(ev/'selected_point_records.jsonl.gz','wt',encoding='utf8') as out:
        for line in f:
            c=json.loads(line)
            if c['decision']=='not_admitted':continue
            i=rows.get(c['source_row'])
            if i is None or c['source_row'] in seen:raise ValueError('Source audit contains an unexpected or duplicate selected row')
            seen.add(c['source_row']);pid=paths[z['path_indices'][i]]
            if pid!=c['path_id'] or pid not in allowed or int(z['pts'][i])!=c['pts'] or str(z['observation_ids'][i])!=c['stable_id'] or not np.array_equal(z['uv'][i],c['uv']):raise ValueError('Point source record differs from delivered input')
            if any(c['source_profile']['flags'].values()):raise ValueError('Unresolved material profile included')
            pt=c['pts']
            if c['source_BGR_sha256']!=r.VIS[pt]['source_BGR_sha256']:raise ValueError('Source image identity mismatch')
            x,y=np.rint(c['uv']).astype(int)
            if r.current(pt)[2][y,x]!=0:raise ValueError('Foreground point included')
            sr=c['source_row']
            if not c['stable_id'].startswith(PREFIX):
                if sr>=len(source['pts']) or not np.array_equal(source['uv'][sr],z['uv'][i]) or int(source['pts'][sr])!=pt or str(source['ids'][sr])!=c['original_point_id']:raise ValueError('Original source point changed')
                original+=1
            else:fresh+=1
            by[pid]+=1;frames[pt]+=1;out.write(json.dumps(c,ensure_ascii=False)+'\n')
    if len(seen)!=len(z['pts']):raise ValueError('Delivered observation has no source record')
    reasons=collections.defaultdict(collections.Counter);nav=collections.Counter()
    with gzip.open(WORK/'coverage_intervals.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            c=json.loads(line);nav[c['path_id']]+=bool(c['source_identity_chart_available']);reasons[c['path_id']].update(c.get('local_visibility_exclusions',{}));reasons[c['path_id']].update(c.get('counts',{}))
    missing=[dict(native_id=p['native_id'],accepted_points=0,source_navigation_frames=nav[p['native_id']],recorded_screening_reasons=dict(reasons[p['native_id']])) for p in registry['paths'] if by[p['native_id']]==0]
    write(ev/'coverage_summary.json',dict(source_frames=793,frames_with_ad_points=len(frames),zero_point_pts=[pt for pt in range(FIRST,LAST+1,256) if not frames[pt]],retained_native_paths=len(allowed),observed_native_paths=len(by),unobserved_retained_native_paths=missing,modeling_exclusions=[dict(ad='Ad3',feature='大吉',status='intentionally_excluded_by_modeling_decision_confirmed_by_user',model_gap=False,points_not_added=True)],complete_coverage_of_every_visible_artwork_detail_claimed=False))
    write(ev/'source_consistency.json',dict(rows=len(seen),source_frames=len(frames),original_source_observations=original,new_actual_RGB_observations=fresh,all_selected_points_have_current_native_path_and_dependency_registry=True,all_source_UV_PTS_original_point_ids_preserved=True,observation_keys_are_original_point_id_plus_PTS=True,foreground_pixel_points=0,unresolved_source_profile_flags=0,all_selected_point_records_equal_data=True,model_authority_files_unchanged=True,native_GPU_evaluation_performed=False,native_joint_linearization_performed=False,native_curve_station_initialization_performed=False,reason='GPU/native validation and initialization skipped at user request while another fit occupies GPU',human_per_point_visual_acceptance=False,fit_accuracy_claimed=False))
    for name in ['authority_inventory.json','source_inventory.json','visibility.json']:shutil.copy2(OUT/name,ev/name)
    for name in ['native_entity_registry.json','coverage_intervals.jsonl.gz','sampling_report.json','measurement_summary.json','perimeter_completion.json','before_ad1_detail_completion_retry_completion.json','reference_repair_merge.json']:shutil.copy2(WORK/name,ev/name)
    for src,name in [(OUT/'atlases.json','source_atlases.json.gz'),(WORK/'additional_source_atlases.json','additional_source_atlases.json.gz'),(WORK/'source_identity_charts.json','source_identity_charts.json.gz')]:
        with src.open('rb') as f,gzip.open(ev/name,'wb') as out:shutil.copyfileobj(f,out)
    external=[OUT/'original_ad_candidates.npz',OUT/'candidate_provenance.jsonl.gz',WORK/'dense_verified_pool/verified_ads.npz',WORK/'pointwise_checks.jsonl.gz',WORK/'perimeter_dense_source_readings.jsonl.gz',WORK/'before_ad1_detail_completion_retry_dense_source_readings.jsonl.gz']
    write(ev/'external_source_evidence.json',dict(files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in external],source_photographs_and_masks='Existing source references are recorded with hashes in the atlas and visibility indexes; original video remains in the programme data directory',dense_evidence_is_not_fit_input=True))
    impl=OUT/'implementation';impl.mkdir(exist_ok=True)
    for name in ['ad_points_s04.py','ad_entity_measure_s01.py','ad_entity_observability_s01.py','ad_entity_select_s01.py','ad_entity_registration_s01.py','ad_entity_tracks_s01.py','ad_entity_ecc_s01.py']:shutil.copy2(PROG/name,impl/name)
    emit('handoff_evidence_collected',rows=len(seen),original=original,new_actual_source=fresh,missing_native_paths=len(missing),native_GPU_used=False)

def seal():
    import zipfile,ast
    authority_check();ads=OUT/'ads_only';evidence=OUT/'evidence';z=dict(np.load(ads/'point_observations.npz'));mapping=b.read(ads/'mapping.json');summary=b.read(WORK/'measurement_summary.json');sampling=b.read(WORK/'sampling_report.json');audit=b.read(evidence/'source_consistency.json');coverage=b.read(evidence/'coverage_summary.json');video=b.read(OUT/'review/video.json');pointsha=sha(ads/'point_observations.npz')
    if video['point_input_sha256']!=pointsha or audit['rows']!=len(z['pts']):raise ValueError('Video/evidence/data versions differ')
    for p in (OUT/'implementation').glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    fixed=int((z['kind']==0).sum());reduction=100*(1-len(z['pts'])/sampling['dense_verified_rows'])
    text=f'''# 第四镜头广告观测交接包

本包只含冈田芽衣短节目第四镜头的广告观测：原片 **58.280–74.120 秒**，PTS **{FIRST}–{LAST}**，时基 1/12800，共 **793 个原始曝光、{len(z['pts']):,} 点**。覆盖 Ad1–Ad5，采用当前 native 保留的路径和参数依赖；不包含人体、冰面或其他镜头数据，不改动场景、相机或轨迹参数。

## 输入及接入状态

数值输入为 `ads_only/point_observations.npz`，SHA256：`{pointsha}`。只含当前接口的七个字段：`source_rows`、`observation_ids`、`uv`、`pts`、`kind`、`stations`、`path_indices`。UV 为原片 1920×1080 的 native_distorted 像素；路径索引只引用随包 `mapping.json`。具体 native 实体及其参数依赖见 `native_entities.json`。

**按用户最新指示，未执行 GPU/native 求值、联合线性化、优化器校验或 native station 初始化。** 未启动拟合，未与隔壁拟合争用 GPU。

`kind=0` 的 {fixed} 个固定材料角保留语义确定的 station。`kind=1` 的 curve station=0.5 是未初始化占位值，不能当作已确认的曲线上位置；原像素和实体归属已在源图侧检查。接入方在运行条件允许时，应使用其当前 native 数值状态，仅在已经绑定的同一条曲线上初始化 station，并保留固定材料角。包内不提供冒充完成初始化的 `initialized_points.npz`。这是广告数据包，不是已启动或已通过运行验证的求解包。

历史密集取点的 point ID 会跨帧重复。输入 observation key 使用“原 point ID＋PTS”，原 ID 单独保存在每点证据的 `original_point_id` 字段；原始 UV、PTS 和 point ID 均保留。`source_rows` 是本包追溯索引，合并其他输入时不能把它误作跨数据包的全局身份。

## 来源、补漏与取舍

源候选来自项目的 `dense_print_v1/final_manifest_anchor.json` 及逐帧成员，并补入 `shot4_20261014` 的实际原始读数和已有材料角；完整文件、哈希及数量见 `evidence/source_inventory.json`。未把用于展示的较小点集当作完整源集合。

每个交付点均有当前帧 RGB、源图哈希、具体 native 路径、材料交界及遮挡检查记录；见 `evidence/selected_point_records.jsonl.gz`。没有使用未拟合 native 投影的贴合程度证明实体身份。新增点来自当前曝光的实际材料交界，参考图与相邻帧只负责定位搜索区域。

对 Ad1 的局部字距和缺失轮廓重新检查，补充了当前原片中明确识别的 I、N、G 轮廓。对 71.400 秒之后的 Ad4，在文字模糊时仍单独读取可辨认的广告上、下边界；只保留少量分散支撑点。小字和细笔画未能分辨时不强取。

从 {sampling['dense_verified_rows']:,} 条合格密集读数中选择 {len(z['pts']):,} 个拟合代表点，减少 {reduction:.1f}% 的冗余。保留可见轮廓的转折、端点、分散的图像区域支撑，以及固定材料角；{sampling['native_entity_frame_pairs_after']:,} 个有合格观测的“native 实体／帧”组合仍有代表点。没有向求解输入添加权重、sigma、容差或鲁棒损失。密集证据池仅用于追溯。

**Ad3「大吉」两字是用户确认的建模取舍，明确不纳入本包，不列为模型缺口。** 当前 native 不保留的其他印刷细节也没有被强绑到其他曲线。

当前 {coverage['retained_native_paths']} 条相关 native 路径中，{coverage['observed_native_paths']} 条有入选点；其余 {len(coverage['unobserved_retained_native_paths'])} 条及筛查记录逐项列在 `evidence/coverage_summary.json`。例如 Ad1 右边界、Ad5 的画外标志与未分辨细部均不能凭模板补出观测。本包不宣称覆盖画面中所有印刷细节。

## 审阅视频

`review/S04_ads_checked.mp4` 仅叠加当前曝光的广告观测。绿点为曲线观测，方框为固定材料角；原始 1920×1080 画面保持完整，底部另加源时间、PTS 和当帧点数。793 帧全部按真实源 PTS 对应，未截掉空白或模糊时段。用 PotPlayer 播放；视频文件时间从 0 开始，真正源时间见底栏及 `frame_index.jsonl`。

全部选中记录做了 CPU 来源一致性和图像材料检查，并审阅了代表帧。此状态不等于逐点人工验收、用户已验收或拟合准确性已通过。GPU 检查按用户要求留待接入方安排。

## 文件与依赖

`evidence/` 保存全部入选点的证据、逐帧覆盖、参数依赖、来源索引、取样说明及明确的未执行检查状态。`implementation/` 是本次实现快照。未入选点和密集历史仍保存在本机原工作目录，地址和哈希见 `external_source_evidence.json`。原片、参考图片与 native 模型继续使用项目已有文件，包内没有替换它们。

当前数据目录：`{ads}`

审阅视频：`{OUT/'review/S04_ads_checked.mp4'}`
'''
    (OUT/'HANDOFF.md').write_text(text,encoding='utf8')
    files=[OUT/'HANDOFF.md']+sorted(ads.glob('*'))+sorted(evidence.glob('*'))+sorted((OUT/'implementation').glob('*'))+[OUT/'review'/n for n in ['S04_ads_checked.mp4','video.json','frame_index.jsonl']]+sorted((OUT/'review').glob('overview_checked_*.jpg'))
    if any(not p.is_file() for p in files):raise ValueError('Missing handoff file')
    package=OUT/'S04_ads_only_20261026.zip'
    if package.exists():raise ValueError('Do not overwrite a sealed handoff')
    manifest=dict(schema='rinkspace.s04_advertisement_source_handoff.v1',source_scope=mapping['source_scope'],rows=len(z['pts']),source_frames=793,native_entity_paths=coverage['observed_native_paths'],fixed_material_corners=fixed,point_input_sha256=pointsha,scene_revision=mapping['scene_revision'],native_GPU_validation_performed=False,native_station_initialization_performed=False,user_requested_GPU_work_skipped=True,files=[dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files]);write(OUT/'delivery_manifest.json',manifest);files.append(OUT/'delivery_manifest.json')
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for p in files:archive.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(package) as archive:
        if archive.testzip() is not None:raise ValueError('Archive integrity failure')
        for row in manifest['files']:
            h=hashlib.sha256()
            with archive.open(row['path']) as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=row['sha256']:raise ValueError('Archived file differs from delivered source')
    digest=dict(package=str(package),sha256=sha(package),bytes=package.stat().st_size,files=len(files),point_input_sha256=pointsha,video=str(OUT/'review/S04_ads_checked.mp4'),video_sha256=video['sha256']);write(OUT/'package_digest.json',digest)
    index=PROG/'source_datasets/ad_points_s04_20261026/result.json'
    if len(str(index).encode('utf-16-le'))//2>=260 or any(len(part.encode('utf-16-le'))//2>255 for part in index.parts):raise ValueError('OneDrive path exceeds Windows limits')
    index.parent.mkdir(exist_ok=True);write(index,dict(schema='rinkspace.programme_ad_source_delivery.v1',shot_id='s04',source_scope=mapping['source_scope'],rows=len(z['pts']),source_frames=793,ads_only=True,result_directory=str(OUT),package=digest,point_input=str(ads/'point_observations.npz'),mapping=str(ads/'mapping.json'),handoff=str(OUT/'HANDOFF.md'),native_GPU_validation_performed=False,native_initialization_performed=False))
    readme=PROG/'README.md';marker='ad_points_s04_20261026/result.json'
    if marker not in readme.read_text(encoding='utf8'):
        with readme.open('a',encoding='utf8') as f:f.write(f'\n\n2026-10-26 第四镜头单纯广告数据包已整理：原片 58.280–74.120 秒，793 帧、{len(z["pts"]):,} 点，绑定当前 native 广告路径和参数依赖。来源及交付入口见 [S04 广告读点包](source_datasets/{marker})。保留清楚的局部边界，去除同一轮廓的冗余密点；Ad3「大吉」按建模取舍排除。按用户指示未执行 GPU/native 校验及 station 初始化，曲线 station 为待接入初始化值；未改动当前拟合输入、相机、场景或轨迹数值。\n')
    emit('handoff_sealed',**digest)

def sample_review():
    r,t,m,o,s,ats,chosen=configure();doc=b.read(WORK/'source_identity_charts.json');rows=collections.defaultdict(list);why=collections.defaultdict(collections.Counter)
    ats+=b.read(WORK/'additional_source_atlases.json')['atlases']
    with gzip.open(WORK/'sample_observability_checks.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            c=json.loads(line)
            if c['decision']!='not_admitted':rows[c['pts']].append(c)
            else:why[c['pts']][c['reason']]+=1
    review=OUT/'review';review.mkdir(exist_ok=True);counts=[]
    for pt,records in sorted(rows.items()):
        by=collections.defaultdict(list);selected=[]
        for c in records:by[c['path_id']].append(c)
        for name,v in by.items():
            keep,_=s.select_path(pt,v,doc,ats);selected.extend(v[i] for i in sorted(keep))
        im=r.current(pt)[0].copy()
        for c in selected:
            xy=tuple(np.rint(c['uv']).astype(int));cv2.circle(im,xy,1,(0,255,0),-1,cv2.LINE_8)
            if c['kind']==0:cv2.rectangle(im,(xy[0]-3,xy[1]-3),(xy[0]+3,xy[1]+3),(0,255,0),1)
        cv2.putText(im,f'S04 source {pt/12800:.3f}s | PTS {pt} | ads {len(selected)}',(18,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),1,cv2.LINE_AA)
        path=review/f'sample_{pt}.jpg';cv2.imwrite(str(path),im,[cv2.IMWRITE_JPEG_QUALITY,96])
        counts.append(dict(pts=pt,source_seconds=pt/12800,qualified=len(records),representatives=len(selected),by_ad=dict(collections.Counter(c['path_id'].split('::')[1].split('.')[0] for c in selected)),by_path=dict(collections.Counter(c['path_id'].split('::')[1] for c in selected)),reasons=dict(why[pt]),review_image=str(path)))
    write(review/'sample_review.json',counts);emit('sample_review',frames=[{k:v for k,v in x.items() if k not in ['by_path','reasons']} for x in counts])

if __name__=='__main__':globals()[sys.argv[1]]()
