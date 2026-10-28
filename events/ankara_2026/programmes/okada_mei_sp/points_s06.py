"""Complete S06 current-native source dataset using existing material readers.

Historical source records are immutable. Only current native path definitions
can receive observations. Motion and model curves remain search evidence.
"""
from pathlib import Path
import argparse,base64,collections,copy,gzip,hashlib,itertools,json,re,shutil,sqlite3,time
import cv2,numpy as np
import points_s02 as shared
from rinkspace.vision_native import NativeSourceGeometry,file_ref
from ice_points_s04 import read,write,rows,select_representatives,material_path

ROOT=shared.ROOT;PROG=shared.PROG;DATA=shared.DATA;REC=shared.REC;SCENE=shared.SCENE
OUT=DATA/'runs/native_points_s06_20261027'
FIRST,LAST,STEP=1301248,1425408,256
SHOT_ID="s06"
FRAME_COUNT=486
CAMERA=PROG/'ad1_cameras/s06.json'
ICE=REC/'dense56_ice_v1/canonical/s06/rows.jsonl.gz'
BOARD=REC/'dense56_board_v1/material_v5/qualified_frames'
MASK_INDEX=REC/'dense56_v1/shoe_fix/effective_v1/s06_index.jsonl.gz'
SURFACE_INDEX=REC/'dense56_v1/near_board_v1/final_v3/index.jsonl.gz'
PRINT_INDEX=REC/'dense56_v1/print_requal_v2/s06/index.jsonl.gz'
PACKETS=PROG/'source_datasets/shot6_20261014/frames'
OLDJOINT=DATA/'runs/okada_joint6_1021b'


def configure():
    shared.OUT=OUT;shared.FIRST=FIRST;shared.LAST=LAST;shared.CAMERA=CAMERA;shared.PACKETS=PACKETS
    shared.verify_authority=verify_authority


def emit(stage,**kw):print(json.dumps(dict(stage=stage,**kw),ensure_ascii=False),flush=True)


def verify_authority():
    a=read(OUT/'evidence/native_authority.json')
    for ref in a['files']+[a['scene_state']]:
        if file_ref(ref['path'])!=ref:raise ValueError('Frozen current native authority changed: '+ref['path'])


def prepare():
    import av
    cv2.setNumThreads(2)
    for name in ['evidence','source_frames','ad_exclusions']:(OUT/name).mkdir(parents=True,exist_ok=True)
    scope=read(PROG/'rebuild_scope.json');shot=next(s for s in scope['shots']if s['id']=='s06');clock=shot['processing']
    if (clock['first_pts'],clock['last_pts'],clock['native_frame_count'])!=(FIRST,LAST,486):raise ValueError('S06 official scope changed')
    video=Path(scope['source']['registered_path'])
    if file_ref(video)['sha256']!=scope['source']['sha256']:raise ValueError('Official original video differs')
    manifest=DATA/'runs/native_joint_s01_s04_20261027/joint_manifest.json';joint=read(manifest)
    geo=NativeSourceGeometry(SCENE,joint['native_library']);authority=OUT/'evidence/native_authority.json'
    if not authority.exists():
        origin=Path(joint['joint_output'])/'accepted_pixels_latest.npz';before=file_ref(origin);state=dict(np.load(origin,allow_pickle=False))
        if file_ref(origin)!=before:raise ValueError('Accepted native state changed during read; retry snapshot')
        if str(state['point_input_sha256'])!=joint['point_input_sha256']:raise ValueError('Accepted current state is not bound to the current manifest')
        target=OUT/'evidence/current_native_geometry_state.npz'
        np.savez_compressed(target,scene_values=state['scene_values'],point_input_sha256=state['point_input_sha256'])
        refs={r['path']:r for r in geo.refs}
        for p in [CAMERA,CAMERA.parent/read(CAMERA)['controls_ref'],Path(joint['native_library'])]:ref=file_ref(p);refs[ref['path']]=ref
        write(authority,dict(files=list(refs.values()),scene_path=str(SCENE),native_library=joint['native_library'],scene_state=file_ref(target),state_origin=before,
            manifest_origin=file_ref(manifest),camera=file_ref(CAMERA),source=scope['source'],shot=shot,current_native_path_count=len(geo.current_paths),geometry_changed=False,camera_seed_only=True))
    else:verify_authority()
    write(OUT/'evidence/native_entities.json',dict(entities=[dict(v,native_path_id=v['id'])for v in geo.current_paths.values()]))
    masks={r['native_pts']:r for _,r in rows(MASK_INDEX)};surface={r['native_pts']:r for _,r in rows(SURFACE_INDEX)};prints={r['native_pts']:r for _,r in rows(PRINT_INDEX)}
    vis=[];ads=[];started=time.monotonic()
    with av.open(str(video))as container:
        stream=container.streams.video[0];container.seek(FIRST,stream=stream)
        for frame in container.decode(stream):
            pt=frame.pts
            if pt<FIRST:continue
            if pt>LAST:break
            im=frame.to_ndarray(format='bgr24');digest=hashlib.sha256(im.tobytes()).hexdigest();m=masks[pt];s=surface[pt];p=prints[pt]
            if not digest==m['source_BGR_sha256']==s['source_BGR_sha256']==p['decoded_native_BGR_sha256']:raise ValueError('S06 historical records refer to different source RGB')
            if file_ref(m['mask_file'])['sha256']!=m['mask_sha256']:raise ValueError('Historical current-pixel actor mask changed')
            image=OUT/f'source_frames/{pt}.png'
            if not image.exists():cv2.imwrite(str(image),im,[cv2.IMWRITE_PNG_COMPRESSION,1])
            elif hashlib.sha256(cv2.imread(str(image)).tobytes()).hexdigest()!=digest:raise ValueError('Cached original image differs')
            mask=cv2.imread(m['mask_file'],0)
            lineage=[dict(path=m['mask_file'],sha256=m['mask_sha256'])]
            for ref in [m.get('source_unknown_exclusion_mask_ref'),s['near_outside_exclusion_mask_ref']]:
                if not ref:continue
                if file_ref(ref['file'])['sha256']!=ref['sha256']:raise ValueError('Historical surface domain changed')
                mask=np.maximum(mask,cv2.imread(ref['file'],0));lineage.append(dict(path=ref['file'],sha256=ref['sha256']))
            combined=OUT/f'ad_exclusions/{pt}.png';cv2.imwrite(str(combined),mask)
            v=dict(pts=pt,shot_id='s06',image=str(image),source_BGR_sha256=digest,source_video_sha256=scope['source']['sha256'],
                mask_path=m['mask_file'],mask_sha256=m['mask_sha256'],unknown_ref=m.get('source_unknown_exclusion_mask_ref'),
                surface_ref=dict(path=s['frame_file'],sha256=s['frame_sha256']),near_outside_ref=s['near_outside_exclusion_mask_ref'],
                ice_visible_ref=s['ice_visible_mask_ref'],board_body_visible_ref=s['board_body_visible_mask_ref'],
                board_source=str(BOARD/f'{pt}.json'),historical_print_frame=p['file'],historical_print_frame_sha256=p['sha256'])
            vis.append(v);ads.append(dict(v,mask_path=str(combined),mask_sha256=file_ref(combined)['sha256'],mask_components=lineage,
                near_deck_and_equipment_excluded_using_current_source_evidence=True))
            if len(vis)%100==0:emit('official_frames',frames=len(vis),total=486,seconds=time.monotonic()-started)
    if [r['pts']for r in vis]!=list(range(FIRST,LAST+1,STEP)):raise ValueError('An official S06 exposure is missing')
    header=dict(source_video=str(video),source_video_sha256=scope['source']['sha256'],source_time_base='1/12800',shot_id='s06')
    write(OUT/'evidence/visibility.json',dict(header,frames=vis));write(OUT/'evidence/ad_visibility.json',dict(header,frames=ads))
    source_refs=[file_ref(p)for p in [MASK_INDEX,SURFACE_INDEX,PRINT_INDEX,REC/'dense56_v1/FINAL_SOURCE_V4.json',REC/'dense56_ice_v1/canonical/s06/source.json']if p.exists()]
    write(OUT/'evidence/history_refs.json',dict(refs=source_refs,source_time_bounds_are_inclusive_native_pts=True,source_seconds=[FIRST/12800,LAST/12800],
        legacy_display_end_111_38_is_not_the_last_exposure_time=True,legacy_500_point_display_budget_is_not_current_sampling_policy=True))
    images=[]
    for index in [0,80,160,240,320,400,450,485]:
        pt=vis[index]['pts'];im=cv2.resize(cv2.imread(vis[index]['image']),(768,432));text=f'S06 {pt/12800:.3f}s PTS {pt}'
        cv2.putText(im,text,(8,422),0,.6,(0,0,0),3);cv2.putText(im,text,(8,422),0,.6,(255,255,255),1);images.append(im)
    cv2.imwrite(str(OUT/'evidence/source_overview.jpg'),np.vstack([np.hstack(images[i:i+2])for i in range(0,8,2)]))
    verify_authority();emit('source_ready',frames=len(vis),native_paths=len(geo.current_paths),source_seconds=[FIRST/12800,LAST/12800])


def inventory():
    counts=collections.Counter();features=collections.Counter();frames=collections.Counter();refs=[file_ref(ICE)]
    for _,r in rows(ICE):
        counts[r['source_kind']]+=1;features[r['feature_id']]+=1;frames[r['native_pts']]+=1
    excluded={};withdrawn={};manual=[]
    p=ICE.parent/'exclusions.jsonl.gz';refs.append(file_ref(p))
    for n,r in rows(p):excluded[r['source_row_id']]=dict(record=r,source=str(p),jsonl_line_0based=n)
    index=DATA/'runs/okada_s02_s09_observations_20261026/source_index.sqlite'
    with sqlite3.connect(index.as_uri()+'?mode=ro',uri=True)as db:
        for key,payload in db.execute("select id,payload from qualification where shot='s06'"):withdrawn[key]=json.loads(payload)
    # Apply later source-store revisions by stable source ID, not fit residual.
    updates={}
    for kind in ['source_commit','positive_source_commit']:
        p=OLDJOINT/kind/'s06/source_request.json';refs.append(file_ref(p))
        for frame in read(p)['changes']:
            for sid,change in frame.get('update',{}).items():updates.setdefault(sid,{}).update(change)
    selected=REC/'dense56_v1/parts_v2/s06/selected_rows.jsonl.gz';refs.append(file_ref(selected))
    review_counts=collections.Counter();qualifications=[]
    for n,r in rows(selected):
        change=updates.get(r['selected_row_id'],{})
        if change:
            qualifications.append(dict(source_row_id=r.get('source_point_id'),selected_row_id=r['selected_row_id'],pts=r['native_pts'],uv=r['uv'],change=change))
            review_counts[change.get('qualification_reason_code','unspecified')]+=1
            if change.get('source_fit_eligible')is False and r.get('source_point_id'):
                withdrawn[r['source_point_id']]=dict(change,source_selected_row_id=r['selected_row_id'],reason=change.get('qualification_reason_code'))
        if r.get('user_confirmed')or'manual'in r.get('source_kind',''):manual.append(dict(source_line=n,record=r))
    with gzip.open(OUT/'evidence/historical_identity_qualification.jsonl.gz','wt',encoding='utf8')as f:
        for r in qualifications:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    # Preserve later fit-side identity revocations independently. They are not
    # an excuse to erase whole materials or import historical geometry.
    rev=[]
    for suffix in 'abcdef':
        p=OLDJOINT/f'native_source_identity_1023{suffix}/revision.json'
        if p.exists():rev.append(dict(ref=file_ref(p),revision=read(p)))
    write(OUT/'evidence/later_identity_revisions.json',dict(revisions=rev,source_only=True))
    write(OUT/'evidence/source_inventory.json',dict(shot='s06',full_exposures=486,raw_ice_rows=sum(counts.values()),source_kinds=dict(counts),
        features=dict(features),frame_counts=dict(frames),historical_corrections=excluded,withdrawals=withdrawn,manual_rows=manual,
        later_qualification_counts=dict(review_counts),source_files=refs,raw_source_files_unchanged=True,human_acceptance_not_inferred=True))
    emit('inventory_ready',raw_ice=sum(counts.values()),frames=len(frames),features=len(features),corrections=len(excluded),
        identity_holds=len(withdrawn),manual_records=len(manual),later_qualification_counts=dict(review_counts))


def baseline():
    from ice_points_s04 import qualify,source_domain,competing_source_ids
    from rinkspace.review.point_connections import polyline_order
    cv2.setNumThreads(2);verify_authority();auth=read(OUT/'evidence/native_authority.json')
    geo=NativeSourceGeometry(SCENE,auth['native_library']);grid=shared.NativeGrid();inv=read(OUT/'evidence/source_inventory.json')
    excluded={k:dict(reason=v['record'].get('reason',v['record'].get('reason_code')))for k,v in inv['historical_corrections'].items()}
    vis={r['pts']:r for r in read(OUT/'evidence/visibility.json')['frames']};ref=file_ref(ICE)
    dest=OUT/'evidence/baseline_ice';dest.mkdir(exist_ok=True);totals=collections.Counter();frames=[];chosen=0
    with gzip.open(OUT/'evidence/historical_ice_dispositions.jsonl.gz','wt',encoding='utf8')as ledger:
        for fi,(pt,entries)in enumerate(itertools.groupby(rows(ICE),key=lambda t:t[1]['native_pts'])):
            v=vis[pt];im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)
            near=cv2.imread(v['near_outside_ref']['file'],0);mask=np.maximum(mask,near)
            graphics,border,distance,_=source_domain(im,mask,read(v['board_source']));by=collections.defaultdict(list)
            for n,r in entries:
                native=material_path(geo,r);reason=qualify(r,native,excluded,{},graphics,border,distance)
                if reason:
                    totals[reason]+=1;ledger.write(json.dumps(dict(source_row_id=r['source_row_id'],pts=pt,feature_id=r['feature_id'],reason=reason))+'\n');continue
                hold=inv['withdrawals'].get(r['source_row_id'])or inv['withdrawals'].get(r.get('source_parent_row_id',''))
                by[native['id']].append(dict(r,native_path_id=native['id'],source_ref=dict(path=str(ICE),jsonl_line_0based=n),
                    historical_identity_hold=hold,not_formal_until_current_source_identity_checked=True))
            projection=grid.project(pt);features={}
            for pid,rr in by.items():
                conflicts=competing_source_ids(rr);rr=[r for r in rr if r['source_row_id']not in conflicts]
                if not rr:continue
                path=geo.current_paths[pid];options={}
                if path['kind']in('ice_circle','ice_ellipse'):
                    curve=projection[pid]['uv'];ok=projection[pid]['valid']
                    if ok.all()and np.isfinite(curve).all():
                        order,_=polyline_order(np.array([r['uv']for r in rr]),curve);arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
                        guide=np.column_stack([np.interp(order,arc,curve[:,k])for k in(0,1)])
                        options=dict(source_curve_guide=guide,source_curve_order=order/max(arc[-1],1e-12)*2*np.pi)
                keep,why=select_representatives(rr,path,**options)
                features[path['label']]=dict(native_path_id=pid,raw_count=len(rr),rows=[dict(rr[k],selection=dict(reasons=sorted(why[k]),original_UV_unchanged=True))for k in keep]);chosen+=len(keep)
            write(dest/f'{pt}.json',dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features=features,source_file=ref,
                model_or_fitted_curve_pixels_emitted=False,human_acceptance=False,source_search_only_until_identity_review=True))
            frames.append(dict(pts=pt,selected=sum(len(f['rows'])for f in features.values()),features={k:len(f['rows'])for k,f in features.items()}))
            if fi%100==0:emit('historical_ice',frames=fi+1,total=FRAME_COUNT,selected=chosen)
    write(OUT/'evidence/baseline_coverage.json',dict(frames=frames,selected=chosen,reasons=dict(totals),full_scope=FRAME_COUNT,complete_accuracy_claimed=False))
    verify_authority();emit('historical_ice_ready',frames=len(frames),selected=chosen)


def ad_configure():
    # The same material readers and atlas IDs used by the repaired S02/S04.
    import ad_replacement_s01 as b,ad_points_s04 as a
    import ad_entity_registration_s01 as r,ad_entity_tracks_s01 as t
    import ad_entity_measure_s01 as m,ad_entity_observability_s01 as o,ad_entity_select_s01 as s
    import ad_replace_package_s01 as pack
    adout=OUT/'ads';work=adout/'source_review';work.mkdir(parents=True,exist_ok=True)
    b.OUT=adout;a.OUT=adout;a.WORK=work;a.FIRST=FIRST;a.LAST=LAST;a.ADNUMBERS=range(1,14);a.PREFIX=SHOT_ID+'_native_source_v1_'
    r.OUT=work;r.VIS={f['pts']:f for f in read(OUT/'evidence/ad_visibility.json')['frames']};r.current.cache_clear()
    for module in(t,m,o,s):module.OUT=work
    pack.OUT=adout
    def current_native():
        from rinkspace.domain.parameters import SceneParameterModel
        from rinkspace.session.gpu_geometry import NativeSceneGPU
        g=shared.NativeGrid();model=SceneParameterModel(read(SCENE),SCENE,optimize_scene=True)
        return g.core,model,NativeSceneGPU(SCENE,g.core,parameter_state=g.parameter_state)
    pack.current_native=current_native
    atlases=read(adout/'atlases.json')['atlases']if(adout/'atlases.json').exists()else[]
    chosen=[x['id']for x in atlases if x['ad']in a.ADNUMBERS and x['id']not in(0,2,5,8,74)]
    t.PRIMARY=chosen;r.ATLAS_IDS=chosen;o.CANONICAL_BY_AD={1:[81,82,83,84],2:[1,3],3:[70,77],4:[4,6,69],5:[7]}
    o.LOCAL_PRINT_SHARPNESS_ADS={5};o.BEST_INDEPENDENT_SHAPE_ADS={1};a.configure=ad_configure
    return r,t,m,o,s,atlases,chosen


def ads_prepare():
    adout=OUT/'ads';(adout/'source_review').mkdir(parents=True,exist_ok=True)
    old=DATA/'runs/native_ad_points_s04_20261026/atlases.json'
    if not(adout/'atlases.json').exists():shutil.copy2(old,adout/'atlases.json')
    write(adout/'source_review/additional_source_atlases.json',dict(atlases=[]))
    shutil.copy2(OUT/'evidence/ad_visibility.json',adout/'visibility.json');ad_configure()
    allrows=[];counts=collections.Counter();packet_ids=set()
    with gzip.open(adout/'candidate_provenance.jsonl.gz','wt',encoding='utf8')as log:
        for fi,(_,v)in enumerate(rows(PRINT_INDEX)):
            d=read(v['file']);pt=d['native_pts'];eligible=np.frombuffer(base64.b64decode(d['eligible_uint8_base64']),np.uint8);ri=0;cells={}
            for mi,member in enumerate(d['source_members']):
                with open(member['file'],'rb')as f:f.seek(member['offset_bytes']);blob=f.read(member['length_bytes'])
                if hashlib.sha256(blob).hexdigest()!=member['sha256']:raise ValueError('Historical print member differs')
                for j,line in enumerate(gzip.decompress(blob).decode('utf8').splitlines()):
                    c=json.loads(line);counts['full_original_print_rows']+=1;ok=bool(eligible[ri]);ri+=1
                    if not ok:counts['historically_excluded_candidates']+=1;continue
                    # Spatial representatives are original pixels. All raw
                    # candidates remain addressable in immutable source members;
                    # new queries subsequently cover complete native contours.
                    label=c.get('feature_id','unassigned_print');cell=(label,*np.floor(np.asarray(c['uv'])/3.).astype(int))
                    if cell in cells:counts['redundant_same_3px_source_cell']+=1;continue
                    cells[cell]=True;match=re.match(r'Ad(\d+)\.',label)
                    item=dict(source_row=len(allrows),stable_id=c['point_id'],pts=pt,uv=c['uv'],label=label,ad=int(match.group(1))if match else 99,kind=0)
                    allrows.append(item);log.write(json.dumps(dict(item,source_frame=v['file'],member_index=mi,member_row=j),ensure_ascii=False)+'\n')
            if ri!=len(eligible):raise ValueError('Historical print eligibility sequence differs')
            if fi%100==0:emit('print_full_pool',frames=fi+1,total=FRAME_COUNT,candidates=len(allrows),original=counts['full_original_print_rows'])
        for p in sorted(PACKETS.glob('*.json')):
            packet=read(p);pt=packet['source_pts']
            for feature in packet['features']:
                label=feature['id'];match=re.match(r'Ad(\d+)\.',label)
                if not match:continue
                for si,seg in enumerate(feature.get('segments',[])):
                    uv=seg.get('source_native_readout_uv')
                    if uv is None and seg.get('pixels_measured_in_this_frame'):uv=seg.get('uv')
                    if uv is None:continue
                    for j,q in enumerate(uv):
                        sid=f'{SHOT_ID}_reviewed_raw_{pt}_{label}_{si}_{j}';item=dict(source_row=len(allrows),stable_id=sid,pts=pt,uv=q,label=label,ad=int(match.group(1)),kind=0)
                        allrows.append(item);log.write(json.dumps(dict(item,source_packet=str(p),segment=si,raw_point=j,human_acceptance_not_inferred=True),ensure_ascii=False)+'\n');counts['named_original_readouts']+=1
    np.savez_compressed(adout/'original_ad_candidates.npz',uv=np.array([r['uv']for r in allrows]),pts=np.array([r['pts']for r in allrows],np.int64),
        ids=np.array([r['stable_id']for r in allrows]),labels=np.array([r['label']for r in allrows]),ad=np.array([r['ad']for r in allrows],np.int16),kind=np.zeros(len(allrows),np.uint8))
    write(adout/'source_inventory.json',dict(full_exposures=FRAME_COUNT,counts=dict(counts),candidate_rows=len(allrows),original_UV_unchanged=True,
        complete_native_contour_queries_follow_this_candidate_inventory=True,source=file_ref(PRINT_INDEX),human_acceptance=False))
    emit('print_inventory_ready',counts=dict(counts),candidates=len(allrows))


def ads_anchors():
    import ad_points_s04 as a
    ad_configure();a.anchors()


def ads_tracks():
    _,t,*_=ad_configure();t.run()


def ads_measure():
    qualify_ad_charts()
    _,_,m,*_=ad_configure();m.run(source=dict(np.load(OUT/'ads/original_ad_candidates.npz')),banned=set(),first_pts=FIRST,source_prefix='s06_native_source_v1_')


def qualify_ad_charts():
    # Source-word context establishes the physical instance. A partial white
    # glyph match cannot turn the ANKARA panel into ISU-SKATING (Ad2).
    work=OUT/'ads/source_review';path=work/'source_identity_charts.json'
    original=work/'source_identity_charts_before_instance_review.json'
    if not original.exists():shutil.copy2(path,original)
    supported=set();refs=[]
    for file in sorted(PACKETS.glob('*.json')):
        packet=read(file);ids=sorted({int(m.group(1))for f in packet['features']if(m:=re.match(r'Ad(\d+)\.',f['id']))})
        supported.update(ids);refs.append(dict(source_ref=file_ref(file),pts=packet['source_pts'],physical_instances=ids))
    # Ad14 shares Ad1's artwork, while keeping its distinct current native ID.
    templates={1 if ad==14 else ad for ad in supported};atlases=read(OUT/'ads/atlases.json')['atlases']
    doc=read(path);previous=read(original);excluded={ai:len(frames)for ai,frames in previous['records'].items()if atlases[int(ai)]['ad']not in templates}
    doc['records']={ai:frames for ai,frames in doc['records'].items()if atlases[int(ai)]['ad']in templates}
    proof=dict(shot_id='s06',physical_instances=sorted(supported),source_template_ids=sorted(templates),source_references=refs,
        excluded_atlas_frame_counts=excluded,reason='no_S06_physical_instance_support_and_Ad2_word_match_is_on_ANKARA_Ad4',
        reviewed_failure_scope=dict(first_pts=1377536,last_pts=1385984,frames=34,actual_word='ANKARA',actual_instance='Ad4',wrong_instance='Ad2'),
        source_review_image=file_ref(OUT/'evidence/Ad2_wrong_binding_review.jpg')if(OUT/'evidence/Ad2_wrong_binding_review.jpg').exists()else None,
        native_geometry_unchanged=True,human_acceptance=False)
    write(OUT/'evidence/ad_physical_instance_review.json',proof);doc['physical_instance_context']=proof;write(path,doc)
    return proof


def ad_identity_reread():
    proof=qualify_ad_charts();scope=proof['reviewed_failure_scope'];wanted=set(range(scope['first_pts'],scope['last_pts']+STEP,STEP))
    rejected=OUT/'evidence/rejected_Ad2_instance_bindings.jsonl.gz'
    if not rejected.exists():
        with gzip.open(rejected,'wt',encoding='utf8')as f:
            for _,row in rows(OUT/'points/point_records.jsonl.gz'):
                if row['native_path_id'].startswith('print_instance/Ad2::'):
                    f.write(json.dumps(dict(row,reason='ANKARA_Ad4_material_was_bound_to_ISU_SKATING_Ad2',source_pixels_preserved=True,
                        binding_retracted_not_the_source_pixels=True,human_acceptance=False),ensure_ascii=False)+'\n')
    r,t,m,o,s,ats,_=ad_configure();work=OUT/'ads/source_review';dest=work/'physical_identity_repair';dest.mkdir(exist_ok=True)
    for name in ['source_identity_charts.json','additional_source_atlases.json','native_entity_registry.json']:shutil.copy2(work/name,dest/name)
    r.OUT=dest;r.VIS={pt:v for pt,v in r.VIS.items()if pt in wanted};r.current.cache_clear()
    for mod in(t,m,o,s):mod.OUT=dest
    source=dict(np.load(OUT/'ads/original_ad_candidates.npz'));source['source_rows']=np.arange(len(source['pts']),dtype=np.int64)
    use=np.isin(source['pts'],list(wanted));source={k:v[use]for k,v in source.items()}
    m.run(source=source,banned=set(),first_pts=FIRST,source_prefix='s06_physical_identity_RGB_')
    o.run();shutil.copy2(dest/'observability_checked_points.jsonl.gz',dest/'pointwise_checks.jsonl.gz');shutil.copy2(dest/'observability_ads.npz',dest/'verified_ads.npz');s.run()
    write(dest/'scope.json',dict(pts=sorted(wanted),replaces_previous_ad_representatives_only_in_this_scope=True,actual_RGB_reread=True))
    ad_configure();ad_records()
    backup=OUT/'nonice_before_instance_review';backup.mkdir(exist_ok=True)
    for pt in sorted(wanted):
        old=OUT/f'nonice/{pt}.json';target=backup/old.name
        if old.exists()and not target.exists():shutil.copy2(old,target)
    import repair_points_s02 as repair
    shared.ad_configure=ad_configure;ad_configure()
    repair.nonice(sorted(wanted),output_dir=OUT,source_records_path=OUT/'evidence/ad_records_before_border_review.jsonl.gz',board_dir=BOARD,
        source_prefix='s06',prepare_context=verify_authority,verify_context=verify_authority)
    verify_authority()


def ad_material_completion():
    """Photographic source aliases for retained word glyphs and badge field."""
    r,t,m,o,s,ats,_=ad_configure();work=OUT/'ads/source_review';added=[];seeds={};failed=[]
    native={e['id'].split('::')[-1]for e in read(OUT/'evidence/native_entities.json')['entities']if'::'in e['id']}
    # These are observed WYNDHAM glyph identities, not the removed Ankara row.
    aliases={'00':'W','01':'Y','02':'N','03':'D','05':'H','06':'A','08':'M'}
    for source in sorted(PACKETS.glob('*.json')):
        packet=read(source);pt=packet['source_pts'];paths=[]
        for f in packet['features']:
            match=re.fullmatch(r'Ad8\.print\.r2_WYNDHAM_(\d+)',f['id'])
            if not match or match.group(1)not in aliases:continue
            name='Ad8.print.glyph_'+aliases[match.group(1)]+'.boundary.0000'
            if name not in native:raise ValueError('Photographic letter is not retained in native')
            for si,seg in enumerate(f['segments']):
                q=seg.get('source_native_readout_uv',seg.get('uv')if seg.get('pixels_measured_in_this_frame')else None)
                if q is None or len(q)<2:continue
                q=np.asarray(q,float);paths.append(dict(path_id=name,pixels=q.tolist(),contour_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
                    proof=dict(kind='source_word_order_and_current_native_retained_letter_identity',word='WYNDHAM',letter=aliases[match.group(1)],
                        source_ref=file_ref(source),source_feature=f['id'],segment=si,source_pts=pt,reference_pixels_are_navigation_only=True)))
        if paths:added.append(dict(ad=8,image=r.VIS[pt]['image'],image_sha256=file_ref(r.VIS[pt]['image'])['sha256'],paths=paths,partial_source_reference=True,source_pts=pt,actual_source_material_colors=True))
    for pt,box in [(1357568,[1580,345,1800,460]),(1424896,[1610,465,1920,675]),(1419520,[15,315,450,615])]:
        im=r.current(pt)[0];x0,y0,x1,y1=box;B,G,R=cv2.split(im[y0:y1,x0:x1].astype(np.float32));chroma=R-(B+G)*.5
        segmentation={}
        if pt==1419520:
            # The fully visible badge has directional blur and white lettering
            # separating its red lobes. Its RGB envelope is a search guide;
            # the shared reader still measures and qualifies each actual edge.
            level,candidate=cv2.threshold(np.clip(chroma,0,255).astype(np.uint8),0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
            curves,_=cv2.findContours(candidate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            curves=[c for c in curves if cv2.contourArea(c)>20]
            if not curves:raise ValueError('Actual red badge source support not found')
            q=cv2.convexHull(np.concatenate(curves))[:,0].astype(float)+[x0,y0]
            segmentation=dict(source_region_method='Otsu_red_components_outer_envelope',Otsu_level=float(level),
                separated_lobes_belong_to_same_observed_badge=True,source_envelope_is_navigation_only=True)
        else:
            candidate=(chroma>12).astype(np.uint8)*255;candidate=cv2.morphologyEx(candidate,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
            curves,_=cv2.findContours(candidate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            if not curves:raise ValueError('Actual red badge source support not found')
            q=max(curves,key=cv2.contourArea)[:,0].astype(float)+[x0,y0]
        q=np.vstack([q,q[0]])
        name='Ad9.print.red_field.boundary.0000'
        if name not in native:raise ValueError('Badge outer field is not retained in native')
        added.append(dict(ad=9,image=r.VIS[pt]['image'],image_sha256=file_ref(r.VIS[pt]['image'])['sha256'],source_pts=pt,partial_source_reference=True,
            paths=[dict(path_id=name,pixels=q.tolist(),contour_sha256=hashlib.sha256(q.tobytes()).hexdigest(),proof=dict(
                kind='current_RGB_red_badge_whole_field_search_only',source_pts=pt,source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],
                search_roi=box,threshold_and_morphology_are_navigation_only=True,no_badge_fine_detail_added=True,**segmentation))]))
    for a in added:
        existing=next((old for old in ats if old.get('S06_material_completion')and old['ad']==a['ad']and old.get('source_pts')==a['source_pts']),None)
        if existing is None:a.update(id=len(ats),S06_material_completion=True);ats.append(a)
        else:a=existing
        pt=a['source_pts'];ref=r.reference(a);score=r.full_pattern_score(ref,np.eye(3),pt,allow_partial=True)
        if score is None:
            failed.append(dict(pts=pt,atlas_id=a['id'],ad=a['ad'],reason='current_source_whole_material_context_not_resolved',points_created=False));continue
        seeds[str(a['id'])]={str(pt):dict(H=np.eye(3).tolist(),atlas_id=a['id'],pts=pt,method='current_source_word_or_badge_material_identity',
            full_pattern_score=score,source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],native_geometry_used=False,human_acceptance=False)}
        preview=r.current(pt)[0].copy()
        for p in a['paths']:cv2.polylines(preview,[np.rint(p['pixels']).astype(int)],False,(0,255,255),1)
        cv2.imwrite(str(OUT/f'evidence/ad_material_navigation_{a["id"]}.jpg'),preview)
    write(OUT/'ads/atlases.json',dict(atlases=ats));write(work/'S06_material_source_seeds.json',dict(records=seeds))
    write(work/'S06_material_source_seed_failures.json',dict(rows=failed,acceptance_thresholds_unchanged=True))
    r,t,m,o,s,ats,_=ad_configure();t.PRIMARY=list(map(int,seeds));t.run(work/'S06_material_source_seeds.json',work/'S06_material_source_charts.json')
    tracked=read(work/'S06_material_source_charts.json');charts=read(work/'source_identity_charts.json');charts['records'].update(tracked['records']);write(work/'source_identity_charts.json',charts)
    r.authority(ad_numbers=range(1,14));dest=work/'material_completion';dest.mkdir(exist_ok=True)
    write(dest/'source_identity_charts.json',tracked)
    for name in ['additional_source_atlases.json','native_entity_registry.json']:shutil.copy2(work/name,dest/name)
    wanted={int(pt)for rr in tracked['records'].values()for pt in rr};r.OUT=dest;r.VIS={pt:v for pt,v in r.VIS.items()if pt in wanted};r.current.cache_clear()
    for mod in(t,m,o,s):mod.OUT=dest
    source=dict(np.load(OUT/'ads/original_ad_candidates.npz'));source['source_rows']=np.arange(len(source['pts']),dtype=np.int64);use=np.isin(source['pts'],list(wanted));source={k:v[use]for k,v in source.items()}
    m.run(source=source,banned=set(),first_pts=FIRST,source_prefix='s06_retained_material_RGB_');o.run()
    shutil.copy2(dest/'observability_checked_points.jsonl.gz',dest/'pointwise_checks.jsonl.gz');shutil.copy2(dest/'observability_ads.npz',dest/'verified_ads.npz');s.run()
    write(dest/'scope.json',dict(pts=sorted(wanted),current_native_only=True,old_Ankara_subtitle_not_added=True,actual_RGB_reread=True))
    ad_configure();ad_records();import repair_points_s02 as repair
    backup=OUT/'nonice_before_material_completion';backup.mkdir(exist_ok=True)
    for pt in sorted(wanted):
        old=OUT/f'nonice/{pt}.json';target=backup/old.name
        if old.exists()and not target.exists():shutil.copy2(old,target)
    shared.ad_configure=ad_configure;ad_configure()
    repair.nonice(sorted(wanted),output_dir=OUT,source_records_path=OUT/'evidence/ad_records_before_border_review.jsonl.gz',board_dir=BOARD,
        source_prefix='s06',prepare_context=verify_authority,verify_context=verify_authority);verify_authority()


def navigation():
    from ice_points_s04 import navigation as run
    run(output_dir=OUT,visibility_path=OUT/'evidence/visibility.json',authority_init=verify_authority,authority_verify=verify_authority,board_dir=BOARD)


class SourceGuides:
    """Reuse the native direction-family audit with S06 source evidence."""
    from repair_points_s02 import SourceGuides as _Common
    projected_directions=_Common.projected_directions
    direction_family=_Common.direction_family

    def __init__(self,grid):
        from ice_points_s04 import FiniteSourceIdentity
        from rinkspace.compat.d78_entities import native_definition
        self.base=FiniteSourceIdentity(output_dir=OUT,visibility_path=OUT/'evidence/visibility.json',packet_dir=PACKETS,first_pts=FIRST)
        self.transforms=self.base.transforms;self.grid=grid;self.paths={};self.axes={};self.rejected=[];self.endpoints=[];self.endpoint_axes=[]
        for p in grid.scene['paths']:
            if p['layer']!='ice':continue
            kind,params=native_definition(grid.scene,p,grid.core)
            if kind!=0:continue
            xyz=np.asarray(grid.core.sample_path(grid.scene['board'],kind,params,[],[0.,1.]));axis=int(np.argmax(abs(xyz[1,:2]-xyz[0,:2])))
            self.paths[p['label']]=p;self.axes[p['label']]=axis;self.endpoints.extend(xyz);self.endpoint_axes.append(axis)
        self.endpoints=np.asarray(self.endpoints)
        for p in sorted((OUT/'evidence/baseline_ice').glob('*.json')):
            packet=read(p)
            for name,f in packet['features'].items():
                direct=[r for r in f['rows']if not r.get('historical_identity_hold')]
                if name not in self.paths or len(direct)<2:continue
                self.base.guides[name].insert(0,dict(pts=packet['source_pts'],path=str(p),spans=[np.array([r['uv']for r in direct])],operator='paint_center',signal=None,transition=None,centre=None,width=None))
        for name,gg in self.base.guides.items():
            if name not in self.paths:continue
            good=[]
            for g in gg:
                actual=self.direction_family(name,g['pts'],np.vstack(g['spans']))
                if actual is not None and actual!=self.axes[name]:self.rejected.append(dict(feature=name,pts=g['pts'],source=g['path'],reason='wrong_native_direction_family'))
                else:good.append(g)
            self.base.guides[name]=good
        write(OUT/'evidence/source_guide_direction_audit.json',dict(rejected=self.rejected,native_geometry_changed=False))

    def at(self,name,pt):return self.base.at(name,pt)


def ice_reread():
    g=shared.NativeGrid();photo=SourceGuides(g)
    shared.ice_reread(board_dir=BOARD,source_prefix=SHOT_ID,photographic=photo)


def line_reference_reread(requested=None):
    from ice_points_s04 import source_domain
    from reread_s04_lines import NativeLineReader
    cv2.setNumThreads(2);verify_authority();grid=shared.NativeGrid();photo=SourceGuides(grid)
    reader=NativeLineReader(grid,photographic=photo,motion_path=OUT/'evidence/source_navigation.json',
        source_prefix=SHOT_ID+'_source_family',source_cross_line_guides=True,source_parallel_references=True)
    for path in reader.paths:path.update(_source_query_spacing_px=35,_separate_profile_support=True)
    vis=read(OUT/'evidence/visibility.json');dest=OUT/'evidence/reread_line_references';dest.mkdir(exist_ok=True);summary=[]
    for v in vis['frames']:
        pt=v['pts']
        if requested is not None and pt not in requested:continue
        im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)>0
        graphics,border,_,_=source_domain(im,mask.astype('uint8')*255,read(v['board_source']))
        excluded=mask|graphics|(np.arange(1080)[:,None]<=border[None])|(cv2.imread(v['near_outside_ref']['file'],0)>0)
        features,diagnostic=reader.read(pt,im,excluded,vis['source_video_sha256'])
        write(dest/f'{pt}.json',dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features=features,diagnostics=diagnostic,
            previous_source=file_ref(OUT/f'evidence/reread_lines/{pt}.json'),source_parallel_reference_ids=sorted(reader.source_parallel_reference_ids),
            material_thresholds_unchanged=True,current_native_geometry_unchanged=True,human_acceptance=False))
        summary.append(dict(pts=pt,counts={k:len(f['rows'])for k,f in features.items()}))
        if len(summary)%50==0 or requested is not None:emit('S06_source_line_family',frames=len(summary),pts=pt,counts=summary[-1]['counts'])
    write(dest/'coverage.json',dict(frames=summary));verify_authority()


def circle_gap_reread():
    """Reuse the material reader at unresolved turns; export no fitted UV."""
    from types import SimpleNamespace
    from ice_points_s04 import source_domain
    grid=shared.NativeGrid();house=shared.SourceHouseReader(grid,motion_path=OUT/'evidence/source_navigation.json')
    original_queries=house.queries
    house.queries=lambda group,index,role,parameters,spacing=35:original_queries(group,index,role,parameters,spacing=8)
    vis=read(OUT/'evidence/visibility.json');dest=OUT/'evidence/reread_circle_gaps';dest.mkdir(exist_ok=True);counts=[]
    for v in vis['frames']:
        pt=v['pts'];source=OUT/f'evidence/reread_circles/{pt}.json';packet=read(source);feature=packet['features'].get('ice.C0:centreline')
        if not feature:continue
        house.prepare_family('C0',pt);parameters=np.asarray(feature['navigation']['shared_source_affine_parameters'])
        im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)>0;near=cv2.imread(v['near_outside_ref']['file'],0)>0
        graphics,border,_,_=source_domain(im,mask.astype('uint8')*255,read(v['board_source']));mask|=near|graphics|(np.arange(1080)[:,None]<=border[None])
        native=SimpleNamespace(frame=SimpleNamespace(rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB),material_signals={}),mask=mask)
        measured,diag=house.measurements(house.groups['C0'],pt,native,parameters)
        if not measured:continue
        curve=house.projection(house.groups['C0'],pt,'outer',np.linspace(0,2*np.pi,513),None)[0]
        error=np.abs(house.transformed_conic_exact_distance(curve,np.array([r['uv']for r in measured]),parameters));actual=[]
        for i,(r,e)in enumerate(zip(measured,error)):
            if e>3.:continue
            r=dict(r,native_path_id=feature['native_path_id'],feature_id='ice.C0:centreline',native_pts=pt,
                source_row_id=f'{SHOT_ID}_current_C0_turn_RGB/{pt}/{r["query_index"]}')
            r['source_evidence']=dict(r['source_evidence'],source_sampling_spacing_px=8,existing_shape_navigation_ref=file_ref(source),model_coordinate_emitted=False)
            actual.append(r)
        if len(actual)<=len(feature['rows']):continue
        write(dest/f'{pt}.json',dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features={'ice.C0:centreline':dict(feature,rows=actual)},
            previous_source=file_ref(source),diagnostics=diag,additional_queries_at_missing_actual_curve_turns=True,source_fit_thresholds_unchanged=True))
        counts.append(dict(pts=pt,old=len(feature['rows']),new=len(actual)))
        if len(counts)%50==0:emit('C0_turn_reread',frames=len(counts),pts=pt)
    write(dest/'summary.json',dict(frames=counts,actual_readout_gain=sum(r['new']-r['old']for r in counts),final_density_selected_after_measurement=True))
    verify_authority();emit('C0_turn_reread_done',frames=len(counts),actual_readout_gain=sum(r['new']-r['old']for r in counts))


def line_gap_reread():
    shared.ice_gap_reread(board_dir=BOARD,source_prefix=SHOT_ID,use_registered_motion=True,
        line_input_dir=OUT/'evidence/reread_line_references'if(OUT/'evidence/reread_line_references').exists()else None)


def ads_complete():
    import ad_points_s04 as a
    r,t,m,o,s,ats,_=ad_configure();before=len(ats);a.complete_references(packet_dir=PACKETS)
    r,t,m,o,s,ats,_=ad_configure();new=list(range(before,len(ats)));seeds={};hashpts={v['source_BGR_sha256']:pt for pt,v in r.VIS.items()}
    for ai in new:
        ref=r.reference(ats[ai]);digest=hashlib.sha256(ref['color'].tobytes()).hexdigest()
        if digest not in hashpts:continue
        pt=hashpts[digest];seeds[str(ai)]={str(pt):dict(H=np.eye(3).tolist(),atlas_id=ai,pts=pt,
            method='same_authentic_source_exposure_identity',source_BGR_sha256=digest,current_native_unique_alias=True,human_acceptance_not_inferred=True)}
    write(r.OUT/'S06_local_source_anchors.json',dict(records=seeds))
    t.PRIMARY=list(map(int,seeds));t.run(r.OUT/'S06_local_source_anchors.json',r.OUT/'S06_local_source_charts.json')
    result=read(r.OUT/'source_identity_charts.json');extra=read(r.OUT/'S06_local_source_charts.json');result['records'].update(extra['records'])
    write(r.OUT/'source_identity_charts.json',result);r.authority(ad_numbers=range(1,14))
    # Existing exact-crop adaptor supplies the current retained ministry mark.
    shared.ad_configure=ad_configure;shared.ads_cropped_artwork()
    emit('S06_photographic_references_completed',new_atlases=new,actual_source_seed_charts=len(seeds))


def ads_qualify():
    _,_,_,o,s,*_=ad_configure();o.run()
    work=OUT/'ads/source_review'
    # Sampling consumes observability-qualified measurements. Keep the complete
    # initial profile ledger, including every original candidate disposition.
    for source,name in [('pointwise_checks.jsonl.gz','initial_pointwise_checks.jsonl.gz'),('verified_ads.npz','initial_verified_ads.npz')]:
        target=work/name
        if not target.exists():shutil.copy2(work/source,target)
    shutil.copy2(work/'observability_checked_points.jsonl.gz',work/'pointwise_checks.jsonl.gz')
    shutil.copy2(work/'observability_ads.npz',work/'verified_ads.npz');s.run()


def ad_records():
    work=OUT/'ads/source_review';sources=[work/'representative_checks.jsonl.gz'];records=[];seen=set()
    if(work/'gap_measurement/representative_checks.jsonl.gz').exists():sources.append(work/'gap_measurement/representative_checks.jsonl.gz')
    repaired=work/'physical_identity_repair';replacement=set()
    if(repaired/'scope.json').exists():replacement=set(read(repaired/'scope.json')['pts']);sources.append(repaired/'representative_checks.jsonl.gz')
    completed=work/'material_completion'
    if(completed/'scope.json').exists():sources.append(completed/'representative_checks.jsonl.gz')
    instance_refs=[file_ref(p)for p in PACKETS.glob('*.json')if any(f['id'].startswith('Ad14.')for f in read(p)['features'])]
    for source in sources:
        ref=file_ref(source)
        for _,r in rows(source):
            if r['decision']=='not_admitted':continue
            identity=r['identity'];pid=r['path_id'];pt=r['pts'];key=(r['stable_id'],pt)
            if pt in replacement and source.parent not in(repaired,completed):continue
            if pid.startswith(('print_instance/Ad2::','print_instance/Ad3::')):continue
            if key in seen:continue
            seen.add(key)
            if pid.startswith('print_instance/Ad1::'):
                # Same artwork is used twice. The S06 source packets identify
                # this visible corner instance as Ad14, beyond Ad13.
                pid=pid.replace('print_instance/Ad1::','print_instance/Ad14::',1)
                r=dict(r,physical_native_path_id=pid,physical_instance_source_refs=instance_refs,template_is_not_instance_identity=True)
            records.append(dict(observation_id=r['stable_id'],source_id=r['stable_id'],pts=pt,uv=r['uv'],native_path_id=pid,
                kind=r['kind'],station=r['station'],origin='verified_original_ad_pixel'if r['decision']=='include_original_pixel'else'new_current_RGB_ad_material',
                human_acceptance=False,source_record=r,source_ref=ref,source_BGR_sha256=r['source_BGR_sha256'],
                source_group_id=f's06/{pt}/{pid}/same_material_profile_family',
                review_order=dict(family='ad',run=pid,atlas=identity['atlas_id'],source_path=identity['source_path_index'],order=identity.get('reference_arc_fraction',.5))))
    with gzip.open(OUT/'evidence/ad_records_before_border_review.jsonl.gz','wt',encoding='utf8')as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    emit('ad_records_ready',rows=len(records),frames=len({r['pts']for r in records}))


def ad_borders():
    import repair_points_s02 as repair
    shared.ad_configure=ad_configure;ad_configure()
    repair.nonice(output_dir=OUT,source_records_path=OUT/'evidence/ad_records_before_border_review.jsonl.gz',board_dir=BOARD,
        source_prefix=SHOT_ID,prepare_context=verify_authority,verify_context=verify_authority)


def assemble():
    from ice_points_s04 import current_graphics
    from calibration.program_tracking.audit_first_shot_circle_material_identity import centreline_house_conflicts
    from rinkspace.review.point_connections import polyline_order
    from prepare_native_s01 import POINT_FIELDS
    cv2.setNumThreads(2);verify_authority();grid=shared.NativeGrid();geo=NativeSourceGeometry(SCENE,grid.manifest['native_library'])
    entities={r['native_path_id']:r for r in read(OUT/'evidence/native_entities.json')['entities']}
    vis=read(OUT/'evidence/visibility.json');folder=OUT/'points';folder.mkdir(exist_ok=True);support=OUT/'support';support.mkdir(exist_ok=True)
    allrows=[];coverage=[];counter=collections.Counter();known={}
    material_review=OUT/'evidence/source_material_review.json'
    material_holds=set(read(material_review)['observation_ids'])if material_review.exists()else set()
    for path in PACKETS.glob('*.json'):
        for f in read(path)['features']:
            if f['id'].startswith('ice.'):known.setdefault(f['id'],[]).append(str(path))
    with gzip.open(OUT/'evidence/final_source_dispositions.jsonl.gz','wt',encoding='utf8')as log:
        def reject(r,reason,pt):
            counter[reason]+=1;log.write(json.dumps(dict(pts=pt,source_row_id=r.get('source_row_id',r.get('observation_id')),uv=r.get('uv'),reason=reason,original_preserved=True),ensure_ascii=False)+'\n')
        for fi,v in enumerate(vis['frames']):
            pt=v['pts'];im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)>0;graphics,_=current_graphics(im);mask|=graphics
            near=cv2.imread(v['near_outside_ref']['file'],0)>0;features={};frame=[];supports={}
            ice_refs={}
            for kind in ['lines','circles']:
                source=OUT/f'evidence/reread_{kind}/{pt}.json'
                if kind=='lines'and(OUT/f'evidence/reread_line_references/{pt}.json').exists():source=OUT/f'evidence/reread_line_references/{pt}.json'
                if kind=='lines'and(OUT/f'evidence/reread_line_final_support/{pt}.json').exists():source=OUT/f'evidence/reread_line_final_support/{pt}.json'
                if kind=='circles'and(OUT/f'evidence/reread_circle_references/{pt}.json').exists():source=OUT/f'evidence/reread_circle_references/{pt}.json'
                packet=read(source);features.update(packet['features'])
                for name in packet['features']:ice_refs[name]=file_ref(source)
            gap=OUT/f'evidence/reread_circle_gaps/{pt}.json'
            if gap.exists():
                packet=read(gap);features.update(packet['features'])
                for name in packet['features']:ice_refs[name]=file_ref(gap)
            gap=OUT/f'evidence/reread_gap_lines/{pt}.json'
            if gap.exists():
                packet=read(gap);features.update(packet['features'])
                for name in packet['features']:ice_refs[name]=file_ref(gap)
            cross=OUT/f'evidence/reread_cross_lines/{pt}.json'
            if cross.exists():
                packet=read(cross);features.update(packet['features'])
                for name in packet['features']:ice_refs[name]=file_ref(cross)
            identity=OUT/f'evidence/reread_material_identity/{pt}.json'
            if identity.exists():
                packet=read(identity);features.update(packet['features'])
                for name in packet['features']:ice_refs[name]=file_ref(identity)
            red=OUT/f'evidence/reread_red_identity/{pt}.json'
            if red.exists():
                packet=read(red);features.update(packet['features'])
                for name in packet['features']:ice_refs[name]=file_ref(red)
            temporal=OUT/f'evidence/reread_temporal_identity/{pt}.json'
            if temporal.exists():
                packet=read(temporal);features.update(packet['features'])
                for name in packet['features']:ice_refs[name]=file_ref(temporal)
            continuous=OUT/f'evidence/reread_continuous_crosslines/{pt}.json'
            if continuous.exists():
                packet=read(continuous);features.update(packet['features'])
                for name in packet['features']:ice_refs[name]=file_ref(continuous)
            final_temporal=OUT/f'evidence/reread_temporal_final/{pt}.json'
            if final_temporal.exists():
                packet=read(final_temporal);features.update(packet['features'])
                for name in packet['features']:ice_refs[name]=file_ref(final_temporal)
            adjacent=OUT/f'evidence/reread_temporal_adjacent/{pt}.json'
            if adjacent.exists():
                packet=read(adjacent);features.update(packet['features'])
                for name in packet['features']:ice_refs[name]=file_ref(adjacent)
            cross_family=OUT/f'evidence/reread_crossline_family/{pt}.json'
            if cross_family.exists():
                packet=read(cross_family);features.update(packet['features'])
                for name in packet['features']:ice_refs[name]=file_ref(cross_family)
            closing=OUT/f'evidence/reread_temporal_closing/{pt}.json'
            if closing.exists():
                packet=read(closing);features.update(packet['features'])
                for name in packet['features']:ice_refs[name]=file_ref(closing)
            straight=OUT/f'evidence/straight_source_selection/{pt}.json'
            if straight.exists():
                packet=read(straight);features.update(packet['features'])
                for name in packet['features']:ice_refs[name]=file_ref(straight)
            reviewed=OUT/f'evidence/final_material_review/{pt}.json'
            if reviewed.exists():
                packet=read(reviewed)
                if packet['source_pts']!=pt or packet['source_BGR_sha256']!=v['source_BGR_sha256']:
                    raise ValueError('Final material review belongs to another source exposure')
                features.update(packet['features'])
                for name in packet['features']:ice_refs[name]=file_ref(reviewed)
            failures,fill,_=centreline_house_conflicts(features,entities,include_crosslines=SHOT_ID in('s03','s05'));bad={r['source_row_id']for r in failures}
            for r in failures:reject(r,r['reason'],pt)
            materialmask=OUT/f'evidence/ice_material_exclusions/{pt}.png';materialmask.parent.mkdir(exist_ok=True);cv2.imwrite(str(materialmask),fill.astype('uint8')*255)
            baseline=read(OUT/f'evidence/baseline_ice/{pt}.json')['features']
            for name,f in features.items():
                pid=f['native_path_id'];path=entities[pid];raw=[]
                # A tiny colour response near a projected but unrecognised arc
                # does not establish a new physical circle/crease identity.
                same_house=path.get('role')=='named_ring_boundary'and any(k.split(':')[0]==name.split(':')[0]for k in known)
                if name not in known and name not in baseline and not same_house:
                    for r in f['rows']:reject(r,'native_search_only_without_independent_photographic_material_identity',pt)
                    continue
                for r in f['rows']:
                    if r['source_row_id']in bad:continue
                    x,y=np.rint(r['uv']).astype(int)
                    if not(0<=x<1920 and 0<=y<1080)or mask[y,x]or near[y,x]:reject(r,'actual_source_visibility_exclusion',pt);continue
                    raw.append(r)
                if not raw:continue
                rr=[dict(r,pixel_support=dict(normal=r['source_evidence']['profile_normal'],query_index=r['source_evidence'].get('query_index',i)))for i,r in enumerate(raw)]
                opts={}
                if path['kind']in('ice_circle','ice_ellipse')and all('theta'in r for r in rr):opts=dict(source_curve_guide=np.asarray([r['source_evidence']['query_uv']for r in rr]),source_curve_order=np.asarray([r['theta']for r in rr]))
                keep,why=select_representatives(rr,path,**opts)
                for i in keep:
                    r=rr[i];sid=r['source_row_id'];frame.append(dict(observation_id=sid,source_id=sid,pts=pt,uv=r['uv'],native_path_id=pid,kind=1,station=.5,
                        origin='current_RGB_ice_material',human_acceptance=False,source_record=r,source_ref=ice_refs[name],
                        source_BGR_sha256=v['source_BGR_sha256'],source_group_id=f'{SHOT_ID}/{pt}/{pid}/same_material_profile_family',selection_reasons=sorted(why[i]),
                        review_order=dict(family='ice',run=pid,order=float(r.get('theta',r['source_evidence'].get('query_index',i))))))
                supports[pid]=dict(rows=raw,kind=path['kind'],closed=bool(grid.cuts[pid]['closed']),source_material_exclusion_mask=str(materialmask)if path.get('measurement')=='centreline'else None)
            npth=OUT/f'nonice/{pt}.json'
            if npth.exists():
                d=read(npth);frame.extend(d['retained_or_remeasured']);ad_ref=file_ref(npth)
                for r in d['dispositions']:reject(r,r['reason'],pt)
                for pid,f in d['borders'].items():
                    rr=f['rows'];supports[pid]=dict(rows=rr,kind='panel_border',closed=False)
                    if not rr:continue
                    keep,why=select_representatives(rr,dict(kind='ice_segment'))
                    for i in keep:
                        r=rr[i];sid=r['source_row_id'];frame.append(dict(observation_id=sid,source_id=sid,pts=pt,uv=r['uv'],native_path_id=pid,kind=1,station=.5,
                            origin='current_RGB_panel_border',human_acceptance=False,source_record=r,source_ref=ad_ref,source_BGR_sha256=v['source_BGR_sha256'],
                            source_group_id=f'{SHOT_ID}/{pt}/{pid}/panel_material',selection_reasons=sorted(why[i]),review_order=dict(family='ad_source_edge',run=pid,order=r['order'])))
            b=read(v['board_source']);br=collections.defaultdict(list);board_ref=file_ref(v['board_source'])
            for r in b['rows']:
                if not r.get('pixels_measured_in_this_frame')or not r.get('pixel_support',{}).get('current_pixel_profile_reread')or r.get('occlusion'):continue
                pid=geo.lookup('board.top'if r['role']=='board.top'else'board.bottom')['id'];x,y=np.rint(r['uv']).astype(int)
                if not(0<=x<1920 and 0<=y<1080)or mask[y,x]or near[y,x]:continue
                br[(pid,'far')].append(dict(r,native_path_id=pid,pixel_support=dict(r['pixel_support'],normal=[0.,1.]),source_ref=board_ref))
            surface=read(v['surface_ref']['path'])
            for r in surface['new_source_rows']:
                if not r.get('pixels_measured_in_this_frame')or r.get('model_or_world_projection_used')or r.get('occlusion'):continue
                x,y=np.rint(r['uv']).astype(int)
                if not(0<=x<1920 and 0<=y<1080)or mask[y,x]:continue
                pid=geo.lookup('board.top')['id'];br[(pid,r['branch_id'])].append(dict(r,native_path_id=pid,source_ref=v['surface_ref']))
            for (pid,branch),rr in br.items():
                # Keep near and far branches separate in both sampling and links.
                keep,why=select_representatives(rr,dict(kind='ice_segment'))
                supports.setdefault(pid,dict(rows=[],kind='board',closed=False,branches=[]))['branches'].append(dict(branch=branch,rows=rr))
                for i in keep:
                    r=rr[i];sid=r['source_row_id'];frame.append(dict(observation_id='board:'+sid,source_id=sid,pts=pt,uv=r['uv'],native_path_id=pid,kind=1,station=.5,
                        origin='preserved_current_RGB_board_material',human_acceptance=False,source_record=r,source_ref=r['source_ref'],source_BGR_sha256=v['source_BGR_sha256'],
                        source_group_id=r.get('segment_id',f'{SHOT_ID}/{pt}/{pid}/{branch}'),review_order=dict(family='board',run=branch,order=float(r['uv'][0])),selection_reasons=sorted(why[i])))
            for r in frame:
                if r['observation_id']in material_holds:
                    reject(r,'source_material_identity_unresolved_after_original_frame_review',pt)
            frame=[r for r in frame if r['observation_id']not in material_holds]
            write(support/f'{pt}.json',supports);allrows.extend(frame)
            count=collections.Counter(entities[r['native_path_id']]['layer']for r in frame)
            coverage.append(dict(pts=pt,source_seconds=pt/12800,counts=dict(count),paths=sorted({r['native_path_id']for r in frame})))
            if fi%100==0:emit('assemble',frames=fi+1,total=FRAME_COUNT,rows=len(allrows),counts=dict(count))
    allrows.sort(key=lambda r:(r['pts'],r['native_path_id'],r['observation_id']));paths=sorted({r['native_path_id']for r in allrows});lookup={p:i for i,p in enumerate(paths)}
    duplicate=collections.Counter(r['observation_id']for r in allrows)
    for r in allrows:
        if duplicate[r['observation_id']]>1:
            r['original_track_id']=r['observation_id'];r['observation_id']+=f'@pts={r["pts"]}'
            r['source_identity_is_track_and_integer_exposure']=True
    ids=[r['observation_id']for r in allrows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate S06 observation IDs')
    z=dict(source_rows=np.arange(len(ids),dtype=np.int64),observation_ids=np.array(ids),uv=np.array([r['uv']for r in allrows]),pts=np.array([r['pts']for r in allrows],np.int64),
        kind=np.array([r['kind']for r in allrows],np.uint8),stations=np.array([r['station']for r in allrows]),path_indices=np.array([lookup[r['native_path_id']]for r in allrows],np.int32))
    if set(z)!=POINT_FIELDS:raise ValueError('Native point fields differ')
    np.savez_compressed(folder/'point_observations.npz',**z);groups=collections.defaultdict(list)
    with gzip.open(folder/'point_records.jsonl.gz','wt',encoding='utf8')as f:
        for i,r in enumerate(allrows):r['source_row']=i;f.write(json.dumps(r,ensure_ascii=False)+'\n');groups[r['source_group_id']].append(r['observation_id'])
    with gzip.open(folder/'source_correlated_groups.jsonl.gz','wt',encoding='utf8')as f:
        for key,ids in groups.items():f.write(json.dumps(dict(group_id=key,observation_ids=ids,independent_samples=False))+'\n')
    from rinkspace.session.gpu_geometry import NativeSceneGPU
    native=NativeSceneGPU(SCENE,grid.core,parameter_state=grid.parameter_state)
    if not set(paths)<=native.paths.keys():raise ValueError('Unknown current native path')
    camera=read(CAMERA);auth=read(OUT/'evidence/native_authority.json')
    write(folder/'mapping.json',dict(schema='rinkspace.native_source_points.v1',shot_id=SHOT_ID,scene_path=str(SCENE),scene_revision=native.scene_revision,scene_file_sha256=file_ref(SCENE)['sha256'],
        camera_path=str(CAMERA),camera_sha256=file_ref(CAMERA)['sha256'],control_sha256=file_ref(CAMERA.parent/camera['controls_ref'])['sha256'],
        path_ids=paths,point_input_sha256=file_ref(folder/'point_observations.npz')['sha256'],source_video=vis['source_video'],source_video_sha256=vis['source_video_sha256'],
        source_scope=dict(first_pts=FIRST,last_pts=LAST,time_base='1/12800',frames=FRAME_COUNT),pixel_domain='native_distorted',image_size=[1920,1080],native_numeric_state=auth['scene_state'],
        original_sources_rewritten=False,projected_or_flow_UV_emitted=False,human_acceptance_not_inferred=True))
    result=dict(frames=FRAME_COUNT,source_seconds=[FIRST/12800,LAST/12800],rows=len(allrows),native_paths=len(paths),coverage=coverage,
        counts=dict(collections.Counter(entities[r['native_path_id']]['layer']for r in allrows)),origins=dict(collections.Counter(r['origin']for r in allrows)),
        dispositions=dict(counter),original_source_files_changed=False,native_geometry_changed=False,machine_results_user_accepted=False)
    write(OUT/'result.json',result);verify_authority();emit('assembled',**{k:v for k,v in result.items()if k not in('coverage','dispositions')})


def integrate():
    from rinkspace.session.gpu_geometry import NativeSceneGPU,initialize_curve_stations
    from prepare_native_s01 import load_pixel_input
    g=shared.NativeGrid();native=NativeSceneGPU(SCENE,g.core,parameter_state=g.parameter_state);folder=OUT/'points';z=dict(np.load(folder/'point_observations.npz'));mapping=read(folder/'mapping.json')
    camera=read(CAMERA);controls=np.load(CAMERA.parent/camera['controls_ref']);paths=[mapping['path_ids'][i]for i in z['path_indices']]
    stations,report=initialize_curve_stations(native,camera=camera,controls=controls,path_ids=paths,stations=z['stations'],pts=z['pts'],observed=z['uv'],kinds=z['kind'],backend='cpu',progress=lambda r:emit('station_init',**r))
    np.savez_compressed(folder/'initialized_points.npz',stations=stations);load_pixel_input(folder)
    with native.resident(camera=camera,controls=controls,path_ids=paths,stations=stations,pts=z['pts'],observed=z['uv'])as resident:result=resident.evaluate('cpu')
    np.savez_compressed(folder/'native_evaluation.npz',status=result['status'],projected_UV=result['values'][:,3:5],residual=result['values'][:,9:11])
    report.update(rows=len(z['pts']),invalid=int(np.count_nonzero(result['status'])),geometry_or_camera_updated=False,projection_error_is_not_pixel_accuracy=True)
    if report['invalid']:
        # An immutable current installed-artwork path can lie entirely below
        # its board domain. Preserve every reading separately; never alter it,
        # its source pixels, or the native status to make ingestion pass.
        bad=np.asarray(result['status'])!=0;unsupported={m for m in set(np.asarray(paths)[bad])}
        if not np.all(np.asarray(result['status'])[bad]==3)or any(not pid.startswith('print_instance/')or native.paths[pid]['kind']!=8 for pid in unsupported):
            write(OUT/'native_integration.json',report);raise ValueError('Unexpected native-domain failure: '+str(unsupported))
        for pid in unsupported:
            if not all(vertex[1]<0 for vertex in native.paths[pid]['vertices'])or not np.all(np.asarray(result['status'])[np.asarray(paths)==pid]==3):
                raise ValueError('Native failure needs separate diagnosis: '+pid)
        hold=OUT/'native_domain_hold';hold.mkdir(exist_ok=True)
        np.savez_compressed(hold/'actual_points.npz',**{k:v[bad]for k,v in z.items()})
        write(hold/'mapping.json',mapping)
        accepted=[];held=[]
        for _,r in rows(folder/'point_records.jsonl.gz'):
            i=r['source_row']
            if bad[i]:held.append(dict(r,solver_admission='current_native_geometry_outside_board_domain',native_status=3,
                native_path_vertices=native.paths[r['native_path_id']]['vertices'],source_measurement_not_retracted=True,geometry_not_changed=True))
            else:r=dict(r,pre_native_domain_partition_row=i,source_row=len(accepted));accepted.append(r)
        for file,records in [(hold/'actual_records.jsonl.gz',held),(folder/'point_records.jsonl.gz',accepted)]:
            with gzip.open(file,'wt',encoding='utf8')as f:
                for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        shutil.copy2(folder/'source_correlated_groups.jsonl.gz',hold/'all_measured_source_groups.jsonl.gz')
        groups=collections.defaultdict(list)
        for r in accepted:groups[r['source_group_id']].append(r['observation_id'])
        with gzip.open(folder/'source_correlated_groups.jsonl.gz','wt',encoding='utf8')as f:
            for key,ids in groups.items():f.write(json.dumps(dict(group_id=key,observation_ids=ids,independent_samples=False))+'\n')
        zz={k:v[~bad]for k,v in z.items()};zz['source_rows']=np.arange(len(zz['pts']),dtype=np.int64)
        active_paths=sorted({r['native_path_id']for r in accepted});zz['path_indices']=np.array([active_paths.index(r['native_path_id'])for r in accepted],np.int32)
        np.savez_compressed(folder/'point_observations.npz',**zz);np.savez_compressed(folder/'initialized_points.npz',stations=stations[~bad])
        mapping.update(path_ids=active_paths,point_input_sha256=file_ref(folder/'point_observations.npz')['sha256'],current_native_domain_holds=str(hold/'actual_records.jsonl.gz'))
        write(folder/'mapping.json',mapping)
        # Re-evaluate the actual deliverable, not a filtered status report.
        with native.resident(camera=camera,controls=controls,path_ids=[r['native_path_id']for r in accepted],stations=stations[~bad],pts=zz['pts'],observed=zz['uv'])as resident:checked=resident.evaluate('cpu')
        np.savez_compressed(folder/'native_evaluation.npz',status=checked['status'],projected_UV=checked['values'][:,3:5],residual=checked['values'][:,9:11])
        report.update(all_measured_rows=len(z['pts']),native_domain_held_rows=int(bad.sum()),rows=len(zz['pts']),invalid=int(np.count_nonzero(checked['status'])),
            withheld_evidence=file_ref(hold/'actual_records.jsonl.gz'),native_domain_problem=dict(paths=sorted(unsupported),vertices_by_path={pid:native.paths[pid]['vertices']for pid in sorted(unsupported)},
                remedy='Requires a separately authorized correction of current native numeric placement; source data is preserved',source_pixel_accuracy_not_disproved=True))
        if len(unsupported)==1:
            pid=next(iter(unsupported));report['native_domain_problem'].update(path=pid,vertices=native.paths[pid]['vertices'])
        result_doc=read(OUT/'result.json');result_doc['all_measured_rows']=result_doc['rows'];result_doc['rows']=len(accepted);result_doc['native_paths']=len(active_paths);result_doc['native_domain_held_rows']=len(held)
        result_doc['counts']['advertisement']-=len(held);result_doc['origins']=dict(collections.Counter(r['origin']for r in accepted));frames=collections.defaultdict(list)
        for r in accepted:frames[r['pts']].append(r)
        entities={r['id']:r for r in read(OUT/'evidence/native_entities.json')['entities']}
        result_doc['coverage']=[dict(pts=pt,source_seconds=pt/12800,counts=dict(collections.Counter(entities[r['native_path_id']]['layer']for r in rr)),paths=sorted({r['native_path_id']for r in rr}))for pt,rr in sorted(frames.items())]
        write(OUT/'result.json',result_doc);emit('native_domain_readings_preserved',held=len(held),paths=sorted(unsupported),formal_rows=len(accepted))
    write(OUT/'native_integration.json',report);verify_authority();emit('native_ingestion',rows=report['rows'],invalid=report['invalid'])


def verify():
    from prepare_native_s01 import load_pixel_input
    verify_authority();folder=OUT/'points';z,m,_=load_pixel_input(folder);vis={v['pts']:v for v in read(OUT/'evidence/visibility.json')['frames']}
    entities={v['id']:v for v in read(OUT/'evidence/native_entities.json')['entities']};counts=collections.Counter();groups=collections.defaultdict(list);refs={};seen=set();by=collections.defaultdict(list)
    if not np.array_equal(np.unique(z['pts']),np.arange(FIRST,LAST+1,STEP)):raise ValueError('Missing official S06 exposure')
    if not np.isfinite(z['uv']).all()or np.any(z['uv']<0)or np.any(z['uv']>[1919,1079]):raise ValueError('Source pixel outside image')
    for _,r in rows(folder/'point_records.jsonl.gz'):
        i=r['source_row'];seen.add(i);pt=r['pts'];pid=r['native_path_id'];source=r['source_record'];by[pt].append(r)
        if r['observation_id']!=str(z['observation_ids'][i])or pt!=int(z['pts'][i])or not np.array_equal(r['uv'],z['uv'][i])or pid!=m['path_ids'][int(z['path_indices'][i])]:raise ValueError('Input and source evidence differ')
        if r['source_BGR_sha256']!=vis[pt]['source_BGR_sha256']:raise ValueError('Source frame differs')
        if r['origin']=='reuse_authenticated_current_RGB_profile_measurement':actual=source['source_profile']['actual_readout_uv']
        else:actual=source['uv']
        if not np.array_equal(r['uv'],actual):raise ValueError('Actual source UV was replaced')
        if r.get('human_acceptance'):raise ValueError('No full S06 human acceptance was issued')
        if source.get('source_uv_from_model')or source.get('model_or_world_projection_used'):raise ValueError('Projected UV is not an observation')
        ev=source.get('source_evidence',{})
        if entities[pid].get('measurement')=='centreline'and ev.get('actual_source_sides_uv'):
            if not np.allclose(np.mean(ev['actual_source_sides_uv'],axis=0),r['uv'],atol=1e-3,rtol=0):raise ValueError('Centreline is not the actual material midpoint')
        if SHOT_ID=='s06'and pid.startswith('print_instance/Ad1::'):raise ValueError('Ad14 was confused with repeated artwork Ad1')
        if SHOT_ID=='s06'and pid.startswith(('print_instance/Ad2::','print_instance/Ad3::')):raise ValueError('Source-word instance lacks S06 physical support')
        groups[r['source_group_id']].append(r['observation_id']);counts[r['origin']]+=1
        ref=r.get('source_ref')
        if ref and ref.get('sha256'):refs[ref['path']]=ref['sha256']
    if seen!=set(range(len(z['pts']))):raise ValueError('Missing source row evidence')
    declared={r['group_id']:r['observation_ids']for _,r in rows(folder/'source_correlated_groups.jsonl.gz')}
    if dict(groups)!=declared:raise ValueError('Correlated source grouping changed')
    for p,sha in refs.items():
        if file_ref(p)['sha256']!=sha:raise ValueError('Source evidence changed: '+p)
    visibility_failures=[];line_audit=[];missing=collections.defaultdict(list)
    for pt,records in by.items():
        mask=cv2.imread(vis[pt]['mask_path'],0)>0;near=cv2.imread(vis[pt]['near_outside_ref']['file'],0)>0;by_path=collections.defaultdict(list)
        for r in records:
            x,y=np.rint(r['uv']).astype(int);path=entities[r['native_path_id']]
            if mask[y,x]or(path['layer']!='board'and near[y,x]):visibility_failures.append(dict(pts=pt,observation_id=r['observation_id'],uv=r['uv']))
            by_path[r['native_path_id']].append(r)
        for pid,rr in by_path.items():
            if entities[pid]['kind']!='ice_segment'or len(rr)<3:continue
            q=np.array([r['uv']for r in rr]);_,_,v=np.linalg.svd(q-q.mean(0),full_matrices=False);error=np.abs((q-q.mean(0))@v[1])
            line_audit.append(dict(pts=pt,path_id=pid,readouts=len(rr),straight_material_max_normal_deviation_px=float(error.max()),p95_px=float(np.percentile(error,95))))
        for name,f in read(OUT/f'evidence/baseline_ice/{pt}.json')['features'].items():
            if f['native_path_id']not in by_path:missing[name].append(pt)
    if visibility_failures:raise ValueError('Formal readings intersect actual source exclusion: '+str(visibility_failures[:3]))
    statuses=np.load(folder/'native_evaluation.npz')['status']
    if len(statuses)!=len(z['pts']):raise ValueError('Native did not evaluate every row')
    if np.any(statuses):raise ValueError('Native initialization has invalid rows: '+str(collections.Counter(statuses.tolist())))
    intervals=[]
    for name,pts in missing.items():
        for group in np.split(np.array(pts),np.flatnonzero(np.diff(pts)!=STEP)+1):
            intervals.append(dict(feature=name,first_pts=int(group[0]),last_pts=int(group[-1]),source_seconds=[float(group[0]/12800),float(group[-1]/12800)],exposures=len(group),
                reason='historical_candidate_exists_but_current_whole_material_or_identity_not_recovered',old_candidates_preserved=True,
                diagnostics=[str(OUT/'evidence/reread_lines'),str(OUT/'evidence/reread_circles')],not_an_accepted_absence_verdict=True))
    write(OUT/'evidence/remaining_coverage.json',dict(whole_shot_exposures=FRAME_COUNT,ice_historical_candidate_gaps=intervals,
        ad_exposures_without_qualified_rows=[pt for pt,rr in by.items()if not any('::'in r['native_path_id']for r in rr)],
        candidate_count_is_not_a_ground_truth_visibility_denominator=True,unresolved_material_not_filled_by_projection=True))
    write(OUT/'evidence/whole_line_audit.json',dict(rows=line_audit,coordinates_not_straightened=True,material_fit_is_not_an_independent_accuracy_certificate=True))
    summary=dict(frames=FRAME_COUNT,rows=len(z['pts']),paths=len(m['path_ids']),source_refs_verified=len(refs),native_invalid_rows=int(np.count_nonzero(statuses)),
        visibility_exclusion_intersections=len(visibility_failures),origins=dict(counts),correlated_groups=len(groups),
        native_authority_unchanged=True,original_UV_preserved_or_new_actual_RGB_readout=True,integer_PTS_verified=True,human_acceptance=False,
        historical_material_gap_intervals=len(intervals),numerical_ingestion_is_not_visual_acceptance=True)
    write(OUT/'verification.json',summary);verify_authority();emit('verified',**summary)


def review(video=False):
    import package_s02_repaired as package
    from fractions import Fraction
    import av
    from rinkspace.review.render import source_frames,validate_video
    package.OUT=OUT;package.old.OUT=OUT;cv2.setNumThreads(2);r=package.Review();dest=OUT/'review';dest.mkdir(exist_ok=True)
    pts=np.arange(FIRST,LAST+1,STEP);samples=list(pts[::30])+[LAST]
    if not video:
        thumbnails=[]
        for pt in samples:
            canvas,ledger=r.frame(int(pt),cv2.imread(r.vis[int(pt)]['image']));cv2.imwrite(str(dest/f'{pt}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,94])
            thumbs=cv2.resize(canvas[:1080,:1920],(640,360));cv2.putText(thumbs,f'{pt/12800:.2f}s PTS {pt}',(7,350),0,.6,(0,0,0),3);cv2.putText(thumbs,f'{pt/12800:.2f}s PTS {pt}',(7,350),0,.6,(255,255,255),1);thumbnails.append(thumbs)
        while len(thumbnails)%3:thumbnails.append(np.zeros_like(thumbnails[0]))
        cv2.imwrite(str(dest/'whole_shot_contact.jpg'),np.vstack([np.hstack(thumbnails[i:i+3])for i in range(0,len(thumbnails),3)]))
        emit('review_samples',images=len(samples));return
    path=dest/(SHOT_ID.upper()+'_points_connected.mp4');tmp=path.with_suffix('.partial.mp4');displayed=0;start=time.monotonic()
    with gzip.open(dest/'frame_index.jsonl.gz','wt',encoding='utf8')as ledger,av.open(str(tmp),'w')as container:
        stream=container.add_stream('libx264',rate=50);stream.width=2400;stream.height=1160;stream.pix_fmt='yuv420p';stream.time_base=Fraction(1,12800);stream.options={'crf':'17','preset':'fast'}
        for i,(pt,im)in enumerate(source_frames(Path(r.mapping['source_video']),dict(time_base_num=1,time_base_den=12800,width=1920,height=1080),pts)):
            canvas,links=r.frame(pt,im);ledger.write(json.dumps(dict(pts=pt,entities=links),ensure_ascii=False)+'\n');displayed+=len(r.by[pt])
            frame=av.VideoFrame.from_ndarray(canvas,format='bgr24');frame.pts=pt-FIRST;frame.time_base=Fraction(1,12800)
            for packet in stream.encode(frame):container.mux(packet)
            if i%100==0:emit('connected_video',frames=i+1,total=FRAME_COUNT,seconds=time.monotonic()-start)
        for packet in stream.encode():container.mux(packet)
    check=validate_video(tmp,pts,Fraction(1,12800));tmp.replace(path)
    if displayed!=len(r.z['pts']):raise ValueError('Review video omitted delivered points')
    write(dest/'video.json',dict(video=file_ref(path),validation=check,points_displayed=displayed,point_input_sha256=r.mapping['point_input_sha256'],
        projection_shown=False,all_official_exposures=True,source_seconds=[FIRST/12800,LAST/12800]));emit('video_ready',path=str(path),points=displayed)


def video():review(video=True)


def deliver():
    import zipfile
    verify_authority();result=read(OUT/'result.json');check=read(OUT/'verification.json');media=read(OUT/'review/video.json')
    target=OUT/'S06_native_points_20261027.zip';files=[]
    for folder in ['points','support','native_domain_hold']:
        files.extend(p for p in (OUT/folder).rglob('*')if p.is_file())
    files += [OUT/p for p in ['result.json','verification.json','native_integration.json','evidence/native_authority.json','evidence/current_native_geometry_state.npz',
        'evidence/history_refs.json','evidence/native_entities.json','evidence/source_inventory.json','evidence/historical_ice_dispositions.jsonl.gz',
        'evidence/historical_identity_qualification.jsonl.gz','evidence/later_identity_revisions.json','evidence/ad_physical_instance_review.json',
        'evidence/rejected_Ad2_instance_bindings.jsonl.gz','evidence/source_family_repair_comparison.json',
        'ads/source_review/S06_material_source_seeds.json','ads/source_review/S06_material_source_seed_failures.json',
        'ads/source_review/material_completion/scope.json','ads/source_review/material_completion/sampling_report.json',
        'ads/source_review/physical_identity_repair/scope.json','ads/source_review/physical_identity_repair/sampling_report.json',
        'evidence/whole_line_audit.json','evidence/remaining_coverage.json','evidence/final_source_dispositions.jsonl.gz','review/video.json','review/frame_index.jsonl.gz']]
    manifest=dict(schema='rinkspace.native_source_dataset_package.v1',shot_id='s06',point_folder='points',
        source_seconds=result['source_seconds'],official_exposures=486,rows=result['rows'],native_paths=result['native_paths'],
        current_native_domain_held_rows=result.get('native_domain_held_rows',0),native_geometry_unchanged=True,human_acceptance=False,
        source_video_and_current_native_dependencies_are_external=True,
        files=[dict(archive=str(p.relative_to(OUT)).replace('\\','/'),**file_ref(p))for p in files])
    with zipfile.ZipFile(target.with_suffix('.partial.zip'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3)as archive:
        for file in files:archive.write(file,str(file.relative_to(OUT)))
        archive.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    target.with_suffix('.partial.zip').replace(target)
    with zipfile.ZipFile(target)as archive:
        if archive.testzip()is not None:raise ValueError('S06 archive CRC failed')
        if hashlib.sha256(archive.read('points/point_observations.npz')).hexdigest()!=read(OUT/'points/mapping.json')['point_input_sha256']:
            raise ValueError('Packed native input differs from evaluated source input')
    index=PROG/'source_datasets/native_points_s06_20261027/result.json'
    if len(str(index.resolve()).encode('utf-16-le'))//2>=260:raise ValueError('OneDrive path too long')
    index.parent.mkdir(exist_ok=True)
    write(index,dict(shot_id='s06',source_seconds=result['source_seconds'],official_exposures=486,point_input=file_ref(OUT/'points/point_observations.npz'),
        mapping=file_ref(OUT/'points/mapping.json'),records=file_ref(OUT/'points/point_records.jsonl.gz'),archive=file_ref(target),video=media['video'],
        verification=file_ref(OUT/'verification.json'),coverage=file_ref(OUT/'evidence/remaining_coverage.json'),rows=result['rows'],native_paths=result['native_paths'],
        current_native_geometry_unchanged=True,human_acceptance=False))
    emit('delivered',archive=str(target),video=media['video']['path'],rows=result['rows'],paths=result['native_paths'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','inventory','baseline','navigation','ads_prepare','ads_anchors','ads_tracks','ads_complete','ads_measure','ads_qualify','ad_identity_reread','ad_material_completion','ice_reread','line_reference_reread','circle_gap_reread','line_gap_reread','ad_records','ad_borders','assemble','integrate','verify','review','video','deliver']);args=parser.parse_args();configure();globals()[args.stage]()
