"""S03/S05 source recovery using the shared native material readers.

Source clocks, masks, revisions and physical advertisement instances are scoped
to each shot. Native geometry and neighbouring fit inputs are read-only.
"""
from pathlib import Path
import argparse,collections,gzip,hashlib,json,shutil,sqlite3,time
import re
import cv2,numpy as np
import points_s06 as common
from ice_points_s04 import read,write,rows
from rinkspace.vision_native import NativeSourceGeometry,file_ref

ROOT=common.ROOT;PROG=common.PROG;DATA=common.DATA;REC=common.REC;SCENE=common.SCENE
PEER=DATA/'runs/native_joint_s01_s02_s04_s06_20261028'


def configure(shot):
    global SHOT,OUT,FIRST,LAST,COUNT,PRINT_INDEX,MASK_INDEX,ICE,PACKETS,CAMERA,BOARD
    SHOT=shot;scope=read(PROG/'rebuild_scope.json');spec=next(s for s in scope['shots']if s['id']==shot)
    clock=spec['processing'];FIRST=clock['first_pts'];LAST=clock['last_pts'];COUNT=clock['native_frame_count']
    OUT=DATA/f'runs/native_points_{shot}_20261028'
    ICE=REC/('dense_ice_v1/canonical_v3/s03/rows.jsonl.gz'if shot=='s03'else'dense56_ice_v1/canonical/s05/rows.jsonl.gz')
    PRINT_INDEX=OUT/'evidence/historical_print_index.jsonl.gz'
    MASK_INDEX=None if shot=='s03'else REC/'dense56_v1/shoe_fix/effective_v2/s05_index.jsonl.gz'
    PACKETS=PROG/f'source_datasets/shot{int(shot[1:])}_20261014/frames'
    CAMERA=PROG/('ad1_cameras/s03.json'if shot=='s03'else'ad_cameras/s05.json')
    BOARD=REC/('dense_board_v1/qualified_frames'if shot=='s03'else'dense56_board_v1/material_v5/qualified_frames')
    for key in ['OUT','FIRST','LAST','ICE','PRINT_INDEX','PACKETS','CAMERA','BOARD']:
        setattr(common,key,globals()[key])
    common.SHOT_ID=shot;common.FRAME_COUNT=COUNT;common.configure()
    return scope,spec


def prepare():
    import av
    cv2.setNumThreads(2)
    scope=read(PROG/'rebuild_scope.json');spec=next(s for s in scope['shots']if s['id']==SHOT)
    for name in ['evidence','source_frames','ad_exclusions','evidence/native','evidence/surface']:(OUT/name).mkdir(parents=True,exist_ok=True)
    video=Path(scope['source']['registered_path']);authfile=OUT/'evidence/native_authority.json'
    if not authfile.exists():
        manifest=read(PEER/'joint_manifest.json');origin=PEER/'batch5_baseline_probe/accepted_pixels_latest.npz'
        before=file_ref(origin);state=dict(np.load(origin,allow_pickle=False))
        if file_ref(origin)!=before:raise ValueError('Native snapshot changed during read')
        if str(state['point_input_sha256'])!=manifest['point_input_sha256']:raise ValueError('Snapshot input binding differs; choose an accepted snapshot explicitly')
        native=Path(manifest['native_library']);original_dll=file_ref(native);frozen=OUT/'evidence/native/rinkspace_core.dll'
        shutil.copy2(native,frozen)
        if file_ref(frozen)['sha256']!=original_dll['sha256']:raise ValueError('Native library changed during copy')
        geo=NativeSourceGeometry(SCENE,str(frozen));target=OUT/'evidence/current_native_geometry_state.npz'
        np.savez_compressed(target,scene_values=state['scene_values'],point_input_sha256=state['point_input_sha256'])
        refs={r['path']:r for r in geo.refs}
        for p in [CAMERA,CAMERA.parent/read(CAMERA)['controls_ref'],frozen]:r=file_ref(p);refs[r['path']]=r
        write(authfile,dict(files=list(refs.values()),scene_path=str(SCENE),native_library=str(frozen),native_library_origin=original_dll,
            scene_state=file_ref(target),state_origin=before,manifest_origin=file_ref(PEER/'joint_manifest.json'),camera=file_ref(CAMERA),
            source=scope['source'],shot=spec,current_native_path_count=len(geo.current_paths),geometry_changed=False,camera_seed_only=True))
        write(OUT/'evidence/native_entities.json',dict(entities=[dict(v,native_path_id=v['id'])for v in geo.current_paths.values()]))
    common.verify_authority()
    original_index=REC/('dense_print_v1/final_index_anchor.jsonl.gz'if SHOT=='s03'else'dense56_v1/print_requal_v2/s05_leg_v4/index.jsonl.gz')
    prints={r['native_pts']:r for _,r in rows(original_index)if FIRST<=r['native_pts']<=LAST}
    with gzip.open(PRINT_INDEX,'wt',encoding='utf8')as f:
        for pt in sorted(prints):f.write(json.dumps(prints[pt])+'\n')
    masks={}if MASK_INDEX is None else{r['native_pts']:r for _,r in rows(MASK_INDEX)}
    blank=OUT/'evidence/surface/no_near_branch.png';cv2.imwrite(str(blank),np.zeros((1080,1920),np.uint8));blankref=dict(file=str(blank),sha256=file_ref(blank)['sha256'])
    surface=OUT/'evidence/surface/no_added_readings.json'
    write(surface,dict(new_source_rows=[],reason='no independently measured near-board branch supplied; existing board source retained',human_acceptance=False))
    vis=[];ads=[];started=time.monotonic();overview=[]
    sample=set(np.linspace(0,COUNT-1,12).round().astype(int))
    with av.open(str(video))as container:
        stream=container.streams.video[0];container.seek(FIRST,stream=stream)
        for frame in container.decode(stream):
            pt=frame.pts
            if pt<FIRST:continue
            if pt>LAST:break
            im=frame.to_ndarray(format='bgr24');digest=hashlib.sha256(im.tobytes()).hexdigest();p=prints[pt]
            if digest!=p['decoded_native_BGR_sha256']:raise ValueError('Historical source exposure differs')
            m=masks.get(pt);maskref=dict(file=m['mask_file'],sha256=m['mask_sha256'])if m else p['mask_ref']
            if file_ref(maskref['file'])['sha256']!=maskref['sha256']:raise ValueError('Historical source mask differs')
            mask=cv2.imread(maskref['file'],0);unknown=m.get('source_unknown_exclusion_mask_ref')if m else p.get('unknown_exclusion_ref')
            components=[maskref]
            if unknown:
                if file_ref(unknown['file'])['sha256']!=unknown['sha256']:raise ValueError('Historical unknown mask differs')
                mask=np.maximum(mask,cv2.imread(unknown['file'],0));components.append(unknown)
            combined=OUT/f'ad_exclusions/{pt}.png';cv2.imwrite(str(combined),mask)
            image=OUT/f'source_frames/{pt}.png'
            if not image.exists():cv2.imwrite(str(image),im,[cv2.IMWRITE_PNG_COMPRESSION,1])
            elif hashlib.sha256(cv2.imread(str(image)).tobytes()).hexdigest()!=digest:raise ValueError('Cached source differs')
            v=dict(pts=pt,shot_id=SHOT,image=str(image),source_BGR_sha256=digest,source_video_sha256=scope['source']['sha256'],
                mask_path=str(combined),mask_sha256=file_ref(combined)['sha256'],mask_components=components,unknown_ref=unknown,
                surface_ref=file_ref(surface),near_outside_ref=blankref,ice_visible_ref=None,board_body_visible_ref=None,
                board_source=str(BOARD/f'{pt}.json'),historical_print_frame=p['file'],historical_print_frame_sha256=p['sha256'])
            vis.append(v);ads.append(v)
            if len(vis)-1 in sample:
                thumb=cv2.resize(im,(640,360));cv2.putText(thumb,f'{SHOT} {pt/12800:.2f}s PTS {pt}',(8,348),0,.6,(0,0,0),3);cv2.putText(thumb,f'{SHOT} {pt/12800:.2f}s PTS {pt}',(8,348),0,.6,(255,255,255),1);overview.append(thumb)
            if len(vis)%100==0:common.emit('official_source',shot=SHOT,frames=len(vis),total=COUNT,seconds=time.monotonic()-started)
    if [v['pts']for v in vis]!=list(range(FIRST,LAST+1,256)):raise ValueError('Official exposure missing')
    header=dict(source_video=str(video),source_video_sha256=scope['source']['sha256'],source_time_base='1/12800',shot_id=SHOT)
    for name,frames in [('visibility',vis),('ad_visibility',ads)]:write(OUT/f'evidence/{name}.json',dict(header,frames=frames))
    write(OUT/'evidence/history_refs.json',dict(refs=[file_ref(p)for p in [original_index,MASK_INDEX]if p is not None],
        original_history_unchanged=True,source_seconds=[FIRST/12800,LAST/12800],human_acceptance_not_inferred=True))
    cv2.imwrite(str(OUT/'evidence/source_overview.jpg'),np.vstack([np.hstack(overview[i:i+3])for i in range(0,12,3)]))
    common.verify_authority();common.emit('source_ready',shot=SHOT,frames=len(vis))


def inventory():
    kinds=collections.Counter();features=collections.Counter();frames=collections.Counter();ids=set()
    for _,r in rows(ICE):kinds[r['source_kind']]+=1;features[r['feature_id']]+=1;frames[r['native_pts']]+=1;ids.add(r['source_row_id'])
    refs=[file_ref(ICE)];corrections={};withdrawals={};manual=[]
    for name in ['exclusions.jsonl.gz','omissions.jsonl.gz']:
        p=ICE.parent/name
        if not p.exists():continue
        refs.append(file_ref(p))
        for i,r in rows(p):
            if r.get('source_row_id')in ids:corrections[r['source_row_id']]=dict(record=r,source=str(p),jsonl_line_0based=i)
    index=DATA/'runs/okada_s02_s09_observations_20261026/source_index.sqlite'
    if index.exists():
        refs.append(file_ref(index))
        with sqlite3.connect(index.as_uri()+'?mode=ro',uri=True)as db:
            for key,payload in db.execute('select id,payload from qualification where shot=?',(SHOT,)):withdrawals[key]=json.loads(payload)
    updates={}
    for kind in ['source_commit','positive_source_commit']:
        p=common.OLDJOINT/kind/SHOT/'source_request.json'
        if not p.exists():continue
        refs.append(file_ref(p))
        for frame in read(p)['changes']:
            for sid,change in frame.get('update',{}).items():updates.setdefault(sid,{}).update(change)
    selected=REC/f'dense56_v1/parts_v2/{SHOT}/selected_rows.jsonl.gz'
    qualifications=[]
    if selected.exists():
        refs.append(file_ref(selected))
        for i,r in rows(selected):
            change=updates.get(r['selected_row_id'],{})
            if change:
                qualifications.append(dict(source_row_id=r.get('source_point_id'),selected_row_id=r['selected_row_id'],pts=r['native_pts'],uv=r['uv'],change=change))
                if change.get('source_fit_eligible')is False and r.get('source_point_id'):withdrawals[r['source_point_id']]=dict(change,reason=change.get('qualification_reason_code'))
            if r.get('user_confirmed'):manual.append(dict(source_line=i,record=r))
    with gzip.open(OUT/'evidence/historical_identity_qualification.jsonl.gz','wt',encoding='utf8')as f:
        for r in qualifications:f.write(json.dumps(r)+'\n')
    revision=[]
    for p in sorted(common.OLDJOINT.glob('native_source_identity_1023*/revision.json')):revision.append(dict(ref=file_ref(p),revision=read(p)))
    write(OUT/'evidence/later_identity_revisions.json',dict(revisions=revision,source_only=True))
    write(OUT/'evidence/source_inventory.json',dict(shot=SHOT,full_exposures=COUNT,raw_ice_rows=sum(kinds.values()),source_kinds=dict(kinds),features=dict(features),
        frame_counts=dict(frames),historical_corrections=corrections,withdrawals=withdrawals,manual_rows=manual,source_files=refs,
        raw_source_files_unchanged=True,human_acceptance_not_inferred=True))
    common.emit('inventory_ready',shot=SHOT,frames=len(frames),raw_ice=sum(kinds.values()),features=len(features),corrections=len(corrections),withdrawals=len(withdrawals))


def ice_reread():
    grid=common.shared.NativeGrid();photo=common.SourceGuides(grid)
    common.shared.ice_reread(board_dir=BOARD,source_prefix=SHOT,photographic=photo,source_finite_extents=True)


class CompleteSourceGuides(common.SourceGuides):
    """A thinned current reading must not truncate a fuller photographed line."""
    def __init__(self,grid):
        super().__init__(grid)
        self.photographs={name:[g for g in guides if Path(g['path']).parent==PACKETS]
            for name,guides in self.base.guides.items()}
        self.extended=[]

    def at(self,name,pt):
        nearest=super().at(name,pt)
        if nearest is None or name not in self.paths:return nearest
        old,h=nearest;q=cv2.perspectiveTransform(np.vstack(old['spans'])[None],h)[0]
        origin,component=self.transforms[pt]
        candidates=[g for g in self.photographs.get(name,[])if self.transforms[g['pts']][1]==component]
        if not candidates:return nearest
        source=min(candidates,key=lambda g:abs(g['pts']-pt));H=origin@np.linalg.inv(self.transforms[source['pts']][0])
        points=cv2.perspectiveTransform(np.vstack(source['spans'])[None],H)[0]
        if not np.isfinite(points).all() or np.linalg.norm(np.ptp(points,axis=0))<=np.linalg.norm(np.ptp(q,axis=0)):return nearest
        # Reuse the established source support operator and its normal search
        # tolerance. Endpoints stay within actual photographed finite spans.
        if not self.base.support(q,source['spans'],H,normal_tolerance_px=28.).all():return nearest
        self.extended.append(dict(pts=pt,feature=name,thinned_source_pts=old['pts'],complete_source_pts=source['pts'],
            complete_source=source['path'],finite_photographic_bounds_only=True))
        return source,H


def line_completion(requested=None):
    from reread_s04_lines import NativeLineReader
    from ice_points_s04 import source_domain
    cv2.setNumThreads(2);grid=common.shared.NativeGrid();photo=CompleteSourceGuides(grid)
    def new_reader():
        value=NativeLineReader(grid,photographic=photo,motion_path=OUT/'evidence/source_navigation.json',source_prefix=SHOT+'_finite_complete',
            source_cross_line_guides=True,source_parallel_references=True,source_finite_extents=True,recover_partial_blue=True)
        for path in value.paths:path.update(_source_query_spacing_px=35,_separate_profile_support=True)
        return value
    reader=new_reader();last_read=None
    vis=read(OUT/'evidence/visibility.json');dest=OUT/'evidence/reread_line_references';dest.mkdir(exist_ok=True);summary=[]
    for v in vis['frames']:
        pt=v['pts']
        if requested is not None and pt not in requested:continue
        if last_read is not None and pt!=last_read+256:reader=new_reader()
        last_read=pt
        im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)>0
        graphics,border,_,_=source_domain(im,mask.astype('uint8')*255,read(v['board_source']))
        excluded=mask|graphics|(np.arange(1080)[:,None]<=border[None])
        features,diagnostic=reader.read(pt,im,excluded,vis['source_video_sha256'])
        previous=OUT/f'evidence/reread_lines/{pt}.json';old=read(previous)['features'];comparison={}
        for name in ['ice.B1:segment0','ice.B2:segment0']:
            before=old.get(name,{}).get('rows',[]);after=features.get(name,{}).get('rows',[])
            comparison[name]=dict(before=len(before),after=len(after))
        write(dest/f'{pt}.json',dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features=features,diagnostics=diagnostic,
            previous_source=file_ref(previous),source_parallel_reference_ids=sorted(reader.source_parallel_reference_ids),comparison=comparison,
            source_finite_extents=True,actual_source_uv_only=True,human_acceptance=False))
        summary.append(dict(pts=pt,comparison=comparison))
        if len(summary)%100==0 or requested is not None:common.emit('finite_line_completion',shot=SHOT,frames=len(summary),pts=pt,comparison=comparison)
    write(dest/'coverage.json',dict(frames=summary,complete_photographic_guides=photo.extended));common.verify_authority()


def crossline_completion():
    from reread_s04_lines import NativeLineReader
    from ice_points_s04 import source_domain
    cv2.setNumThreads(2);grid=common.shared.NativeGrid();photo=CompleteSourceGuides(grid)
    reader=NativeLineReader(grid,photographic=photo,motion_path=OUT/'evidence/source_navigation.json',source_prefix=SHOT+'_photo_direction',
        source_cross_line_guides=True,source_parallel_references=True,source_finite_extents=True,recover_partial_blue=True,photographic_cross_direction=True)
    reader.paths=[p for p in reader.paths if p['feature_id']in('B1','B2','C')]
    for p in reader.paths:p.update(_source_query_spacing_px=35,_separate_profile_support=True)
    dest=OUT/'evidence/reread_cross_lines';dest.mkdir(exist_ok=True);report=[]
    for v in read(OUT/'evidence/visibility.json')['frames']:
        pt=v['pts'];im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0);graphics,border,_,_=source_domain(im,mask,read(v['board_source']))
        features,diagnostic=reader.read(pt,im,(mask>0)|graphics|(np.arange(1080)[:,None]<=border[None]),v['source_video_sha256'])
        source=OUT/f'evidence/reread_line_references/{pt}.json';previous=read(source)['features'];changes={}
        for p in reader.paths:
            name=p['label'];features.setdefault(name,dict(native_path_id=p['id'],rows=[]))
            changes[name]=dict(before=len(previous.get(name,{}).get('rows',[])),after=len(features[name]['rows']))
        write(dest/f'{pt}.json',dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features=features,diagnostics=diagnostic,
            previous_source=file_ref(source),comparison=changes,current_native_identity_unchanged=True,actual_source_direction_and_RGB=True,human_acceptance=False))
        report.append(dict(pts=pt,comparison=changes))
        if len(report)%100==0:common.emit('photographic_crossline_RGB',shot=SHOT,frames=len(report),total=COUNT)
    write(dest/'coverage.json',dict(frames=report));common.verify_authority();crossline_choose()


def line_support_reread():
    from reread_s04_lines import NativeLineReader,EntryProfiles,NativeProfiles,reread_final_support
    from ice_points_s04 import source_domain
    cv2.setNumThreads(2);grid=common.shared.NativeGrid();reader=NativeLineReader(grid)
    paths={p['label']:p for p in reader.paths};dest=OUT/'evidence/reread_line_final_support';dest.mkdir(exist_ok=True)
    report=[]
    for v in read(OUT/'evidence/visibility.json')['frames']:
        pt=v['pts'];source=OUT/f'evidence/reread_line_references/{pt}.json';packet=read(source);features=packet['features']
        bad={}
        for name,f in features.items():
            q=np.asarray([r['uv']for r in f['rows']])
            if len(q)<3:continue
            _,_,axes=np.linalg.svd(q-q.mean(0),full_matrices=False)
            if np.max(np.abs((q-q.mean(0))@axes[1]))>=4.:bad[name]=f
        existing=dest/f'{pt}.json'
        if existing.exists():
            history=OUT/'evidence/superseded_line_support';history.mkdir(exist_ok=True)
            if not(history/existing.name).exists():shutil.copy2(existing,history/existing.name)
        if not bad:
            if existing.exists():write(existing,dict(packet,previous_source=file_ref(source),changed={},unresolved=[],repair_needed=False))
            continue
        im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0);graphics,border,_,_=source_domain(im,mask,read(v['board_source']))
        excluded=(mask>0)|graphics|(np.arange(1080)[:,None]<=border[None]);changed={};unresolved=[]
        for name,f in bad.items():
            old=dict(f['navigation']['source_line_fit'],direct_readouts=f['rows']);path=paths[name]
            cls=EntryProfiles if path['feature_id']in('B1','B2')else NativeProfiles
            native=cls(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),dict(source_video_sha256=v['source_video_sha256'],native_pts=pt,time_base='1/12800'),excluded)
            feature=reread_final_support(path,old,native)
            if feature is None:unresolved.append(name);continue
            fit=feature['source_line_fit'];rows=[dict(r,native_path_id=path['id'],feature_id=name,native_pts=pt,
                source_row_id=f'{SHOT}_final_support_RGB/{pt}/{path["curve_id"]}/{i}')for i,r in enumerate(fit['direct_readouts'])]
            features[name]=dict(f,rows=rows,navigation=dict(f['navigation'],source_line_fit={k:v for k,v in fit.items()if k!='direct_readouts'},
                final_support_reread=feature['extraction_diagnostic']['final_direction_prior_release_reread']))
            changed[name]=dict(before=len(f['rows']),after=len(rows),before_max=old['residual_max_px'],after_max=fit['residual_max_px'])
        write(dest/f'{pt}.json',dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features=features,previous_source=file_ref(source),
            changed=changed,unresolved=unresolved,actual_source_uv_only=True,human_acceptance=False))
        report.append(dict(pts=pt,changed=changed,unresolved=unresolved))
        if len(report)%50==0:common.emit('final_line_support_RGB',shot=SHOT,frames=len(report),pts=pt)
    write(dest/'coverage.json',dict(frames=report));common.verify_authority()
    common.emit('final_line_support_complete',shot=SHOT,frames=len(report),unresolved=sum(len(r['unresolved'])for r in report))


def crossline_choose():
    """Keep the strongest actual full-line read, not a preferred camera prior."""
    dest=OUT/'evidence/reread_cross_lines';report=[]
    def support_score(feature):
        rows=feature.get('rows',[])
        if len(rows)<2:return float(len(rows))
        q=np.asarray([r['uv']for r in rows]);_,_,axes=np.linalg.svd(q-q.mean(0),full_matrices=False)
        # Same source query spacing in both reads. Reuse the whole-line
        # coverage term; this rank is never a solver weight or new UV.
        return len(rows)+float(np.ptp((q-q.mean(0))@axes[0]))/500.
    for v in read(OUT/'evidence/visibility.json')['frames']:
        pt=v['pts'];file=dest/f'{pt}.json';packet=read(file)
        candidate=packet.get('photographic_direction_candidates',packet['features']);previous={};refs=[]
        for folder in ['reread_line_references','reread_line_final_support','reread_gap_lines']:
            source=OUT/f'evidence/{folder}/{pt}.json'
            if source.exists():previous.update(read(source)['features']);refs.append(file_ref(source))
        selected={};decision={}
        for name,f in candidate.items():
            old=previous.get(name,dict(native_path_id=f['native_path_id'],rows=[]));a=support_score(old);b=support_score(f)
            use_new=b>a;selected[name]=f if use_new else old
            decision[name]=dict(previous_actual_rows=len(old['rows']),candidate_actual_rows=len(f['rows']),
                previous_support_score=a,candidate_support_score=b,selected='photographic_direction_RGB'if use_new else'preserved_existing_actual_material',
                no_coordinate_or_identity_change_to_selected_readout=True)
        packet.update(features=selected,photographic_direction_candidates=candidate,actual_material_selection=decision,previous_actual_sources=refs)
        write(file,packet);report.append(dict(pts=pt,selection=decision))
    write(dest/'selection.json',dict(frames=report,all_candidates_and_original_sources_retained=True));common.verify_authority()


def source_line_frame(points):
    q=np.asarray(points,float)
    if len(q)<2:return None
    centre=q.mean(0);_,_,axes=np.linalg.svd(q-centre,full_matrices=False)
    tangent=axes[0]
    if tangent[0]<0:tangent=-tangent
    normal=np.array([-tangent[1],tangent[0]])
    return centre,tangent,normal,float(np.ptp((q-centre)@tangent))


def crossline_source_side(rows, boundary, side):
    """Keep actual blue readouts on their native side of actual red C.

    This is an entity-order test, not a residual or confidence threshold. A
    blue line cannot cross the red centre line to become the other blue line.
    """
    origin,_,normal,_=boundary
    return [r for r in rows if side*float((np.asarray(r['uv'])-origin)@normal)>0.]


def material_identity_repair(requested=None):
    """Resolve current-frame material identity before ranking read coverage."""
    from reread_s04_lines import NativeLineReader,Extractor,EntryProfiles,NativeProfiles
    from ice_points_s04 import source_domain
    cv2.setNumThreads(2);grid=common.shared.NativeGrid();photo=CompleteSourceGuides(grid)
    reader=NativeLineReader(grid,photographic=photo,motion_path=OUT/'evidence/source_navigation.json',source_prefix=SHOT+'_material_identity',
        source_cross_line_guides=True,source_parallel_references=True,source_finite_extents=True,recover_partial_blue=True)
    paths={p['label']:p for p in reader.paths};dest=OUT/'evidence/reread_material_identity';dest.mkdir(exist_ok=True)
    report=[];held=[]
    for v in read(OUT/'evidence/visibility.json')['frames']:
        pt=v['pts']
        if requested is not None and pt not in requested:continue
        features={};refs=[]
        for folder in ['reread_line_references','reread_line_final_support','reread_gap_lines','reread_cross_lines']:
            source=OUT/f'evidence/{folder}/{pt}.json'
            if source.exists():features.update(read(source)['features']);refs.append(file_ref(source))
        changes={};decisions={};reassigned=collections.defaultdict(list);cross=read(OUT/f'evidence/reread_cross_lines/{pt}.json')
        centre=features.get('ice.C:segment0',{}).get('rows',[])
        boundary=source_line_frame([r['uv']for r in centre])
        if boundary is not None and boundary[3]<180:boundary=None
        projections=grid.project(pt);im=None;excluded=None
        def native_level(label):
            z=projections[paths[label]['id']];q=z['uv'];valid=z['valid']&np.isfinite(q).all(1)&(q[:,0]>=0)&(q[:,0]<=1920)
            q=q[valid]
            if len(q)<2:return None
            frame=source_line_frame(q);o,t,_,_=frame
            return o+(960-o[0])/t[0]*t if abs(t[0])>1e-6 else None
        nc=native_level('ice.C:segment0');nb=[native_level(n)for n in ['ice.B1:segment0','ice.B2:segment0']]
        if boundary is not None and nc is not None and all(q is not None for q in nb):
            actual=boundary[0]+(960-boundary[0][0])/boundary[1][0]*boundary[1]
            # A short red circle lip must not redefine the centre stripe. The
            # model is used only to choose between named search regions.
            if abs(actual[1]-nc[1])>min(abs(actual[1]-q[1])for q in nb):boundary=None
        boundary_origin='current_actual_red_C_material'
        if boundary is None:
            z=projections[paths['ice.C:segment0']['id']];q=z['uv'];valid=z['valid']&np.isfinite(q).all(1)&(q[:,0]>=0)&(q[:,0]<=1920)
            boundary=source_line_frame(q[valid]);boundary_origin='current_native_C_search_partition_only_no_C_observation_created'
        for name,f in features.items():
            if name not in paths:continue
            path=paths[name];guide=photo.at(name,pt)
            if not f.get('rows')and path['feature_id']not in('B1','B2'):continue
            # Global occurrence of a name does not identify a current exposure.
            # Rejected historical directions must not be revived by a model-only
            # image search and relabelled as measured source identity.
            if guide is None and path['feature_id']not in('B1','B2','C'):
                changes[name]=dict(f,rows=[])
                decisions[name]=dict(reason='no_qualified_current_photographic_identity_in_registration_component',previous=len(f['rows']),after=0)
                held.extend(dict(pts=pt,feature=name,reason=decisions[name]['reason'],actual_readout=r)for r in f['rows'])
                continue
            if path['feature_id']not in('B1','B2')or boundary is None:continue
            nl=native_level(name);nc=native_level('ice.C:segment0')
            if nl is None or nc is None:continue
            side=float(np.sign((nl-nc)@boundary[2]))
            if not side:continue
            actual=crossline_source_side(f['rows'],boundary,side)
            if len(actual)==len(f['rows'])and actual:continue
            candidates=[dict(f,rows=actual)]
            alternate=cross.get('photographic_direction_candidates',{}).get(name)
            if alternate:candidates.append(dict(alternate,rows=crossline_source_side(alternate['rows'],boundary,side)))
            # Re-read the actual blue paint in its physical half-plane. Source
            # and native lines guide only the search; every returned UV still
            # comes from current-frame material interfaces.
            if im is None:
                im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)
                graphics,border,_,_=source_domain(im,mask,read(v['board_source']))
                excluded=(mask>0)|graphics|(np.arange(1080)[:,None]<=border[None])
            yy,xx=np.mgrid[:1080,:1920]
            region=side*((xx-boundary[0][0])*boundary[2][0]+(yy-boundary[0][1])*boundary[2][1])<=0
            binding=dict(source_video_sha256=v['source_video_sha256'],native_pts=pt,time_base='1/12800')
            rgb=NativeProfiles(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),binding,excluded|region)
            best=max(candidates,key=lambda a:len(a['rows']))
            if len(best['rows'])>=3:
                q=np.array([r['uv']for r in best['rows']]);nav=source_line_frame(q);s=(q-nav[0])@nav[1]
                search=nav[0]+np.linspace(s.min(),s.max(),257)[:,None]*nav[1]
            else:
                q=projections[path['id']]['uv'];valid=np.isfinite(q).all(1)&(q[:,0]>=0)&(q[:,0]<=1920)
                search=q[valid]+(boundary[0]+(960-boundary[0][0])/boundary[1][0]*boundary[1]-nc)
            if len(search)>=2:
                local=dict(path,_source_query_spacing_px=35,_separate_profile_support=True,
                    _wide_local_identity_reason='current_native_B1_C_B2_order_and_actual_C_material_separate_blue_branches')
                delta=search[-1]-search[0];slope=float(delta[1]/delta[0])if abs(delta[0])>1e-6 else None
                value,_,_=Extractor.line(None,local,search,rgb,family_slope=slope);fit=value.get('source_line_fit')
                if fit:
                    rr=[dict(r,native_path_id=path['id'],feature_id=name,native_pts=pt,
                        source_row_id=f'{SHOT}_material_identity_RGB/{pt}/{path["curve_id"]}/{i}')for i,r in enumerate(fit['direct_readouts'])]
                    candidates.append(dict(native_path_id=path['id'],rows=crossline_source_side(rr,boundary,side),
                        navigation=dict(actual_C_material_halfplane=True,current_native_order_only=True,projected_UV_not_emitted=True)))
            best=max(candidates,key=lambda a:len(a['rows']))
            fit=source_line_frame([r['uv']for r in best['rows']])
            if fit is not None:
                combined={r['source_row_id']:r for r in best['rows']}
                for candidate in candidates:
                    for r in candidate['rows']:
                        if abs(float((np.asarray(r['uv'])-fit[0])@fit[2]))<=4.:
                            combined.setdefault(r['source_row_id'],r)
                # Representative selection later retains spatial coverage and
                # one correlated family; repeated RGB reads are not weights.
                best=dict(best,rows=sorted(combined.values(),key=lambda r:float((np.asarray(r['uv'])-fit[0])@fit[1])))
            changes[name]=best
            decisions[name]=dict(reason='blue_candidate_crossed_actual_C_into_other_native_material_halfplane',previous=len(f['rows']),
                previously_on_correct_side=len(actual),after=len(best['rows']),native_side=side,actual_C_rows=len(centre),actual_C_span=boundary[3],
                search_partition_origin=boundary_origin)
            if not f['rows']:decisions[name]['reason']='missing_blue_material_reread_in_current_native_search_region'
            kept={r['source_row_id']for r in best['rows']}
            wrong=[r for r in f['rows']if r not in actual]
            other='ice.B2:segment0'if path['feature_id']=='B1'else'ice.B1:segment0'
            other_rows=crossline_source_side(features.get(other,{}).get('rows',[]),boundary,-side)
            reference=source_line_frame([r['uv']for r in other_rows])
            # Preserve a real blue band's original UV when the identity is the
            # other blue line. If that line has no current reading, independently
            # read its chromatic material first; a grey response alone is not
            # sufficient to change an entity's identity.
            if (reference is None or reference[3]<180)and len(wrong)>=3:
                nav=source_line_frame([r['uv']for r in wrong])
                if nav[3]>=180:
                    q=np.asarray([r['uv']for r in wrong]);s=(q-nav[0])@nav[1];search=nav[0]+np.linspace(s.min(),s.max(),257)[:,None]*nav[1]
                    other_profile=NativeProfiles(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),binding,excluded|~region)
                    target=dict(paths[other],_source_query_spacing_px=35,_separate_profile_support=True)
                    value,_,_=Extractor.line(None,target,search,other_profile,family_slope=float(nav[1][1]/nav[1][0]));fit=value.get('source_line_fit')
                    if fit:reference=source_line_frame([r['uv']for r in fit['direct_readouts']])
            rebound=set()
            if reference is not None and reference[3]>=180:
                for r in wrong:
                    if abs(float((np.asarray(r['uv'])-reference[0])@reference[2]))<=4.:
                        reassigned[other].append(dict(r,native_path_id=paths[other]['id'],feature_id=other,
                            source_identity_revision=dict(original_native_path_id=path['id'],original_feature_id=name,
                                reason='same_actual_blue_band_confirmed_on_opposite_side_of_C',original_UV_and_PTS_unchanged=True)))
                        rebound.add(r['source_row_id'])
            decisions[name]['actual_points_rebound_to_other_blue_line']=len(rebound)
            held.extend(dict(pts=pt,feature=name,reason=decisions[name]['reason'],actual_readout=r,
                rebound_to_native_path=paths[other]['id']if r['source_row_id']in rebound else None)for r in f['rows']if r['source_row_id']not in kept)
        for name,rr in reassigned.items():
            base=changes.get(name,features.get(name,dict(native_path_id=paths[name]['id'],rows=[])))
            combined={r['source_row_id']:r for r in base['rows']}
            combined.update({r['source_row_id']:r for r in rr})
            changes[name]=dict(base,rows=list(combined.values()))
        write(dest/f'{pt}.json',dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features=changes,decisions=decisions,
            original_actual_sources=refs,current_native_geometry_unchanged=True,model_or_flow_observations_generated=False,human_acceptance=False))
        report.append(dict(pts=pt,decisions=decisions))
        if len(report)%100==0 or requested is not None:common.emit('material_identity',shot=SHOT,frames=len(report),pts=pt,decisions=decisions)
    with gzip.open(dest/'previous_readouts.jsonl.gz','wt',encoding='utf8')as stream:
        for r in held:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    write(dest/'coverage.json',dict(frames=report,previous_actual_readouts_preserved=len(held),human_acceptance=False));common.verify_authority()


def temporal_line_repair(final_pass=False,adjacent_pass=False,closing_pass=False):
    """Reuse the source-identity filter with both adjacent actual line reads."""
    from reread_s04_lines import NativeLineReader,Extractor,NativeProfiles,EntryProfiles
    from ice_points_s04 import source_domain
    cv2.setNumThreads(2);grid=common.shared.NativeGrid();photo=CompleteSourceGuides(grid)
    reader=NativeLineReader(grid,photographic=photo,motion_path=OUT/'evidence/source_navigation.json')
    paths={p['label']:p for p in reader.paths};vis={f['pts']:f for f in read(OUT/'evidence/visibility.json')['frames']}
    pairs={p['source_pts']:p for p in read(OUT/'evidence/source_navigation.json')['pairs']}
    audit='pending_peer_temporal_audit'if adjacent_pass else'latest_peer_temporal_audit'
    requests=read(OUT/f'evidence/{audit}/diagnostic.json')['review_candidates']
    dest=OUT/('evidence/reread_temporal_closing'if closing_pass else'evidence/reread_temporal_adjacent'if adjacent_pass else'evidence/reread_temporal_final'if final_pass else'evidence/reread_temporal_identity');dest.mkdir(exist_ok=True);report=[];cache={};preserved=[]
    anchors=collections.defaultdict(dict)
    if final_pass and not adjacent_pass:
        # Anchor selection controls navigation only. A long but sparse false
        # branch must not outweigh a fully observed blue/red strip nearby.
        # Twelve real profiles at the existing 35 px spacing support the
        # already-used 400 px whole-material guide, without inventing pixels.
        for pt in vis:
            file=OUT/f'evidence/reread_continuous_crosslines/{pt}.json';packet=read(file)
            for name,feature in packet['features'].items():
                actual=[r for r in feature['rows']if r['source_row_id'].startswith(SHOT+'_continuous_crossline_')]
                q=np.asarray([r['uv']for r in actual]);frame=source_line_frame(q)
                if frame is None or len(q)<12 or frame[3]<400:continue
                if np.percentile(np.abs((q-frame[0])@frame[2]),90)>4:continue
                anchors[name][pt]=(q,file_ref(file))
        requests=[r for r in requests if r['label']not in anchors]
        requests.extend(dict(pts=pt,label=name)for name in anchors for pt in vis)
    def features(pt):
        if pt not in cache:
            values={}
            folders=['reread_line_references','reread_line_final_support','reread_gap_lines','reread_cross_lines','reread_material_identity','reread_red_identity']
            if final_pass:folders+=['reread_temporal_identity','reread_continuous_crosslines']
            if adjacent_pass:folders+=['reread_temporal_final']
            if closing_pass:folders+=['reread_temporal_adjacent','reread_crossline_family']
            for folder in folders:
                file=OUT/f'evidence/{folder}/{pt}.json'
                if file.exists():
                    for name,value in read(file)['features'].items():values[name]=(value,file_ref(file))
            cache[pt]=values
        return cache[pt]
    packets={}
    for request in requests:
        pt,name=request['pts'],request['label'];current,original=features(pt)[name]
        if not current['rows']and name not in anchors:continue
        neighbours=[];refs=[];errors=[]
        reference_pts=[pt-256,pt+256]
        if final_pass and name in anchors:
            component=photo.transforms[pt][1]
            same=[p for p in anchors[name]if photo.transforms[p][1]==component]
            earlier=[p for p in same if p<pt];later=[p for p in same if p>pt]
            if not earlier or not later:continue
            reference_pts=[max(earlier),min(later)]
        for previous in reference_pts:
            if previous not in vis:break
            if final_pass and name in anchors:
                actual,ref=anchors[name][previous]
            else:
                if name not in features(previous):break
                previous_rows,ref=features(previous)[name]
                if len(previous_rows['rows'])<3:break
                actual=np.asarray([r['uv']for r in previous_rows['rows']])
            H,component=photo.transforms[pt];old,old_component=photo.transforms[previous]
            if component!=old_component:break
            moved=cv2.perspectiveTransform(actual[None],H@np.linalg.inv(old))[0]
            frame=source_line_frame(moved)
            if frame[3]<60:break
            neighbours.append((frame,moved));refs.append(ref)
            links=[pairs[p]for p in range(min(pt,previous)+256,max(pt,previous)+1,256)]
            if any(p['status']!='candidate'or p['diagnostic_median_reference_px']>2 for p in links):break
            errors.append(2.*sum(p['diagnostic_median_reference_px']**2 for p in links)**.5)
        if len(neighbours)!=2:
            report.append(dict(pts=pt,feature=name,status='two_adjacent_material_supports_unavailable'));continue
        error=max(errors);tolerance=min(14.,6.+2.*error);radius=min(24.,14.+2.*error)
        first,second=neighbours[0][0],neighbours[1][0]
        if first[1]@second[1]<0:second=(second[0],-second[1],-second[2],second[3])
        tangent=first[1]+second[1];tangent/=np.linalg.norm(tangent);normal=np.array([-tangent[1],tangent[0]])
        centre=(first[0]+second[0])/2.;station=np.array([((q-centre)@tangent).min()for _,q in neighbours]+[((q-centre)@tangent).max()for _,q in neighbours])
        support=centre+np.linspace(max(station[:2]),min(station[2:]),257)[:,None]*tangent
        if len(support)<2 or np.linalg.norm(support[-1]-support[0])<35:continue
        disagreement=max(float(np.max(np.abs((support-frame[0])@frame[2])))for frame,_ in neighbours)
        independent_guides_required=disagreement>tolerance
        if independent_guides_required and not final_pass:
            report.append(dict(pts=pt,feature=name,status='adjacent_materials_disagree',maximum_disagreement_px=disagreement));continue
        q=np.asarray([r['uv']for r in current['rows']]).reshape(-1,2);distance=np.abs((q-centre)@normal)
        if len(distance)and np.percentile(distance,90)<=tolerance:
            report.append(dict(pts=pt,feature=name,status='actual_rows_inside_existing_temporal_identity_tolerance',tolerance=tolerance));continue
        v=vis[pt];im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0);graphics,border,_,_=source_domain(im,mask,read(v['board_source']))
        profile_type=EntryProfiles if final_pass and path_is_blue(name)else NativeProfiles
        native=profile_type(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),dict(source_video_sha256=v['source_video_sha256'],native_pts=pt,time_base='1/12800'),
            (mask>0)|graphics|(np.arange(1080)[:,None]<=border[None]))
        policy=dict(current_identity_constrained=True,identity_source='two_source_materials_bracketing_current_frame'if final_pass else'two_adjacent_actual_materials_registered_into_current_frame',
            reference_frame=reference_pts,normal_identity_tolerance_px=tolerance,search_radius_px=radius,registration_error_px=error,
            native_image_coordinates_not_generated=True)
        path=paths[name];local=dict(path,_source_query_spacing_px=35,_separate_profile_support=True,_line_identity_policy=policy)
        if final_pass and path_is_blue(name):local['_wide_local_identity_reason']='two_whole_source_blue_materials_registered_into_current_frame'
        value,_,_=Extractor.line(None,local,support,native,family_slope=float(tangent[1]/tangent[0]));fit=value.get('source_line_fit')
        if independent_guides_required:
            attempts=[]
            for frame,_ in neighbours:
                guide=support-((support-frame[0])@frame[2])[:,None]*frame[2]
                other,_,_=Extractor.line(None,local,guide,native,family_slope=float(frame[1][1]/frame[1][0]))
                found=other.get('source_line_fit')
                if found is None:break
                actual=found['direct_readouts'];observed=source_line_frame([r['uv']for r in actual])
                if observed is None:break
                attempts.append((actual,observed))
            agreement=float('inf')
            if len(attempts)==2:
                agreement=max(float(np.max(np.abs((np.asarray([r['uv']for r in rr])-other[0])@other[2])))
                    for (rr,_),(_,other)in [attempts,attempts[::-1]])
            if agreement>4.:
                report.append(dict(pts=pt,feature=name,status='independent_current_RGB_guides_do_not_confirm_same_material',
                    source_disagreement_px=disagreement,current_RGB_agreement_px=agreement if np.isfinite(agreement)else None,
                    source_neighbours=refs));continue
            fit=dict(direct_readouts=[r for actual,_ in attempts for r in actual])
            policy['independent_current_RGB_material_agreement_px']=agreement
        rr=[]
        if fit:
            prefix=SHOT+('_closing_RGB'if closing_pass else'_adjacent_final_RGB'if adjacent_pass else'_bracketed_identity_RGB'if final_pass else'_bidirectional_identity_RGB')
            rr=[dict(r,native_path_id=path['id'],feature_id=name,native_pts=pt,
                source_row_id=f'{prefix}/{pt}/{path["curve_id"]}/{i}')for i,r in enumerate(fit['direct_readouts'])]
        # Preserve original material positions which already satisfy the same
        # named temporal branch. Do not shift them to either neighbour's UV.
        measured=source_line_frame([r['uv']for r in rr])
        rr.extend(r for r,d in zip(current['rows'],distance)if d<=tolerance and
            (measured is None or abs(float((np.asarray(r['uv'])-measured[0])@measured[2]))<=4.))
        feature=dict(current,rows=list({r['source_row_id']:r for r in rr}.values()),
            temporal_identity_revision=dict(policy=policy,source_neighbours=refs,original_actual_source=original))
        packet=packets.setdefault(pt,dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features={},human_acceptance=False))
        packet['features'][name]=feature
        retained={r['source_row_id']for r in feature['rows']}
        preserved.extend(dict(pts=pt,feature=name,actual_readout=r,reason='left_two_adjacent_actual_material_identity_support')for r in current['rows']if r['source_row_id']not in retained)
        report.append(dict(pts=pt,feature=name,status='current_RGB_reread',before=len(current['rows']),after=len(feature['rows']),
            tolerance=tolerance,source_neighbours=refs,original_actual_source=original))
    for pt,packet in packets.items():write(dest/f'{pt}.json',packet)
    with gzip.open(dest/'previous_readouts.jsonl.gz','wt',encoding='utf8')as stream:
        for row in preserved:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    write(dest/'coverage.json',dict(review=report,changed_exposures=sorted(packets),previous_actual_readouts_preserved=len(preserved),
        original_UV_not_moved=True,all_added_UV_reread_from_current_RGB=True));common.verify_authority()
    common.emit('temporal_identity',shot=SHOT,requests=len(requests),changed_exposures=len(packets),previous_rows_preserved=len(preserved))


def final_temporal_line_repair():
    temporal_line_repair(final_pass=True)


def adjacent_temporal_line_repair():
    temporal_line_repair(final_pass=True,adjacent_pass=True)


def closing_temporal_line_repair():
    temporal_line_repair(final_pass=True,adjacent_pass=True,closing_pass=True)


def path_is_blue(name):
    return name in('ice.B1:segment0','ice.B2:segment0')


def crossline_family_repair(only_missing=False):
    """Use two other real cross-lines to identify a faint blue strip.

    The current native family supplies physical identity and finite search
    bounds. Same-frame measured red/blue lines align that search to the image.
    Every resulting point is read again from the original RGB.
    """
    from reread_s04_lines import NativeLineReader,Extractor,NativeProfiles,EntryProfiles
    from ice_points_s04 import source_domain
    cv2.setNumThreads(2);grid=common.shared.NativeGrid();reader=NativeLineReader(grid)
    paths={p['label']:p for p in reader.paths};names=['ice.B1:segment0','ice.B2:segment0','ice.C:segment0']
    slices={};xyz=[];offset=0
    for name in names:
        cut=grid.cuts[paths[name]['id']];points=grid.samples['xyz'][cut['begin']:cut['end']]
        xyz.append(points);slices[paths[name]['id']]=slice(offset,offset+len(points));offset+=len(points)
    xyz=np.vstack(xyz)
    dest=OUT/'evidence/reread_crossline_family';dest.mkdir(exist_ok=True);report=[];preserved=[]
    if only_missing and(dest/'coverage.json').exists():
        report=read(dest/'coverage.json')['frames']
        with gzip.open(dest/'previous_readouts.jsonl.gz','rt',encoding='utf8')as stream:preserved=[json.loads(line)for line in stream]
    for v in read(OUT/'evidence/visibility.json')['frames']:
        pt=v['pts'];features={};refs={}
        existing=read(dest/f'{pt}.json')if only_missing and(dest/f'{pt}.json').exists()else None
        if existing and all(name in existing['features']for name in names[:2]):continue
        for folder in ['reread_continuous_crosslines','reread_temporal_final','reread_temporal_adjacent']:
            file=OUT/f'evidence/{folder}/{pt}.json'
            if file.exists():
                for name,value in read(file)['features'].items():features[name]=value;refs[name]=file_ref(file)
        source={};native_frames={};projected=grid.project_xyz(pt,xyz)
        projection={pid:dict(uv=projected['uv'][s],valid=projected['status'][s]==0)for pid,s in slices.items()}
        for name in names:
            rr=features.get(name,{}).get('rows',[]);q=np.asarray([r['uv']for r in rr]);frame=source_line_frame(q)
            if frame is not None and len(q)>=12 and frame[3]>=400 and np.percentile(np.abs((q-frame[0])@frame[2]),90)<=4:
                source[name]=frame
            z=projection[paths[name]['id']];q=z['uv'];good=z['valid']&np.isfinite(q).all(1)&(q[:,0]>=-120)&(q[:,0]<=2039)
            frame=source_line_frame(q[good])
            if frame is not None:native_frames[name]=(frame,q[good])
        if len(source)<2 or len(native_frames)!=3:continue
        def line_y(frame):
            o,t,_,_=frame;return float(o[1]+(960-o[0])*t[1]/t[0])
        packet=existing or dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features={},human_acceptance=False)
        im=None
        for name in names[:2]:
            if name in packet['features']:continue
            other=[n for n in names if n!=name]
            if not all(n in source for n in other):continue
            actual_y=np.array([line_y(source[n])for n in other]);native_y=np.array([line_y(native_frames[n][0])for n in other])
            if abs(np.diff(actual_y)[0])<100 or np.diff(actual_y)[0]*np.diff(native_y)[0]<=0:continue
            shift=float(np.median(actual_y-native_y));level=line_y(native_frames[name][0])+shift
            slopes=np.array([source[n][1][1]/source[n][1][0]for n in other]);a=float(np.diff(slopes)[0]/np.diff(actual_y)[0]);slope=float(slopes[0]+a*(level-actual_y[0]))
            x0,x1=np.clip([native_frames[name][1][:,0].min(),native_frames[name][1][:,0].max()],20,1900)
            if x1-x0<400:continue
            xx=np.linspace(x0,x1,257);guide=np.c_[xx,level+slope*(xx-960)]
            if im is None:
                im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0);graphics,border,_,_=source_domain(im,mask,read(v['board_source']))
                excluded=(mask>0)|graphics|(np.arange(1080)[:,None]<=border[None]);rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            partition=reader.crossline_partition(paths[name],projection);blocked=excluded if partition is None else excluded|partition
            meta=dict(source_video_sha256=v['source_video_sha256'],native_pts=pt,time_base='1/12800')
            policy=dict(current_identity_constrained=True,identity_source='two_same_frame_actual_crosslines_align_current_native_family',reference_frame=pt,
                normal_identity_tolerance_px=14.,search_radius_px=24.,native_projection_is_navigation_only=True)
            local=dict(paths[name],_source_query_spacing_px=35,_separate_profile_support=True,_line_identity_policy=policy)
            chroma,_,_=Extractor.line(None,local,guide,NativeProfiles(rgb,meta,blocked),family_slope=slope);chroma_fit=chroma.get('source_line_fit')
            if chroma_fit:
                # Actual blue response fixes the branch before recovering its
                # washed-out stretches in luminance. The family projection
                # must not keep the grayscale search on a neighbouring halo.
                chroma_q=np.asarray([r['uv']for r in chroma_fit['direct_readouts']])
                measured_level=float(np.median(chroma_q[:,1]-slope*(chroma_q[:,0]-960)))
                guide=np.c_[xx,measured_level+slope*(xx-960)]
                policy=dict(policy,identity_source='same_frame_actual_blue_material_with_crossline_family_direction',
                    normal_identity_tolerance_px=6.,search_radius_px=14.)
                local=dict(local,_line_identity_policy=policy)
            alternative,_,_=Extractor.line(None,dict(local,_wide_local_identity_reason=policy['identity_source']),guide,EntryProfiles(rgb,meta,blocked),family_slope=slope)
            fit=alternative.get('source_line_fit')or chroma_fit
            if fit is None:continue
            actual=fit['direct_readouts'];observed=source_line_frame([r['uv']for r in actual])
            if observed is None or len(actual)<12 or observed[3]<400:continue
            if np.max(np.abs((np.asarray([r['uv']for r in actual])-observed[0])@observed[2]))>4:continue
            if chroma_fit and fit is not chroma_fit:
                chroma_q=np.asarray([r['uv']for r in chroma_fit['direct_readouts']])
                measured_q=np.asarray([r['uv']for r in actual]);stations=(measured_q-observed[0])@observed[1]
                color_stations=(chroma_q-observed[0])@observed[1]
                matched=[]
                for q,station in zip(chroma_q,color_stations):
                    nearest=int(np.argmin(abs(stations-station)));r=actual[nearest]
                    sides=r.get('source_evidence',{}).get('actual_source_sides_uv')
                    if sides is None:continue
                    # Hue may occupy only one part of a washed-out blue strip.
                    # Confirm it lies in the measured paint band; do not
                    # replace the physical midpoint with the hue midpoint.
                    offsets=(np.asarray(sides)-measured_q[nearest])@observed[2]
                    distance=float((q-measured_q[nearest])@observed[2])
                    if float(offsets.min())-r['sigma_px']<=distance<=float(offsets.max())+r['sigma_px']:matched.append(float(station))
                if len(matched)<2 or np.ptp(matched)<180:continue
            previous=features[name];old=previous['rows'];kept=[r for r in old if abs(float((np.asarray(r['uv'])-observed[0])@observed[2]))<=4]
            new=[dict(r,native_path_id=paths[name]['id'],feature_id=name,native_pts=pt,
                source_row_id=f'{SHOT}_crossline_family_RGB/{pt}/{paths[name]["curve_id"]}/{i}')for i,r in enumerate(actual)]
            packet['features'][name]=dict(previous,rows=list({r['source_row_id']:r for r in kept+new}.values()),
                physical_family_identity=dict(actual_reference_sources=[refs[n]for n in other],previous_source=refs[name],
                    source_direction=slope,source_alignment_shift_px=shift,actual_RGB_reread=True))
            kept_ids={r['source_row_id']for r in kept}
            preserved.extend(dict(pts=pt,feature=name,actual_readout=r,reason='superseded_by_current_RGB_crossline_family_identity')for r in old if r['source_row_id']not in kept_ids)
            report.append(dict(pts=pt,feature=name,before=len(old),retained_actual=len(kept),new_actual=len(new),
                chromatic_readouts=len(chroma_fit['direct_readouts'])if chroma_fit else 0,actual_reference_sources=[refs[n]for n in other]))
        if packet['features']:write(dest/f'{pt}.json',packet)
        if(pt-FIRST)//256%100==0:common.emit('crossline_family_progress',pts=pt,source_seconds=pt/12800,material_frames=len(report))
    with gzip.open(dest/'previous_readouts.jsonl.gz','wt',encoding='utf8')as stream:
        for r in preserved:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
    write(dest/'coverage.json',dict(frames=report,previous_actual_readouts_preserved=len(preserved),native_geometry_unchanged=True))
    common.verify_authority();common.emit('crossline_family_RGB',shot=SHOT,changed_material_frames=len(report),previous_readouts_preserved=len(preserved))


def crossline_family_complete():
    crossline_family_repair(only_missing=True)


def straight_source_selection():
    """Do not replace an intact actual straight trace with a mixed reread.

    Uses the existing 4 px whole-line support criterion, without straightening
    UV or choosing individual convenient vertices from a failed attempt.
    """
    folders=['reread_line_references','reread_line_final_support','reread_gap_lines','reread_cross_lines',
        'reread_material_identity','reread_red_identity','reread_temporal_identity','reread_continuous_crosslines',
        'reread_temporal_final','reread_temporal_adjacent','reread_crossline_family','reread_temporal_closing']
    dest=OUT/'evidence/straight_source_selection';dest.mkdir(exist_ok=True);report=[]
    for v in read(OUT/'evidence/visibility.json')['frames']:
        pt=v['pts'];history=collections.defaultdict(list)
        for folder in folders:
            file=OUT/f'evidence/{folder}/{pt}.json'
            if file.exists():
                packet=read(file);ref=file_ref(file)
                for name,f in packet['features'].items():history[name].append((f,ref))
        packet=dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features={},decisions={},human_acceptance=False)
        for name,attempts in history.items():
            current,ref=attempts[-1];q=np.asarray([r['uv']for r in current['rows']]);frame=source_line_frame(q)
            if frame is None or len(q)<3:continue
            error=float(np.max(np.abs((q-frame[0])@frame[2])))
            if error<=4:continue
            selected=None;candidate_error=None;source=None
            for earlier,previous_ref in reversed(attempts[:-1]):
                if not earlier['rows']:break
                old=np.asarray([r['uv']for r in earlier['rows']]);line=source_line_frame(old)
                if line is None or len(old)<3:continue
                deviation=float(np.max(np.abs((old-line[0])@line[2])))
                if deviation<=4:
                    selected=earlier;source=previous_ref;candidate_error=deviation;break
            decision=dict(reason='mixed_material_reread_cannot_replace_coherent_actual_straight_source',
                latest_source=ref,latest_actual_rows=len(q),latest_max_normal_error_px=error,
                selected_source=source,selected_max_normal_error_px=candidate_error,
                original_UV_preserved=True,raw_failed_attempt_preserved=True)
            if selected is None:selected=dict(current,rows=[]);decision['reason']='no_coherent_actual_straight_material_in_current_source_lineage'
            packet['features'][name]=dict(selected,straight_source_selection=decision);packet['decisions'][name]=decision
            report.append(dict(pts=pt,feature=name,after=len(selected['rows']),**decision))
        if packet['features']:write(dest/f'{pt}.json',packet)
    write(dest/'coverage.json',dict(decisions=report,existing_whole_material_tolerance_px=4.,UV_straightening=False))
    common.verify_authority();common.emit('straight_source_selection',shot=SHOT,material_frames=len(report),held=sum(r['after']==0 for r in report))


def red_center_identity_repair():
    """Read C between the two blue lines, excluding red dots/rings beyond them."""
    from reread_s04_lines import NativeLineReader,Extractor,NativeProfiles
    from ice_points_s04 import source_domain
    cv2.setNumThreads(2);grid=common.shared.NativeGrid();photo=CompleteSourceGuides(grid)
    reader=NativeLineReader(grid,photographic=photo,motion_path=OUT/'evidence/source_navigation.json')
    paths={p['label']:p for p in reader.paths};name='ice.C:segment0';path=paths[name]
    dest=OUT/'evidence/reread_red_identity';dest.mkdir(exist_ok=True);report=[];preserved=[]
    for v in read(OUT/'evidence/visibility.json')['frames']:
        pt=v['pts'];source=OUT/f'evidence/reread_cross_lines/{pt}.json';features=read(source)['features'];refs=[file_ref(source)]
        identity=OUT/f'evidence/reread_material_identity/{pt}.json'
        if identity.exists():features.update(read(identity)['features']);refs.append(file_ref(identity))
        old=features.get(name,dict(native_path_id=path['id'],rows=[]));projections=grid.project(pt);native={}
        for key in [name,'ice.B1:segment0','ice.B2:segment0']:
            z=projections[paths[key]['id']];q=z['uv'];good=z['valid']&np.isfinite(q).all(1)&(q[:,0]>=0)&(q[:,0]<=1920)
            frame=source_line_frame(q[good]);native[key]=(frame,q[good])
        if any(v[0]is None for v in native.values()):continue
        centre=native[name][0];side_frames=[];shifts=[]
        for key in ['ice.B1:segment0','ice.B2:segment0']:
            frame=native[key][0];rows=features.get(key,{}).get('rows',[]);actual=source_line_frame([r['uv']for r in rows])
            if actual is not None and actual[3]>=180:
                if abs(float((actual[0]-frame[0])@frame[2]))<abs(float((actual[0]-centre[0])@centre[2])):
                    shifts.append(float((actual[0]-frame[0])@frame[2]));frame=actual
            sign=float(np.sign((centre[0]-frame[0])@frame[2]));side_frames.append((frame,sign))
        def inside(r):return all(sign*float((np.asarray(r['uv'])-frame[0])@frame[2])>0 for frame,sign in side_frames)
        retained=[r for r in old['rows']if inside(r)]
        if retained and len(retained)==len(old['rows']):continue
        im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0);graphics,border,_,_=source_domain(im,mask,read(v['board_source']))
        excluded=(mask>0)|graphics|(np.arange(1080)[:,None]<=border[None]);yy,xx=np.mgrid[:1080,:1920]
        for frame,sign in side_frames:excluded|=sign*((xx-frame[0][0])*frame[2][0]+(yy-frame[0][1])*frame[2][1])<=0
        rgb=NativeProfiles(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),dict(source_video_sha256=v['source_video_sha256'],native_pts=pt,time_base='1/12800'),excluded)
        search=native[name][1].copy()
        if shifts:search+=float(np.median(shifts))*centre[2]
        local=dict(path,_source_query_spacing_px=35,_separate_profile_support=True)
        value,_,_=Extractor.line(None,local,search,rgb,family_slope=float(centre[1][1]/centre[1][0]));fit=value.get('source_line_fit');rr=[]
        if fit:
            rr=[dict(r,native_path_id=path['id'],feature_id=name,native_pts=pt,source_row_id=f'{SHOT}_center_identity_RGB/{pt}/C:segment0/{i}')
                for i,r in enumerate(fit['direct_readouts'])if inside(r)]
        measured=source_line_frame([r['uv']for r in rr])
        rr.extend(r for r in retained if measured is None or abs(float((np.asarray(r['uv'])-measured[0])@measured[2]))<=4.)
        selected=dict(native_path_id=path['id'],rows=rr,navigation=dict(physical_identity='red_C_between_current_native_blue_lines',
            native_search_partition_only=True,all_coordinates_actual_current_RGB=True))
        reason='red_candidate_outside_the_two_blue_lines'if old['rows']else'missing_red_C_reread_from_current_RGB'
        write(dest/f'{pt}.json',dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features={name:selected},original_actual_sources=refs,
            reason=reason,model_or_flow_observations_generated=False,human_acceptance=False))
        kept={r['source_row_id']for r in rr};preserved.extend(dict(pts=pt,feature=name,reason=reason,actual_readout=r)for r in old['rows']if r['source_row_id']not in kept)
        report.append(dict(pts=pt,before=len(old['rows']),after=len(rr),reason=reason))
    with gzip.open(dest/'previous_readouts.jsonl.gz','wt',encoding='utf8')as stream:
        for row in preserved:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    write(dest/'coverage.json',dict(frames=report,previous_actual_readouts_preserved=len(preserved)));common.verify_authority()
    common.emit('red_identity',shot=SHOT,changed_exposures=len(report),previous_actual_readouts_preserved=len(preserved))


def crossline_continuous_reread():
    from reread_s04_lines import NativeLineReader
    from ice_points_s04 import source_domain
    cv2.setNumThreads(2);grid=common.shared.NativeGrid();photo=CompleteSourceGuides(grid)
    reader=NativeLineReader(grid,photographic=photo,motion_path=OUT/'evidence/source_navigation.json',source_prefix=SHOT+'_continuous_crossline',
        source_cross_line_guides=True,source_parallel_references=True,source_finite_extents=True,recover_partial_blue=True,
        photographic_cross_direction=True,native_crossline_partitions=True,temporal_achromatic_blue=True)
    reader.paths=[p for p in reader.paths if p['feature_id']in('B1','B2','C')]
    for path in reader.paths:path.update(_source_query_spacing_px=35,_separate_profile_support=True)
    dest=OUT/'evidence/reread_continuous_crosslines';dest.mkdir(exist_ok=True);report=[]
    for v in read(OUT/'evidence/visibility.json')['frames']:
        pt=v['pts'];im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0);graphics,border,_,_=source_domain(im,mask,read(v['board_source']))
        features,diagnostic=reader.read(pt,im,(mask>0)|graphics|(np.arange(1080)[:,None]<=border[None]),v['source_video_sha256'])
        previous={};refs=[];comparison={}
        for folder in ['reread_cross_lines','reread_material_identity','reread_red_identity','reread_temporal_identity']:
            file=OUT/f'evidence/{folder}/{pt}.json'
            if file.exists():previous.update(read(file)['features']);refs.append(file_ref(file))
        for path in reader.paths:
            name=path['label'];value=features.setdefault(name,dict(native_path_id=path['id'],rows=[]));actual=value['rows'];prior=previous.get(name,{}).get('rows',[])
            measured=source_line_frame([r['uv']for r in actual]);same=[]
            if measured is not None:
                partition=reader.crossline_partition(path,reader.current_projections)
                for row in prior:
                    x,y=np.rint(row['uv']).astype(int)
                    if partition is not None and(0<=x<1920 and 0<=y<1080)and partition[y,x]:continue
                    if abs(float((np.asarray(row['uv'])-measured[0])@measured[2]))<=4.:same.append(row)
            combined={r['source_row_id']:r for r in actual};combined.update({r['source_row_id']:r for r in same});value['rows']=list(combined.values())
            comparison[name]=dict(previous=len(prior),current_RGB=len(actual),preserved_same_material=len(same),after=len(value['rows']))
        write(dest/f'{pt}.json',dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features=features,diagnostics=diagnostic,
            previous_actual_sources=refs,comparison=comparison,temporal_identity_retained_at_crossline_entry=True,
            native_geometry_search_only=True,actual_current_RGB_coordinates_only=True,human_acceptance=False))
        report.append(dict(pts=pt,comparison=comparison))
        if len(report)%100==0:common.emit('continuous_crossline_RGB',shot=SHOT,frames=len(report),pts=pt,comparison=comparison)
    write(dest/'coverage.json',dict(frames=report,original_sources_retained=True));common.verify_authority()


def continuous_source_selection():
    """A failed new read does not erase an existing actual source reading."""
    dest=OUT/'evidence/reread_continuous_crosslines';report=[]
    for v in read(OUT/'evidence/visibility.json')['frames']:
        pt=v['pts'];file=dest/f'{pt}.json';packet=read(file);previous={}
        for ref in packet['previous_actual_sources']:
            if file_ref(ref['path'])!=ref:raise ValueError('Prior actual source changed')
            previous.update(read(ref['path'])['features'])
        decisions={}
        for name,feature in packet['features'].items():
            if not feature['rows']and previous.get(name,{}).get('rows'):
                packet.setdefault('unrecovered_continuous_candidates',{})[name]=feature
                packet['features'][name]=previous[name]
                decisions[name]=dict(reason='new_reader_did_not_supersede_existing_actual_source_evidence',
                    previous_rows_preserved=len(previous[name]['rows']),temporal_recovery_confirmed=False)
        packet['empty_read_dispositions']=decisions;write(file,packet);report.append(dict(pts=pt,decisions=decisions))
    write(dest/'selection.json',dict(frames=report,no_existing_rows_deleted_due_to_empty_new_read=True));common.verify_authority()


def circle_source_pool():
    """Keep qualified full curve support for navigation, before density thinning."""
    import itertools
    from ice_points_s04 import qualify,source_domain,competing_source_ids,material_path
    cv2.setNumThreads(2);common.verify_authority();inv=read(OUT/'evidence/source_inventory.json')
    excluded={k:dict(reason=v['record'].get('reason',v['record'].get('reason_code')))for k,v in inv['historical_corrections'].items()}
    vis={v['pts']:v for v in read(OUT/'evidence/visibility.json')['frames']};geo=NativeSourceGeometry(SCENE,read(OUT/'evidence/native_authority.json')['native_library'])
    dest=OUT/'evidence/circle_source_pool';dest.mkdir(exist_ok=True);ref=file_ref(ICE);counts=[]
    for fi,(pt,entries)in enumerate(itertools.groupby(rows(ICE),key=lambda t:t[1]['native_pts'])):
        v=vis[pt];im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0);graphics,border,distance,_=source_domain(im,mask,read(v['board_source']))
        grouped=collections.defaultdict(list)
        for n,r in entries:
            if not re.fullmatch(r'ice\.[FR][1-4]:outer',r['feature_id']):continue
            if r['source_row_id']in inv['withdrawals']or r.get('source_parent_row_id')in inv['withdrawals']:continue
            path=material_path(geo,r)
            if qualify(r,path,excluded,{},graphics,border,distance):continue
            grouped[path['label']].append(r)
        features={}
        for name,rr in grouped.items():
            conflicts=competing_source_ids(rr);rr=[r for r in rr if r['source_row_id']not in conflicts]
            features[name]=dict(rows=[dict(uv=r['uv'],source_row_id=r['source_row_id'])for r in rr])
        write(dest/f'{pt}.json',dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features=features,source_file=ref,
            actual_source_navigation_only=True,withdrawn_identities_excluded=True,points_not_emitted_by_this_stage=True))
        counts.append(dict(pts=pt,counts={k:len(v['rows'])for k,v in features.items()}))
        if fi%100==0:common.emit('full_circle_source_pool',shot=SHOT,frames=fi+1,total=COUNT)
    write(dest/'coverage.json',dict(frames=counts));common.verify_authority()


def circle_source_reread(requested=None):
    from types import SimpleNamespace
    from ice_points_s04 import source_domain
    from calibration.program_tracking.audit_first_shot_circle_material_identity import conflicts
    cv2.setNumThreads(2);grid=common.shared.NativeGrid()
    house=common.shared.SourceHouseReader(grid,motion_path=OUT/'evidence/source_navigation.json',source_pool_dir=OUT/'evidence/circle_source_pool')
    dest=OUT/'evidence/reread_circle_references';dest.mkdir(exist_ok=True);summary=[]
    for v in read(OUT/'evidence/visibility.json')['frames']:
        pt=v['pts']
        if requested is not None and pt not in requested:continue
        previous=OUT/f'evidence/reread_circles/{pt}.json';saved=read(previous);features=dict(saved['features']);changed={};shape=[]
        im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0);graphics,border,_,_=source_domain(im,mask,read(v['board_source']))
        native=SimpleNamespace(frame=SimpleNamespace(rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB),material_signals={}),mask=(mask>0)|graphics|(np.arange(1080)[:,None]<=border[None]))
        for fid in house.groups:
            if not re.fullmatch(r'[FR][1-4]',fid):continue
            house.prepare_family(fid,pt)
            if (pt,fid)not in house.registration_proof:continue
            result,diagnostic=house.read(fid,pt,native);before={name:len(f['rows'])for name,f in features.items()if name.startswith('ice.'+fid+':')}
            features={name:f for name,f in features.items()if not name.startswith('ice.'+fid+':')}
            for role,data in result.items():
                path=house.native_paths[(fid,role)][2];actual=[]
                for row in data['rows']:
                    r={k:v for k,v in row.items()if k not in('representative_uv','representative_to_actual_material_px')}
                    r.update(native_path_id=path['id'],feature_id=path['label'],native_pts=pt,source_row_id=f'{SHOT}_full_source_house_RGB/{pt}/{fid}/{role}/{row["query_index"]}');actual.append(r)
                data['joint_group_navigation']['source_registration']=house.registration_proof[(pt,fid)]
                features[path['label']]=dict(native_path_id=path['id'],rows=actual,navigation=data['joint_group_navigation'])
                e=cv2.fitEllipse(np.asarray(data['curve'],np.float32));shape.append(dict(target_id=path['label'],measurement_operator='material_interface',
                    segments=[dict(uv=[r['uv']for r in actual],source_evidence=[r['source_evidence']for r in actual])],
                    geometric_interpretation=dict(kind='projected_physical_circle_ellipse',center_uv=e[0],axes_px=e[1],angle_degrees=e[2])))
            house.record_source_family(fid,pt,result)
            changed[fid]=dict(before=before,after={name:len(f['rows'])for name,f in features.items()if name.startswith('ice.'+fid+':')},diagnostic=diagnostic)
        rejected={}
        for r in conflicts(shape):
            name=r['target_id'];rejected[name]=dict(reason=r,feature=features.pop(name,None))
        if changed:
            write(dest/f'{pt}.json',dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features=features,previous_source=file_ref(previous),
                changed_families=changed,rejected_source_candidates=rejected,native_geometry_changed=False,only_actual_RGB_readouts=True))
        summary.append(dict(pts=pt,changed={fid:{k:v for k,v in d.items()if k!='diagnostic'}for fid,d in changed.items()}))
        if len(summary)%50==0 or requested is not None:common.emit('full_source_circle_reread',shot=SHOT,frames=len(summary),pts=pt,changed=summary[-1]['changed'])
    write(dest/'coverage.json',dict(frames=summary));common.verify_authority()


def ads_local_references():
    """Physical source packets seed tracks; shared artwork is not an instance."""
    import ad_replacement_s01 as b
    r,t,m,o,s,ats,_=common.ad_configure();assets=b.assets();groups={}
    for file in sorted(PACKETS.glob('*.json')):
        d=read(file);pt=d['source_pts']
        for feature in d['features']:
            match=re.match(r'Ad(\d+)\.',feature['id'])
            if not match:continue
            instance=int(match[1]);base=instance-13 if instance>=14 else instance
            name=re.sub(r'^Ad\d+\.',f'Ad{base}.',feature['id']);aliases=set(assets[base]['aliases'].get(name,[]))
            if len(aliases)!=1:continue
            pid=aliases.pop()
            for si,seg in enumerate(feature.get('segments',[])):
                q=seg.get('source_native_readout_uv')
                if q is None and seg.get('pixels_measured_in_this_frame'):q=seg.get('uv')
                if q is None or len(q)<2:continue
                q=np.asarray(q,float);image=OUT/f'source_frames/{pt}.png';key=(instance,pt)
                a=groups.setdefault(key,dict(ad=base,physical_instance=instance,image=str(image),image_sha256=file_ref(image)['sha256'],paths=[],
                    partial_source_reference=True,source_pts=pt,local_physical_reference=True,actual_source_material_colors=True))
                a['paths'].append(dict(path_id=pid,pixels=q.tolist(),contour_sha256=hashlib.sha256(q.tobytes()).hexdigest(),proof=dict(
                    kind='same_shot_actual_readout_unique_current_native_alias',source_packet=file_ref(file),original_feature=feature['id'],
                    physical_instance=f'Ad{instance}',segment=si,source_pts=pt,reference_only_not_current_measurement=True)))
    existing={(a.get('physical_instance'),a.get('source_pts')) for a in ats if a.get('local_physical_reference')};seeds={}
    for key,a in groups.items():
        if key not in existing:a['id']=len(ats);ats.append(a)
    for a in ats:
        if not a.get('local_physical_reference'):continue
        ai=a['id'];pt=a['source_pts'];seeds[str(ai)]={str(pt):dict(H=np.eye(3).tolist(),atlas_id=ai,pts=pt,physical_instance=a['physical_instance'],
            method='same_authentic_source_exposure_identity',source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],human_acceptance=False)}
    write(OUT/'ads/atlases.json',dict(atlases=ats));write(r.OUT/'local_physical_anchors.json',dict(records=seeds))
    common.emit('local_physical_references',shot=SHOT,atlases=len(seeds),instances=sorted({a['physical_instance']for a in ats if a.get('local_physical_reference')}))


def ads_anchors():
    import ad_points_s04 as a
    r,t,m,o,s,ats,chosen=common.ad_configure()
    allowed={at['ad']for at in ats if at.get('local_physical_reference')}
    # Register global material references independently. Local physical charts
    # have exactly their photographed exposure as root; no global reacquisition.
    selected=[i for i in chosen if ats[i]['ad']in allowed and not ats[i].get('local_physical_reference')]
    original=a.configure
    def scoped():
        rr,tt,mm,oo,ss,aa,_=common.ad_configure();tt.PRIMARY=selected;rr.ATLAS_IDS=selected
        return rr,tt,mm,oo,ss,aa,selected
    a.configure=scoped
    try:a.anchors()
    finally:a.configure=original


def ads_tracks():
    r,t,m,o,s,ats,chosen=common.ad_configure()
    t.PRIMARY=list(map(int,read(r.OUT/'source_pattern_anchors.json')['records']));t.run()
    t.PRIMARY=list(map(int,read(r.OUT/'local_physical_anchors.json')['records']))
    t.run(r.OUT/'local_physical_anchors.json',r.OUT/'local_physical_charts.json')
    charts=read(r.OUT/'source_identity_charts.json');local=read(r.OUT/'local_physical_charts.json');charts['records'].update(local['records'])
    write(r.OUT/'source_identity_charts.json',charts);r.authority(ad_numbers=range(1,14))


def ads_measure():
    r,t,m,o,s,ats,_=common.ad_configure()
    m.run(source=dict(np.load(OUT/'ads/original_ad_candidates.npz')),banned=set(),first_pts=FIRST,source_prefix=SHOT+'_native_source_v1_')


def ads_select():
    r,t,m,o,s,ats,_=common.ad_configure();o.run()
    shutil.copy2(r.OUT/'observability_checked_points.jsonl.gz',r.OUT/'pointwise_checks.jsonl.gz')
    shutil.copy2(r.OUT/'observability_ads.npz',r.OUT/'verified_ads.npz');s.run()


def photographic_path_agreement(source_paths, physical_paths, tolerance):
    from rinkspace.review.point_connections import polyline_order
    distances={}
    for name,curves in physical_paths.items():
        if name not in source_paths:continue
        q=np.vstack(curves)
        nearest=np.min([polyline_order(q,p)[1]for p in source_paths[name]if len(p)>=2],axis=0)
        distances[name]=float(np.median(nearest))
    matched={k:v for k,v in distances.items()if v<=tolerance}
    two_dimensional=[]
    for name in matched:
        q=np.vstack(physical_paths[name]);span=np.ptp(q,axis=0)
        if len(q)>=3 and np.linalg.matrix_rank(q-q.mean(0),tol=1e-6)==2 and min(span)>=7:two_dimensional.append(name)
    return dict(supported=len(matched)>=2 or bool(two_dimensional),matched_named_paths=matched,independent_named_source_paths=len(matched),
        shared_resolved_two_dimensional_source_shapes=two_dimensional,
        source_search_radius_px=tolerance,uses_photographic_material_only=True)


def ad_records():
    r,t,m,o,s,ats,_=common.ad_configure();charts=read(r.OUT/'local_physical_charts.json')['records'];by=collections.defaultdict(list)
    measured_charts=read(r.OUT/'source_identity_charts.json')['records'];mapped={};agreement={}
    def mapped_paths(ai,pt,record):
        key=(ai,pt)
        if key not in mapped:
            grouped=collections.defaultdict(list)
            for path in ats[ai]['paths']:
                grouped[path['path_id']].append(cv2.perspectiveTransform(np.asarray(path['pixels'],float)[None],np.asarray(record['H'],float))[0])
            mapped[key]=grouped
        return mapped[key]
    for ai,frames in charts.items():
        a=ats[int(ai)];points=np.vstack([p['pixels']for p in a['paths']])
        for pt,record in frames.items():
            q=cv2.perspectiveTransform(points.astype(float)[None],np.asarray(record['H'],float))[0]
            by[(int(pt),a['ad'])].append(dict(instance=a['physical_instance'],atlas=int(ai),uv=q,hull=cv2.convexHull(q.astype(np.float32)),
                paths=mapped_paths(int(ai),int(pt),record),anchor_pts=record.get('anchor_pts',a['source_pts'])))
    sources=[r.OUT/'representative_checks.jsonl.gz'];extra=r.OUT/'asset_completion/representative_checks.jsonl.gz'
    if extra.exists():sources.append(extra)
    records=[];pending=[];seen=set();source_refs={str(p):file_ref(p)for p in sources}
    for source,_,row in ((p,n,row)for p in sources for n,row in rows(p)):
        ref=source_refs[str(source)]
        if row['decision']=='not_admitted':continue
        pt=row['pts'];identity=row['identity'];ai=identity['atlas_id'];base=ats[ai]['ad'];possible=[]
        radius=max(32.,float(row['source_profile'].get('search_radius_px',0)))
        for candidate in by[(pt,base)]:
            inside=cv2.pointPolygonTest(candidate['hull'],tuple(map(float,row['uv'])),True)>=-radius
            proof=dict(inside_photographic_footprint=inside,source_search_radius_px=radius)
            if not inside:
                key=(pt,ai,candidate['atlas'],radius)
                if key not in agreement:
                    chart=measured_charts.get(str(ai),{}).get(str(pt))
                    agreement[key]=photographic_path_agreement(mapped_paths(ai,pt,chart),candidate['paths'],radius)if chart else dict(supported=False)
                proof.update(agreement[key])
                if not proof['supported']:continue
            possible.append(dict(candidate,physical_source_support=proof))
        physical={c['instance']for c in possible}
        chosen=None
        if len(physical)==1:chosen=next(iter(physical))
        elif physical:
            # Different copies may share artwork. Compare their continuously
            # transported source footprints, never their native residuals.
            q=np.asarray(row['uv']);distance={inst:min(float(np.linalg.norm(c['uv']-q,axis=1).min())for c in possible if c['instance']==inst)for inst in physical}
            ranked=sorted(distance,key=distance.get)
            if distance[ranked[1]]-distance[ranked[0]]>35:chosen=ranked[0]
        if chosen is None:
            pending.append(dict(row,reason='physical_instance_not_resolved_by_adjacent_source_tracks',possible_instances=sorted(physical),
                actual_photographic_footprint_or_shared_named_paths_required=True));continue
        pid=re.sub(r'^print_instance/Ad\d+::',f'print_instance/Ad{chosen}::',row['path_id'])
        key=(row['stable_id'],pt,pid)
        if key in seen:continue
        seen.add(key)
        ev=dict(row,physical_native_path_id=pid,physical_instance_source_tracks=[{k:v for k,v in c.items()if k not in('uv','paths','hull')}for c in possible if c['instance']==chosen],template_is_not_instance_identity=True)
        records.append(dict(observation_id=row['stable_id'],source_id=row['stable_id'],pts=pt,uv=row['uv'],native_path_id=pid,kind=row['kind'],station=row['station'],
            origin='verified_original_ad_pixel'if row['decision']=='include_original_pixel'else'new_current_RGB_ad_material',human_acceptance=False,source_record=ev,source_ref=ref,
            source_BGR_sha256=row['source_BGR_sha256'],source_group_id=f'{SHOT}/{pt}/{pid}/same_material_profile_family',
            review_order=dict(family='ad',run=pid,atlas=identity['atlas_id'],source_path=identity['source_path_index'],order=identity.get('reference_arc_fraction',.5))))
    if extra.exists():
        grouped=collections.defaultdict(list);selected=[];doc=read(r.OUT/'source_identity_charts.json')
        for record in records:grouped[(record['pts'],record['native_path_id'])].append(record)
        for (pt,pid),items in grouped.items():
            indices,_=s.select_path(pt,[v['source_record']for v in items],doc,ats)
            selected.extend(items[i]for i in sorted(indices))
        records=selected
    for name,data in [('ad_records_before_border_review',records),('ad_instance_pending',pending)]:
        with gzip.open(OUT/f'evidence/{name}.jsonl.gz','wt',encoding='utf8')as f:
            for row in data:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    common.emit('physical_ad_records',shot=SHOT,rows=len(records),pending=len(pending),frames=len({r['pts']for r in records}))


def ad_identity_repair():
    import repair_points_s02 as repair
    source=OUT/'evidence/ad_records_before_border_review.jsonl.gz'
    previous=OUT/'evidence/ad_records_before_physical_support.jsonl.gz'
    if not previous.exists():shutil.copy2(source,previous)
    before=collections.defaultdict(dict)
    for _,r in rows(source):before[r['pts']][(r['observation_id'],r['native_path_id'])]=r
    ad_records();after=collections.defaultdict(dict)
    for _,r in rows(source):after[r['pts']][(r['observation_id'],r['native_path_id'])]=r
    changed=[pt for pt in sorted(set(before)|set(after))if set(before[pt])!=set(after[pt])or any(
        before[pt][key]['uv']!=after[pt][key]['uv']for key in set(before[pt])&set(after[pt]))]
    initial=collections.defaultdict(dict)
    for _,r in rows(previous):initial[r['pts']][(r['observation_id'],r['native_path_id'])]=r
    pending={(r['pts'],r['stable_id'])for _,r in rows(OUT/'evidence/ad_instance_pending.jsonl.gz')}
    decisions=[]
    for pt in initial:
        for key in set(initial[pt])-set(after[pt]):
            reason='physical_source_support_not_established'if(pt,key[0])in pending else'representative_reselected_after_physical_source_check'
            decisions.append(dict(initial[pt][key],reason=reason,original_preserved=True))
    ledger=OUT/'evidence/ad_physical_identity_repair.jsonl.gz';history=OUT/'evidence/ad_physical_identity_initial_review.jsonl.gz'
    if ledger.exists()and not history.exists():shutil.copy2(ledger,history)
    with gzip.open(ledger,'wt',encoding='utf8')as f:
        for record in decisions:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    if changed:
        common.shared.ad_configure=common.ad_configure;common.ad_configure()
        repair.nonice(requested=changed,output_dir=OUT,source_records_path=source,board_dir=BOARD,
            source_prefix=SHOT,prepare_context=common.verify_authority,verify_context=common.verify_authority)
    write(OUT/'evidence/ad_physical_identity_review.json',dict(changed_exposures=changed,previous_records=file_ref(previous),
        removed_or_reselected=len(decisions),previous_review=file_ref(history)if history.exists()else None,
        original_sources_unchanged=True,native_geometry_changed=False))
    common.emit('ad_physical_identity_repair',shot=SHOT,changed_exposures=len(changed),removed_or_reselected=len(decisions))


def ads_asset_references():
    """Use current asset observation mappings, including renamed old contours."""
    import ad_replacement_s01 as b
    r,t,m,o,s,ats,_=common.ad_configure();groups={};assets=b.assets()
    for base,asset in assets.items():
        for ob in asset['asset'].get('observations',[]):
            pt=ob.get('pts');pid=ob.get('feature_id');instance=ob.get('instance_id','').split('/')[-1]
            if pt not in r.VIS or pid not in asset['paths']or not re.fullmatch(r'Ad\d+',instance):continue
            q=np.asarray(ob.get('source_native_uv',[]),float)
            if q.ndim!=2 or q.shape[1]!=2 or len(q)<2:continue
            num=int(instance[2:]);key=(num,pt);image=OUT/f'source_frames/{pt}.png'
            a=groups.setdefault(key,dict(ad=base,physical_instance=num,image=str(image),image_sha256=file_ref(image)['sha256'],paths=[],
                partial_source_reference=True,source_pts=pt,local_physical_reference=True,current_asset_observation_reference=True))
            a['paths'].append(dict(path_id=pid,pixels=q.tolist(),contour_sha256=hashlib.sha256(q.tobytes()).hexdigest(),proof=dict(
                kind='current_native_asset_explicit_physical_source_observation',asset=file_ref(asset['path']),observation_id=ob['id'],
                original_feature=ob.get('original_feature_id'),source_pts=pt,physical_instance=instance,human_acceptance_not_inferred=True)))
    # Whole photographic outlines establish Ad25's physical location even
    # where its old component numbering has no retained native alias. These
    # contours never enter the material query/measurement charts.
    for file in PACKETS.glob('*.json'):
        d=read(file);pt=d['source_pts'];points=[]
        for f in d['features']:
            if not f['id'].startswith('Ad25.'):continue
            for seg in f.get('segments',[]):
                q=seg.get('source_native_readout_uv')
                if q is None and seg.get('pixels_measured_in_this_frame'):q=seg.get('uv')
                if q is not None and len(q)>=2:points.append(np.asarray(q,float))
        if not points:continue
        q=np.vstack(points);image=OUT/f'source_frames/{pt}.png';key=(25,pt)
        if key not in groups:groups[key]=dict(ad=12,physical_instance=25,image=str(image),image_sha256=file_ref(image)['sha256'],source_pts=pt,
            partial_source_reference=True,local_physical_reference=True,context_only=True,current_asset_observation_reference=True,
            paths=[dict(path_id='source_context/Ad25',pixels=q.tolist(),contour_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
                proof=dict(kind='photographic_panel_identity_context_only',source_packet=file_ref(file),physical_instance='Ad25',source_pts=pt,not_a_native_material_path=True))])
    previous={(a.get('physical_instance'),a.get('source_pts')):a for a in ats if a.get('current_asset_observation_reference')};added=[];seeds={}
    for key,a in groups.items():
        if key in previous:a=previous[key]
        else:a['id']=len(ats);ats.append(a)
        added.append(a['id']);pt=a['source_pts'];seeds[str(a['id'])]={str(pt):dict(H=np.eye(3).tolist(),atlas_id=a['id'],pts=pt,
            physical_instance=a['physical_instance'],method='same_authentic_source_exposure_identity',source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'])}
    write(OUT/'ads/atlases.json',dict(atlases=ats));write(r.OUT/'asset_observation_anchors.json',dict(records=seeds))
    write(r.OUT/'asset_observation_reference_index.json',dict(atlas_ids=added,instances=sorted({ats[i]['physical_instance']for i in added}),
        context_only_atlas_ids=[i for i in added if ats[i].get('context_only')],native_geometry_unchanged=True))
    common.emit('current_asset_sources_reconnected',shot=SHOT,atlases=len(added),instances=sorted({ats[i]['physical_instance']for i in added}))


def ads_asset_tracks():
    r,t,m,o,s,ats,_=common.ad_configure();t.PRIMARY=list(map(int,read(r.OUT/'asset_observation_anchors.json')['records']))
    t.run(r.OUT/'asset_observation_anchors.json',r.OUT/'asset_observation_charts.json')


def ads_missing_global():
    import ad_points_s04 as a
    r,t,m,o,s,ats,chosen=common.ad_configure();selected=[i for i in chosen if ats[i]['ad']in(8,9,10,12)and not ats[i].get('local_physical_reference')]
    original=a.configure;work=r.OUT/'asset_completion';work.mkdir(exist_ok=True)
    def scoped():
        rr,tt,mm,oo,ss,aa,_=common.ad_configure();tt.PRIMARY=selected;rr.ATLAS_IDS=selected;rr.OUT=work
        a.WORK=work
        return rr,tt,mm,oo,ss,aa,selected
    oldwork=a.WORK;a.WORK=work;a.configure=scoped
    try:
        if not (work/'source_pattern_anchors.json').exists():a.anchors()
    finally:a.WORK=oldwork;a.configure=original
    r,t,m,o,s,ats,_=common.ad_configure();t.PRIMARY=selected
    t.run(work/'source_pattern_anchors.json',work/'source_identity_charts.json')


def ads_merge_assets():
    bank=read(OUT/'ads/atlases.json')
    for atlas in bank['atlases']:
        if atlas.get('local_physical_reference')and not atlas.get('context_only'):atlas['actual_source_material_colors']=True
    write(OUT/'ads/atlases.json',bank)
    r,t,m,o,s,ats,_=common.ad_configure();work=r.OUT/'asset_completion'
    master=read(r.OUT/'source_identity_charts.json');physical=read(r.OUT/'local_physical_charts.json');additional=read(r.OUT/'asset_observation_charts.json')
    physical['records'].update(additional['records']);write(r.OUT/'local_physical_charts.json',physical)
    additions=read(work/'source_identity_charts.json')
    additions['records'].update({ai:rr for ai,rr in additional['records'].items()if not ats[int(ai)].get('context_only')})
    write(work/'source_identity_charts.json',additions)
    master['records'].update(additions['records']);write(r.OUT/'source_identity_charts.json',master);r.authority(ad_numbers=range(1,14))
    for name in ['additional_source_atlases.json','native_entity_registry.json']:shutil.copy2(r.OUT/name,work/name)
    wanted={int(pt)for rr in additions['records'].values()for pt in rr};r.OUT=work;r.VIS={pt:v for pt,v in r.VIS.items()if pt in wanted};r.current.cache_clear()
    for module in(t,m,o,s):module.OUT=work
    source=dict(np.load(OUT/'ads/original_ad_candidates.npz'));source['source_rows']=np.arange(len(source['pts']),dtype=np.int64)
    use=np.isin(source['pts'],list(wanted));source={k:v[use]for k,v in source.items()}
    m.run(source=source,banned=set(),first_pts=FIRST,source_prefix=SHOT+'_current_asset_RGB_');o.run()
    shutil.copy2(work/'observability_checked_points.jsonl.gz',work/'pointwise_checks.jsonl.gz');shutil.copy2(work/'observability_ads.npz',work/'verified_ads.npz');s.run()
    # Preserve the original selected rows; the merged selection can de-duplicate
    # overlapping measurements by material and source exposure afterward.
    common.emit('current_asset_reread_complete',shot=SHOT,exposures=len(wanted))


def provenance():
    refs=read(OUT/'evidence/history_refs.json');additional=[]
    if SHOT=='s03':
        for name in ['s03_continuous_source_v1/typed_init_v2/source_rows.jsonl','s03_print_source_v1/native_q2_after_prefix/qualified_rows.jsonl']:
            p=REC/name
            if p.exists():additional.append(dict(ref=file_ref(p),disposition='retained historical source; unknown texture/latent paths cannot extend current native geometry; existing board material is independently re-read'))
    refs.update(additional_historical_sources=additional)
    write(OUT/'evidence/history_refs.json',refs)
    common.verify_authority()


def deliver():
    import zipfile
    common.verify_authority();provenance()
    result=read(OUT/'result.json');check=read(OUT/'verification.json');media=read(OUT/'review/video.json');mapping=read(OUT/'points/mapping.json')
    if (check['frames']!=COUNT or check['native_invalid_rows'] or media['points_displayed']!=result['rows']
        or media['point_input_sha256']!=mapping['point_input_sha256']):raise ValueError('Incomplete final dataset')
    temporal=read(OUT/'evidence/latest_peer_temporal_audit/diagnostic.json')
    if temporal.get('point_input_sha256')!=mapping['point_input_sha256']:
        raise ValueError('Temporal diagnostic belongs to another point input')
    selection=read(OUT/'evidence/straight_source_selection/coverage.json')
    final_review=read(OUT/'evidence/final_material_review/coverage.json')if(OUT/'evidence/final_material_review/coverage.json').exists()else dict(decisions=[])
    labels={e['id']:e['label']for e in read(OUT/'evidence/native_entities.json')['entities']}
    admitted={(f['pts'],labels[pid])for f in result['coverage']for pid in f['paths']}
    recovered={(r['pts'],r['feature'])for r in final_review['decisions']if r['status']=='current_RGB_reread'and r.get('after',0)>0}&admitted
    quality=dict(schema='rinkspace.source_quality.v1',shot_id=SHOT,official_exposures=COUNT,
        temporal_compared_material_frames=temporal['compared_material_frames'],temporal_review_candidates=temporal['review_candidates'],
        source_lineage_without_coherent_straight_read=[dict(r,source_seconds=r['pts']/12800)for r in selection['decisions']if r['after']==0 and(r['pts'],r['feature'])not in recovered],
        final_material_review=final_review,
        native_domain_held_rows=result.get('native_domain_held_rows',0),
        source_material_held_rows=result.get('source_material_held_rows',0),
        source_material_review=read(OUT/'source_material_hold/review.json')if(OUT/'source_material_hold/review.json').exists()else None,
        diagnostic_candidate_is_not_automatic_rejection=True,all_visible_materials_recovered_not_claimed=True,human_acceptance=False)
    write(OUT/'quality.json',quality)
    target=OUT/f'{SHOT.upper()}_native_points_20261028.zip';files=[]
    for folder in ['points','support','native_domain_hold','source_material_hold','evidence/final_material_review','evidence/goal_end_native_adoption']:
        files.extend(p for p in (OUT/folder).rglob('*')if p.is_file())
    names=['result.json','verification.json','quality.json','native_integration.json','evidence/native_authority.json','evidence/current_native_geometry_state.npz',
        'evidence/native_constraint_adoption.json','evidence/native_authority_before_ad26_parallel.json',
        'evidence/native_model_at_source_read.json','evidence/navigation_geometry_before_ad26_parallel.npz',
        'evidence/history_refs.json','evidence/native_entities.json','evidence/source_inventory.json','evidence/historical_ice_dispositions.jsonl.gz',
        'evidence/historical_identity_qualification.jsonl.gz','evidence/later_identity_revisions.json','evidence/whole_line_audit.json',
        'evidence/remaining_coverage.json','evidence/final_source_dispositions.jsonl.gz','evidence/ad_instance_pending.jsonl.gz',
        'evidence/ad_physical_identity_review.json','evidence/ad_physical_identity_repair.jsonl.gz','evidence/ad_physical_identity_initial_review.jsonl.gz',
        'evidence/reread_line_references/coverage.json','evidence/reread_line_final_support/coverage.json','evidence/reread_cross_lines/coverage.json','evidence/reread_cross_lines/selection.json','evidence/reread_circle_references/coverage.json','evidence/circle_source_pool/coverage.json',
        'evidence/reread_material_identity/coverage.json','evidence/reread_material_identity/previous_readouts.jsonl.gz','evidence/latest_peer_temporal_audit/diagnostic.json',
        'evidence/reread_temporal_identity/coverage.json','evidence/reread_temporal_identity/previous_readouts.jsonl.gz',
        'evidence/reread_temporal_final/coverage.json','evidence/reread_temporal_final/previous_readouts.jsonl.gz',
        'evidence/reread_temporal_adjacent/coverage.json','evidence/reread_temporal_adjacent/previous_readouts.jsonl.gz',
        'evidence/reread_crossline_family/coverage.json','evidence/reread_crossline_family/previous_readouts.jsonl.gz',
        'evidence/reread_temporal_closing/coverage.json','evidence/reread_temporal_closing/previous_readouts.jsonl.gz',
        'evidence/straight_source_selection/coverage.json',
        'evidence/reread_red_identity/coverage.json','evidence/reread_red_identity/previous_readouts.jsonl.gz',
        'evidence/reread_continuous_crosslines/coverage.json',
        'evidence/reread_continuous_crosslines/selection.json',
        'evidence/latest_peer_error_classes.json',
        'evidence/source_material_review.json',
        'evidence/source_guide_direction_audit.json','evidence/temporal_support_gaps.json','review/video.json','review/frame_index.jsonl.gz']
    files.extend(OUT/name for name in names if (OUT/name).exists())
    manifest=dict(schema='rinkspace.native_source_dataset_package.v1',shot_id=SHOT,point_folder='points',
        source_seconds=result['source_seconds'],official_exposures=COUNT,rows=result['rows'],native_paths=result['native_paths'],
        all_measured_rows=result.get('all_measured_rows',result['rows']),native_domain_held_rows=result.get('native_domain_held_rows',0),
        source_material_held_rows=result.get('source_material_held_rows',0),
        native_geometry_unchanged=True,human_acceptance=False,source_video_and_current_native_dependencies_are_external=True,
        files=[dict(archive=p.relative_to(OUT).as_posix(),**file_ref(p))for p in files])
    note=f'''# {SHOT.upper()} 当前 native 取点输入

原片 {FIRST/12800:.3f}–{LAST/12800:.3f} 秒；整数 PTS {FIRST}–{LAST}，步长 256，time_base=1/12800，共 {COUNT} 个官方曝光。

`points/` 可直接交给现有 `prepare_native_s01.load_pixel_input`；`mapping.json` 列出当前 native 路径、相机引用及本次固定数值状态。几何与相机均未由本次工作改写。逐条原片 UV、来源和相关组保存在同目录。初始化 stations 仅供求解导航，不是实测坐标。

全段已生成实际取点连线视频；`review/frame_index.jsonl.gz` 对应逐帧实体。视频不叠加不准确的模型投影，连线不增加求解观测。

`evidence/remaining_coverage.json` 保留仍未恢复的历史候选区间；这些候选不是已确认可见材料的分母。原始材料、纠错和撤销记录没有覆写。native 导入成功不代表视觉质量已经人工通过。

`evidence/reread_material_identity/` 和 `evidence/reread_temporal_identity/` 记录后续实体顺序与连续性检查。已确认的错误绑定可保留原 UV 改绑；尚无当前材料身份依据的实际读数单独保留，不能把“有本帧数据”写成“所有可见实体已无漏读”。

`quality.json` 列出仍需说明的跨帧异常候选及没有合格直线实测的具体曝光和实体。当前有 {len(quality['temporal_review_candidates'])} 个跨帧诊断候选、{len(quality['source_lineage_without_coherent_straight_read'])} 个未恢复的直线材料曝光；这些数量不是可见材料漏读率，也没有写成人工通过。
'''
    if result.get('native_domain_held_rows'):
        note+=f"\n另有 {result['native_domain_held_rows']} 条实测保存在 `native_domain_hold/`。其当前 native 路径整体位于围挡几何域之外，不能加入本次 `points/`；`native_integration.json` 列出路径及模型顶点数值。原始 UV、PTS、身份和相关组均保留，没有改动几何或把这些点写成错点。检查视频展示可直接导入的 {result['rows']} 点，这些单独保留的读数不混入该视频。\n"
    if result.get('source_material_held_rows'):
        note+=f"\n另有 {result['source_material_held_rows']} 条细连接线候选在原片及相邻曝光中缺少独立细线材料支持，保存在 `source_material_hold/`；`review.json` 给出逐点 ID、具体源时间、审阅图与原因。这些读数没有被改成圈上观测，也没有被写成人工否定或通过。\n"
    write(OUT/'package_manifest.json',manifest)
    (OUT/'README_dataset.md').write_text(note,encoding='utf8')
    with zipfile.ZipFile(target.with_suffix('.partial.zip'),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3)as archive:
        for file in files:archive.write(file,file.relative_to(OUT).as_posix())
        archive.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');archive.writestr('README.md',note)
    target.with_suffix('.partial.zip').replace(target)
    with zipfile.ZipFile(target)as archive:
        if archive.testzip()is not None:raise ValueError('Dataset archive CRC failed')
        if hashlib.sha256(archive.read('points/point_observations.npz')).hexdigest()!=mapping['point_input_sha256']:raise ValueError('Packed input differs from evaluated input')
    index=PROG/f'source_datasets/native_points_{SHOT}_20261028/result.json'
    if len(str(index.resolve()).encode('utf-16-le'))//2>=260:raise ValueError('OneDrive path too long')
    index.parent.mkdir(exist_ok=True)
    write(index,dict(shot_id=SHOT,source_seconds=result['source_seconds'],official_exposures=COUNT,point_input=file_ref(OUT/'points/point_observations.npz'),
        mapping=file_ref(OUT/'points/mapping.json'),records=file_ref(OUT/'points/point_records.jsonl.gz'),archive=file_ref(target),video=media['video'],
        verification=file_ref(OUT/'verification.json'),quality=file_ref(OUT/'quality.json'),coverage=file_ref(OUT/'evidence/remaining_coverage.json'),rows=result['rows'],native_paths=result['native_paths'],
        all_measured_rows=result.get('all_measured_rows',result['rows']),native_domain_held_rows=result.get('native_domain_held_rows',0),
        source_material_held_rows=result.get('source_material_held_rows',0),
        current_native_geometry_unchanged=True,human_acceptance=False))
    common.emit('delivered',shot=SHOT,archive=str(target),video=media['video']['path'],rows=result['rows'],paths=result['native_paths'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('shot',choices=['s03','s05']);p.add_argument('stages',nargs='+');args=p.parse_args();configure(args.shot)
    for stage in args.stages:
        common.emit('start',shot=SHOT,stage_name=stage)
        (globals().get(stage)or getattr(common,stage))()
