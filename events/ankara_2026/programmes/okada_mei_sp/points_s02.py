"""Whole S02 actual-source dataset; current native geometry is read-only.

The historical full-resolution pools, corrections and manual edits remain
immutable. Shared material readers supply actual RGB observations. Camera,
source registration and optical flow only navigate searches and review.
"""
from pathlib import Path
import argparse
import collections
import gzip
import hashlib
import itertools
import json
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
PROG = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(PROG)]
import cv2
import numpy as np
from rinkspace.vision_native import NativeSourceGeometry, file_ref
from ice_points_s04 import read, write, rows, material_path, select_representatives
from review_points_s04 import CurrentNativeGrid as SharedNativeGrid

DATA = Path('D:/CodexData/RinkSpace_Data')
OUT = DATA/'runs/native_points_s02_20261027'
REC = DATA/'runs/okada_rec_1019/recover_v2'
SCENE = ROOT/'events/ankara_2026/scene/field/model.json'
CAMERA = PROG/'ad_cameras/s02.json'
FIRST, LAST, STEP = 575744, 720640, 256
PACKETS = PROG/'source_datasets/shot2_r2_20261014/frames'


def emit(stage, **kw):
    print(json.dumps(dict(stage=stage, **kw), ensure_ascii=False), flush=True)


def prepare():
    import av
    cv2.setNumThreads(2)
    (OUT/'evidence').mkdir(parents=True, exist_ok=True)
    (OUT/'source_frames').mkdir(exist_ok=True)
    scope=read(PROG/'rebuild_scope.json')
    shot=next(s for s in scope['shots'] if s['id']=='s02')
    assert shot['processing']['native_frame_count']==567
    assert (shot['processing']['first_pts'],shot['processing']['last_pts'])==(FIRST,LAST)
    video=Path(scope['source']['registered_path'])
    if file_ref(video)['sha256']!=scope['source']['sha256']:
        raise ValueError('Official source video identity differs')
    manifest=DATA/'runs/native_joint_s01_s04_20261027/joint_manifest.json'
    joint=read(manifest)
    geo=NativeSourceGeometry(SCENE,joint['native_library'])
    refs={v['path']:v for v in geo.refs}
    for p in [CAMERA,CAMERA.parent/read(CAMERA)['controls_ref'],Path(joint['native_library'])]:
        ref=file_ref(p);refs[ref['path']]=ref
    authority=OUT/'evidence/native_authority.json'
    if not authority.exists():
        # Freeze the explicitly registered current geometry state. The S02
        # camera seed is independent of the S01/S04 motion and is not fitted.
        state_path=Path(joint['joint_output'])/'accepted_pixels_latest.npz'
        blob=state_path.read_bytes();state_ref=file_ref(state_path)
        if hashlib.sha256(blob).hexdigest()!=state_ref['sha256']:
            raise ValueError('Current state changed during snapshot; retry safely')
        target=OUT/'evidence/current_native_geometry_state.npz';target.write_bytes(blob)
        z=dict(np.load(target,allow_pickle=False))
        if str(z['point_input_sha256'])!=joint['point_input_sha256']:
            raise ValueError('Selected accepted state does not match current manifest')
        write(authority,dict(files=list(refs.values()),scene_path=str(SCENE),
            native_library=joint['native_library'],scene_state=file_ref(target),
            state_origin=state_ref,manifest_origin=file_ref(manifest),camera=file_ref(CAMERA),
            source=scope['source'],shot=shot,current_native_path_count=len(geo.current_paths),
            geometry_changed=False,camera_seed_only=True))
    else:
        verify_authority()
    write(OUT/'evidence/native_entities.json',dict(entities=list(geo.current_paths.values())))
    index={r['native_pts']:r for _,r in rows(REC/'dense_print_v1/final_index_anchor.jsonl.gz') if r['shot_id']=='s02'}
    vis=[];start=time.monotonic()
    with av.open(str(video)) as c:
        stream=c.streams.video[0];c.seek(FIRST,stream=stream)
        for f in c.decode(stream):
            pt=f.pts
            if pt<FIRST:continue
            if pt>LAST:break
            im=f.to_ndarray(format='bgr24');digest=hashlib.sha256(im.tobytes()).hexdigest();old=index[pt]
            if digest!=old['decoded_native_BGR_sha256']:raise ValueError('Source pixels differ from historical raw support')
            mask=old['mask_ref']
            if file_ref(mask['file'])['sha256']!=mask['sha256']:raise ValueError('Historical silhouette changed')
            dst=OUT/'source_frames'/f'{pt}.png'
            if not dst.exists():cv2.imwrite(str(dst),im,[cv2.IMWRITE_PNG_COMPRESSION,1])
            elif hashlib.sha256(cv2.imread(str(dst)).tobytes()).hexdigest()!=digest:raise ValueError('Cached RGB changed')
            vis.append(dict(pts=pt,shot_id='s02',image=str(dst),source_BGR_sha256=digest,
                source_video_sha256=scope['source']['sha256'],mask_path=mask['file'],mask_sha256=mask['sha256'],
                historical_frame_record=old['file'],historical_frame_record_sha256=old['sha256']))
            if len(vis)%150==0:emit('official_frames',frames=len(vis),total=567,seconds=time.monotonic()-start)
    if [r['pts']for r in vis]!=list(range(FIRST,LAST+1,STEP)):raise ValueError('Missing native exposure')
    write(OUT/'evidence/visibility.json',dict(frames=vis,source_video=str(video),source_video_sha256=scope['source']['sha256'],source_time_base='1/12800',shot_id='s02'))
    chosen=[FIRST,598784,623872,649472,675072,700672,LAST]
    images=[]
    for pt in chosen:
        im=cv2.resize(cv2.imread(str(OUT/'source_frames'/f'{pt}.png')),(768,432))
        cv2.putText(im,f'S02 {pt/12800:.3f}s PTS {pt}',(8,422),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),3)
        cv2.putText(im,f'S02 {pt/12800:.3f}s PTS {pt}',(8,422),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        images.append(im)
    images.append(np.zeros_like(images[0]))
    cv2.imwrite(str(OUT/'evidence/source_overview.jpg'),np.vstack([np.hstack(images[i:i+2])for i in range(0,8,2)]))
    emit('source_ready',frames=len(vis),seconds=time.monotonic()-start)


def verify_authority():
    authority=read(OUT/'evidence/native_authority.json')
    for ref in authority['files']+[authority['scene_state']]:
        if file_ref(ref['path'])!=ref:raise ValueError('Current native authority changed: '+ref['path'])


def inventory():
    counts=collections.Counter();features=collections.Counter();frames=collections.Counter();origins=collections.Counter()
    p=REC/'dense_ice_v1/canonical_v3/s02/rows.jsonl.gz'
    for _,r in rows(p):
        counts[r['source_kind']]+=1;features[r['feature_id']]+=1;frames[r['native_pts']]+=1
        origins[str(r.get('navigation',{}).get('source_file',''))]+=1
    excluded={};extra=collections.Counter();refs=[file_ref(p)]
    for rel in ['dense_ice_v1/branch_delta_v2/manifest.json','source_overlay_check_v1/exclusions_manifest.json']:
        q=REC/rel;d=read(q);refs.append(file_ref(q))
        ep=Path(d.get('exclusions_file',d.get('exclusions',{}).get('file','')))
        expected=d.get('exclusions_sha256',d.get('exclusions',{}).get('sha256'))
        ref=file_ref(ep)
        if ref['sha256']!=expected:raise ValueError('Historical correction changed')
        refs.append(ref)
        for n,r in rows(ep):
            if r.get('shot_id')=='s02' or FIRST<=r.get('native_pts',0)<=LAST:
                excluded[r['source_row_id']]=dict(record=r,source=str(ep),jsonl_line_0based=n)
        if 'rows_file'in d:
            refs.append(file_ref(d['rows_file']))
            for _,r in rows(d['rows_file']):
                if r['shot_id']=='s02':extra[r['feature_id']]+=1
    withdrawn={};index=DATA/'runs/okada_s02_s09_observations_20261026/source_index.sqlite'
    with sqlite3.connect(index.as_uri()+'?mode=ro',uri=True)as db:
        for key,payload in db.execute("select id,payload from qualification where shot='s02'"):
            withdrawn[key]=json.loads(payload)
    manual=[]
    for n,r in rows(REC/'dense234_v1/generation_v2/selected_rows.jsonl.gz'):
        if r.get('shot_id')=='s02' and 'manual' in r.get('source_kind',''):manual.append(dict(source_line=n,record=r))
    write(OUT/'evidence/source_inventory.json',dict(shot='s02',full_exposures=567,raw_ice_rows=sum(counts.values()),
        source_kinds=dict(counts),features=dict(features),frame_counts=dict(frames),navigation_origins=dict(origins),
        historical_corrections=excluded,branch_repair_rows=dict(extra),withdrawals=withdrawn,manual_rows=manual,
        source_files=refs,source_candidates_are_not_automatically_observations=True))
    emit('inventory_ready',raw_ice=sum(counts.values()),frames=len(frames),features=len(features),
        inferred=sum(v for k,v in counts.items()if 'inferred'in k),corrections=len(excluded),withdrawals=len(withdrawn),manual=len(manual))


def navigation():
    from ice_points_s04 import navigation as shared
    shared(output_dir=OUT,visibility_path=OUT/'evidence/visibility.json',
        authority_init=verify_authority,authority_verify=verify_authority)


def native_preview():
    grid=NativeGrid();images=[]
    for pt in [FIRST,598784,623872,649472,675072,700672,LAST]:
        im=cv2.imread(str(OUT/'source_frames'/f'{pt}.png'))
        for key,p in grid.project(pt).items():
            q=p['uv'];ok=p['valid']&np.isfinite(q).all(1)
            for a,b,good in zip(q[:-1],q[1:],ok[:-1]&ok[1:]):
                if good and np.max(abs(np.r_[a,b]))<100000:
                    cv2.line(im,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),(255,255,255),1,cv2.LINE_AA)
        im=cv2.resize(im,(960,540));cv2.putText(im,f'S02 {pt/12800:.3f}s',(10,530),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),3)
        images.append(im)
    images.append(np.zeros_like(images[0]));cv2.imwrite(str(OUT/'evidence/native_navigation_overview.jpg'),np.vstack([np.hstack(images[i:i+2])for i in range(0,8,2)]))
    grid.unchanged();emit('native_navigation_preview',paths=len(grid.cuts))


def ad_configure():
    """Configure existing readers only inside this S02 worker process."""
    import ad_replacement_s01 as b
    import ad_points_s04 as shared
    import ad_entity_registration_s01 as r,ad_entity_tracks_s01 as t
    import ad_entity_measure_s01 as m,ad_entity_observability_s01 as o,ad_entity_select_s01 as s
    import ad_replace_package_s01 as pack
    adout=OUT/'ads';work=adout/'source_review'
    adout.mkdir(exist_ok=True);work.mkdir(exist_ok=True)
    b.OUT=adout;shared.OUT=adout;shared.WORK=work
    shared.FIRST=FIRST;shared.LAST=LAST;shared.ADNUMBERS=range(1,14);shared.PREFIX='s02_native_source_v1_'
    r.OUT=work;r.VIS={f['pts']:f for f in read(OUT/'evidence/visibility.json')['frames']};r.current.cache_clear()
    for module in (t,m,o,s):module.OUT=work
    pack.OUT=adout
    def current_native():
        from rinkspace.domain.parameters import SceneParameterModel
        from rinkspace.session.gpu_geometry import NativeSceneGPU
        grid=NativeGrid();model=SceneParameterModel(read(SCENE),SCENE,optimize_scene=True)
        return grid.core,model,NativeSceneGPU(SCENE,grid.core,parameter_state=grid.parameter_state)
    pack.current_native=current_native
    atlases=read(adout/'atlases.json')['atlases'] if (adout/'atlases.json').exists() else []
    chosen=[a['id']for a in atlases if a['ad']in shared.ADNUMBERS and a['id']not in (0,2,5,8,74)]
    t.PRIMARY=chosen;r.ATLAS_IDS=chosen
    o.CANONICAL_BY_AD={1:[81,82,83,84],2:[1,3],3:[70,77],4:[4,6,69],5:[7]}
    o.LOCAL_PRINT_SHARPNESS_ADS={5};o.BEST_INDEPENDENT_SHAPE_ADS={1}
    shared.configure=ad_configure
    return r,t,m,o,s,atlases,chosen


def ads_prepare():
    import shutil
    adout=OUT/'ads';(adout/'source_review').mkdir(parents=True,exist_ok=True)
    # Retain atlas IDs referenced by the existing material-identity validators.
    old=DATA/'runs/native_ad_points_s04_20261026/atlases.json'
    if not (adout/'atlases.json').exists():shutil.copy2(old,adout/'atlases.json')
    write(adout/'source_review/additional_source_atlases.json',dict(atlases=[]))
    shutil.copy2(OUT/'evidence/visibility.json',adout/'visibility.json')
    import ad_replacement_s01 as b
    assets=b.assets()
    write(adout/'authority_inventory.json',dict(scene=file_ref(SCENE),camera=file_ref(CAMERA),
        assets=[dict(ad=n,**file_ref(v['path']))for n,v in assets.items()],
        source_scope=read(OUT/'evidence/native_authority.json')['shot']['processing']))
    ad_configure()
    import ad_points_s04 as shared
    shared.inventory(shot_id='s02',packet_dir=PACKETS)


def ads_anchors():
    ad_configure()
    import ad_points_s04 as shared
    shared.anchors()


def ads_tracks():
    r,t,m,o,s,ats,chosen=ad_configure();t.run()


def ads_complete_patterns():
    r,t,m,o,s,ats,chosen=ad_configure();cv2.setNumThreads(2)
    seed=read(r.OUT/'whole_artwork_seed_trial.json')['records'];doc=read(r.OUT/'source_pattern_anchors.json')
    for row in seed:doc['records'].setdefault(str(row['atlas_id']),{})[str(row['pts'])]=row
    write(r.OUT/'completed_artwork_anchors.json',doc)
    t.PRIMARY=sorted({r['atlas_id']for r in seed})
    t.run(r.OUT/'completed_artwork_anchors.json',r.OUT/'completed_artwork_charts.json')
    old=read(r.OUT/'source_identity_charts.json');addition=read(r.OUT/'completed_artwork_charts.json')
    original=r.OUT/'before_artwork_completion.json'
    if not original.exists():write(original,old)
    for ai in t.PRIMARY:old['records'][str(ai)]=addition['records'][str(ai)]
    old.setdefault('source_completion',[]).append(dict(anchors=file_ref(r.OUT/'completed_artwork_anchors.json'),
        additional_charts=file_ref(r.OUT/'completed_artwork_charts.json'),actual_material_still_reread_per_exposure=True))
    write(r.OUT/'source_identity_charts.json',old)
    write(r.OUT/'additional_source_atlases.json',dict(atlases=[]))
    emit('whole_artwork_navigation_complete',frames_by_atlas={ai:len(old['records'][str(ai)])for ai in t.PRIMARY})


def ads_measure():
    r,t,m,o,s,ats,chosen=ad_configure()
    m.run(source=dict(np.load(OUT/'ads/original_ad_candidates.npz')),banned=set(),
        first_pts=FIRST,source_prefix='s02_native_source_v1_')


def ads_source_completion():
    """Add existing exact S02 material references, not model contours."""
    import re
    import ad_replacement_s01 as b
    r,t,m,o,s,ats,chosen=ad_configure();assets=b.assets();groups={}
    def add(ad,pt,image,pid,q,proof,closed=False):
        q=np.asarray(q,float)
        if q.ndim!=2 or q.shape[1]!=2 or len(q)<2 or not np.isfinite(q).all():return
        if closed and not np.allclose(q[0],q[-1]):q=np.vstack([q,q[:1]])
        key=(ad,str(image));a=groups.setdefault(key,dict(ad=ad,image=str(image),image_sha256=b.sha(image),paths=[],partial_source_reference=True,source_pts=pt))
        a['paths'].append(dict(path_id=pid,pixels=q.tolist(),contour_sha256=hashlib.sha256(q.tobytes()).hexdigest(),proof=proof))
    for f in sorted(PACKETS.glob('*.json')):
        d=read(f);pt=d['source_pts']
        for feature in d['features']:
            match=re.match(r'Ad(\d+)\.',feature['id'])
            if not match:continue
            instance=int(match[1]);base=instance-13 if instance>=14 else instance
            name=re.sub(r'^Ad\d+\.',f'Ad{base}.',feature['id'])
            aliases=set(assets[base]['aliases'].get(name,[]))
            if len(aliases)!=1:continue
            pid=aliases.pop()
            for si,seg in enumerate(feature['segments']):
                q=seg.get('source_native_readout_uv')
                if q is None and seg.get('pixels_measured_in_this_frame'):q=seg['uv']
                if q is None:continue
                add(base,pt,OUT/f'source_frames/{pt}.png',pid,q,dict(kind='earlier_actual_S02_readout_with_unique_current_asset_alias',
                    source_packet=file_ref(f),original_feature=feature['id'],physical_instance=f'Ad{instance}',segment=si,source_pts=pt))
    # Ad13/Ad26 was outside the old atlas builder's range(2,13). Reuse the
    # current asset's own photographed evidence for every retained detail.
    asset=assets[13]
    for ob in asset['asset'].get('observations',[]):
        source=asset['asset'].get('sources',{}).get(ob.get('source_id'),{})
        image=ob.get('source_frame_path')or source.get('frame_path')
        if image and ob['feature_id']in asset['paths']:
            add(13,ob.get('pts'),image,ob['feature_id'],ob['source_native_uv'],dict(kind='current_asset_source_observation',
                asset=file_ref(asset['path']),observation_id=ob['id'],source_pts=ob.get('pts')),ob.get('closed',False))
    doc=read(OUT/'ads/atlases.json');anchors=read(r.OUT/'source_pattern_anchors.json');added=[]
    old_keys={(a['ad'],a['image'])for a in ats}
    for key,a in groups.items():
        if key in old_keys:continue
        a['id']=len(doc['atlases']);doc['atlases'].append(a);added.append(a['id'])
        pt=a['source_pts']
        if pt in r.VIS and hashlib.sha256(cv2.imread(a['image']).tobytes()).hexdigest()==r.VIS[pt]['source_BGR_sha256']:
            anchors['records'][str(a['id'])]={str(pt):dict(H=np.eye(3).tolist(),atlas_id=a['id'],pts=pt,
                method='same_authentic_source_exposure_identity',source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'],
                current_native_alias_verified=True,source_coordinates_are_navigation_only=True)}
    write(OUT/'ads/atlases.json',doc);write(r.OUT/'S02_local_source_anchors.json',anchors)
    write(r.OUT/'S02_local_source_references.json',dict(added_atlas_ids=added,
        refs=[dict(atlas_id=a['id'],ad=a['ad'],pts=a['source_pts'],paths=len(a['paths']))for a in doc['atlases']if a['id']in added]))
    emit('S02_source_references_added',atlases=len(added),paths=sum(len(a['paths'])for a in doc['atlases']if a['id']in added))


def board_texture_step(ref,H,previous,pt):
    """Use saved actual board LK support when a thin glyph cannot define affine motion."""
    import ad_entity_tracks_s01 as t
    import ad_entity_registration_s01 as r
    result,why=t._s02_original_step(ref,H,previous,pt)
    if result is not None:return result,why
    file=REC/f'dense_print_v1/source_anchor_motion/{max(previous,pt)}.json'
    if abs(previous-pt)!=STEP or not file.exists():return None,why
    packet=read(file)
    if packet.get('cross_cut')or packet.get('previous_native_pts')!=min(previous,pt):return None,why
    old=np.asarray(packet.get('source_uv',[]),np.float32);new=np.asarray(packet.get('current_uv',[]),np.float32)
    if old.ndim!=2 or old.shape!=new.shape or len(old)<12:return None,why
    if pt<previous:old,new=new,old
    q=cv2.perspectiveTransform(ref['points'][None],H)[0];lo=q.min(0)-[220,35];hi=q.max(0)+[220,35]
    keep=(old>=lo).all(1)&(old<=hi).all(1)&(np.asarray(packet['forward_backward_error_px'])<.8)
    old,new=old[keep],new[keep]
    if len(old)<12:return None,why
    A,inlier=cv2.estimateAffine2D(old,new,method=cv2.RANSAC,ransacReprojThreshold=1.25,maxIters=2500,confidence=.999)
    if A is None or inlier.sum()<10 or inlier.mean()<.65:return None,why
    HH=np.vstack([A,[0,0,1]])@H
    if r.geometry(HH,ref)is None:return None,why
    score=r.full_pattern_score(ref,HH,pt,allow_partial=True)
    if score is None:return None,'adjacent_board_context_does_not_reidentify_current_material'
    error=np.linalg.norm(cv2.transform(old[None],A)[0]-new,axis=1)
    return dict(H=HH.tolist(),feature_inliers=int(inlier.sum()),feature_pairs=len(old),source_track_p95=float(np.percentile(error[inlier[:,0]>0],95)),
        source_context_motion=file_ref(file),full_pattern_score=score,original_local_failure=why,
        source_navigation_only=True),None


def ads_complete_tracks():
    r,t,m,o,s,ats,chosen=ad_configure()
    if not hasattr(t,'_s02_original_step'):t._s02_original_step=t.step
    t.step=board_texture_step
    added=read(r.OUT/'S02_local_source_references.json')['added_atlas_ids']
    # Complete the failed official whole-word tracks too, using the same
    # physical board context followed by current pattern verification.
    doc=read(r.OUT/'S02_local_source_anchors.json')
    for row in read(r.OUT/'whole_artwork_seed_trial.json')['records']:
        doc['records'].setdefault(str(row['atlas_id']),{})[str(row['pts'])]=row
    write(r.OUT/'S02_completion_anchors.json',doc)
    t.PRIMARY=added+[71,76]
    t.run(r.OUT/'S02_completion_anchors.json',r.OUT/'S02_completion_tracks.json')
    result=read(r.OUT/'source_identity_charts.json');new=read(r.OUT/'S02_completion_tracks.json')
    for ai in t.PRIMARY:result['records'].setdefault(str(ai),{}).update(new['records'].get(str(ai),{}))
    write(r.OUT/'source_identity_charts.json',result)
    r.authority(ad_numbers=range(1,14))
    registry=read(r.OUT/'native_entity_registry.json');from rinkspace.domain.parameters import SceneParameterModel
    model=SceneParameterModel(read(SCENE),SCENE,optimize_scene=True);rebind={9:22,10:23,11:24,12:25,13:26}
    for row in registry['paths']:
        n=row['ad']
        if n in rebind:
            row['template_ad']=n;row['ad']=rebind[n];row['native_id']=row['native_id'].replace(f'print_instance/Ad{n}::',f'print_instance/Ad{rebind[n]}::',1)
            row['parameter_dependencies']=list(model.dependencies(row['native_id']))
            row['physical_instance_evidence']=[file_ref(p)for p in PACKETS.glob('*.json')if any(f['id'].startswith(f'Ad{rebind[n]}.')for f in read(p)['features'])]
    registry['shared_template_is_not_instance_identity']=True;write(r.OUT/'native_entity_registry.json',registry)
    emit('S02_actual_ad_instances_registered',template_to_instance=rebind,charts=sum(len(v)for v in result['records'].values()))


def ads_cropped_artwork():
    import ad_replacement_s01 as b,ad_source_verify_s01 as old
    r,t,m,o,s,ats,chosen=ad_configure();cv2.setNumThreads(2);asset=b.assets()[13];groups={}
    for ob in asset['asset']['observations']:
        src=asset['asset']['sources'][ob['source_id']]
        if not src.get('source_crop'):continue
        crop=src['source_crop'];box=src['source_crop_box']
        if b.sha(crop['path'])!=crop['sha256']:raise ValueError('Original ministry source crop changed')
        q=np.asarray(ob['source_native_uv'])-box[:2]
        a=groups.setdefault(crop['path'],dict(id=len(ats)+len(groups),ad=13,image=crop['path'],image_sha256=crop['sha256'],paths=[],partial_source_reference=False))
        a['paths'].append(dict(path_id=ob['feature_id'],pixels=q.tolist(),contour_sha256=hashlib.sha256(np.asarray(q,float).tobytes()).hexdigest(),
            proof=dict(kind='current_asset_observation_with_exact_source_crop_offset',asset=file_ref(asset['path']),observation_id=ob['id'],source_pts=ob['pts'],source_crop_box=box)))
    doc=read(OUT/'ads/atlases.json');known={a['image']:a for a in doc['atlases']};selected=[]
    for a in groups.values():
        if a['image']in known:selected.append(known[a['image']]['id'])
        else:a['id']=len(doc['atlases']);doc['atlases'].append(a);selected.append(a['id'])
    write(OUT/'ads/atlases.json',doc);ats=doc['atlases'];sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.008);matcher=cv2.BFMatcher(cv2.NORM_L2)
    refs={ai:r.reference(ats[ai])for ai in selected};prepared={};anchors={'records':{}}
    for ai,ref in refs.items():
        kp,ds=sift.detectAndCompute(ref['gray'],None);prepared[ai]=dict(keypoints=np.asarray([k.pt for k in kp]),descriptors=ds)
    for pt in sorted(r.VIS)[::10]:
        im,gray,mask=r.current(pt);keep=np.ones_like(gray,np.uint8)*255;keep[:185]=0;keep[cv2.dilate(mask,np.ones((9,9),np.uint8))>0]=0
        kp,ds=sift.detectAndCompute(gray,keep);xy=np.asarray([k.pt for k in kp])
        for ai,ref in refs.items():
            best=None
            for c in old.image_candidates(prepared[ai],xy,ds,matcher):
                H=np.asarray(c['H'])
                if c['inliers']<8 or r.geometry(H,ref)is None:continue
                score=r.full_pattern_score(ref,H,pt,allow_partial=True)
                if score is None:continue
                if best is None or score['pattern_correlation_median']>best['full_pattern_score']['pattern_correlation_median']:
                    best=dict(c,atlas_id=ai,pts=pt,full_pattern_score=score,method='actual_source_pattern_correspondence',source_BGR_sha256=r.VIS[pt]['source_BGR_sha256'])
            if best:anchors['records'].setdefault(str(ai),{})[str(pt)]=best
    write(r.OUT/'ministry_source_anchors.json',anchors)
    t.PRIMARY=selected;t.run(r.OUT/'ministry_source_anchors.json',r.OUT/'ministry_source_tracks.json')
    result=read(r.OUT/'source_identity_charts.json');new=read(r.OUT/'ministry_source_tracks.json')
    result['records'].update(new['records']);write(r.OUT/'source_identity_charts.json',result)
    reg=read(r.OUT/'native_entity_registry.json')
    for row in reg['paths']:
        name=row['native_id'].split('::')[1]
        for ai in selected:
            for path in ats[ai]['paths']:
                if path['path_id']==name:row['source_references'].append(dict(atlas_id=ai,source_image=ats[ai]['image'],source_image_sha256=ats[ai]['image_sha256'],source_contour_sha256=path['contour_sha256'],proof=path['proof']))
    reg['missing_source_reference']=[x['native_id']for x in reg['paths']if not x['source_references']];write(r.OUT/'native_entity_registry.json',reg)
    emit('current_retained_ministry_materials',anchors=sum(map(len,anchors['records'].values())),charts=sum(map(len,new['records'].values())),unreferenced=len(reg['missing_source_reference']))


def baseline():
    """Qualify the full original pool before deciding which gaps need rereading."""
    from ice_points_s04 import qualify,source_domain,competing_source_ids
    from rinkspace.review.point_connections import polyline_order
    cv2.setNumThreads(2);verify_authority();auth=read(OUT/'evidence/native_authority.json')
    geo=NativeSourceGeometry(SCENE,auth['native_library']);grid=NativeGrid()
    inv=read(OUT/'evidence/source_inventory.json')
    excluded={k:dict(reason=v['record'].get('reason',v['record'].get('reason_code')))for k,v in inv['historical_corrections'].items()}
    visibility={r['pts']:r for r in read(OUT/'evidence/visibility.json')['frames']}
    original=REC/'dense_ice_v1/canonical_v3/s02/rows.jsonl.gz';ref=file_ref(original)
    groups=iter(itertools.groupby(rows(original),key=lambda t:t[1]['native_pts']));current=next(groups,None)
    extras=collections.defaultdict(list)
    delta=read(REC/'dense_ice_v1/branch_delta_v2/manifest.json')
    for n,r in rows(delta['rows_file']):
        if r['shot_id']=='s02':extras[r['native_pts']].append((n,r,delta['rows_file']))
    totals=collections.Counter();frames=[];start=time.monotonic();chosen=0
    dest=OUT/'evidence/baseline_ice';dest.mkdir(exist_ok=True)
    with gzip.open(OUT/'evidence/historical_ice_dispositions.jsonl.gz','wt',encoding='utf8')as ledger:
        for fi,pt in enumerate(range(FIRST,LAST+1,STEP)):
            entries=[(n,r,str(original))for n,r in current[1]] if current and current[0]==pt else []
            if current and current[0]==pt:current=next(groups,None)
            entries+=extras[pt];v=visibility[pt]
            im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)
            graphics,border,distance,receipt=source_domain(im,mask,read(REC/f'dense_board_v1/qualified_frames/{pt}.json'))
            by=collections.defaultdict(list);decisions=[]
            for n,r,p in entries:
                native=material_path(geo,r)
                reason=qualify(r,native,excluded,inv['withdrawals'],graphics,border,distance)
                if reason:
                    totals[reason]+=1
                    ledger.write(json.dumps(dict(source_row_id=r['source_row_id'],pts=pt,feature_id=r['feature_id'],
                        source_ref=dict(path=p,jsonl_line_0based=n),reason=reason),ensure_ascii=False)+'\n')
                else:by[native['id']].append(dict(r,native_path_id=native['id'],source_ref=dict(path=p,jsonl_line_0based=n)))
            projections=grid.project(pt);features={}
            for pid,records in by.items():
                conflicts=competing_source_ids(records)
                if conflicts:
                    totals['competing_same_query_actual_profiles']+=sum(r['source_row_id']in conflicts for r in records)
                    records=[r for r in records if r['source_row_id']not in conflicts]
                if not records:continue
                path=geo.current_paths[pid];options={}
                if path['kind']in ('ice_circle','ice_ellipse'):
                    curve=projections[pid]['uv'];ok=projections[pid]['valid']
                    if ok.all() and np.isfinite(curve).all():
                        order,dist=polyline_order(np.asarray([r['uv']for r in records]),curve)
                        length=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
                        guide=np.column_stack([np.interp(order,length,curve[:,k])for k in (0,1)])
                        options=dict(source_curve_guide=guide,source_curve_order=order/max(length[-1],1e-12)*2*np.pi)
                keep,why=select_representatives(records,path,**options)
                features[path['label']]=dict(native_path_id=pid,raw_count=len(records),rows=[dict(records[k],
                    selection=dict(reasons=sorted(why[k]),original_UV_unchanged=True))for k in keep])
                chosen+=len(keep)
            packet=dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features=features,
                source_file=ref,model_or_fitted_curve_pixels_emitted=False,human_acceptance=False)
            write(dest/f'{pt}.json',packet)
            frames.append(dict(pts=pt,eligible=sum(len(r)for r in by.values()),selected=sum(len(f['rows'])for f in features.values()),
                features={k:len(f['rows'])for k,f in features.items()}))
            if fi%100==0:emit('historical_ice_qualification',frames=fi+1,total=567,selected=chosen,seconds=time.monotonic()-start)
    write(OUT/'evidence/baseline_coverage.json',dict(frames=frames,selected=chosen,reasons=dict(totals),full_scope=567,
        stage='original_source_qualification_before_whole_shape_and_gap_repairs',complete_accuracy_claimed=False))
    verify_authority();emit('historical_ice_ready',frames=len(frames),selected=chosen,reasons=dict(totals))


def ice_reread(requested=None, *, board_dir=None, source_prefix='s02', photographic=None, source_finite_extents=False):
    from types import SimpleNamespace
    from ice_points_s04 import FiniteSourceIdentity,source_domain
    from reread_s04_lines import NativeLineReader
    from calibration.program_tracking.audit_first_shot_circle_material_identity import conflicts
    cv2.setNumThreads(2);grid=NativeGrid();motion=OUT/'evidence/source_navigation.json'
    photo=photographic if photographic is not None else FiniteSourceIdentity(output_dir=OUT,visibility_path=OUT/'evidence/visibility.json',packet_dir=PACKETS,first_pts=FIRST)
    # Every original full-pool line can guide its own exposure. Sparse 16-frame
    # review packets must not be the only identity entry to the dense reader.
    for p in ([] if photographic is not None else sorted((OUT/'evidence/baseline_ice').glob('*.json'))):
        packet=read(p);pt=packet['source_pts']
        for name,f in packet['features'].items():
            path=grid.cuts[f['native_path_id']]
            if path['kind']!='ice_segment' or len(f['rows'])<2:continue
            uv=np.asarray([r['uv']for r in f['rows']]);photo.guides[name].insert(0,dict(pts=pt,path=str(p),spans=[uv],operator='paint_center',signal=None,transition=None,centre=None,width=None))
    lines=NativeLineReader(grid,photographic=photo,motion_path=motion,source_prefix=source_prefix,source_cross_line_guides=True,
        source_parallel_references=source_finite_extents,source_finite_extents=source_finite_extents)
    for path in lines.paths:
        path['_source_query_spacing_px']=35
        if source_finite_extents:path['_separate_profile_support']=True
    house=SourceHouseReader(grid,motion_path=motion)
    vis=read(OUT/'evidence/visibility.json');lookup={r['pts']:r for r in vis['frames']}
    requested=sorted(lookup)if requested is None else sorted(requested)
    for kind in ['lines','circles']:(OUT/f'evidence/reread_{kind}').mkdir(exist_ok=True)
    algorithm=file_ref(Path(__file__))
    for fi,pt in enumerate(requested):
        start=time.monotonic();v=lookup[pt];im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)
        if hashlib.sha256(im.tobytes()).hexdigest()!=v['source_BGR_sha256']:raise ValueError('Wrong source RGB')
        board_path=(Path(board_dir)/f'{pt}.json')if board_dir is not None else REC/f'dense_board_v1/qualified_frames/{pt}.json'
        graphics,border,_,_=source_domain(im,mask,read(board_path))
        excluded=(mask>0)|graphics|(np.arange(1080)[:,None]<=border[None])
        if v.get('near_outside_ref'):
            near=cv2.imread(v['near_outside_ref']['file'],0)
            if near is None or near.shape!=excluded.shape:raise ValueError('Missing current-source near-board domain')
            excluded|=near>0
        lf,ld=lines.read(pt,im,excluded,vis['source_video_sha256'])
        write(OUT/f'evidence/reread_lines/{pt}.json',dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],
            features=lf,diagnostics=ld,algorithm=algorithm,model_or_fitted_curve_pixels_emitted=False))
        native=SimpleNamespace(frame=SimpleNamespace(rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB),material_signals={}),mask=excluded)
        features={};diagnostics={};frame_results={};shape_features=[]
        for fid in house.groups:
            house.prepare_family(fid,pt)
            curves=[house.projection(house.groups[fid],pt,role,np.linspace(0,2*np.pi,513))[0]for role in house.groups[fid]['radii']]
            if not all(np.isfinite(q).all()for q in curves):
                diagnostics[fid]=dict(status='native_family_outside_valid_projection');continue
            cloud=np.vstack(curves)
            if (cloud[:,0].max()<-75 or cloud[:,0].min()>1995 or cloud[:,1].max()<-75 or cloud[:,1].min()>1155):
                diagnostics[fid]=dict(status='native_family_outside_source_image');continue
            result,diag=house.read(fid,pt,native);diagnostics[fid]=diag;frame_results[fid]=result
            for role,data in result.items():
                path=house.native_paths[(fid,role)][2];observed=[]
                for row in data['rows']:
                    r={k:v for k,v in row.items()if k not in ('representative_uv','representative_to_actual_material_px')}
                    r.update(native_path_id=path['id'],feature_id=path['label'],native_pts=pt,
                        source_row_id=f'{source_prefix}_joint_house_RGB/{pt}/{fid}/{role}/{row["query_index"]}')
                    observed.append(r)
                data['joint_group_navigation']['source_registration']=house.registration_proof.get((pt,fid))
                features[path['label']]=dict(native_path_id=path['id'],rows=observed,navigation=data['joint_group_navigation'])
                ellipse=cv2.fitEllipse(np.asarray(data['curve'],np.float32))
                shape_features.append(dict(target_id=path['label'],measurement_operator='paint_center'if house.groups[fid]['outline']else'material_interface',
                    segments=[dict(uv=[r['uv']for r in observed],source_evidence=[r['source_evidence']for r in observed])],
                    geometric_interpretation=dict(kind='projected_physical_circle_ellipse',center_uv=ellipse[0],axes_px=ellipse[1],angle_degrees=ellipse[2])))
        material_conflicts=conflicts(shape_features);bad={r['target_id']for r in material_conflicts}
        rejected={name:features.pop(name)for name in bad}
        for fid,result in frame_results.items():
            if not any(name.startswith('ice.'+fid+':')for name in bad):house.record_source_family(fid,pt,result)
        write(OUT/f'evidence/reread_circles/{pt}.json',dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],
            features=features,diagnostics=diagnostics,algorithm=algorithm,material_identity_conflicts=material_conflicts,
            rejected_source_candidates=rejected,model_or_fitted_curve_pixels_emitted=False))
        if fi%25==0 or len(requested)<20 or fi==len(requested)-1:
            emit('current_RGB_ice',frame=fi+1,total=len(requested),pts=pt,lines={k:len(f['rows'])for k,f in lf.items()},
                circles={k:len(f['rows'])for k,f in features.items()},seconds=time.monotonic()-start)
    verify_authority()


def ice_gap_reread(*, board_dir=None, source_prefix='s02', use_registered_motion=False, line_input_dir=None):
    """Use actual source reads on both sides of gaps to navigate fresh RGB reads."""
    from ice_points_s04 import source_domain
    from reread_s04_lines import NativeLineReader
    from calibration.program_tracking.extract_first_shot_ice import Extractor,NativeProfiles
    from calibration.program_tracking.native_ice_line_continuity import straight_guide
    cv2.setNumThreads(2);grid=NativeGrid()
    # This reader contributes native finite paths and the existing family
    # audit. No S04 source guides are used by the gap operation below.
    from ice_points_s04 import FiniteSourceIdentity
    photo=FiniteSourceIdentity(output_dir=OUT,visibility_path=OUT/'evidence/visibility.json',packet_dir=PACKETS,first_pts=FIRST)
    reader=NativeLineReader(grid,photographic=photo,motion_path=OUT/'evidence/source_navigation.json',source_prefix=source_prefix)
    originals={};guides=collections.defaultdict(list)
    for pt in range(FIRST,LAST+1,STEP):
        packet=read((Path(line_input_dir)/f'{pt}.json')if line_input_dir is not None else OUT/f'evidence/reread_lines/{pt}.json');originals[pt]=packet
        for name,f in packet['features'].items():
            if len(f['rows'])<2:continue
            q=np.asarray([r['uv']for r in f['rows']]);centre=q.mean(0);direction=np.linalg.svd(q-centre,full_matrices=False)[2][0]
            if direction[0]<0:direction*=-1
            tt=(q-centre)@direction;guides[name].append(dict(pts=pt,ends=centre+np.array([tt.min(),tt.max()])[:,None]*direction))
    folder=OUT/'evidence/reread_gap_lines';folder.mkdir(exist_ok=True);summary=[];visibility=read(OUT/'evidence/visibility.json')
    for fi,v in enumerate(visibility['frames']):
        pt=v['pts'];im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)
        board_path=(Path(board_dir)/f'{pt}.json')if board_dir is not None else REC/f'dense_board_v1/qualified_frames/{pt}.json'
        graphics,border,_,_=source_domain(im,mask,read(board_path))
        excluded=(mask>0)|graphics|(np.arange(1080)[:,None]<=border[None])
        if v.get('near_outside_ref'):excluded|=cv2.imread(v['near_outside_ref']['file'],0)>0
        native=NativeProfiles(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),dict(source_video_sha256=visibility['source_video_sha256'],native_pts=pt,time_base='1/12800'),
            excluded)
        projections=grid.project(pt);reader.current_projections=projections;features=[];searches={};attempts={};brackets={};before=originals[pt]['features']
        if line_input_dir is not None:reader.source_parallel_reference_ids=set(originals[pt].get('source_parallel_reference_ids',[]))
        for path in reader.paths:
            name=path['label'];local=dict(path,_source_query_spacing_px=35);proj=projections[path['id']]
            if line_input_dir is not None:local['_separate_profile_support']=True
            good=proj['valid']&(proj['uv']>=[-120,-120]).all(1)&(proj['uv']<=[2039,1199]).all(1)
            if good.sum()<2:continue
            finite=straight_guide(proj['uv'][good],proj['uv'][good])
            if name in before:
                old=before[name];features.append(dict(target_id=name,world_curve=path['curve_id'],point_count=len(old['rows']),
                    source_line_fit=dict(old['navigation']['source_line_fit'],direct_readouts=old['rows'])))
                searches[path['curve_id']]=(local,finite,'existing_actual_current_RGB');continue
            gg=guides[name];past=[g for g in gg if g['pts']<pt];future=[g for g in gg if g['pts']>pt]
            if not past or not future:continue
            a,c=past[-1],future[0];alpha=(pt-a['pts'])/(c['pts']-a['pts'])
            if use_registered_motion:
                H,component=photo.transforms[pt];Ha,ca=photo.transforms[a['pts']];Hc,cc=photo.transforms[c['pts']]
                if component!=ca or component!=cc:
                    attempts[name]=dict(reason='two_sided_source_identity_crosses_failed_image_registration');continue
                qa=cv2.perspectiveTransform(a['ends'][None],H@np.linalg.inv(Ha))[0]
                qc=cv2.perspectiveTransform(c['ends'][None],H@np.linalg.inv(Hc))[0]
                q=qa*(1-alpha)+qc*alpha
            else:q=a['ends']*(1-alpha)+c['ends']*alpha
            # Observed extent interpolates only inside the finite current path.
            guide=straight_guide(q,q)
            if len(guide)<2:continue
            slope=None;delta=finite[-1]-finite[0]
            if abs(delta[0])>1e-8:slope=float(delta[1]/delta[0])
            f,_,_=Extractor.line(None,local,guide,native,family_slope=slope)
            attempts[name]=f['extraction_diagnostic'];searches[path['curve_id']]=(local,guide,'two_sided_actual_source_gap_navigation')
            brackets[name]=dict(before_pts=a['pts'],after_pts=c['pts'],fraction=alpha,source_positions_only_navigate=True,
                actual_adjacent_image_registration_used=use_registered_motion)
            if f.get('source_line_fit'):features.append(f)
        checked,rejected,audit=Extractor.family_audit(reader,pt,features,searches,native)
        result={}
        for f in checked:
            name=f['target_id']
            if name in before:continue
            path=next(q for q in reader.paths if q['label']==name);rr=[]
            for j,r in enumerate(f['source_line_fit']['direct_readouts']):
                rr.append(dict(r,native_path_id=path['id'],feature_id=name,native_pts=pt,source_row_id=f'{source_prefix}_gap_RGB/{pt}/{path["curve_id"]}/{j}'))
            result[name]=dict(native_path_id=path['id'],rows=rr,navigation=dict(source_bracket=brackets[name],
                source_line_fit={k:v for k,v in f['source_line_fit'].items()if k!='direct_readouts'},native_finite_shape_unchanged=True))
        write(folder/f'{pt}.json',dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features=result,attempts=attempts,
            rejected_candidates=rejected,model_or_fitted_curve_pixels_emitted=False))
        summary.append(dict(pts=pt,rows=sum(len(f['rows'])for f in result.values()),paths=list(result)))
        if fi%100==0:emit('ice_gap_current_RGB',frames=fi+1,rows=sum(r['rows']for r in summary))
    write(folder/'coverage.json',dict(frames=summary));verify_authority()


def red_arc_reread():
    from types import SimpleNamespace
    from ice_points_s04 import source_domain
    cv2.setNumThreads(2);grid=NativeGrid();house=SourceHouseReader(grid,motion_path=OUT/'evidence/source_navigation.json')
    folder=OUT/'evidence/red_arc_RGB';folder.mkdir(exist_ok=True);counts=[]
    for v in read(OUT/'evidence/visibility.json')['frames']:
        pt=v['pts'];im=cv2.imread(v['image']);mask=cv2.imread(v['mask_path'],0)
        graphics,border,_,_=source_domain(im,mask,read(REC/f'dense_board_v1/qualified_frames/{pt}.json'))
        native=SimpleNamespace(frame=SimpleNamespace(rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB),material_signals={}),mask=(mask>0)|graphics|(np.arange(1080)[:,None]<=border[None]))
        features={};diagnostics={}
        for fid in ('A1','A2','A3','A4'):
            house.prepare_family(fid,pt);group=house.groups[fid]
            curve,depth=house.projection(group,pt,'outer',np.linspace(0,2*np.pi,513))
            if not np.isfinite(curve).all() or not ((curve>=[-75,-75]).all(1)&(curve<=[1995,1155]).all(1)&(depth>.05)).any():continue
            result,diag=house.read(fid,pt,native);diagnostics[fid]=diag
            for role,data in result.items():
                path=house.native_paths[(fid,role)][2];rr=[]
                for r in data['rows']:
                    row={k:v for k,v in r.items()if k not in ('representative_uv','representative_to_actual_material_px')}
                    row.update(native_path_id=path['id'],feature_id=path['label'],native_pts=pt,source_row_id=f's02_red_arc_RGB/{pt}/{fid}/{r["query_index"]}')
                    rr.append(row)
                features[path['label']]=dict(native_path_id=path['id'],rows=rr,navigation=data['joint_group_navigation'])
        write(folder/f'{pt}.json',dict(source_pts=pt,source_BGR_sha256=v['source_BGR_sha256'],features=features,diagnostics=diagnostics,
            algorithm=file_ref(Path(__file__)),actual_material_signal='red',current_native_shape_unchanged=True,model_or_fitted_curve_pixels_emitted=False))
        counts.append(dict(pts=pt,features={k:len(f['rows'])for k,f in features.items()}))
        if len(counts)%100==0:emit('actual_red_arc',frames=len(counts),total=567,nonempty=sum(bool(c['features'])for c in counts))
    write(folder/'coverage.json',dict(frames=counts));verify_authority()


from reread_s04_house import NativeHouseReader


class SourceHouseReader(NativeHouseReader):
    """Register sufficiently observed source circles, retaining native geometry.

    The existing full-pool measured curve initializes the source search, just
    as photographic word charts initialize advertisements. Its image affine
    map is not a scene parameter or a delivered point. All layers share it.
    Clipped/unresolved source shapes keep the existing fixed native search.
    """
    def __init__(self,grid,source_pool_dir=None,**kw):
        super().__init__(grid,**kw);self.registration={};self.registration_proof={};self.active_family=None
        self.source_pool_dir=Path(source_pool_dir)if source_pool_dir is not None else None
        from rinkspace.compat.d78_entities import native_definition
        for path in self.scene['paths']:
            if path['layer']!='ice':continue
            kind,params=native_definition(self.scene,path,grid.core)
            if kind!=6:continue
            fid=path['label'].removeprefix('ice.')
            self.groups[fid]=dict(center=np.array(params[:2]),radii={'outer':max(params[3:5])},anchor=FIRST,
                authority='current_native_half_ellipse_material_boundary',paths=[dict(path,feature_id=fid)],outline=False)
            self.native_paths[(fid,'outer')]=(kind,params,path)

    def prepare_family(self,fid,pt):
        self.active_family=fid;key=(pt,fid)
        if key in self.registration:return
        self.registration[key]=np.eye(3)
        group=self.groups[fid]
        if self.native_paths[(fid,'outer')][0]!=1:return
        base=super().projection(group,pt,'outer',np.linspace(0,2*np.pi,513))[0]
        if not np.isfinite(base).all():return
        packet_path=(self.source_pool_dir/f'{pt}.json')if self.source_pool_dir is not None else OUT/f'evidence/baseline_ice/{pt}.json';packet=read(packet_path)
        name=self.native_paths[(fid,'outer')][2]['label'];feature=packet['features'].get(name)
        if not feature or len(feature['rows'])<6:return
        uv=np.asarray([r['uv']for r in feature['rows']if not r.get('historical_identity_hold')],np.float32)
        if len(uv)<6:return
        try:
            target=cv2.fitEllipse(uv);centre,axes,angle=target
            if min(axes)<=0:return
            closest=cv2.getClosestEllipsePoints(target,uv).reshape(-1,2)
            good=np.linalg.norm(closest-uv,axis=1)<=3.
            if good.sum()<6 or good.mean()<.75:return
            target=cv2.fitEllipse(uv[good]);centre,axes,angle=target
            # The refit changes the curve. Its old inlier mask is not proof
            # that the same source points still support the final ellipse.
            final_closest=cv2.getClosestEllipsePoints(target,uv[good]).reshape(-1,2)
            if np.any(np.linalg.norm(final_closest-uv[good],axis=1)>3.):return
            rotation=cv2.getRotationMatrix2D((0,0),-angle,1.)[:,:2]
            local=(uv[good]-centre)@rotation
            # Both material turns and both sides constrain the shape. A short
            # arc does not determine a replacement ellipse from a few pixels.
            extent=np.ptp(local,axis=0)/axes
            if np.min(extent)<.80:return
            original=cv2.fitEllipse(np.asarray(base,np.float32))
            def transform(e):
                c,ax,a=e;a=np.deg2rad(a);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
                T=np.eye(3);T[:2,:2]=R@np.diag(np.asarray(ax)*.5);T[:2,2]=c;return T
            H=transform(target)@np.linalg.inv(transform(original))
            # Eigenvector signs do not change an ellipse; choose the map near
            # the original source orientation, never a 180-degree reflection.
            if np.trace(H[:2,:2])<0:
                T=transform(target);T[:2,:2]*=-1;H=T@np.linalg.inv(transform(original))
            if np.linalg.det(H[:2,:2])<=0:return
        except (cv2.error,np.linalg.LinAlgError):return
        self.registration[key]=H;self.cache.clear()
        self.registration_proof[key]=dict(H_source_from_native_search=H.tolist(),source_ref=file_ref(packet_path),
            original_actual_material_points=len(uv),coherent_source_points=int(good.sum()),source_shape_axis_coverage=extent.tolist(),
            one_registration_for_all_native_radii=True,coordinates_not_observations=True,scene_values_not_changed=True)

    def _project(self,pts,xyz):
        uv,depth=super()._project(pts,xyz)
        H=self.registration.get((pts,self.active_family))
        if H is not None:uv=cv2.perspectiveTransform(np.asarray(uv,float)[None],H)[0]
        return uv,depth

    def projection(self,group,index,role,theta,parameters=None):
        self.active_family=group['paths'][0]['feature_id']
        kind,params,path=self.native_paths[(self.active_family,role)]
        if kind==6:
            # A native half ellipse is an open arc. Wrapping its endpoint
            # would introduce a nonexistent closing chord into the reader.
            stations=np.clip(np.asarray(theta)/(2*np.pi),0.,1.)
            xyz=np.asarray(self.grid.core.sample_path(self.scene['board'],kind,params,[],stations.tolist()))
            uv,depth=self._project(index,xyz)
            return self.source_transform(uv,parameters),depth
        return super().projection(group,index,role,theta,parameters)

    def measurements(self,group,index,native,parameters):
        fid=group['paths'][0]['feature_id']
        if self.native_paths[(fid,'outer')][0]==6:
            from calibration.program_tracking.extract_first_shot_ice import NativeProfiles
            from rinkspace.review.point_connections import polyline_order
            # Reuse the historical S02 blue material-interface operator for
            # the current native open half ellipse. The grey house's outer
            # fill operator is not the same material, despite a shared role.
            curve=self.projection(group,index,'outer',np.linspace(0,2*np.pi,513),parameters)[0]
            if not np.isfinite(curve).all():return [],{}
            if np.linalg.norm(np.diff(curve,axis=0),axis=1).sum()<=1e-9:
                return [],{'outer':dict(reason='current_native_search_projection_has_no_finite_arc',historical_actual_support_preserved=True)}
            local=dict(feature_id=fid,operator='material_interface',material_signal='blue',material_transition='light_to_dark',
                _circle=True,_maximum_width=35,_minimum_contrast=1.3,_maximum_sigma=14)
            reader=NativeProfiles(native.frame.rgb,dict(source_video_sha256=read(OUT/'evidence/visibility.json')['source_video_sha256'],native_pts=index,time_base='1/12800'),native.mask)
            records,diagnostic=reader.sample(local,curve,radius=48,spacing=35)
            pool=getattr(self,'source_material_pool',{}).get(index,{}).get(fid,[])
            if pool:
                actual=np.asarray([r['uv']for r in records]).reshape(-1,2)
                for r in pool:
                    if len(actual) and np.linalg.norm(actual-r['uv'],axis=1).min()<2:continue
                    records.append(dict(r,source_evidence=dict(r['source_evidence'])))
                diagnostic['restored_original_actual_blue_material_rows']=len(pool)
            if records:
                arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
                order,_=polyline_order(np.asarray([r['source_evidence']['query_uv']for r in records]),curve)
                for i,(r,d)in enumerate(zip(records,order)):
                    r.update(role='outer',theta=float(np.interp(d,arc,np.linspace(0,2*np.pi,513))),query_index=i)
                    r['source_evidence'].update(source_operator_origin='existing_S02_blue_half_ellipse_material_interface',declared_inside_outside_materials=['blue_goal_crease_fill','ice'])
            return records,{'outer':diagnostic}
        if fid in ('A1','A2','A3','A4'):
            from calibration.program_tracking.extract_first_shot_ice import NativeProfiles
            from rinkspace.review.point_connections import polyline_order
            # Historical S02 source guides identify these as red painted
            # outlines. Reuse its signed RGB two-flank operator, retaining
            # the exact current native curve rather than a grey-house reader.
            curve=self.projection(group,index,'outer',np.linspace(0,2*np.pi,513),parameters)[0]
            if not np.isfinite(curve).all():return [],{}
            local=dict(feature_id=fid,operator='paint_center',material_signal='red',material_transition=None,
                _circle=True,_maximum_width=35,_minimum_contrast=1.3,_maximum_sigma=14)
            reader=NativeProfiles(native.frame.rgb,dict(source_video_sha256=read(OUT/'evidence/visibility.json')['source_video_sha256'],
                native_pts=index,time_base='1/12800'),native.mask)
            records,diagnostic=reader.sample(local,curve,radius=48,spacing=35)
            if records:
                arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
                order,_=polyline_order(np.asarray([r['source_evidence']['query_uv']for r in records]),curve)
                for i,(r,d)in enumerate(zip(records,order)):
                    r.update(role='outer',theta=float(np.interp(d,arc,np.linspace(0,2*np.pi,513))),query_index=i)
                    ev=r['source_evidence'];ev['narrow_paint_width_px']=float(np.linalg.norm(np.diff(ev['actual_source_sides_uv'],axis=0)))
                    ev['source_operator_origin']='existing_S02_red_outline_reader_with_current_native_curve'
            return records,{'outer':diagnostic}
        records,diagnostic=super().measurements(group,index,native,parameters)
        if group is self.groups.get('C0'):
            from calibration.program_tracking.repair_shot2_ice import filter_grey_circle_readouts
            records,rejected=filter_grey_circle_readouts(records,native.frame.rgb)
            diagnostic.setdefault('outer',{}).update(eligible_native_material_readouts=len(records),crossing_red_band_candidates=rejected)
        return records,diagnostic

    def read(self,fid,index,native):
        # The registered complete source family uses the established joint
        # reader. The fixed-shape fallback remains for clipped or weak arcs.
        if fid.startswith('A'):
            from calibration.program_tracking.native_ice_short_arc_repair import short_native_outer_arc
            return short_native_outer_arc(self,fid,index,native,allow_outline=True,preserve_native_projection=True)
        if (index,fid)in self.registration_proof:
            from calibration.program_tracking.native_ice_material_repair import NestedHouseReader
            return NestedHouseReader.read(self,fid,index,native)
        return super().read(fid,index,native)


class NativeGrid:
    def __init__(self, *, full_model=False, observed_path_ids=None):
        from rinkspace.core.native import NativeCore
        from rinkspace.domain.parameters import SceneParameterModel,ParameterState
        from rinkspace.compat.d78_entities import sample_entities
        from rinkspace.review.scene import sample_review_scene
        auth=read(OUT/'evidence/native_authority.json');verify_authority()
        scene=read(SCENE);self.core=NativeCore(auth['native_library'])
        model=SceneParameterModel(scene,SCENE,optimize_scene=True)
        state=dict(np.load(auth['scene_state']['path'],allow_pickle=False))
        self.parameter_state=ParameterState(model.initial_state.layout,state['scene_values'])
        self.scene=self.parameter_state.apply_scene(scene)
        camera=read(CAMERA)
        self.state=dict(scene_values=state['scene_values'],rig=np.asarray(camera['rig']),controls=np.load(CAMERA.parent/camera['controls_ref']))
        self.first=camera['first_control_time'];self.dt=camera['control_dt'];self.reference=camera['reference_pts']
        self.manifest=dict(scene_path=str(SCENE),native_library=auth['native_library'])
        # The full drawing includes unmeasured portions of each path. The
        # existing renderer can show exact native SZ beyond the board domain;
        # this is not a fit constraint. Actual observation stations are checked
        # separately by the unmodified native resident evaluator.
        self.samples=(sample_review_scene(scene,self.core,SCENE,parameter_state=self.parameter_state,observed_path_ids=set())
            if full_model else sample_entities(self.scene,self.core,samples=257))
        self.cuts={r['id']:r for r in self.samples['cuts']}
        if observed_path_ids is not None and not set(observed_path_ids)<=self.cuts.keys():raise ValueError('Observed paths missing from full current model')
        self.receipt=dict(source_files=auth['files'],geometry_numeric_source=auth['scene_state'],
            camera_and_motion_source=file_ref(CAMERA),all_current_paths=len(self.cuts),
            missing_entities=self.samples['missing_entities'],native_geometry_or_parameters_changed=False,
            drawing_surface_extensions=self.samples.get('unobserved_print_surface_extensions',[]),
            actual_station_domain_check='points/native_evaluation.npz; complete drawing is not observation support',
            projection_is_navigation_and_review_not_a_measurement=True)

    @staticmethod
    def order(points,path):
        return SharedNativeGrid.order(points,path)

    def time(self,pt):return (pt-self.reference)/12800

    def project_xyz(self,pt,xyz):
        return self.core.project_motion(self.state['rig'],self.state['controls'],self.first,self.dt,1080,self.time(pt),xyz)

    def project(self,pt):
        r=self.project_xyz(pt,self.samples['xyz'])
        return {key:dict(cut,uv=r['uv'][cut['begin']:cut['end']],valid=r['status'][cut['begin']:cut['end']]==0)for key,cut in self.cuts.items()}

    def unchanged(self):verify_authority()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','inventory','navigation','native_preview','ads_prepare','ads_anchors','ads_tracks','ads_complete_patterns','ads_measure','baseline','ice_reread'])
    parser.add_argument('--pts',type=int,nargs='+')
    args=parser.parse_args()
    if args.stage=='ice_reread':ice_reread(args.pts)
    else:globals()[args.stage]()
