"""Actual native 2D print-texture candidates with unknown board coordinates.

Source-reviewed print interiors supply material ownership at the seed only.
They never inherit an advertisement world corner, an ice H, or a world XY.
"""
from pathlib import Path
from collections import defaultdict,Counter
import gzip,json,hashlib
import av,cv2,numpy as np
from .io_utils import read_json,read_jsonl,write_json,file_sha256
from .source_texture_tracklets import _source_mask
from .native_entity_tracklets import NAVIGATION,_ncc
from .source_worker import _channels,_remeasure


def _ref(path):
    p=Path(path).resolve();return dict(file=str(p),sha256=file_sha256(p))


def acquire_board_texture_candidates(scope_file,roles_file,seed_review,actor_ledgers,output):
    out=Path(output).resolve()
    if out.exists():raise ValueError('New immutable board-texture candidate generation required')
    scope=read_json(scope_file);source=scope['source'];review=read_json(seed_review)
    if (review.get('schema')!='rinkfit.native_board_print_texture_seed_review.v1'
        or review.get('authority')!='assistant_native_pixel_review' or not review.get('source_material_seed_ownership_reviewed')
        or review['source_video_sha256']!=source['sha256'] or file_sha256(source['path'])!=source['sha256']):
        raise ValueError('Exact source-reviewed board print seeds and actual native programme video required')
    roles=read_json(roles_file)['frame_roles'];clock={p:s['id']for s in scope['shots']for p in range(s['first_pts'],s['last_pts']+1,256)}
    regions=defaultdict(list)
    for region in review['regions']:
        if (region.get('unknown_board_surface_position')is not True or region.get('known_world_corner_correspondence')is not False
            or not region['board_instance_id'].startswith('Ad') or region['shot_id']=='s01'
            or clock.get(region['native_pts'])!=region['shot_id']):raise ValueError('Board print texture must own an unknown surface location')
        for name in ('native_png','same_instance_source_qualification'):
            ref=region[name]
            if file_sha256(ref['file'])!=ref['sha256']:raise ValueError('Changed seed material/source ownership evidence')
        regions[region['native_pts']].append(region)
    ledgers={}
    for path in actor_ledgers:
        for row in read_jsonl(path):ledgers[row['native_pts']]=row
    out.mkdir(parents=True);(out/'ownership_crops').mkdir()
    active={};completed=[];rejections=Counter();counts=Counter();pairs=defaultdict(set);seed_status=[]
    previous=None;oldpts=None;oldsid=None;next_id=0;total=0;previous_rgb=None
    params=dict(winSize=(31,31),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
    def patch(gray,uv):return cv2.getRectSubPix(gray.astype(np.float32),(33,33),tuple(np.asarray(uv,float)))
    def crop(rgb,t,tag):
        x,y=map(int,np.rint(t['uv']));bounds=[max(0,x-64),max(0,y-64),min(1920,x+65),min(1080,y+65)]
        p=out/'ownership_crops'/(t['short_id']+'_'+tag+'.png');cv2.imwrite(str(p),rgb[bounds[1]:bounds[3],bounds[0]:bounds[2],::-1])
        return dict(**_ref(p),native_pts=t['last_pts'],current_rgb_sha256=t['last_rgb_sha256'],native_crop_xyxy=bounds)
    def finish(k,reason,pts,diagnosis=None):
        t=active.pop(k);r=dict(track_id=k,shot_id=t['region']['shot_id'],board_instance_id=t['region']['board_instance_id'],
            first_pts=t['first_pts'],last_pts=t['last_pts'],rows=t['rows'],terminal_native_pts=pts,terminal_reason=reason,
            terminal_source_diagnosis=diagnosis,ownership_seed=t['ownership_seed'],seed_review_region=t['region'],
            current_surface_point_coordinates='unknown shared continuous board [s,h]',known_world_corner_correspondence=False,
            source_geometry_qualified=False)
        if previous_rgb is not None and oldpts==t['last_pts']:r['ownership_last']=crop(previous_rgb,t,'last')
        completed.append(r);rejections[reason]+=1
    def emit(handle,t,pts,digest,fb,ncc,diagnostic,is_seed):
        nonlocal total
        if t['rows'] and pts-t['last_pts']==256:pairs[t['region']['shot_id']].add(t['last_pts'])
        t['last_pts']=pts;t['last_rgb_sha256']=digest;t['rows']+=1;counts[t['region']['shot_id']]+=1;total+=1
        row=dict(track_id=t['track_id'],shot_id=t['region']['shot_id'],native_pts=pts,time_base=[1,12800],
            source_uv=np.asarray(t['uv']).tolist(),current_uv=np.asarray(t['uv']).tolist(),current_rgb_sha256=digest,
            source_video_sha256=source['sha256'],surface_class='board',surface_instance_id=t['region']['board_instance_id'],
            source_material='physical printed graphic texture',source_measurement='current_native_RGB_2D_corner'if is_seed else'current_native_RGB_pyramidal_LK_2D_texture',
            forward_backward_px=fb,patch_ncc=ncc,two_direction_tensor_diagnostic=diagnostic,
            material_ownership_at_seed='assistant_native_pixel_review_of_exact_source_print_interior',
            continuation_material_ownership='candidate_pending_exact_track_source_review',
            seed_review_region=t['region'],fit_role='development',source_geometry_eligible=False,
            solver_eligible=False,statistical_objective_eligible=False,numeric_statistics_ready=False,
            precision_status='engineering_source_texture_no_covariance',pixels_from_model=False,
            known_world_corner_correspondence=False,existing_world_corner_or_XY_used=False,ice_H_used=False,
            physical_position='one shared unknown continuous-board [s,h] per track; not yet installed in model',
            actor_graphics_exclusion='current native RGB-bound independent source actor ledger and existing mask policy')
        handle.write(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    cv2.setNumThreads(2)
    with gzip.open(out/'candidate_rows.jsonl.gz','wt',encoding='utf8')as handle,av.open(source['path'])as con:
        stream=con.streams.video[0]
        if str(stream.time_base)!='1/12800':raise ValueError('Wrong native source clock')
        for frame in con.decode(stream):
            pts=int(frame.pts)
            if not active and pts not in regions:
                if pts>max(regions):break
                previous=previous_rgb=None;oldpts=oldsid=None;continue
            sid=clock.get(pts)
            if sid!=oldsid or oldpts is None or pts!=oldpts+256:
                for k in list(active):finish(k,'native_cut_or_source_gap',pts)
                previous=None
            ld=ledgers.get(pts)
            if sid is None or roles.get(str(pts),{}).get('role')!='development_candidate' or ld is None or ld.get('role')!='development_candidate' or ld['shot_id']!=sid:
                for k in list(active):finish(k,'guard_heldout_or_missing_independent_source_ledger',pts)
                previous=previous_rgb=None;oldpts=oldsid=None;continue
            rgb=frame.to_ndarray(format='rgb24');gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);mask,digest=_source_mask(rgb,ld,source,pts,sid);channels=_channels(rgb)
            if active:
                keys=list(active);p=np.asarray([active[k]['uv']for k in keys],np.float32).reshape(-1,1,2)
                q,status,_=cv2.calcOpticalFlowPyrLK(previous,gray,p,None,**params)
                back,bstatus,_=cv2.calcOpticalFlowPyrLK(gray,previous,q,None,**params)
                for j,k in enumerate(keys):
                    t=active[k];uv=q[j,0].astype(float);x,y=map(int,np.rint(uv));fb=float(np.linalg.norm(back[j,0]-p[j,0]));ncc=None;reason=None
                    if not status[j,0] or not bstatus[j,0] or fb>NAVIGATION['reciprocal_max_px']:reason='actual_native_forward_backward_navigation_failed'
                    elif not(0<=x<1920 and 0<=y<1080)or mask[y,x]:reason='current_texture_support_actor_overlay_image_edge_or_excluded'
                    else:
                        current_patch=patch(gray,uv);ncc=_ncc(t['patch'],current_patch)
                        if ncc<NAVIGATION['patch_ncc_min']:reason='current_native_patch_navigation_below_existing075'
                    diagnostic=_remeasure(uv,[0,0],channels,'texture',corner=True)if reason is None else None
                    if diagnostic is not None and diagnostic['status']!='entity_corner_two_direction_support':reason='current_texture_no_two_direction_information'
                    if reason is not None:finish(k,reason,pts,dict(native_pts=pts,current_rgb_sha256=digest,navigation_uv=uv.tolist(),forward_backward_px=fb,patch_ncc=ncc,two_direction_tensor_diagnostic=diagnostic));continue
                    t['uv']=uv;t['patch']=current_patch;emit(handle,t,pts,digest,fb,ncc,diagnostic,False)
            for region in regions.get(pts,[]):
                if digest!=region['current_RGB_sha256']:raise ValueError('Seed print review is not this exact native source RGB')
                area=np.zeros((1080,1920),np.uint8);poly=np.rint(region['seed_region_native_polygon_uv']).astype(np.int32);cv2.fillPoly(area,[poly],255);area[mask>0]=0
                corners=cv2.goodFeaturesToTrack(gray,maxCorners=0,qualityLevel=.015,minDistance=31,mask=area,blockSize=5)
                accepted=0;weak=0
                for point in [] if corners is None else corners[:,0]:
                    uv=point.astype(float);diagnostic=_remeasure(uv,[0,0],channels,'texture',corner=True)
                    if diagnostic['status']!='entity_corner_two_direction_support':weak+=1;continue
                    short=sid+'_'+region['board_instance_id']+'_'+str(next_id);next_id+=1;k='board_print_texture/'+short
                    t=dict(track_id=k,short_id=short,region=region,uv=uv,patch=patch(gray,uv),first_pts=pts,last_pts=pts,last_rgb_sha256=digest,rows=0)
                    t['ownership_seed']=crop(rgb,t,'seed');active[k]=t;emit(handle,t,pts,digest,None,None,diagnostic,True);accepted+=1
                seed_status.append(dict(native_pts=pts,board_instance_id=region['board_instance_id'],detected_native_2D_candidates=0 if corners is None else len(corners),two_direction_supported_seeds=accepted,weak_2D_not_started=weak))
            previous=gray;previous_rgb=rgb;oldpts=pts;oldsid=sid
    for k in list(active):finish(k,'source_or_scope_end',oldpts)
    write_json(out/'tracks.json',completed)
    coverage={s['id']:dict(possible_native_pairs=(s['last_pts']-s['first_pts'])//256,candidate_source_pairs=len(pairs[s['id']]),new_material_qualified_pairs=0,unknown_qualified_source_motion_pairs=(s['last_pts']-s['first_pts'])//256)for s in scope['shots']if s['id']!='s01'}
    result=dict(schema='rinkfit.native_board_print_unknown_surface_texture_candidate_generation.v1',source_video=source,
        scope=_ref(scope_file),roles=_ref(roles_file),source_seed_review=_ref(seed_review),independent_actor_ledgers=[_ref(p)for p in actor_ledgers],
        candidate_rows=_ref(out/'candidate_rows.jsonl.gz'),tracks=_ref(out/'tracks.json'),rows=total,track_count=len(completed),by_shot=dict(counts),
        source_motion_candidate_coverage=coverage,seed_dispositions=seed_status,terminal_reasons=dict(rejections),
        existing_ice_or_board_world_corner_measurements_changed=False,world_or_ice_H_used=False,
        geometry_installed_in_model=False,source_continuation_qualification_issued=False,statistical_covariance_issued=False,
        surface_position='unknown shared continuous-board [s,h] per texture; no existing world-corner correspondence',
        no_full_source_frame_cache_written=True,reader_modules=[_ref(__file__),_ref(Path(__file__).with_name('source_texture_tracklets.py'))],
        limitations=['Seed material ownership was reviewed in actual source pixels. Continuing printed-feature identity remains candidate evidence.',
            'The old world-corner material is never reused for a nearby printed texture.',
            'Existing FB/NCC/local-2D criteria are unchanged; failed or excluded frames terminate rather than receive model UV.'])
    write_json(out/'manifest.json',result);return result
