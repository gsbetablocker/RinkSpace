"""Source-only 2D ice-paint texture tracks next to qualified material supports.

This is a recovery front end. Each track owns one unknown fixed ice position;
its UV values are current RGB measurements, never interpolated projections.
The source admission used to seed a paint component does not supply precision.
"""
from pathlib import Path
from collections import Counter
import argparse, gzip, hashlib, json
import av, cv2, numpy as np
from .io_utils import read_json, read_jsonl, file_sha256, write_json
from .native_entity_tracklets import NAVIGATION, _ncc

VERSION = 'native_qualified_ice_paint_texture_tracks_v1'


def _source_mask(rgb, ledger, source, pts, sid):
    digest = hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
    bgr = hashlib.sha256(np.ascontiguousarray(rgb[:,:,::-1]).tobytes()).hexdigest()
    if bgr != ledger['decoded_native_BGR_sha256']:
        raise ValueError('Current native RGB differs from the bound actor ledger')
    mask = np.zeros(rgb.shape[:2], np.uint8)
    # Existing broadcast-overlay exclusions, also used by dense_native.
    mask[:270,:620] = 255; mask[900:,1600:] = 255
    for actor in ledger.get('current_person_boxes', []):
        b = actor['source_binding']
        if (b['source_pts'] != pts or b['shot_id'] != sid or
            b['source_sha256'] != source['sha256'] or b['decoded_native_rgb_sha256'] != digest):
            raise ValueError('Actor exclusion belongs to a different exposure')
        x1,y1,x2,y2 = actor['box_xyxy']
        a = np.floor([x1-4,y1-4]).astype(int); z = np.ceil([x2+4,y2+4]).astype(int)
        mask[max(0,a[1]):min(1080,z[1]),max(0,a[0]):min(1920,z[0])] = 255
    # Keep the entire LK support, not just its centre, outside exclusions.
    mask = cv2.dilate(mask, np.ones((33,33),np.uint8))
    mask[:17] = 255; mask[-17:] = 255; mask[:,:17] = 255; mask[:,-17:] = 255
    return mask, digest


def _paint_signal(rgb):
    a = rgb.astype(np.int16)
    # A material-colour discriminator, not a position covariance or fit threshold.
    return (a.max(2)-a.min(2) >= 12).astype(np.uint8)*255


def acquire_ice_texture_tracks(*, scope_file, active_arrays, active_metadata,
                               actor_ledgers, output):
    out = Path(output)
    if out.exists(): raise ValueError('New immutable track generation required')
    scope = read_json(scope_file); source = scope['source']
    if file_sha256(source['path']) != source['sha256']: raise ValueError('Wrong original video')
    ledger = {}
    for path in actor_ledgers:
        for row in read_jsonl(path): ledger[int(row['native_pts'])] = row
    with np.load(active_arrays) as packed:
        data = {k:packed[k] for k in packed.files}
    meta = read_json(active_metadata)
    ice = {x['material_index']:x['label'] for x in meta['material_map'] if x['label'].startswith('ice.')}
    anchors = {}
    for i in np.flatnonzero(np.isin(data['material_index'], list(ice))):
        anchors.setdefault(int(data['native_pts'][i]), []).append(i)
    clock = {p:s['id'] for s in scope['shots'] for p in range(s['first_pts'],s['last_pts']+1,256)}
    out.mkdir(parents=True); count = Counter(); rejection = Counter(); tracks = {}; previous = None
    old_pts = None; old_sid = None; next_id = 0; pairs = set(); shots = Counter(); lengths = Counter()
    all_anchors = []; frames = []; cv2.setNumThreads(2)
    options = dict(winSize=(31,31), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
    def emit(handle, row):
        handle.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    with gzip.open(out/'tracks.jsonl.gz','wt',encoding='utf8') as rows, av.open(source['path']) as con:
        st=con.streams.video[0]
        if str(st.time_base) != '1/12800': raise ValueError('Wrong native source clock')
        for frame in con.decode(st):
            pts=int(frame.pts)
            if pts not in clock:
                if pts > max(clock): break
                continue
            sid=clock[pts]; count['decoded_programme_frames']+=1
            ld=ledger.get(pts)
            if ld is None or ld.get('role') != 'development_candidate':
                tracks={};previous=None;old_pts=old_sid=None
                count['source_actor_or_development_scope_unknown']+=1;continue
            rgb=frame.to_ndarray(format='rgb24');gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            mask,digest=_source_mask(rgb,ld,source,pts,sid);paint=_paint_signal(rgb)
            if old_sid!=sid or old_pts is None or pts!=old_pts+256: tracks={};previous=None
            retained={}
            if tracks:
                keys=list(tracks);p=np.asarray([tracks[k]['uv'] for k in keys],np.float32).reshape(-1,1,2)
                q,ok,_=cv2.calcOpticalFlowPyrLK(previous,gray,p,None,**options)
                back,bok,_=cv2.calcOpticalFlowPyrLK(gray,previous,q,None,**options)
                for j,k in enumerate(keys):
                    t=tracks[k]; uv=q[j,0].astype(float); xy=np.rint(uv).astype(int)
                    fb=float(np.linalg.norm(back[j,0]-p[j,0]))
                    reason=None
                    if not ok[j,0] or not bok[j,0] or fb>NAVIGATION['reciprocal_max_px']: reason='forward_backward_failed'
                    elif not (0<=xy[0]<1920 and 0<=xy[1]<1080) or mask[xy[1],xy[0]]: reason='actor_overlay_or_image_exit'
                    # Retain exact old patch in compact form; subpixel current patch is re-read.
                    b=cv2.getRectSubPix(gray.astype(np.float32),(33,33),tuple(uv)) if reason is None else None
                    ncc=_ncc(t['patch'],b) if b is not None else -1.
                    if reason is None and ncc<NAVIGATION['patch_ncc_min']:reason='current_patch_appearance_changed'
                    if reason is None and not paint[max(0,xy[1]-2):xy[1]+3,max(0,xy[0]-2):xy[0]+3].any():reason='current_paint_material_support_lost'
                    if reason:
                        rejection[reason]+=1;continue
                    row=dict(track_id=k,shot_id=sid,native_pts=pts,source_uv=uv.tolist(),
                        surface_class='unknown',surface_instance_id=None,material_instance_candidate=t['material'],surface_class_candidate='ice',
                        anchor_raw_index=t['anchor_raw_index'],anchor_native_pts=t['anchor_native_pts'],
                        previous_native_pts=old_pts,previous_source_uv=t['uv'],
                        forward_backward_px=fb,patch_ncc=ncc,current_rgb_sha256=digest,
                        source_video_sha256=source['sha256'],time_base=[1,12800],
                        source_measurement='current_native_RGB_pyramidal_LK_2D_texture',
                        stationary_material_evidence='texture_candidate_near_qualified_ice_source_not_material_qualification',
                        fit_role='development',precision_status='engineering_source_texture_no_covariance',static_material_qualified=False,
                        world_position_is_shared_track_latent=True,pixels_from_model=False)
                    emit(rows,row);lengths[k]+=1;shots[sid]+=1;pairs.add((sid,old_pts))
                    retained[k]=dict(t,uv=uv.tolist(),patch=b)
            if pts in anchors:
                region=np.zeros((1080,1920),np.uint8)
                for i in anchors[pts]:
                    uv=data['uv'][i]
                    if np.isfinite(uv).all() and 0<=uv[0]<1920 and 0<=uv[1]<1080:
                        cv2.circle(region,tuple(np.rint(uv).astype(int)),12,255,-1)
                region=cv2.bitwise_and(region,cv2.dilate(paint,np.ones((5,5),np.uint8)))
                region[mask>0]=0
                for t in retained.values(): cv2.circle(region,tuple(np.rint(t['uv']).astype(int)),31,0,-1)
                corners=cv2.goodFeaturesToTrack(gray,maxCorners=0,qualityLevel=.015,minDistance=31,mask=region,blockSize=5)
                for point in [] if corners is None else corners[:,0]:
                    distances=np.linalg.norm(data['uv'][anchors[pts]]-point,axis=1);j=int(np.argmin(distances));i=anchors[pts][j]
                    k=sid+'/ice_texture/'+str(next_id);next_id+=1
                    material=ice[int(data['material_index'][i])]
                    patch=cv2.getRectSubPix(gray.astype(np.float32),(33,33),tuple(point))
                    t=dict(uv=point.astype(float).tolist(),patch=patch,material=material,
                        anchor_raw_index=int(data['raw_index'][i]),anchor_native_pts=pts)
                    retained[k]=t
                    row=dict(track_id=k,shot_id=sid,native_pts=pts,source_uv=t['uv'],surface_class='unknown',
                        surface_instance_id=None,material_instance_candidate=material,surface_class_candidate='ice',anchor_raw_index=t['anchor_raw_index'],
                        anchor_native_pts=pts,current_rgb_sha256=digest,source_video_sha256=source['sha256'],
                        time_base=[1,12800],source_measurement='current_native_RGB_2D_corner',
                        stationary_material_evidence='texture_candidate_near_qualified_ice_source_not_material_qualification',fit_role='development',
                        precision_status='engineering_source_texture_no_covariance',static_material_qualified=False,pixels_from_model=False,
                        world_position_is_shared_track_latent=True)
                    emit(rows,row);all_anchors.append(row);shots[sid]+=1
            tracks=retained;previous=gray;old_pts=pts;old_sid=sid
            frames.append(dict(native_pts=pts,shot_id=sid,RGB_sha256=digest,tracks=len(tracks)))
            if count['decoded_programme_frames']%1000==0:
                print(json.dumps(dict(decoded=count['decoded_programme_frames'],tracks=len(tracks),observations=sum(shots.values()))),flush=True)
    write_json(out/'anchors.json',all_anchors)
    write_json(out/'frame_bindings.json',frames)
    possible={s['id']:s['native_frame_count']-1 for s in scope['shots']}
    report=dict(schema='rinkfit.source_ice_texture_tracklet_generation.v1',version=VERSION,
        source_video=source,scope=dict(file=str(scope_file),sha256=file_sha256(scope_file)),
        anchor_source=dict(arrays=str(active_arrays),sha256=file_sha256(active_arrays),metadata=str(active_metadata),metadata_sha256=file_sha256(active_metadata)),
        actor_ledgers=[dict(file=str(p),sha256=file_sha256(p)) for p in actor_ledgers],
        observations=sum(shots.values()),tracks=next_id,tracks_with_multiple_exposures=sum(v>0 for v in lengths.values()),
        by_shot={sid:dict(source_rows=shots[sid],covered_pairs=sum(s==sid for s,p in pairs),possible_pairs=possible[sid]) for sid in possible},
        source_status_counts=dict(count),terminal_reasons=dict(rejection),
        no_cross_cut_or_scope_gap=True,source_precision_calibrated=False,independent_validation=False,
        caveats=['Neighbouring colour/texture does not qualify static material; per-track semantic review is required before fitting.',
                 'No tracks are invented for protected exposures or missing source actor ledgers.',
                 'A fixed unknown ice position is shared per track; ice H is never used for board points.'],
        tracks_file=dict(file=str(out/'tracks.jsonl.gz'),sha256=file_sha256(out/'tracks.jsonl.gz')))
    write_json(out/'manifest.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scope',required=True);p.add_argument('--arrays',required=True)
    p.add_argument('--metadata',required=True);p.add_argument('--actor-ledger',action='append',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(acquire_ice_texture_tracks(scope_file=a.scope,active_arrays=a.arrays,
        active_metadata=a.metadata,actor_ledgers=a.actor_ledger,output=a.out)),flush=True)
