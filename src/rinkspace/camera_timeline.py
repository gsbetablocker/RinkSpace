"""Target-ice, source-PTS timelines; estimates never imply verified camera changes.

Read-only reuse of survey inputs. Spatial/colour features supplement (not replace)
continuous original PTS evidence. Ignored and pending footage never enters the
detector. All frame intervals are half-open and retain original source indices.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

SOURCES = ('Rd66CneD9io','2U7RhoKchXA','DXiFMHULDZ8','oxXu1avhrvs',
           'uaod0sM-9Hw','mEoZsjFK4r0','M_2a8gDUyrQ','yS-iLdqoHtg')
VERSION = 'rinkspace.camera-timeline.v1'
FEATURE_VERSION = 1


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.pending')
    temp.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def relative(path, root):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def load_inputs(root: Path, source_id: str):
    base = root/'outputs/event_survey_v11'/source_id
    summary = read_json(base/'summary.json')
    chunks, arrays, scores, legacy_cuts, candidates, features = [], [], [], [], [], []
    previous = None
    for manifest_path in sorted((base/'chunks').glob('*/manifest.json')):
        m = read_json(manifest_path)
        npz_path = manifest_path.parent/'decoded_frame_pts_and_cut_scores.npz'
        with np.load(npz_path) as saved:
            pts = saved['source_pts'].copy()
            cut = saved['cut_score'].copy()
        tb = Fraction(m['source_time_base'])
        step = int(np.median(np.diff(pts))) if len(pts)>1 else None
        audit = {'manifest_path':relative(manifest_path,root),
                 'pts_archive':relative(npz_path,root),'pts_archive_sha256':sha256(npz_path),
                 'source_path':m['source_path'],'source_sha256':m['source_sha256'],
                 'time_base':str(tb),'requested_interval_seconds':m['requested_interval_seconds'],
                 'first_pts':int(pts[0]),'last_pts':int(pts[-1]),'frame_count':len(pts),
                 'record_frame_count_matches':len(pts)==m['decoded_frame_count'],
                 'strictly_increasing':bool(np.all(np.diff(pts)>0)),
                 'internal_gap_frame_indices':(np.flatnonzero(np.diff(pts)>step)+1).tolist(),
                 'previous_chunk_delta_pts':int(pts[0]-previous) if previous is not None else None,
                 'expected_step_pts':step}
        chunks.append(audit)
        arrays.append(pts)
        scores.append(cut)
        previous=int(pts[-1])
        for row in read_json(manifest_path.parent/'cut_candidates.json'):
            row=dict(row); row['legacy_index_path']=relative(manifest_path.parent/'cut_candidates.json',root)
            legacy_cuts.append(row)
        candidates.extend(read_json(manifest_path.parent/'candidates.json'))
        features.extend(read_json(manifest_path.parent/'one_second_features.json'))
    pts=np.concatenate(arrays)
    if not np.all(np.diff(pts)>0):
        raise ValueError(f'{source_id}: chunks contain duplicate or reversed PTS')
    identities={(r['source_path'],r['source_sha256'],r['time_base']) for r in chunks}
    if len(identities)!=1:
        raise ValueError(f'{source_id}: multiple track identities must be reconciled explicitly')
    path, digest, tb = next(iter(identities))
    media=root/path
    if not media.exists():
        raise FileNotFoundError(media)
    actual_hash=sha256(media)
    if actual_hash!=digest:
        raise ValueError(f'{source_id}: media SHA-256 differs from PTS index')
    candidates.sort(key=lambda row:row['source_pts'])
    features.sort(key=lambda row:row['source_pts'])
    return {'summary':summary,'pts':pts,'legacy_score':np.concatenate(scores),'chunks':chunks,
            'candidate_pts':np.asarray([r['source_pts'] for r in candidates],dtype=np.int64),
            'feature_pts':np.asarray([r['source_pts'] for r in features],dtype=np.int64),
            'legacy_cuts':legacy_cuts,'candidates':candidates,'second_features':features,
            'track':{'source_id':source_id,'source_path':path,'source_sha256':digest,
                     'sha256_meaning':'entire_local_encoded_video_file','sha256_verified':True,
                     'time_base':tb,'format_id':'134' if source_id=='oxXu1avhrvs' else '133'}}


def spatial_colour_feature(frame):
    """Read Y/U/V planes directly; no expensive full-frame colour conversion."""
    if frame.format.name not in ('yuv420p','yuvj420p'):
        frame=frame.reformat(format='yuv420p')
    small=[]
    for plane in frame.planes[:3]:
        raw=np.frombuffer(plane,dtype=np.uint8).reshape(plane.height,plane.line_size)[:,:plane.width]
        small.append(cv2.resize(raw,(32,18),interpolation=cv2.INTER_AREA))
    grid=np.stack([cv2.resize(x,(8,6),interpolation=cv2.INTER_AREA) for x in small],axis=-1)
    histogram=np.concatenate([cv2.calcHist([x],[0],None,[16],[0,256]).ravel() for x in small])
    histogram=np.rint(histogram/576*255).astype(np.uint8)
    return grid,histogram


def scope_frame_runs(pts, scope, track):
    """Validate the source identity and map target PTS to original frame runs."""
    if scope.get('source_sha256') != track['source_sha256'] or Fraction(scope['time_base']) != Fraction(track['time_base']):
        raise ValueError('rink scope does not identify the current source track')
    step=int(np.median(np.diff(pts))) if len(pts)>1 else 1
    source_end=int(pts[-1]+step) if len(pts) else 0
    masks={name:np.zeros(len(pts),dtype=bool) for name in ('target','ignored','pending')}
    for name,mask in masks.items():
        for interval in scope.get(f'{name}_intervals',[]):
            start,end=int(interval['start_pts']),int(interval['end_pts_exclusive'])
            if start>=end or not len(pts) or start<int(pts[0]) or end>source_end:
                raise ValueError(f'invalid {name} scope interval: {start}, {end}')
            a,b=np.searchsorted(pts,[start,end])
            if (a>=len(pts) or int(pts[a])!=start or (b<len(pts) and int(pts[b])!=end)
                    or (b==len(pts) and end!=source_end)):
                raise ValueError('scope endpoints must use original source frame PTS')
            if np.any(mask[a:b]):raise ValueError(f'overlapping {name} scope intervals')
            mask[a:b]=True
    allocation=sum(mask.astype(np.uint8) for mask in masks.values())
    if np.any(allocation>1):raise ValueError('target, ignored and pending scope ranges overlap')
    # Unlisted frames are pending content review, never implicitly target ice.
    masks['pending'] |= allocation==0
    edges=np.flatnonzero(np.diff(np.r_[False,masks['target'],False].astype(np.int8)))
    return [(int(a),int(b)) for a,b in zip(edges[::2],edges[1::2])],masks


def feature_cache(root, source_id, inputs, out, retained_runs=None):
    npz_path=out/f'{source_id}_features.npz'
    meta_path=out/f'{source_id}_features.json'
    track=inputs['track']
    wanted=np.zeros(len(inputs['pts']),dtype=bool)
    for a,b in retained_runs if retained_runs is not None else [(0,len(wanted))]:wanted[a:b]=True
    historical=root/'outputs/camera_classification/timelines'
    for archive,metadata in [(npz_path,meta_path),(historical/npz_path.name,historical/meta_path.name)]:
        if not archive.exists() or not metadata.exists():continue
        meta=read_json(metadata)
        if meta.get('feature_version')==FEATURE_VERSION and meta.get('source_sha256')==track['source_sha256']:
            with np.load(archive) as saved:
                pts=saved['source_pts'].copy()
                available=saved['valid_frame_mask'] if 'valid_frame_mask' in saved.files else np.ones(len(pts),dtype=bool)
                if np.array_equal(pts,inputs['pts']) and np.all(available[wanted]):
                    return {k:saved[k].copy() for k in saved.files},dict(meta,
                        feature_archive=relative(archive,root),cache_reused_read_only=True,
                        selected_frame_count=int(wanted.sum()),new_feature_frame_count=0,
                        selection='Only target-ice frame runs enter segmentation; historical features are read-only inputs',
                        selected_feature_pts_match_original=True)
    n=len(inputs['pts'])
    pts=inputs['pts'].copy()
    grid=np.zeros((n,6,8,3),dtype=np.uint8)
    hist=np.zeros((n,48),dtype=np.uint8)
    t=time.perf_counter(); count=0; decoded_count=0
    with av.open(str(root/track['source_path'])) as container:
        stream=container.streams.video[0]
        stream.thread_type='AUTO'; stream.codec_context.thread_count=2
        actual_tb=str(stream.time_base)
        if actual_tb!=track['time_base']:
            raise ValueError('actual time base differs from survey')
        probe={'width':stream.width,'height':stream.height,'average_rate':str(stream.average_rate),
               'codec':stream.codec_context.name,'stream_index':stream.index,
               'stream_duration_pts':stream.duration,'stream_start_pts':stream.start_time}
        for a,b in retained_runs if retained_runs is not None else [(0,n)]:
            if a==b:continue
            container.seek(int(pts[a]),stream=stream,backward=True)
            i=a
            for frame in container.decode(stream):
                decoded_count+=1
                if frame.pts<int(pts[a]):continue  # codec preroll is not feature processing
                if frame.pts!=int(pts[i]):raise ValueError('target feature decode differs from original PTS index')
                grid[i],hist[i]=spatial_colour_feature(frame)
                count+=1;i+=1
                if i==b:break
            if i!=b:raise ValueError('target feature decode ended before retained interval')
    if count!=int(wanted.sum()):raise ValueError('not every retained frame has features')
    np.savez_compressed(npz_path,source_pts=pts,spatial_yuv=grid,histogram_yuv=hist,valid_frame_mask=wanted)
    meta={'feature_version':FEATURE_VERSION,'source_id':source_id,**track,'probe':probe,
          'decoded_frame_count':decoded_count,'feature_frame_count':count,
          'all_pts_equal_legacy_index':bool(wanted.all()),
          'selected_feature_pts_match_original':True,'selected_frame_count':count,
          'new_feature_frame_count':count,'cache_reused_read_only':False,
          'feature_archive':relative(npz_path,root),
          'feature_definition':{'spatial_yuv':'6 x 8 spatial means for Y/U/V planes, uint8; limited/full range retained as encoded',
                                'histogram_yuv':'16 bins for each 32x18 Y/U/V plane, normalized to 255 and quantized'},
          'selection':'Only target-ice frame runs; excluded and pending frames have no computed features',
          'elapsed_seconds':round(time.perf_counter()-t,3)}
    dump(meta_path,meta)
    return {'source_pts':pts,'spatial_yuv':grid,'histogram_yuv':hist,'valid_frame_mask':wanted},meta


def seam_luma_cache(root,source_id,inputs,out,retained_runs=None):
    """Repair the historical first-frame reset using actual adjacent source PTS."""
    path=out/f'{source_id}_seam_features.json'
    historical=root/'outputs/camera_classification/timelines'/path.name
    cache_path=path if path.exists() else historical
    meta=read_json(cache_path) if cache_path.exists() else None
    selected=np.zeros(len(inputs['pts']),dtype=bool)
    for a,b in retained_runs if retained_runs is not None else [(0,len(selected))]:selected[a:b]=True
    relevant=[]
    for chunk in inputs['chunks'][1:]:
        i=int(np.searchsorted(inputs['pts'],chunk['first_pts']))
        if 0<i<len(selected) and selected[i-1] and selected[i]:relevant.append(chunk)
    required={c['first_pts'] for c in relevant}
    if (not meta or meta.get('source_sha256')!=inputs['track']['source_sha256']
            or not required.issubset({r['source_pts'] for r in meta['seams']})):
        pts=inputs['pts'];records=[]
        with av.open(str(root/inputs['track']['source_path'])) as container:
            stream=container.streams.video[0];stream.thread_type='AUTO';stream.codec_context.thread_count=2
            for chunk in relevant:
                current=chunk['first_pts'];i=int(np.searchsorted(pts,current));previous=int(pts[i-1]);images={}
                container.seek(previous,stream=stream,backward=True)
                for frame in container.decode(stream):
                    if frame.pts in (previous,current):
                        if frame.format.name not in ('yuv420p','yuvj420p'):frame=frame.reformat(format='yuv420p')
                        plane=frame.planes[0]
                        raw=np.frombuffer(plane,dtype=np.uint8).reshape(plane.height,plane.line_size)[:,:plane.width]
                        tiny=cv2.resize(raw,(80,46),interpolation=cv2.INTER_AREA)
                        if frame.format.name!='yuvj420p':tiny=np.clip((tiny.astype(np.float32)-16)*(255/219),0,255).astype(np.uint8)
                        images[frame.pts]=tiny
                    if frame.pts>=current:break
                score=float(cv2.absdiff(images[current],images[previous]).mean()/255)
                records.append({'source_pts':current,'previous_pts':previous,
                                'source_seconds':float(current*Fraction(inputs['track']['time_base'])),
                                'historical_chunk_first_luma_score':float(inputs['legacy_score'][i]),
                                'recomputed_luma_score':score})
        meta={**inputs['track'],'method':'Exact original adjacent PTS frames across every historical chunk seam; same 80x46 luma metric as survey, with full/limited range respected','seams':records}
        dump(path,meta)
        cache_path=path
    meta=dict(meta,seams=[s for s in meta['seams'] if s['source_pts'] in required])
    corrected=inputs['legacy_score'].copy()
    for seam in meta['seams']:
        i=int(np.searchsorted(inputs['pts'],seam['source_pts']))
        if i>=len(corrected) or int(inputs['pts'][i])!=seam['source_pts'] or int(inputs['pts'][i-1])!=seam['previous_pts']:
            raise ValueError('cached seam PTS do not match current track')
        corrected[i]=seam['recomputed_luma_score']
    return corrected,dict(meta,cache_path=relative(cache_path,root),selection='adjacent target-ice frames only')


def change_signals(grid,hist):
    n=len(grid)
    bands=np.zeros((n,3),np.float32)
    colour=np.zeros(n,np.float32)
    # Blocks keep the expanded float array small for long broadcasts.
    for start in range(1,n,30000):
        end=min(n,start+30000)
        delta=np.abs(grid[start:end].astype(np.float32)-grid[start-1:end-1].astype(np.float32))/255
        weighted=delta[:,:,:,0]*.65+delta[:,:,:,1]*.175+delta[:,:,:,2]*.175
        bands[start:end]=weighted.reshape(-1,3,2,8).mean(axis=(2,3))
        hd=np.abs(hist[start:end].astype(np.float32)-hist[start-1:end-1].astype(np.float32))
        colour[start:end]=hd.sum(axis=1)/(2*255*3)
    return bands,colour


def detect_boundaries(pts, old_score, grid, hist, tb, legacy_cuts, source_id, root):
    if len(pts)<2:
        return [],[],{'status_counts':{},'selection':'single retained frame has no internal cut'}
    bands,colour=change_signals(grid,hist)
    spatial=bands.mean(axis=1)
    n=len(pts); fps=1/(float(tb)*float(np.median(np.diff(pts))))
    step=int(np.median(np.diff(pts)))
    candidates=np.flatnonzero((old_score>=.07)|(spatial>=.055)|(colour>=.14))
    legacy_by_pts={int(r['source_pts']):r for r in legacy_cuts if pts[0]<int(r['source_pts'])<=pts[-1]}
    candidates=np.unique(np.concatenate((candidates,np.searchsorted(pts,list(legacy_by_pts)))))
    rows=[]
    for i in candidates:
        if i<=0 or i>=n: continue
        score=float(max(old_score[i],spatial[i]*1.7,colour[i]*.65))
        lo=max(1,i-10); hi=min(n,i+11)
        background=np.concatenate((old_score[lo:max(lo,i-1)],old_score[min(hi,i+2):hi]))
        baseline=float(np.median(background)) if len(background) else 0
        ratio=float(old_score[i]/max(.009,baseline))
        before=grid[max(0,i-3):i].astype(np.float32).mean(axis=0)
        after=grid[i:min(n,i+3)].astype(np.float32).mean(axis=0)
        persistence=float(np.mean(np.abs(after-before))/255)
        rebound=float(np.mean(np.abs(grid[min(n-1,i+2)].astype(np.float32)-grid[max(0,i-2)].astype(np.float32)))/255)
        reasons=[]
        flash=False
        if i+1<n and old_score[i+1]>.06:
            flash=float(np.mean(np.abs(grid[i+1].astype(np.float32)-grid[i-1].astype(np.float32)))/255)<.022
        if flash:
            status='unresolved_short_event'; reasons=['one-frame appearance returns; flash and genuine brief inserted shot cannot be distinguished from features alone']
        elif ratio>=2.6 and (old_score[i]>=.09 or spatial[i]>=.060 or colour[i]>=.16) and persistence>.023:
            status='machine_supported_cut'; reasons=['isolated temporal discontinuity','spatial/colour change persists across neighbouring frames']
        elif (old_score[i]>=.105 or spatial[i]>=.085 or colour[i]>=.20) and persistence>.025:
            status='unresolved_transition'; reasons=['substantial change also compatible with motion, occlusion or gradual transition']
        else:
            status='rejected_motion_or_weak_candidate'; reasons=['insufficient isolated persistent discontinuity']
        row={'boundary_id':f'{source_id}:b:{int(pts[i])}','source_pts':int(pts[i]),
             'source_seconds':float(pts[i]*tb),'frame_index':int(i),'status':status,
             'uncertainty_interval_seconds':[float(pts[max(0,i-1)]*tb),float(pts[min(n-1,i+1)]*tb)],
             'scores':{'legacy_luma':round(float(old_score[i]),6),'spatial_yuv':round(float(spatial[i]),6),
                       'colour_histogram':round(float(colour[i]),6),'local_spike_ratio':round(ratio,4),
                       'persistent_colour_spatial':round(persistence,6),'return_after_two_frames':round(rebound,6),
                       'top_middle_bottom_change':[round(float(x),6) for x in bands[i]]},
             'reason':reasons,'legacy_candidate':legacy_by_pts.get(int(pts[i]))}
        rows.append(row)
    # A returned one-frame appearance needs both possible entry and exit. The
    # latter may be below the general candidate threshold, so retain it explicitly.
    by_index={row['frame_index']:row for row in rows}
    for entry in [r for r in rows if r['status']=='unresolved_short_event']:
        i=entry['frame_index']+1
        if i>=n:continue
        counterpart=by_index.get(i)
        if counterpart is None:
            counterpart={'boundary_id':f'{source_id}:b:{int(pts[i])}','source_pts':int(pts[i]),
                         'source_seconds':float(pts[i]*tb),'frame_index':i,
                         'uncertainty_interval_seconds':[float(pts[i-1]*tb),float(pts[min(n-1,i+1)]*tb)],
                         'scores':{'legacy_luma':round(float(old_score[i]),6),'spatial_yuv':round(float(spatial[i]),6),
                                   'colour_histogram':round(float(colour[i]),6)},
                         'legacy_candidate':legacy_by_pts.get(int(pts[i]))}
            rows.append(counterpart);by_index[i]=counterpart
        counterpart.update(status='unresolved_short_event',reason=['possible one-frame event return; keep both entry and exit visible'])
    # Nearby responses share an uncertainty range without discarding any split.
    active=sorted([r for r in rows if r['status']=='machine_supported_cut' or r['status'].startswith('unresolved')],key=lambda row:row['frame_index'])
    groups=[]
    for r in active:
        if groups and r['frame_index']-groups[-1][-1]['frame_index']<=2:
            groups[-1].append(r)
        else:groups.append([r])
    selected=[]
    for group in groups:
        if len(group)>1:
            interval=[float(pts[max(0,group[0]['frame_index']-1)]*tb),
                      float(pts[min(n-1,group[-1]['frame_index']+1)]*tb)]
            # Do not impose a minimum shot length: A/B/A inserts and dissolves both
            # produce clustered responses. Keep every possible split visible.
            event_id=f'{source_id}:short-event:{group[0]["source_pts"]}'
            for row in group:
                row['uncertainty_interval_seconds']=interval
                row['status']='unresolved_short_event'
                row['event_group_id']=event_id
                row['reason'].append('clustered response retains all possible splits; brief insert, flash, occlusion or transition unresolved')
        selected.extend(group)
    # Search wider temporal windows as well as adjacent frames. Stable endpoints
    # joined by changing colour/spatial structure identify possible gradual edits.
    # Camera movement can also satisfy this test, so every such event is unresolved.
    # This deliberately does not turn a midpoint into an asserted exact cut.
    centres=np.arange(15,n-15,3,dtype=np.int64)
    gradual_candidates=[]
    for start in range(0,len(centres),20000):
        ci=centres[start:start+20000]
        matches=np.zeros(len(ci),dtype=bool)
        for span in (3,6,9,15):
            # Different zoom/camera views can share ice and board colours. Compare
            # multi-scale endpoint displacement against nearby continuous motion,
            # rather than requiring one large absolute colour difference.
            valid=ci[(ci-span-6>=0)&(ci+span+6<n)]
            across=np.abs(grid[valid+span].astype(np.float32)-grid[valid-span].astype(np.float32)).mean(axis=(1,2,3))/255
            pre=np.abs(grid[valid-span].astype(np.float32)-grid[valid-span-6].astype(np.float32)).mean(axis=(1,2,3))/255
            post=np.abs(grid[valid+span+6].astype(np.float32)-grid[valid+span].astype(np.float32)).mean(axis=(1,2,3))/255
            good=(across>.030)&(pre<.028)&(post<.028)&(across>1.4*(2*span/6)*np.minimum(pre,post))
            matches[np.searchsorted(ci,valid[good])]=True
        for i in ci[matches]:
            i=int(i)
            local=spatial[i-9:i+10]
            if np.count_nonzero(local>.004)<4 or float(np.max(old_score[i-9:i+10]))>=.09:
                continue
            gradual_candidates.append(i)
    gradual_groups=[]
    for i in gradual_candidates:
        if gradual_groups and i-gradual_groups[-1][-1]<=12:gradual_groups[-1].append(i)
        else:gradual_groups.append([i])
    active_indices=np.asarray([r['frame_index'] for r in selected],dtype=np.int64)
    for group in gradual_groups:
        left=max(1,group[0]-15); right=min(n-1,group[-1]+15)
        if len(active_indices) and np.any((active_indices>=left)&(active_indices<=right)):
            # Existing uncertain response records already expose this same interval.
            for row in selected:
                if left<=row['frame_index']<=right:
                    extent=[float(pts[left]*tb),float(pts[right]*tb)]
                    row['possible_gradual_transition_interval_seconds']=extent
                    row['uncertainty_interval_seconds']=[min(row['uncertainty_interval_seconds'][0],extent[0]),max(row['uncertainty_interval_seconds'][1],extent[1])]
                    row['status']='unresolved_transition'
                    row['reason'].append('multi-scale gradual-transition response extends the uncertain source range')
            continue
        i=group[len(group)//2]
        row={'boundary_id':f'{source_id}:gradual:{int(pts[i])}','source_pts':int(pts[i]),
             'source_seconds':float(pts[i]*tb),'frame_index':i,'status':'unresolved_gradual_transition',
             'uncertainty_interval_seconds':[float(pts[left]*tb),float(pts[right]*tb)],
             'provisional_split_not_exact_cut':True,
             'scores':{'legacy_luma':round(float(old_score[i]),6),'spatial_yuv':round(float(spatial[i]),6),
                       'colour_histogram':round(float(colour[i]),6)},
             'reason':['stable endpoint appearances differ across a multi-frame interval',
                       'dissolve, fade and camera motion require actual sequence review; midpoint only allocates frames'],
             'legacy_candidate':legacy_by_pts.get(int(pts[i]))}
        selected.append(row);rows.append(row)
    # New, explicitly sourced visual findings are classification data. They do
    # not modify historical survey records or claim exact unsampled blend edges.
    development_review_path=root/'outputs/camera_classification/timelines/development_reviews/boundary_reviews.json'
    if development_review_path.exists():
        for review in read_json(development_review_path)['reviews']:
            if review['source_id']!=source_id:continue
            left,right=review['source_interval_seconds']
            left_pts,right_pts=review.get('source_interval_pts',[int(round(left/float(tb)/step))*step,int(round(right/float(tb)/step))*step])
            if right_pts<int(pts[0]) or left_pts>int(pts[-1]):continue
            if review['kind'].startswith('continuous_'):
                for row in rows:
                    if left_pts<=row['source_pts']<=right_pts:
                        row.update(status='rejected_model_visually_reviewed_continuity',
                                   review_reference=relative(development_review_path,root),
                                   review_evidence_paths=review['evidence_paths'],reason=[review['reason']])
                selected=[r for r in selected if not left_pts<=r['source_pts']<=right_pts]
                continue
            if review['kind']=='hard_cut':
                if not int(pts[0])<right_pts<=int(pts[-1]):continue
                i=int(np.searchsorted(pts,right_pts))
                if i>=n or int(pts[i])!=right_pts:raise ValueError('visual hard-cut PTS not on current track')
                row=next((r for r in rows if r['source_pts']==right_pts),None)
                if row is None:
                    row={'boundary_id':f'{source_id}:b:{right_pts}','source_pts':right_pts,
                         'source_seconds':float(right_pts*tb),'frame_index':i,'scores':{},
                         'legacy_candidate':legacy_by_pts.get(right_pts)}
                    rows.append(row)
                row.update(status='visually_reviewed_cut',uncertainty_interval_seconds=[float(pts[i-1]*tb),float(pts[i]*tb)],
                           reason=[review['reason']],review_reference=relative(development_review_path,root),
                           review_evidence_paths=review['evidence_paths'])
                selected=[r for r in selected if r is not row and not left_pts<=r['source_pts']<=right_pts]
                selected.append(row)
                continue
            left=max(left,float(pts[0]*tb));right=min(right,float(pts[-1]*tb))
            left_pts=max(left_pts,int(pts[0]));right_pts=min(right_pts,int(pts[-1]))
            affected=[r for r in selected if left_pts<=r['source_pts']<=right_pts]
            if not affected:
                i=min(n-1,int(np.searchsorted(pts,int(round((left+right)/2/float(tb))))))
                row={'boundary_id':f'{source_id}:review-transition:{int(pts[i])}','source_pts':int(pts[i]),
                     'source_seconds':float(pts[i]*tb),'frame_index':i,'status':'unresolved_transition',
                     'uncertainty_interval_seconds':[left,right],'scores':{},'reason':[],
                     'legacy_candidate':legacy_by_pts.get(int(pts[i]))}
                rows.append(row);selected.append(row);affected=[row]
            for row in affected:
                interval=row['uncertainty_interval_seconds']
                row.update(status='unresolved_transition',transition_kind=review['kind'],
                           visual_review_status='model_visually_supported_transition_range',
                           review_reference=relative(development_review_path,root),
                           review_evidence_paths=review['evidence_paths'],
                           uncertainty_interval_seconds=[min(left,interval[0]),max(right,interval[1])])
                row['reason'].append(review['reason'])
    # Explicit historical review, scoped to its actual evidence and original tracks.
    if source_id=='2U7RhoKchXA':
        review_path=root/'outputs/event_reuse_v11/men_15557_shot_review/shot_boundary_review.json'
        review=read_json(review_path)
        ref=relative(review_path,root)
        reviewed_start=review['containing_shot_lowres']['first_new_pts']
        reviewed_end=review['containing_shot_lowres']['next_shot_first_pts']
        for row in rows:
            if reviewed_start<row['source_pts']<reviewed_end:
                row.update(status='rejected_reviewed_continuous_shot',review_reference=ref,
                           reason=['historical continuous complete-shot review supports no edit inside this interval; physical installation remains unknown'])
            for false in review['false_cut_clusters']:
                if false['start_s']-.04<=row['source_seconds']<=false['end_s']+.04:
                    row.update(status='rejected_reviewed_foreground_occlusion',review_reference=ref,
                               reason=[false['reason']])
        selected=[r for r in selected if not r['status'].startswith('rejected')]
        for kind,key in [('start','first_new_pts'),('end','next_shot_first_pts')]:
            p=review['containing_shot_lowres'][key]; i=int(np.searchsorted(pts,p))
            if not 0<i<n or int(pts[i])!=p:continue
            existing=next((r for r in rows if r['source_pts']==p),None)
            if existing is None:
                existing={'boundary_id':f'{source_id}:b:{p}','source_pts':p,'source_seconds':float(p*tb),
                          'frame_index':i,'scores':{},'legacy_candidate':legacy_by_pts.get(p)}
                rows.append(existing)
            existing.update(status='visually_reviewed_cut',reason=['existing source-neighbour review'],
                            review_reference=ref,uncertainty_interval_seconds=[float((p-step)*tb),float(p*tb)],
                            distinct_hd_boundary={'source_format_id':'399','time_base':review['containing_shot']['time_base'],
                              'source_pts':review['containing_shot']['first_new_pts' if kind=='start' else 'next_shot_first_pts'],
                              'hash_kind':'range_set_sha256','range_set_sha256':review['HD_boundary_evidence']['range_set_sha256']})
            for row in selected:
                if row is not existing and abs(row['frame_index']-i)<=2:
                    row.update(status='rejected_reviewed_neighbour_response',review_reference=ref,
                               reason=['reviewed exact proxy neighbour frames resolve response to nearby true boundary'],
                               grouped_into_boundary_id=existing['boundary_id'])
            selected=[r for r in selected if abs(r['frame_index']-i)>2]
            selected.append(existing)
    # A reviewed exact cut clips any overlapping estimated transition range.
    # This keeps the verified first-new-frame boundary intact on its own track.
    verified=[r for r in selected if r['status']=='visually_reviewed_cut']
    for row in selected:
        if not row['status'].startswith('unresolved'):continue
        lower,upper=row['uncertainty_interval_seconds']
        for cut in verified:
            if row['frame_index']<cut['frame_index'] and lower<cut['source_seconds']<=upper:
                upper=float(pts[cut['frame_index']-1]*tb)
            elif row['frame_index']>cut['frame_index'] and lower<cut['source_seconds']<upper:
                lower=cut['source_seconds']
        row['uncertainty_interval_seconds']=[lower,upper]
    return sorted(selected,key=lambda r:r['frame_index']),sorted(rows,key=lambda r:r['frame_index']),{
        'adjustable_detector_parameters':{'candidate_luma':.07,'candidate_spatial':.055,'candidate_histogram':.14,
          'isolated_spike_ratio':2.6,'local_context_frames_each_side':10,'response_group_gap_frames':2,'gradual_window_frames':30,'gradual_comparison_spans_frames':[3,6,9,15],'gradual_endpoint_change':.030,'gradual_endpoint_stability':.028,'gradual_change_to_neighbour_motion_ratio':1.4,'gradual_motion_comparison_normalized_by_frame_span':True,'gradual_baseline':'lower of before/after motion; a dissolve may join two moving views'},
        'parameter_basis':'High recall beyond original .105 luma threshold. Initial .055 gradual endpoint displacement missed visually reviewed same-ice dissolves at dance 7007.3–7007.5 and 7021.6–7021.8. Moving-view dissolve M_2 at 1636.233–1636.533 also requires the lower of before/after motion as candidate baseline. Revision applies .030 displacement plus speed-normalized neighbour-motion ratio at four temporal scales across all eight broadcasts; extra candidates remain unresolved. Values are adjustable technical parameters, not user constraints.',
        'status_meaning':'machine_supported_cut is an algorithmic estimate, not human-confirmed. unresolved_transition/short_event/gradual_transition retain intervals and provisional frame allocations; multiple possible short-event splits remain visible.',
        'status_counts':dict(Counter(r['status'] for r in rows)),
        'limitations':['Slow dissolves without sufficient per-frame change can be missed; low contrast cuts and occlusions remain imperfectly separated.',
                       'Real-video independent checks are reported separately; complete frame allocation is not proof of correct shot boundaries.']}


def frame_partition(pts, boundaries, tb, source_step=None):
    """Partition frames into shot intervals and unioned, explicit uncertain events.

    Candidate instants within a short event are evidence, not exact shot breaks.
    Keep their PTS inside the event instead of manufacturing one-frame shots.
    """
    n=len(pts)
    if not n:return []
    step=source_step if source_step is not None else int(np.median(np.diff(pts))) if n>1 else 1
    ranges=[]
    for boundary in boundaries:
        if not boundary.get('status','').startswith('unresolved'):continue
        lower,upper=boundary['uncertainty_interval_seconds']
        lo_pts=int(round(lower/float(tb)));hi_pts=int(round(upper/float(tb)))
        left=max(0,int(np.searchsorted(pts,lo_pts,side='left')))
        right=min(n,int(np.searchsorted(pts,hi_pts,side='right')))
        left=min(left,int(boundary['frame_index']));right=max(right,int(boundary['frame_index'])+1)
        ranges.append([left,right,[boundary]])
    merged=[]
    for left,right,rows in sorted(ranges,key=lambda item:item[0]):
        if merged and left<=merged[-1][1]:
            merged[-1][1]=max(right,merged[-1][1]);merged[-1][2].extend(rows)
        else:merged.append([left,right,list(rows)])
    indices={0,n}
    for left,right,_ in merged:indices.update((left,right))
    for boundary in boundaries:
        i=int(boundary['frame_index'])
        if not any(left<i<right for left,right,_ in merged):indices.add(i)
    # A candidate inside its unresolved range never becomes a partition edge.
    indices={i for i in indices if not any(left<i<right for left,right,_ in merged)}
    indices=sorted(indices)
    result=[]
    for a,b in zip(indices,indices[1:]):
        if a==b:continue
        p=int(pts[a]);q=int(pts[b]) if b<n else int(pts[-1]+step)
        event=next((r for r in merged if r[0]==a and r[1]==b),None)
        row={'start_frame_index':a,'end_frame_index_exclusive':b,'frame_count':b-a,
             'start_pts':p,'end_pts_exclusive':q,'start_seconds':float(p*tb),'end_seconds':float(q*tb),
             'interval_type':'shot'}
        if event:
            candidates=[r for r in boundaries if a<=r['frame_index']<b]
            types={r.get('status') for r in candidates}
            row.update(interval_type='transition' if types=={'unresolved_gradual_transition'} else 'boundary_unresolved',
                       boundary_status='unresolved_range',
                       candidate_points=[{'boundary_id':r['boundary_id'],'source_pts':r['source_pts'],
                         'source_seconds':r['source_seconds'],'frame_index':r['frame_index'],'status':r['status']} for r in candidates],
                       transition_kinds=sorted(set(r['transition_kind'] for r in candidates if r.get('transition_kind'))),
                       review_evidence_paths=sorted(set(path for r in candidates for path in r.get('review_evidence_paths',[]))),
                       uncertainty_reason='One or more edits, occlusions, brief inserts, flashes or gradual transitions remain unresolved inside this explicit source interval; internal candidates are not asserted shot boundaries.')
        result.append(row)
    return result


def observation_selection(segment, inputs, grid):
    pts=inputs['pts']; a=segment['start_frame_index']; b=segment['end_frame_index_exclusive']
    # Reuse all existing in-segment evidence. Add temporal endpoints and composition changes.
    left,right=np.searchsorted(inputs['candidate_pts'],[segment['start_pts'],segment['end_pts_exclusive']])
    old=[r for r in inputs['candidates'][left:right] if Path(r['image_path']).exists()]
    chosen={int(np.searchsorted(pts,r['source_pts'])):dict(r,origin='existing_survey_candidate') for r in old}
    for i in [min(b-1,a+3),(a+b-1)//2,max(a,b-4)]:
        if not any(abs(i-j)<=15 for j in chosen):chosen[i]={'origin':'new_segment_temporal_coverage'}
    last=a
    for i in range(a,b,30):
        delta=float(np.abs(grid[i].astype(np.float32)-grid[last].astype(np.float32)).mean()/255)
        # Sample when composition changes or a long stretch otherwise lacks observations.
        if delta>.09 or i-last>=900:
            if not any(abs(i-j)<=15 for j in chosen):chosen[i]={'origin':'new_composition_change' if delta>.09 else 'new_long_shot_temporal_coverage'}
            last=i
    return sorted(chosen.items())


def scoped_partition(inputs, features, transition_luma, tb, retained_runs, source_id, root):
    """Detect within separate original-frame runs, never across omitted footage."""
    pts=inputs['pts'];step=int(np.median(np.diff(pts))) if len(pts)>1 else 1
    segments,boundaries,candidates,reports=[],[],[],[]
    for a,b in retained_runs:
        run_pts=pts[a:b]
        scores=transition_luma[a:b].copy();scores[0]=0
        selected,rows,report=detect_boundaries(run_pts,scores,features['spatial_yuv'][a:b],
            features['histogram_yuv'][a:b],tb,inputs['legacy_cuts'],source_id,root)
        end_pts=int(pts[b]) if b<len(pts) else int(pts[-1]+step)
        for row in {id(r):r for r in rows+selected}.values():
            if row.get('uncertainty_interval_seconds'):
                lo,hi=row['uncertainty_interval_seconds']
                row['uncertainty_interval_seconds']=[max(lo,float(run_pts[0]*tb)),min(hi,float(run_pts[-1]*tb))]
        local=frame_partition(run_pts,selected,tb,source_step=step)
        for segment in local:
            segment['start_frame_index']+=a;segment['end_frame_index_exclusive']+=a
            if segment['end_frame_index_exclusive']==b:
                segment.update(end_pts_exclusive=end_pts,end_seconds=float(end_pts*tb),scope_end=True)
            if segment['start_frame_index']==a:segment['scope_start']=True
            segment['scope_run_start_frame_index']=a
            segment['scope_run_end_frame_index_exclusive']=b
            for point in segment.get('candidate_points',[]):point['frame_index']+=a
        for row in {id(r):r for r in rows+selected}.values():
            row['frame_index']+=a
            row['scope_run_start_frame_index']=a
            row['scope_run_end_frame_index_exclusive']=b
        segments.extend(local);boundaries.extend(selected);candidates.extend(rows)
        reports.append({'start_frame_index':a,'end_frame_index_exclusive':b,**report})
    return segments,boundaries,candidates,{
        'selection':'Each target-ice run is detected independently; ignored and pending footage is not segmented',
        'runs':reports,'status_counts':dict(Counter(r['status'] for r in candidates))}


def extract_images(root, track, requested, out):
    """Seek grouped requests. Native proxy JPEGs retain their exact source PTS."""
    if not requested:return
    pts_to_paths={p:paths for p,paths in requested.items() if any(not x.exists() for x in paths)}
    if not pts_to_paths:return
    ordered=sorted(pts_to_paths)
    tb=Fraction(track['time_base']); groups=[]
    for p in ordered:
        if groups and float((p-groups[-1][-1])*tb)<3:
            groups[-1].append(p)
        else:groups.append([p])
    with av.open(str(root/track['source_path'])) as container:
        stream=container.streams.video[0]; stream.thread_type='AUTO'; stream.codec_context.thread_count=2
        for group in groups:
            pending=set(group)
            container.seek(group[0],stream=stream,backward=True)
            for frame in container.decode(stream):
                if frame.pts in pending:
                    rgb=frame.to_ndarray(format='bgr24')
                    for path in pts_to_paths[frame.pts]:
                        path.parent.mkdir(parents=True,exist_ok=True)
                        cv2.imwrite(str(path),rgb,[cv2.IMWRITE_JPEG_QUALITY,90])
                    pending.remove(frame.pts)
                if not pending or frame.pts>group[-1]:break
            if pending:raise ValueError(f'Failed exact-PTS image extraction: {sorted(pending)}')


def build_source(root,source_id,out):
    root=Path(root).resolve(); out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=True)
    if out==root/'outputs/camera_classification/timelines':
        raise ValueError('Historical timelines are read-only; use outputs/camera_classification/rink_timelines')
    from .camera_scope import load_source_scope
    scope=load_source_scope(root,source_id,out.parent)
    cv2.setNumThreads(1)
    print(f'{source_id} loading original indices',flush=True)
    inputs=load_inputs(root,source_id)
    retained_runs,scope_masks=scope_frame_runs(inputs['pts'],scope,inputs['track'])
    features,feature_meta=feature_cache(root,source_id,inputs,out,retained_runs)
    pts=inputs['pts']; grid=features['spatial_yuv']; tb=Fraction(inputs['track']['time_base'])
    transition_luma,seam_meta=seam_luma_cache(root,source_id,inputs,out,retained_runs)
    segments,boundaries,candidates,detector=scoped_partition(inputs,features,transition_luma,tb,retained_runs,source_id,root)
    seam_by_pts={r['source_pts']:r for r in seam_meta['seams']}
    for boundary in candidates:
        if boundary['source_pts'] in seam_by_pts:
            seam=seam_by_pts[boundary['source_pts']]
            boundary['scores']['legacy_luma']=seam['historical_chunk_first_luma_score']
            boundary['scores']['recomputed_chunk_seam_luma']=seam['recomputed_luma_score']
            boundary['luma_origin']='historical chunk-start reset repaired from exact adjacent original source frames'
    boundary_by_i={r['frame_index']:r for r in boundaries}
    requested={}
    reusable_images={int(row['source_pts']):row['image_path'] for row in inputs['candidates'] if (root/row['image_path']).exists()}
    def image_for(p,kind):
        if p in reusable_images:return reusable_images[p]
        alternatives=[out/source_id/folder/f'pts_{p:012d}.jpg' for folder in ('observations','boundaries')]
        path=next((path for path in alternatives if path.exists()),out/source_id/kind/f'pts_{p:012d}.jpg')
        if not path.exists():requested.setdefault(p,[]).append(path)
        reusable_images[p]=relative(path,root)
        return reusable_images[p]
    for ix,segment in enumerate(segments):
        segment.update({'source_id':source_id,'shot_id':f'{source_id}:shot:{segment["start_pts"]}',
                        'ordinal':ix,'source_path':inputs['track']['source_path'],
                        'source_sha256':inputs['track']['source_sha256'],'time_base':str(tb),
                        'boundary_status':segment.get('boundary_status',boundary_by_i.get(segment['start_frame_index'],{}).get('status','scope_start' if segment.get('scope_start') else 'after_unresolved_interval')),
                        'start_boundary_id':boundary_by_i.get(segment['start_frame_index'],{}).get('boundary_id'),
                        'end_boundary_id':boundary_by_i.get(segment['end_frame_index_exclusive'],{}).get('boundary_id'),
                        'classification_status':'unprocessed','observations':[]})
        for i,row in observation_selection(segment,inputs,grid):
            p=int(pts[i]); existing=row.get('image_path')
            obs={'source_id':source_id,'source_pts':p,'source_seconds':float(p*tb),'time_base':str(tb),
                 'frame_index':i,'image_path':existing or image_for(p,'observations'),'origin':row['origin'],
                 'legacy_shot_id':row.get('shot_id'),'legacy_camera_group_id':row.get('camera_group_id'),
                 'legacy_camera_group_status':row.get('camera_group_status'),
                 'legacy_installation_id':row.get('installation_id'),
                 'source_sha256':inputs['track']['source_sha256'],
                 'spatial_yuv_descriptor':grid[i].reshape(-1).tolist()}
            segment['observations'].append(obs)
        left,right=np.searchsorted(inputs['feature_pts'],[segment['start_pts'],segment['end_pts_exclusive']])
        segment['legacy_shot_ids']=sorted(set([r['legacy_shot_id'] for r in segment['observations'] if r['legacy_shot_id']]+[r['shot_id'] for r in inputs['second_features'][left:right]]))
    for boundary in boundaries:
        i=boundary['frame_index']
        boundary['neighbour_frames']=[{'source_pts':int(pts[j]),'source_seconds':float(pts[j]*tb),
             'image_path':image_for(int(pts[j]),'boundaries'),'frame_index':int(j)}
             for j in sorted({max(boundary['scope_run_start_frame_index'],i-1),i})]
    print(f'{source_id} {len(segments)} segments; extracting {len(requested)} exact-PTS images',flush=True)
    extract_images(root,inputs['track'],requested,out)
    image_hashes={}
    for row in [o for seg in segments for o in seg['observations']]+[o for boundary in boundaries for o in boundary['neighbour_frames']]:
        path=row['image_path']
        if path not in image_hashes:image_hashes[path]=sha256(root/path)
        row['image_sha256']=image_hashes[path]
        row['image_sha256_meaning']='encoded JPEG evidence file, not source video hash'
    step=int(np.median(np.diff(pts)))
    gaps=np.flatnonzero(np.diff(pts)>step)
    allocation=np.zeros(len(pts),dtype=np.uint8)
    for segment in segments:allocation[segment['start_frame_index']:segment['end_frame_index_exclusive']]+=1
    valid={'every_target_frame_assigned_once':bool(np.all(allocation[scope_masks['target']]==1)),
           'no_ignored_or_pending_frame_segmented':bool(np.all(allocation[~scope_masks['target']]==0)),
           'target_frame_count':int(scope_masks['target'].sum()),
           'ignored_frame_count':int(scope_masks['ignored'].sum()),
           'pending_frame_count':int(scope_masks['pending'].sum()),
           'gap_count':len(gaps),'gap_intervals_pts':[[int(pts[i]+step),int(pts[i+1])] for i in gaps],
           'overlap_count':0,'first_pts':int(pts[0]),'end_pts_exclusive':int(pts[-1]+step),
           'source_duration_seconds':inputs['summary']['duration_seconds'],
           'allocated_duration_seconds':sum(s['end_seconds']-s['start_seconds'] for s in segments),
           'first_and_last_match_declared_duration':abs(float(pts[0]*tb))<1e-6 and abs(float((pts[-1]+step)*tb)-inputs['summary']['duration_seconds'])<.002,
           'selected_feature_pts_match_original':feature_meta['selected_feature_pts_match_original'],
           'all_observation_images_exist':all((root/o['image_path']).exists() for s in segments for o in s['observations']),
           'all_boundary_neighbour_images_exist':all((root/o['image_path']).exists() for boundary in boundaries for o in boundary['neighbour_frames']),
           'all_chunk_manifests_match_saved_frame_count':all(c['record_frame_count_matches'] for c in inputs['chunks'])}
    record={'schema_version':VERSION,'source_id':source_id,'title':inputs['summary']['title'],
            'track':inputs['track'],'feature_cache':feature_meta,'chunk_seam_features':seam_meta,'chunk_audit':inputs['chunks'],
            'boundaries':boundaries,'boundary_candidates':candidates,'segments':segments,
            'rink_scope':scope,'retained_frame_count':int(scope_masks['target'].sum()),
            'ignored_frame_count':int(scope_masks['ignored'].sum()),'pending_frame_count':int(scope_masks['pending'].sum()),
            'frame_count':len(pts),'segment_count':len(segments),
            'shot_count':sum(s['interval_type']=='shot' for s in segments),
            'interval_type_counts':dict(Counter(s['interval_type'] for s in segments)),
            'unresolved_intervals':[dict((key,s[key]) for key in ('shot_id','interval_type','start_pts','end_pts_exclusive','start_seconds','end_seconds','frame_count','candidate_points')) for s in segments if s['interval_type']!='shot'],
            'duration_seconds':inputs['summary']['duration_seconds'],'timeline_validation':valid,
            'detector':detector,'scope':{'entire_broadcast_processed':False,'target_ice_only':True,
             'all_boundaries_visually_confirmed':False,'complete_frame_allocation_is_not_correct_shot_segmentation':True,
             'non_ice_material_retained':False,'pending_material_segmented':False,'geometry_model_modified':False}}
    dump(out/f'{source_id}.json',record)
    print(f'{source_id} complete: {len(segments)} segments, {sum(len(s["observations"]) for s in segments)} observations',flush=True)
    return {'source_id':source_id,'path':relative(out/f'{source_id}.json',root),'segment_count':len(segments),
            'frame_count':len(pts),'duration_seconds':inputs['summary']['duration_seconds'],
            'retained_frame_count':int(scope_masks['target'].sum()),
            'ignored_frame_count':int(scope_masks['ignored'].sum()),'pending_frame_count':int(scope_masks['pending'].sum()),
            'shot_count':sum(s['interval_type']=='shot' for s in segments),
            'interval_type_counts':dict(Counter(s['interval_type'] for s in segments)),
            'unresolved_interval_seconds':sum(s['end_seconds']-s['start_seconds'] for s in segments if s['interval_type']!='shot'),
            'observation_count':sum(len(s['observations']) for s in segments),
            'boundary_status_counts':dict(Counter(b['status'] for b in boundaries)),
            'timeline_validation':valid}


def audit_timelines(root,out,source_ids=None):
    """Independent reconstruction of per-frame allocation from delivered JSON."""
    root=Path(root).resolve();out=Path(out).resolve();reports=[];errors=[]
    source_ids=source_ids or SOURCES
    for source_id in source_ids:
        path=out/f'{source_id}.json';record=read_json(path)
        archive=record.get('feature_cache',{}).get('feature_archive')
        with np.load(root/archive if archive else out/f'{source_id}_features.npz') as data:pts=data['source_pts'].copy()
        allocation=np.zeros(len(pts),dtype=np.uint16);tb=Fraction(record['track']['time_base']);step=int(np.median(np.diff(pts)))
        if 'rink_scope' in record:
            _,scope_masks=scope_frame_runs(pts,record['rink_scope'],record['track'])
        else:
            scope_masks={'target':np.ones(len(pts),dtype=bool),'ignored':np.zeros(len(pts),dtype=bool),'pending':np.zeros(len(pts),dtype=bool)}
        bad_intervals=[];bad_observations=[];bad_boundaries=[];image_paths=set()
        for seg in record['segments']:
            a,b=seg['start_frame_index'],seg['end_frame_index_exclusive']
            if not 0<=a<b<=len(pts):bad_intervals.append(seg['shot_id']);continue
            allocation[a:b]+=1
            ep=int(pts[b]) if b<len(pts) else int(pts[-1]+step)
            if seg['start_pts']!=int(pts[a]) or seg['end_pts_exclusive']!=ep or seg['frame_count']!=b-a:bad_intervals.append(seg['shot_id'])
            for obs in seg['observations']:
                i=obs['frame_index'];p=obs['source_pts'];image_paths.add(obs['image_path'])
                if not (a<=i<b and int(pts[i])==p and abs(obs['source_seconds']-float(p*tb))<1e-8 and obs['source_sha256']==record['track']['source_sha256']):
                    bad_observations.append({'shot_id':seg['shot_id'],'source_pts':p})
        for boundary in record['boundaries']:
            i=boundary['frame_index']
            if not 0<=i<len(pts) or int(pts[i])!=boundary['source_pts'] or not scope_masks['target'][i]:bad_boundaries.append(boundary['boundary_id'])
            for frame in boundary['neighbour_frames']:
                image_paths.add(frame['image_path'])
                if frame['source_pts']!=int(pts[frame['frame_index']]) or not scope_masks['target'][frame['frame_index']]:bad_boundaries.append(boundary['boundary_id'])
        missing_images=[p for p in sorted(image_paths) if not (root/p).exists()]
        flags={'every_decoded_frame_assigned_exactly_once':bool(np.all(allocation==1)),
               'every_target_frame_assigned_exactly_once':bool(np.all(allocation[scope_masks['target']]==1)),
               'no_ignored_or_pending_frame_segmented':bool(np.all(allocation[~scope_masks['target']]==0)),
               'retained_frame_count':int(scope_masks['target'].sum()),
               'ignored_frame_count':int(scope_masks['ignored'].sum()),
               'pending_frame_count':int(scope_masks['pending'].sum()),
               'unassigned_target_frame_count':int(np.count_nonzero((allocation==0)&scope_masks['target'])),
               'unassigned_frame_count':int(np.count_nonzero(allocation==0)),
               'multiply_assigned_frame_count':int(np.count_nonzero(allocation>1)),
               'segment_pts_or_frame_count_errors':bad_intervals,'observation_pts_or_identity_errors':bad_observations,
               'boundary_pts_errors':bad_boundaries,'missing_image_paths':missing_images,
               'all_chunk_seams_pts_continuous':all(c['previous_chunk_delta_pts'] in (None,c['expected_step_pts']) for c in record['chunk_audit']),
               'seams_recomputed_count':len(record['chunk_seam_features']['seams']),
               'continuous_redecode_matches_historical_pts':record['feature_cache'].get('all_pts_equal_legacy_index',False),
               'selected_feature_pts_match_original':record['feature_cache'].get('selected_feature_pts_match_original',record['feature_cache'].get('all_pts_equal_legacy_index',False))}
        ok=(flags['every_target_frame_assigned_exactly_once'] and flags['no_ignored_or_pending_frame_segmented']
            and not bad_intervals and not bad_observations and not bad_boundaries and not missing_images and flags['all_chunk_seams_pts_continuous'])
        if not ok:errors.append(source_id)
        reports.append({'source_id':source_id,'timeline_json_sha256':sha256(path),'passed':ok,
                        'frame_count':len(pts),'duration_seconds':record['duration_seconds'],
                        'segment_count':len(record['segments']),'shot_count':record['shot_count'],
                        'interval_type_counts':record['interval_type_counts'],
                        'uncertain_interval_duration_seconds':sum(s['end_seconds']-s['start_seconds'] for s in record['segments'] if s['interval_type']!='shot'),
                        'unique_referenced_images':len(image_paths),**flags})
    men=read_json(out/'2U7RhoKchXA.json') if '2U7RhoKchXA' in source_ids else None
    containing=next((s for s in men['segments'] if s['start_seconds']<=15557<s['end_seconds']),None) if men else None
    if containing and not containing.get('scope_start') and not containing.get('scope_end'):
        regression={'source_id':'2U7RhoKchXA','contains_15557_as_one_reviewed_continuous_proxy_shot':containing['interval_type']=='shot' and containing['start_pts']==238497280 and containing['end_pts_exclusive']==239108096,
                'proxy_true_cut_first_new_pts_30fps':239108096,'distinct_hd_true_cut_first_new_pts_60fps':239107840,
                'no_supported_cut_at_reviewed_15545_or_15565_occlusions':not any(b['status'] in ('machine_supported_cut','visually_reviewed_cut') and (15545.45<=b['source_seconds']<=15545.7 or 15564.85<=b['source_seconds']<=15565.2) for b in men['boundaries'])}
        if not regression['contains_15557_as_one_reviewed_continuous_proxy_shot'] or not regression['no_supported_cut_at_reviewed_15545_or_15565_occlusions']:errors.append('men_15557_regression')
    else:
        regression={'source_id':'2U7RhoKchXA','status':'not_applicable_to_selected_scope',
                    'reason':'The full historical regression interval is not included in these target-ice timelines'}
    result={'schema_version':'rinkspace.camera-timeline-integrity-validation.v1','passed':not errors,'errors':errors,
            'scope':'Programmatic media/PTS/frame-allocation and evidence-pointer checks; not a claim of visually correct boundaries or physical camera identity',
            'sources':reports,'regression':regression,
            'total_frames':sum(r['frame_count'] for r in reports),'total_seconds':round(sum(r['duration_seconds'] for r in reports),6),
            'total_retained_frames':sum(r['retained_frame_count'] for r in reports),
            'total_ignored_frames':sum(r['ignored_frame_count'] for r in reports),
            'total_pending_frames':sum(r['pending_frame_count'] for r in reports),
            'total_segments':sum(r['segment_count'] for r in reports),'total_shot_intervals':sum(r['shot_count'] for r in reports),
            'total_uncertain_interval_seconds':sum(r['uncertain_interval_duration_seconds'] for r in reports),
            'total_seams_recomputed':sum(r['seams_recomputed_count'] for r in reports)}
    dump(out/'integrity_validation.json',result)
    return result
