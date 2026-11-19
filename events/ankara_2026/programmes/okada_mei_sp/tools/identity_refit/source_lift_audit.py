"""Screen boot lift against raw-image motion, retaining unknown/overlap cases.

Model geometry selects patches and predicts their motion. Original-image LK
tracks are an imperfect comparison cue, never a source observation or a pass.
The upper excludes the metal runner; saturated reflections are excluded from
feature selection. Every native exposure and both physical feet are counted.
"""
import argparse
from collections import deque
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare import bind
from motion_overlay import project
from source_review import SourceVideo, crop_bounds, digest, tracks


def projection(args):
    ref, camera, *_ = bind(args.prepared, args.source, args.parameter)
    upper = []
    for first in range(0, len(ref.times), 64):
        frames = np.arange(first, min(first + 64, len(ref.times)))
        assets = project(ref, camera, frames, include_body=False)
        upper.append(np.stack((assets[2][1], assets[2][3]), axis=1).astype(np.float32))
        if first % 1024 == 0:
            print(json.dumps(dict(projected=first + len(frames), total=len(ref.times))), flush=True)
    screen = np.load(args.work / 'source_review/shoe_screen_after.npz')
    np.testing.assert_array_equal(ref.p['pts'], screen['pts'])
    np.savez_compressed(args.work / 'source_review/current_lift_upper.npz',
                        upper=np.concatenate(upper), pts=ref.p['pts'], shot_ids=screen['shot_ids'],
                        parameter_sha256=digest(args.parameter))


def motion_patch(previous, current, rgb, a, b, other):
    bounds = crop_bounds(np.concatenate((a, b)), previous.shape, 60)
    x, y, x1, y1 = bounds
    if x1 - x < 12 or y1 - y < 12:
        return None
    origin = np.array([x, y], np.float32)
    mask = np.zeros((y1-y, x1-x), np.uint8)
    hull = cv2.convexHull(a.astype(np.float32))
    cv2.fillConvexPoly(mask, np.rint(hull[:, 0] - origin).astype(np.int32), 255)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    other_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(other_mask, np.rint(cv2.convexHull(other.astype(np.float32))[:, 0]-origin).astype(np.int32), 255)
    overlap = float(np.count_nonzero(mask & other_mask) / max(np.count_nonzero(mask), 1))
    patch = rgb[y:y1, x:x1]
    # Avoid bright metal flashes, colored skin and background advertising.
    white = (patch.max(2) - patch.min(2) < 48) & (patch.mean(2) > 80) & (patch.max(2) < 247)
    mask[~white] = 0
    points = cv2.goodFeaturesToTrack(previous[y:y1, x:x1], 40, .015, 3, mask=mask, blockSize=3)
    if points is None or len(points) < 4:
        return None
    points = points[:, 0] + origin
    target, fb, photo = tracks(previous, current, points, bounds)
    # A 2-D affine approximation of the actual projected upper is only a
    # screening model. Large perspective/deformation error remains explicit.
    design = np.c_[a, np.ones(len(a))]
    affine = np.linalg.lstsq(design, b, rcond=None)[0]
    affine_error = float(np.sqrt(np.mean(np.sum((design @ affine-b)**2, axis=1))))
    prediction = np.c_[points, np.ones(len(points))] @ affine
    valid = (fb < 1.25) & (photo < 22.) & np.isfinite(target).all(1)
    if valid.sum() < 4:
        return None
    residual = target[valid] - prediction[valid]
    median = np.median(residual, axis=0)
    spread = np.median(abs(residual - median), axis=0)
    return dict(error=median, spread=spread, count=int(valid.sum()), overlap=overlap,
                affine_error=affine_error, source_step=np.median(target[valid]-points[valid],axis=0),
                model_step=np.median(prediction[valid]-points[valid],axis=0))


def scan(args):
    cache = dict(np.load(args.work / 'source_review/current_lift_upper.npz'))
    assert str(cache['parameter_sha256']) == digest(args.parameter)
    source = json.loads(args.registration.read_text(encoding='utf8'))['primary_source']
    upper, pts, shots = (cache[k] for k in ('upper', 'pts', 'shot_ids'))
    n = len(pts)
    error = np.full((n, 2, 2), np.nan, np.float32)
    spread = error.copy(); source_step = error.copy(); model_step = error.copy()
    count = np.zeros((n, 2), np.int16)
    overlap = np.full((n, 2), np.nan, np.float32); affine = overlap.copy()
    queue = deque(maxlen=5)
    reader = SourceVideo(source['path'], [1, 12800])
    cv2.setNumThreads(1)
    try:
        for i, stamp in enumerate(pts):
            rgb = reader.read(int(stamp)); gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if len(queue) >= 4 and shots[i] == shots[i-4]:
                j, old_gray, old_rgb = queue[-4]
                assert i-j == 4 and pts[i]-pts[j] == 1024
                for foot in (0, 1):
                    value = motion_patch(old_gray, gray, old_rgb, upper[j,foot], upper[i,foot], upper[j,1-foot])
                    if value is None:
                        continue
                    for name, dst in (('error',error),('spread',spread),('source_step',source_step),('model_step',model_step)):
                        dst[i,foot] = value[name]
                    count[i,foot] = value['count']; overlap[i,foot] = value['overlap']; affine[i,foot] = value['affine_error']
            queue.append((i, gray, rgb))
            if i % 600 == 0:
                print(json.dumps(dict(scanned=i+1,total=n,source_seconds=float(stamp/12800))),flush=True)
    finally:
        reader.close()
    usable = (count >= 4) & (spread.max(2) < 2.5) & (affine < 1.5) & (overlap < .35)
    # Positive residual y means the model rose farther than tracked pixels.
    suspect = usable & (error[:,:,1] > 3.)
    path = args.work / 'source_review/current_lift_screen.npz'
    np.savez_compressed(path,pts=pts,shot_ids=shots,error=error,spread=spread,count=count,
                        source_step=source_step,model_step=model_step,overlap=overlap,
                        affine_error=affine,usable=usable,suspect=suspect,
                        parameter_sha256=str(cache['parameter_sha256']),source_sha256=source['sha256'])
    events=[]
    for foot in (0,1):
        ids=np.flatnonzero(suspect[:,foot])
        cuts=(np.diff(ids)>8)|(shots[ids[1:]]!=shots[ids[:-1]])
        for group in np.split(ids,np.flatnonzero(cuts)+1):
            if not len(group):continue
            peak=int(group[np.argmax(error[group,foot,1])])
            events.append(dict(foot=foot,first_index=int(group[0]),last_index=int(group[-1]),
                peak_index=peak,seconds=(pts[group[[0,-1]]]/12800).tolist(),
                peak_seconds=float(pts[peak]/12800),screen_overlift_px=float(error[peak,foot,1]),
                evidence='raw LK screening only; requires source-neighbour review',review='pending'))
    report=dict(schema='rinkspace.upper_lift_screen.v1',parameter_sha256=str(cache['parameter_sha256']),
        source_sha256=source['sha256'],source_exposures=n,physical_foot_exposures=2*n,
        comparable_foot_exposures=int(usable.sum()),unknown_foot_exposures=int((~usable).sum()),
        source_seconds=(pts[[0,-1]]/12800).tolist(),comparison_interval_seconds=.08,
        screen_threshold_px=3.,threshold_is_navigation_not_accuracy=True,
        masks_and_model_are_only_patch_proposals=True,metal_excluded_from_geometry=True,
        events=sorted(events,key=lambda r:r['peak_seconds']))
    (args.work/'source_review/current_lift_review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k!='events'}),flush=True)
    print(json.dumps(sorted(events,key=lambda r:-r['screen_overlift_px'])[:20]),flush=True)


def boards(args):
    """Raw source beside unchanged saved geometry, including adjacent frames."""
    path=args.work/'source_review/current_lift_review.json'
    report=json.loads(path.read_text(encoding='utf8'))
    screen=dict(np.load(args.work/'source_review/shoe_screen_after.npz'))
    assert str(screen['parameter_sha256'])==report['parameter_sha256']
    cue=dict(np.load(args.work/'source_review/current_lift_screen.npz'))
    upper=dict(np.load(args.work/'source_review/current_lift_upper.npz'))['upper']
    # A rotating 3-D shoe is often not affine in the image. Those comparisons
    # stay unknown; review a candidate in EVERY chronological bin as well.
    # This navigation addition never promotes an uncertain track to evidence.
    if not report.get('all_chronological_bins_included'):
        for first in range(0,len(screen['pts']),100):
            stop=min(first+100,len(screen['pts']))
            if any(first<=r['peak_index']<stop for r in report['events']):continue
            score=np.where((cue['count'][first:stop]>=4)&(cue['spread'][first:stop,:,1]<2.5),
                           cue['error'][first:stop,:,1],-np.inf)
            reason='uncertain raw/model motion cue; perspective, overlap and reflection need visual review'
            if not np.isfinite(score).any():
                centre=upper.mean(2)
                score=np.zeros((stop-first,2))
                for k in range(first,stop):
                    j=max(0,k-4)
                    if screen['shot_ids'][k]==screen['shot_ids'][j]:score[k-first]=centre[j,:,1]-centre[k,:,1]
                reason='model lift selects raw frames only; source motion unavailable'
            offset,foot=np.unravel_index(np.argmax(score),score.shape);i=first+int(offset)
            report['events'].append(dict(foot=int(foot),first_index=i,last_index=i,peak_index=i,
                seconds=[float(screen['pts'][i]/12800)]*2,peak_seconds=float(screen['pts'][i]/12800),
                screen_overlift_px=None,evidence=reason,review='pending',chronological_navigation=True))
        report['events'].sort(key=lambda r:r['peak_seconds'])
        report['all_chronological_bins_included']=True
        report['chronological_bin_seconds']=2.
    source=json.loads(args.registration.read_text(encoding='utf8'))['primary_source']
    reader=SourceVideo(source['path'],[1,12800])
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',20)
    pages=[]
    try:
        for start in range(0,len(report['events']),6):
            canvas=Image.new('RGB',(3300,1860),(20,25,30));draw=ImageDraw.Draw(canvas)
            entries=[]
            for slot,row in enumerate(report['events'][start:start+6]):
                centre=row['peak_index'];foot=row['foot'];records=[]
                for col,delta in enumerate((-4,0,4)):
                    i=int(np.clip(centre+delta,0,len(screen['pts'])-1))
                    if screen['shot_ids'][i]!=screen['shot_ids'][centre]:i=centre
                    raw=reader.read(int(screen['pts'][i]));v=screen['contour'][i,foot]
                    box=crop_bounds(v,raw.shape[:2],45)
                    x0,y0,x1,y1=box;tile=raw[y0:y1,x0:x1]
                    scale=min(535/tile.shape[1],265/tile.shape[0])
                    draw.text((col*1100+5,slot*310+3),f"E{start+slot:03d} foot {foot}  {screen['pts'][i]/12800:.2f}s RAW / CURRENT",font=font,fill='white')
                    origins=[]
                    for k in range(2):
                        value=tile.copy()
                        if k:
                            for f in (0,1):
                                cv2.polylines(value,[np.rint(screen['contour'][i,f]-[x0,y0]).astype(np.int32)],True,
                                              (255,145,20) if f==foot else (25,210,255),1)
                        im=Image.fromarray(value).resize((round(tile.shape[1]*scale),round(tile.shape[0]*scale)))
                        origin=[col*1100+k*550+(550-im.width)//2,slot*310+32]
                        canvas.paste(im,tuple(origin));origins.append(origin)
                    records.append(dict(pts=int(screen['pts'][i]),seconds=float(screen['pts'][i]/12800),
                                        bounds=box,scale=scale,raw_origin=origins[0]))
                entries.append(dict(event=start+slot,foot=foot,frames=records))
            output=args.work/f'source_review/current_lift_events_{start//6}.jpg'
            canvas.save(output,quality=94)
            pages.append(dict(path=str(output),sha256=digest(output),entries=entries))
            print(json.dumps(dict(page=start//6,events=len(entries))),flush=True)
    finally:reader.close()
    report['pages']=pages
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('work','prepared','source','parameter','registration'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--project',action='store_true')
    p.add_argument('--boards',action='store_true')
    a=p.parse_args()
    if a.boards:
        boards(a)
        return
    if a.project:projection(a)
    scan(a)


if __name__=='__main__':
    main()
