"""Track source-verified skate edges on the native programme clock."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import json

import av
import cv2
import numpy as np


SEMANTICS = {'boot_sole': 'outsole', 'front_stanchion': 'front_mount',
             'rear_stanchion': 'rear_mount', 'runner_lower_edge': 'runner_lower_edge',
             'runner_upper_edge': 'runner_upper_edge'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def native_frames(path, pts, time_base, wanted):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        container.seek(int(int(pts[0])*time_base/stream.time_base), stream=stream, backward=True)
        i = 0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            stamp = frame.pts*frame.time_base/time_base
            if stamp < pts[0]:
                continue
            if stamp != pts[i]:
                raise ValueError(f'Changed source frame at index {i}')
            if i in wanted:
                yield i, frame.to_ndarray(format='gray')
            i += 1
            if i == len(pts):
                return
    raise ValueError('Incomplete native programme decode')


def resample(points, step=3.):
    u = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    q = np.linspace(0., u[-1], max(4, int(np.ceil(u[-1]/step))+1))
    return np.stack([np.interp(q, u, x) for x in points.T], axis=-1)


def normals(points):
    tangent = np.gradient(points, axis=0)
    normal = np.c_[-tangent[:, 1], tangent[:, 0]]
    return normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)


def profile(image, points, normal, shifts):
    xy = points[:, None, None, :] + normal[:, None, None, :]*(np.asarray(shifts)[None, :, None, None] +
                                                    np.array([-2.5, -1.2, 1.2, 2.5])[None, None, :, None])
    coord = xy.reshape(len(points), -1, 2).astype(np.float32)
    value = cv2.remap(image.astype(np.float32), coord[..., 0], coord[..., 1],
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    value = value.reshape(len(points), len(shifts), 4)
    return value[..., 2:].mean(-1)-value[..., :2].mean(-1)


def advance(parts, before, after):
    # LK and the motion-consensus test use one native crop for the whole shoe.
    pts = np.concatenate([p['uv'] for p in parts])
    lo = np.maximum(np.floor(pts.min(0)-94).astype(int), 0)
    hi = np.minimum(np.ceil(pts.max(0)+94).astype(int), before.shape[::-1])
    x0, y0 = lo
    x1, y1 = hi
    a, b = (f[y0:y1, x0:x1] for f in (before, after))
    if min(a.shape) < 32:
        return []
    p0 = (pts-lo).astype(np.float32)
    mask = np.zeros(a.shape, np.uint8)
    for part in parts:
        cv2.polylines(mask, [np.rint(part['uv']-lo).astype(np.int32)], False, 255, 33)
    texture = cv2.goodFeaturesToTrack(a, 46, .011, 4, mask=mask)
    if texture is not None:
        p0 = np.r_[p0, texture[:, 0]]
    p0 = np.ascontiguousarray(p0[:, None])
    opts = dict(winSize=(23, 23), maxLevel=3)
    f, sf, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, **opts)
    if f is None:
        return []
    back, sb, _ = cv2.calcOpticalFlowPyrLK(b, a, f, None, **opts)
    if back is None:
        return []
    err = np.linalg.norm((back-p0)[:, 0], axis=1)
    valid = (sf[:, 0] != 0) & (sb[:, 0] != 0) & (err < 1.3)
    if valid.sum() < 6:
        return []
    pose, inliers = cv2.estimateAffinePartial2D(p0[valid, 0], f[valid, 0], method=cv2.RANSAC,
                                                ransacReprojThreshold=2.8, maxIters=400)
    if pose is None or inliers.mean() < .66 or not .76 < np.linalg.norm(pose[0, :2]) < 1.27:
        return []
    predicted = p0[:, 0]@pose[:, :2].T+pose[:, 2]
    valid &= np.linalg.norm(predicted-f[:, 0], axis=1) < 3.8
    result = []
    shifts = np.array([-2., -1., 0., 1., 2.])
    start = 0
    for part in parts:
        end = start+len(part['uv'])
        old = part['uv']-lo
        q = f[start:end, 0]
        last = profile(a, old, normals(old), [0.])[:, 0]
        normal = normals(q)
        strength = profile(b, q, normal, shifts)*np.sign(last)[:, None]
        best = np.argmax(strength - abs(shifts)[None]*1.4, axis=1)
        actual = strength[np.arange(len(q)), best]
        good = valid[start:end] & (abs(last) >= 5.) & (actual >= 6.) & (actual >= .39*abs(last))
        start = end
        runs = np.split(np.flatnonzero(good), np.flatnonzero(np.diff(np.flatnonzero(good)) != 1)+1)
        idx = max(runs, key=len)
        if len(idx) < 3 or len(idx) < .45*part['seed_points']:
            continue
        q = q+normal*shifts[best, None]+lo
        line = q[idx]
        if np.linalg.norm(line[-1]-line[0]) < 7 or np.min(np.linalg.norm(np.diff(line, axis=0), axis=1)) < 1.:
            continue
        result.append(dict(part, uv=line,
                           confidence=float(min(1., .55*inliers.mean()+.45*len(idx)/len(q)))))
    return result


def make_anchors(document, manual):
    evidence = document['programme_skate_evidence']
    anchors = []
    for r in document.get('full_source_skate_structures', {}).get('rows', []):
        sem = SEMANTICS.get(r['structure_feature'])
        if sem is None or r['role'] != 'fit' or r.get('optimization_enabled') is False:
            continue
        if r.get('bounded_ambiguous_boundary') or r.get('source_status') == 'ambiguous':
            continue
        src_order = r.get('source_order', 'unknown')
        direction = 'rear_to_front' if src_order in ('front_to_rear', 'rear_to_front') and sem.startswith('runner_') else 'image_trace'
        u = r['polyline_uv'][::-1] if src_order == 'front_to_rear' and direction == 'rear_to_front' else r['polyline_uv']
        anchors.append(dict(anchor_id=r['id'], kind='legacy', pts=r['pts'], foot=r['foot'],
                            role='fit', semantic=sem, source_order=direction, uv=u,
                            radius=float(r['pixel_uncertainty_radius_px'])))
    for r in [*evidence['ordered_contours'], *manual]:
        if r['role'] not in ('fit', 'check') or r['foot'] not in (0, 1):
            continue
        provenance = r.get('provenance', {})
        if r not in manual and provenance.get('kind') != 'direct source visual read':
            continue
        anchors.append(dict(anchor_id=r['raw_group_id'], kind='manual' if r in manual else 'ordered',
                            pts=r['pts'], foot=r['foot'], role=r['role'], semantic=r['semantic'],
                            source_order=r['source_order'], uv=r['polyline_uv'], radius=float(r['radius_px'])))
    return anchors


def track(frames, pts, index, rows, window, hash_cache, result):
    initial = []
    for row in rows:
        p = np.asarray(row['uv'], float)
        length = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        if length < 10:
            continue
        u = resample(p)
        initial.append(dict(row, uv=u, seed_points=len(u)))
    for direction in (-1, 1):
        live = initial
        previous = index
        for step in range(1, window+1):
            current = index+direction*step
            if current not in frames or not live:
                break
            live = advance(live, frames[previous], frames[current])
            previous = current
            if step%2:
                continue
            for r in live:
                if step > 16 and r['confidence'] < .81:
                    continue
                if current not in hash_cache:
                    hash_cache[current] = sha256(np.ascontiguousarray(frames[current]).tobytes()).hexdigest()
                result.append(dict(r, pts=int(pts[current]), distance=step, native_luma_sha256=hash_cache[current]))


def main():
    p = ArgumentParser()
    for name in ('source', 'prepared', 'parent', 'manual', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir()
    cv2.setNumThreads(2)
    cv2.setRNGSeed(431)
    doc = read(args.parent)
    evidence = doc['programme_skate_evidence']
    manual = read(args.manual)
    with np.load(args.prepared/'input.npz') as f:
        pts, shots = (f[k] for k in ('pts', 'shot_ids'))
    ix = {int(p): i for i, p in enumerate(pts)}
    window = 21
    anchors = make_anchors(doc, manual)
    key = lambda r: (r['pts'], r['foot'], r['semantic'])
    # One seed per source group and material. No duplicate views of a legacy outline.
    by_anchor = {}
    for row in sorted(anchors, key=lambda r: (r['kind'] == 'manual', r['role'] == 'check', len(r['uv']))):
        by_anchor[key(row)] = row
    groups = defaultdict(list)
    for row in by_anchor.values():
        groups[(row['pts'], row['foot'], row['role'])].append(row)
    schedule = defaultdict(list)
    demand = Counter()
    for (stamp, _, _), rows in groups.items():
        at = ix[stamp]
        span = tuple(i for i in range(max(0, at-window), min(len(pts), at+window+1))
                     if shots[i] == shots[at])
        demand.update(span)
        schedule[span[-1]].append((at, rows, span))
    candidate = []
    frames, hashes = {}, {}
    for i, frame in native_frames(args.source, pts, Fraction(*evidence['time_base']), set(demand)):
        frames[i] = frame
        for at, rows, span in schedule[i]:
            track({j: frames[j] for j in span}, pts, at, rows, window, hashes, candidate)
            for j in span:
                demand[j] -= 1
                if demand[j] == 0:
                    del frames[j]
        if i%500 < 2:
            print(json.dumps(dict(event='native_source_tracking', index=i, measured=len(candidate))), flush=True)
    existing = {key(r) for r in [*evidence['ordered_contours'], *manual] if r['role'] in ('fit', 'check')}
    existing.update(key(r) for r in by_anchor.values() if r['kind']=='legacy')
    check_windows = {(r['foot'], ix[r['pts']]+dt) for r in manual if r['role']=='check'
                     for dt in range(-10, 11)}
    choices = defaultdict(list)
    for r in candidate:
        if key(r) in existing or (r['role']=='fit' and (r['foot'], ix[r['pts']]) in check_windows):
            continue
        choices[key(r)].append(r)
    events = []
    for shot in dict.fromkeys(shots.tolist()):
        pp = pts[shots == shot]
        events.append(dict(id='temporal_skate_'+shot, start_pts=int(pp[0]), end_pts=int(pp[-1]),
                           active_feet=[0,1], description_zh='原片连续可见鞋刀跟踪'))
    for r in manual:
        shot = shots[ix[r['pts']]]
        r['event_id'] = 'temporal_skate_'+shot
        r['track_id'] = f'temporal-skate:{shot}:{r["foot"]}:{r["semantic"]}'
    added = []
    disagreements = 0
    for parts in choices.values():
        parts.sort(key=lambda r: (r['role']=='check', r['confidence']-.012*r['distance']), reverse=True)
        best = parts[0]
        if len(parts)>1 and parts[1]['role']==best['role']:
            other=parts[1]
            distance = np.linalg.norm(best['uv'][:,None]-other['uv'],axis=-1)
            if max(np.median(distance.min(0)),np.median(distance.min(1))) > max(5.,best['radius']+1.):
                disagreements += 1
                continue
        t, f, sem = best['pts'], best['foot'], best['semantic']
        shot = shots[ix[t]]
        added.append(dict(raw_group_id=f'visible-track:{t}:{f}:{sem}', event_id='temporal_skate_'+shot,
            track_id=f'temporal-skate:{shot}:{f}:{sem}', pts=t, foot=f, semantic=sem, role=best['role'],
            phase='continuous_visible', polyline_uv=best['uv'].tolist(), source_order=best['source_order'],
            complete_visible=False, visible_range='contiguous native arc with signed edge response',
            visibility='partial', radius_px=best['radius']+1.3+.055*best['distance'],
            relative_weight=max(.34, best['confidence']*(1-.019*best['distance'])),
            provenance=dict(kind='native optical motion plus signed edge measured at this exposure',
                            track_source_anchor_id=best['anchor_id'], track_source_anchor_kind=best['kind'],
                            source_luma_sha256=best['native_luma_sha256'])))
    amended = dict(schema='rinkspace.programme-skate-amendment.v1', time_base=evidence['time_base'],
        source_video_sha256=evidence['source_video_sha256'], events=events, manual_landmarks=[],
        ordered_contours=[*manual,*added], exposure_review=[],
        method_zh='真实源图短时正反运动及有符号像素边界支持的有序分件弧。相邻曝光单独核对，来源种子继承拟合/检查角色。',
        tracked_sequence_policy=dict(schema='rinkspace.skate-tracked-sequence.v1', max_anchor_offset_s=.42,
            boot_axes_enabled=True, legacy_source_guard=True))
    write(args.output/'amendment.json',amended)
    report = dict(rows=len(added), direct_rows=len(manual), disagreed_views=disagreements, decoded_exposures=len(pts),
        by_shot={s:{role:{'rows':sum(r['event_id']=='temporal_skate_'+s and r['role']==role for r in added),
                       'exposures':len({r['pts'] for r in added if r['event_id']=='temporal_skate_'+s and r['role']==role})}
                 for role in ('fit','check')} for s in dict.fromkeys(shots.tolist())})
    write(args.output/'coverage.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
