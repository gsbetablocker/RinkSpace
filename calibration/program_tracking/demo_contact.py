"""Presentation-only support-blade candidates during explicitly selected spins.

The raw candidate is the bottom-three-row midpoint of the largest original
SAM2 component. It is a segmentation heuristic, not a measured blade label.
Original source ankle observations diagnose missing boots; invalid candidates
remain recorded and replacements are explicitly inferred. No body/model input
is changed and nothing outside the chosen grounded spin windows is replaced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

ROOT = Path(__file__).resolve().parents[2]
for location in (ROOT, ROOT / 'src', ROOT / 'calibration/program_tracking'):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))
from calibration.program_tracking.demo_source import DEFAULT_OUTPUT, LEGACY_EXPORT, read, path, ref, check_output_path
from calibration.program_tracking.render_result import PersonMasks, DEFAULT_MASKS, DEFAULT_PEOPLE
from calibration.program_tracking.projection import ground_xy

DEFAULT_WINDOWS = ((156.4, 165.9), (171.92, 184.9))
REASONS = {1: 'missing or invalid SAM2 mask', 2: 'largest component too small',
           4: 'mask bottom clipped by image', 8: 'original ankle observation unavailable',
           16: 'mask/ankle boot region inconsistent', 32: 'ground inverse invalid',
           64: 'isolated world-coordinate jump'}


class RawMasks:
    def __init__(self):
        self.people = PersonMasks(DEFAULT_MASKS, DEFAULT_PEOPLE)
        self.active_path, self.data = None, None

    def get(self, pts):
        item = self.people.lookup.get(int(pts))
        if item is None:
            return None, 'missing'
        file, i, status = item
        if status != 'measured_mask':
            return None, status
        if self.active_path != file:
            with np.load(file, allow_pickle=False) as z:
                self.data = {k: z[k].copy() for k in ('pts', 'masks')}
            self.active_path = file
        if int(self.data['pts'][i]) != int(pts):
            return None, 'wrong_pts'
        return np.unpackbits(self.data['masks'][i], axis=1)[:, :960].astype(np.uint8), status


def _intervals(times, selected):
    ix = np.flatnonzero(selected)
    if not len(ix):
        return []
    runs = np.split(ix, np.flatnonzero(np.diff(ix) > 1) + 1)
    return [{'start_seconds': float(times[r[0]]), 'last_seconds': float(times[r[-1]]),
             'frames': len(r)} for r in runs]


def _local_median(values, radius=25):
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        chunk = values[max(0, i-radius): min(len(values), i+radius+1)]
        good = chunk[np.isfinite(chunk)]
        if len(good):
            out[i] = np.median(good)
    return out


def build_packet(output_dir=DEFAULT_OUTPUT, windows=DEFAULT_WINDOWS, blend_seconds=.25):
    output_dir = path(output_dir)
    npz_path = check_output_path(output_dir / 'contact_packet.npz')
    json_path = check_output_path(output_dir / 'contact_packet.json')
    source_path = output_dir / 'source_packet.npz'
    with np.load(source_path, allow_pickle=False) as z:
        source = {k: z[k].copy() for k in z.files}
    command = read(LEGACY_EXPORT / 'export.command.json')
    body_path = path(command[command.index('--body') + 1])
    with np.load(body_path, allow_pickle=False) as z:
        body = {k: z[k].copy() for k in ('actual_pts', 'source_COCO17_native_uv',
                                        'source_COCO17_qualified', 'source_COCO17_score')}
    body_rows = {int(p): i for i, p in enumerate(body['actual_pts'])}
    scene = read(str(source['scene_path']))
    times, pts = source['time_seconds'], source['pts']
    unit = float(source['unit'])
    n = len(pts)
    raw_uv = np.full((n, 2), np.nan)
    raw_world = np.full((n, 2), np.nan)
    ankles = np.full((n, 2, 2), np.nan)
    ankle_scores = np.full((n, 2), np.nan)
    ankle_qualified = np.zeros((n, 2), bool)
    support_ankle_index = np.full(n, -1, np.int8)
    component_area = np.zeros(n, np.int32)
    boot_gap = np.full(n, np.nan)
    boot_expected = np.full(n, np.nan)
    reason = np.zeros(n, np.uint16)
    spin_id = np.zeros(n, np.int8)
    candidate_valid = np.zeros(n, bool)
    inferred = np.zeros(n, bool)
    method = np.zeros(n, np.uint8)  # 0 untouched,1 mask,2 neighbour interpolation,3 ankle-assisted inference
    filtered_uv = np.full((n, 2), np.nan)
    contact_world = np.full((n, 2), np.nan)
    correction_weight = np.zeros(n, float)
    xy = source['xy_m'].copy()
    masks = RawMasks()
    for sid, (begin, end) in enumerate(windows, 1):
        if begin >= end:
            raise ValueError('Each spin needs increasing grounded source times')
        rows = np.flatnonzero((times >= begin-1e-8) & (times <= end+1e-8))
        if not len(rows) or np.any(source['shot_index'][rows] != 9):
            raise ValueError('This preview adapter expects spin windows entirely in official shot9')
        spin_id[rows] = sid
        for i in rows:
            legacy = int(source['legacy_mask_pts'][i])
            b = body_rows.get(legacy)
            if b is not None:
                ankles[i] = body['source_COCO17_native_uv'][b, 15:17]
                ankle_scores[i] = body['source_COCO17_score'][b, 15:17]
                ankle_qualified[i] = body['source_COCO17_qualified'][b, 15:17]
                good = ankle_qualified[i] & (ankle_scores[i] >= .45) & np.isfinite(ankles[i]).all(axis=1)
                if good.any():
                    support_ankle_index[i] = int(np.argmax(np.where(good, ankles[i, :, 1], -np.inf)))
                else:
                    reason[i] |= 8
            else:
                reason[i] |= 8
            mask, status = masks.get(legacy)
            if mask is None:
                reason[i] |= 1
                continue
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if count < 2:
                reason[i] |= 2
                continue
            label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            component_area[i] = int(stats[label, cv2.CC_STAT_AREA])
            if component_area[i] < 800:
                reason[i] |= 2
                continue
            ys, xs = np.nonzero(labels == label)
            bottom = int(ys.max())
            strip = xs[ys >= bottom-2]
            raw_uv[i] = ((float(strip.min())+float(strip.max())), float(bottom*2))
            if bottom >= 538:
                reason[i] |= 4
            if support_ankle_index[i] >= 0:
                boot_gap[i] = raw_uv[i, 1] - ankles[i, support_ankle_index[i], 1]
            world, good = ground_xy(source['camera25'][i], raw_uv[i:i+1], scene)
            if good[0]:
                raw_world[i] = world[0] * unit
            else:
                reason[i] |= 32
        gaps = boot_gap[rows]
        plausible = np.where((gaps >= 30) & (gaps <= 160), gaps, np.nan)
        expected = _local_median(plausible)
        if not np.isfinite(expected).all():
            fallback = np.nanmedian(plausible)
            expected[~np.isfinite(expected)] = fallback
        boot_expected[rows] = expected
        bad_boot = np.isfinite(gaps) & ((gaps < np.maximum(22., expected*.42))
                    | (gaps > np.maximum(140., expected*2.1)))
        reason[rows[bad_boot]] |= 16
        # Reject isolated spikes only, retaining the actual support-foot route.
        raw = raw_world[rows]
        finite_raw = raw.copy()
        finite_good = np.isfinite(finite_raw).all(axis=1)
        for axis in range(2):
            finite_raw[:, axis] = np.interp(times[rows], times[rows[finite_good]], raw[finite_good, axis])
        median = median_filter(finite_raw, size=(21, 1), mode='nearest')
        residual = np.linalg.norm(raw-median, axis=1)
        jump = np.isfinite(residual) & (residual > .35)
        reason[rows[jump]] |= 64
        good = (reason[rows] == 0) & np.isfinite(raw_uv[rows]).all(axis=1)
        if good.sum() < 2:
            raise ValueError(f'Insufficient reliable mask candidates in spin {sid}')
        candidate_valid[rows] = good
        inferred[rows] = ~good
        method[rows[good]] = 1
        filled = raw_uv[rows].copy()
        for axis in range(2):
            filled[:, axis] = np.interp(times[rows], times[rows[good]], raw_uv[rows[good], axis])
        method[rows[~good]] = 2
        # A longer dropout can follow the original lower ankle with the local
        # reliable blade-minus-ankle offset; this remains inferred, never measured.
        support = np.full((len(rows), 2), np.nan)
        for j, i in enumerate(rows):
            if support_ankle_index[i] >= 0:
                support[j] = ankles[i, support_ankle_index[i]]
        offset = raw_uv[rows] - support
        offset[~good] = np.nan
        offset_reference = np.column_stack([_local_median(offset[:, axis], 25) for axis in range(2)])
        good_indices = np.flatnonzero(good)
        for j in np.flatnonzero(~good):
            distance = np.min(np.abs(times[rows[j]]-times[rows[good_indices]]))
            if distance > .16 and np.isfinite(support[j]).all() and np.isfinite(offset_reference[j]).all():
                filled[j] = support[j] + offset_reference[j]
                method[rows[j]] = 3
        filtered = median_filter(filled, size=(5, 1), mode='nearest')
        filtered = gaussian_filter1d(filtered, sigma=1., axis=0, mode='nearest')
        filtered_uv[rows] = filtered
        for j, i in enumerate(rows):
            world, valid = ground_xy(source['camera25'][i], filtered[j:j+1], scene)
            if not valid[0]:
                raise ValueError(f'Filtered contact leaves ice at official PTS {int(pts[i])}')
            contact_world[i] = world[0] * unit
        distance_to_edge = np.minimum(times[rows]-begin, end-times[rows])
        weight = np.clip(distance_to_edge/max(float(blend_seconds), 1e-9), 0., 1.)
        weight = weight*weight*(3.-2.*weight)
        correction_weight[rows] = weight
        xy[rows] = source['xy_m'][rows]*(1.-weight[:, None]) + contact_world[rows]*weight[:, None]
    selected = spin_id > 0
    packet = dict(pts=pts, native_pts=pts, time_seconds=times, xy_m=xy,
                  original_xy_m=source['xy_m'], contact_uv=filtered_uv,
                  raw_uv=raw_uv, filtered_uv=filtered_uv, raw_world_m=raw_world,
                  filtered_world_m=contact_world, candidate_valid=candidate_valid,
                  inferred=inferred, spin_id=spin_id, correction_weight=correction_weight,
                  candidate_invalid_reason=reason, replacement_method=method,
                  original_source_ankles_uv=ankles, original_source_ankles_score=ankle_scores,
                  original_source_ankles_qualified=ankle_qualified,
                  lower_ankle_index=support_ankle_index, boot_gap_px=boot_gap,
                  expected_boot_gap_px=boot_expected, largest_component_area_halfres=component_area,
                  legacy_mask_pts=source['legacy_mask_pts'], time_base=source['time_base'],
                  unit=source['unit'], source_path=source['source_path'], scene_path=source['scene_path'])
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **packet)
    anomalies = {text: _intervals(times, selected & ((reason & bit) != 0)) for bit, text in REASONS.items()}
    inspected = []
    for t in (178.4, 178.6, 178.8, 180.):
        i = int(np.argmin(np.abs(times-t)))
        inspected.append(dict(seconds=float(times[i]), pts=int(pts[i]), legacy_mask_pts=int(source['legacy_mask_pts'][i]),
                              raw_uv=raw_uv[i].tolist(), filtered_uv=filtered_uv[i].tolist(),
                              source_ankles_uv=ankles[i].tolist(), ankle_scores=ankle_scores[i].tolist(),
                              ankle_qualified=ankle_qualified[i].tolist(), boot_gap_px=float(boot_gap[i]),
                              expected_boot_gap_px=float(boot_expected[i]), candidate_valid=bool(candidate_valid[i]),
                              inferred=bool(inferred[i]), invalid_reason=int(reason[i])))
    report = dict(schema='rinkspace.demo-spin-support-candidates.v1', output=ref(npz_path),
                  inputs=dict(source_packet=ref(source_path), original_body=ref(body_path)),
                  sam2_original_mask_directory=str(masks.people.mask_dir),
                  sam2_dilation_applied=False,
                  windows_seconds=[list(map(float, w)) for w in windows], blend_seconds=float(blend_seconds),
                  window_visual_review=dict(source_times_checked_seconds=[155.8, 156.2, 156.6, 157.0],
                      reviewer='main_session_original_source_visual_review',
                      conclusion='150.4 through 155.8 is travelling turns/steps; fixed layback begins near156.4. Only156.4-165.9 is spin1. Spin2 remains171.92-184.9.'),
                  spin_frames=int(selected.sum()), candidate_valid_frames=int(candidate_valid.sum()),
                  inferred_frames=int(inferred.sum()), per_spin={str(s): dict(frames=int(np.sum(spin_id==s)),
                     valid=int(np.sum(candidate_valid & (spin_id==s))),
                     inferred=int(np.sum(inferred & (spin_id==s)))) for s in range(1, len(windows)+1)},
                  method='Largest original undilated half-resolution SAM2 component; midpoint of bottom 3 rows scaled x2; source COCO ankles diagnose truncated boots; ground_xy uses unchanged camera',
                  validity_is_heuristic_not_manual_blade_measurement=True,
                  missing_candidate_policy='Same-spin reliable UV interpolation; longer gaps may use observed lower ankle plus local reliable blade offset, always inferred',
                  smoothing='5-frame median then Gaussian sigma1 frame, same spin only',
                  source_clocks_preserved=True, originals_unmodified=True,
                  airborne_before_grounded_window_not_modified=True,
                  anomaly_reason_bits={str(k): v for k, v in REASONS.items()}, anomalies=anomalies,
                  inspected_samples=inspected,
                  replacement_method={'0': 'untouched', '1': 'raw mask heuristic with light filter',
                                      '2': 'inferred from reliable neighbours', '3': 'ankle-assisted inference'},
                  solver_run=False, body_fit_run=False, video_generated=False)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True) + '\n', encoding='utf8')
    return report


FULL_OUTPUT = DEFAULT_OUTPUT.parent/'demo_full_20261016'
FRONT_WINDOW = (81.56, 92.64)
FRONT_WINDOW_EVIDENCE = dict(
    reviewer='main_session_locate_original_official50_source_visual_review',
    start_pts=1043968, last_pts=1185792, time_base='1/12800',
    excluded_airborne_seconds=[81.44,81.46],
    excluded_uncertain_landing_seconds=[81.48,81.54],
    first_clear_blade_contact_seconds=81.56,
    stable_camel_seconds=81.60,
    final_supported_rotation_seconds=[92.60,92.62,92.64],
    blade_flat_and_exit_begins_seconds=92.66,
    excluded_kneeling_choreography_from_seconds=92.80,
    evidence_path='D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/demo_last_shot_20261016/front_spin_grounded_edges.jpg')


def _component_bottom(labels,stats,label):
    x,y,w,h,_=map(int,stats[label])
    bottom=y+h-1
    strip=labels[max(y,bottom-2):bottom+1,x:x+w]
    _,xx=np.nonzero(strip==label)
    if not len(xx):return np.full(2,np.nan)
    return np.array([float(xx.min()+xx.max()+2*x),float(bottom*2)])


def build_full_packet(output_dir=FULL_OUTPUT,*,source_packet_path=DEFAULT_OUTPUT/'source_packet.npz',
                      baseline_contact_path=DEFAULT_OUTPUT/'contact_packet.npz',front_window=FRONT_WINDOW,
                      blend_seconds=.25,window_evidence=None):
    """Append only the independently reviewed front spin, preserving old rows.

    sid1/2 retain their historical end-spin identities; sid3 is the earlier
    source-time spin. No immutable source or baseline contact file is changed.
    """
    output_dir=path(output_dir)
    output_npz=check_output_path(output_dir/'contact_packet.npz')
    output_json=check_output_path(output_dir/'contact_packet.json')
    if output_npz.resolve()==path(baseline_contact_path).resolve():
        raise ValueError('Full-programme extension must preserve the old baseline contact packet')
    with np.load(source_packet_path,allow_pickle=False) as z:source={k:z[k].copy() for k in z.files}
    with np.load(baseline_contact_path,allow_pickle=False) as z:baseline={k:z[k].copy() for k in z.files}
    if not np.array_equal(source['pts'],baseline['pts']):raise ValueError('Source/baseline official native clocks differ')
    result={k:v.copy() for k,v in baseline.items()}
    times,pts=source['time_seconds'],source['pts']
    n=len(pts)
    begin,end=map(float,front_window)
    rows=np.flatnonzero((times>=begin-1e-8)&(times<=end+1e-8))
    if not len(rows) or len(np.unique(source['shot_index'][rows]))!=1:
        raise ValueError('New grounded spin must stay inside one unchanged source camera shot')
    if np.any(baseline['spin_id'][rows]!=0):raise ValueError('New front spin overlaps an existing immutable contact interval')
    if n!=8434 or np.any(np.diff(pts)!=256):raise ValueError('Complete official50 native programme clock required')
    command=read(LEGACY_EXPORT/'export.command.json')
    body_path=path(command[command.index('--body')+1])
    with np.load(body_path,allow_pickle=False) as z:
        body={k:z[k].copy() for k in ('actual_pts','source_COCO17_native_uv','source_COCO17_qualified','source_COCO17_score')}
    body_rows={int(v):i for i,v in enumerate(body['actual_pts'])}
    scene=read(str(source['scene_path']))
    unit=float(source['unit'])
    masks=RawMasks()
    m=len(rows)
    largest_uv=np.full((m,2),np.nan)
    selected_uv=np.full((m,2),np.nan)
    ankles=np.full((m,2,2),np.nan)
    scores=np.full((m,2),np.nan)
    qualified=np.zeros((m,2),bool)
    support=np.full(m,-1,np.int8)
    area=np.zeros(m,np.int32)
    body_height=np.full(m,np.nan)
    reasons=np.zeros(m,np.uint16)
    raw_world=np.full((m,2),np.nan)
    max_world=np.full((m,2),np.nan)
    components=[]
    for j,i in enumerate(rows):
        legacy=int(source['legacy_mask_pts'][i])
        b=body_rows.get(legacy)
        if b is not None:
            ankles[j]=body['source_COCO17_native_uv'][b,15:17]
            scores[j]=body['source_COCO17_score'][b,15:17]
            qualified[j]=body['source_COCO17_qualified'][b,15:17]
            good=qualified[j]&(scores[j]>=.45)&np.isfinite(ankles[j]).all(axis=1)
            if good.any():support[j]=int(np.argmax(np.where(good,ankles[j,:,1],-np.inf)))
        if support[j]<0:reasons[j]|=8
        mask,status=masks.get(legacy)
        if mask is None:
            reasons[j]|=1;components.append([]);continue
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,connectivity=8)
        if count<2:
            reasons[j]|=2;components.append([]);continue
        largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        area[j]=int(stats[largest,cv2.CC_STAT_AREA])
        if area[j]<800:
            reasons[j]|=2;components.append([]);continue
        largest_uv[j]=_component_bottom(labels,stats,largest)
        selected_uv[j]=largest_uv[j]
        substantial=np.flatnonzero(stats[1:,cv2.CC_STAT_AREA]>=max(8,int(area[j]*.003)))+1
        body_height[j]=float((np.max(stats[substantial,cv2.CC_STAT_TOP]+stats[substantial,cv2.CC_STAT_HEIGHT])
                              -np.min(stats[substantial,cv2.CC_STAT_TOP]))*2)
        records=[]
        for label in substantial:
            uv=_component_bottom(labels,stats,int(label))
            records.append(dict(label=int(label),largest=bool(label==largest),area=int(stats[label,4]),
                                bbox=np.asarray(stats[label,:4],float)*2,uv=uv))
        components.append(records)
        world,valid=ground_xy(source['camera25'][i],largest_uv[j:j+1],scene)
        if valid[0]:max_world[j]=world[0]*unit
    # Learn the boot-to-blade pixel scale from this spin and its native camera.
    # The old near-camera fixed30px plausibility floor is deliberately not used.
    ankle=np.full((m,2),np.nan)
    for j in range(m):
        if support[j]>=0:ankle[j]=ankles[j,support[j]]
    largest_gap=largest_uv[:,1]-ankle[:,1]
    plausible=(np.isfinite(largest_gap)&(largest_gap>=np.maximum(4.,body_height*.015))
               &(largest_gap<=np.maximum(12.,body_height*.24)))
    expected=_local_median(np.where(plausible,largest_gap,np.nan),25)
    global_scale=float(np.nanmedian(largest_gap[plausible])) if plausible.any() else float(np.nanmedian(body_height)*.09)
    expected[~np.isfinite(expected)]=global_scale
    if not np.isfinite(expected).all() or np.any(expected<=0):raise ValueError('Cannot infer this spin camera boot scale')
    recovered=np.zeros(m,bool)
    selected_label=np.full(m,-1,np.int32)
    selected_area=np.zeros(m,np.int32)
    for j,i in enumerate(rows):
        if not components[j]:continue
        largest=next(v for v in components[j] if v['largest'])
        selected_label[j]=largest['label'];selected_area[j]=largest['area']
        low=max(4.,expected[j]*.35)
        high=max(12.,expected[j]*2.1)
        gap=largest_gap[j]
        if support[j]>=0 and (not np.isfinite(gap) or gap<low or gap>high):
            candidates=[]
            for candidate in components[j]:
                if candidate['largest']:continue
                cg=candidate['uv'][1]-ankle[j,1]
                x,y,w,h=candidate['bbox']
                margin=max(8.,expected[j]*.4)
                near=(x-margin<=ankle[j,0]<=x+w+margin and y-margin<=ankle[j,1]<=y+h+margin)
                horizontal=abs(candidate['uv'][0]-ankle[j,0])<=max(18.,expected[j]*1.7)
                if low<=cg<=high and near and horizontal and candidate['area']>=max(12,area[j]*.005):
                    candidates.append((abs(cg-expected[j])/expected[j]
                                       +abs(candidate['uv'][0]-ankle[j,0])/(expected[j]*4),candidate))
            if candidates:
                chosen=min(candidates,key=lambda v:v[0])[1]
                selected_uv[j]=chosen['uv'];selected_label[j]=chosen['label'];selected_area[j]=chosen['area']
                recovered[j]=True
        if selected_uv[j,1]>=1076:reasons[j]|=4
        gap=selected_uv[j,1]-ankle[j,1]
        if np.isfinite(gap) and (gap<low or gap>high):reasons[j]|=16
        world,valid=ground_xy(source['camera25'][i],selected_uv[j:j+1],scene)
        if valid[0]:raw_world[j]=world[0]*unit
        else:reasons[j]|=32
    finite=np.isfinite(raw_world).all(axis=1)
    if finite.sum()<2:raise ValueError('New spin lacks finite raw contact candidates')
    filled_world=np.column_stack([np.interp(times[rows],times[rows[finite]],raw_world[finite,axis]) for axis in range(2)])
    median=median_filter(filled_world,size=(21,1),mode='nearest')
    residual=np.linalg.norm(raw_world-median,axis=1)
    reasons[np.isfinite(residual)&(residual>.35)]|=64
    valid=(reasons==0)&np.isfinite(selected_uv).all(axis=1)
    if valid.sum()<2:raise ValueError('New spin lacks reliable source-mask blade candidates')
    filled=np.column_stack([np.interp(times[rows],times[rows[valid]],selected_uv[valid,axis]) for axis in range(2)])
    method=np.where(valid,1,2).astype(np.uint8)
    offset=selected_uv-ankle
    offset[~valid]=np.nan
    reference_offset=np.column_stack([_local_median(offset[:,axis],25) for axis in range(2)])
    for j in np.flatnonzero(~valid):
        distance=np.min(np.abs(times[rows[j]]-times[rows[valid]]))
        if distance>.16 and np.isfinite(ankle[j]).all() and np.isfinite(reference_offset[j]).all():
            filled[j]=ankle[j]+reference_offset[j]
            method[j]=3
    filtered=gaussian_filter1d(median_filter(filled,size=(5,1),mode='nearest'),sigma=1.,axis=0,mode='nearest')
    filtered_world=np.full((m,2),np.nan)
    for j,i in enumerate(rows):
        world,good=ground_xy(source['camera25'][i],filtered[j:j+1],scene)
        if not good[0]:raise ValueError(f'Inferred/filtered front blade leaves ice at native PTS{int(pts[i])}')
        filtered_world[j]=world[0]*unit
    distance=np.minimum(times[rows]-begin,end-times[rows])
    weight=np.clip(distance/max(float(blend_seconds),1e-9),0.,1.)
    weight=weight*weight*(3.-2.*weight)
    updates=dict(raw_uv=selected_uv,raw_world_m=raw_world,filtered_uv=filtered,contact_uv=filtered,
                 filtered_world_m=filtered_world,candidate_valid=valid,inferred=~valid,
                 spin_id=np.full(m,3,np.int8),correction_weight=weight,candidate_invalid_reason=reasons,
                 replacement_method=method,original_source_ankles_uv=ankles,original_source_ankles_score=scores,
                 original_source_ankles_qualified=qualified,lower_ankle_index=support,
                 boot_gap_px=selected_uv[:,1]-ankle[:,1],expected_boot_gap_px=expected,
                 largest_component_area_halfres=area,
                 xy_m=source['xy_m'][rows]*(1.-weight[:,None])+filtered_world*weight[:,None])
    for key,value in updates.items():result[key][rows]=value
    result['raw_largest_component_uv']=baseline['raw_uv'].copy()
    result['raw_largest_component_world_m']=baseline['raw_world_m'].copy()
    result['raw_largest_component_uv'][rows]=largest_uv
    result['raw_largest_component_world_m'][rows]=max_world
    result['component_recovered_from_ankle_roi']=np.zeros(n,bool)
    result['component_recovered_from_ankle_roi'][rows]=recovered
    result['selected_component_label']=np.full(n,-1,np.int32)
    result['selected_component_area_halfres']=np.zeros(n,np.int32)
    result['selected_component_label'][rows]=selected_label
    result['selected_component_area_halfres'][rows]=selected_area
    result['source_mapping_inferred']=~source['direct_mapping']
    old_spin=baseline['spin_id']>0
    outside=np.ones(n,bool);outside[rows]=False
    preserved=[]
    def same(a,b):
        return np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b)
    for key,value in baseline.items():
        if value.ndim and len(value)==n:
            if not same(result[key][old_spin],value[old_spin]):
                raise AssertionError('Existing end-spin data changed: '+key)
            if not same(result[key][outside],value[outside]):
                raise AssertionError('Unselected programme data changed: '+key)
            preserved.append(key)
        elif not same(result[key],value):raise AssertionError('Baseline scalar metadata changed: '+key)
    if not np.isfinite(result['xy_m']).all():raise ValueError('Full demo trajectory must remain finite')
    output_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output_npz,**result)
    bad=np.zeros(n,bool);bad[rows]=~valid
    per_spin={str(sid):dict(frames=int(np.sum(result['spin_id']==sid)),
                  valid=int(np.sum(result['candidate_valid']&(result['spin_id']==sid))),
                  inferred=int(np.sum(result['inferred']&(result['spin_id']==sid)))) for sid in (1,2,3)}
    report=dict(schema='rinkspace.demo-full-contact-extension.v1',output=ref(output_npz),
                source_packet=ref(source_packet_path),baseline_contact=ref(baseline_contact_path),
                original_body=ref(body_path),source_clock='official native50 PTS1/12800',native_frames=n,
                new_spin_id=3,new_spin_window_seconds=[begin,end],new_spin_first_pts=int(pts[rows[0]]),
                new_spin_last_pts=int(pts[rows[-1]]),new_spin_frames=m,new_spin_valid_frames=int(valid.sum()),
                new_spin_inferred_frames=int((~valid).sum()),component_recovered_frames=int(recovered.sum()),
                candidate_source_mapping_inferred_frames=int(np.sum(result['source_mapping_inferred'][rows])),
                window_visual_review=FRONT_WINDOW_EVIDENCE if window_evidence is None else window_evidence,
                spin_id_source_times={'1':[156.4,165.9],'2':[171.92,184.9],'3':[begin,end]},per_spin=per_spin,
                preserved_existing_end_spin_fields=preserved,existing_end_spins_all_fields_bitexact=True,
                all_frames_outside_new_window_all_baseline_fields_bitexact=True,
                scale_adaptation=dict(expected_boot_gap_global_px=global_scale,
                     expected_boot_gap_range_px=[float(expected.min()),float(expected.max())],
                     low_gap_fraction=.35,high_gap_fraction=2.1,old_fixed30px_floor_used=False),
                raw_largest_component_preserved=True,roi_component_recovery_is_mask_heuristic=True,
                missing_boot_policy='Invalid raw candidate retained; same-spin reliable UV interpolation or longer-gap source-ankle-offset inference, always flagged',
                grounded_only=True,airborne_and_uncertain_landing_not_replaced=True,kneeling_choreography_not_replaced=True,
                metric_scale_conditional=True,solver_modified=False,original_source_or_body_modified=False,
                anomaly_intervals=_intervals(times,bad),
                anomaly_reason_bits={str(k):v for k,v in REASONS.items()},
                implementation=ref(__file__))
    output_json.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    return report


def render_front_contact_review(output_dir=FULL_OUTPUT):
    """Render only the reviewed11.1s front spin with auditable blade markers."""
    import subprocess
    from calibration.program_tracking.render_demo_last_shot import decode,WH,FPS
    from calibration.program_tracking.render_result import _ffmpeg_executable,_available_encoder
    from calibration.program_tracking.demo_particle_pass import _encoder_quality
    output_dir=path(output_dir)
    packet_file=output_dir/'contact_packet.npz'
    manifest=read(output_dir/'contact_packet.json')
    source_packet=path(manifest['source_packet']['path'])
    with np.load(packet_file,allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
    with np.load(source_packet,allow_pickle=False) as z:source=str(z['source_path'])
    rows=np.flatnonzero(p['spin_id']==3)
    first,last=int(p['pts'][rows[0]]),int(p['pts'][rows[-1]])
    lookup={int(stamp):int(i) for stamp,i in zip(p['pts'][rows],rows)}
    video=check_output_path(output_dir/'front_spin_contact_review.mp4')
    log_path=check_output_path(output_dir/'front_spin_contact_review.stderr.log')
    for value in (82.,87.06,90.,92.60):check_output_path(output_dir/f'front_contact_{round(value*1000):06d}.jpg')
    ffmpeg=_ffmpeg_executable();args,name=_available_encoder(ffmpeg)
    args=_encoder_quality(args,name)
    duration=len(rows)/FPS
    command=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','bgr24',
             '-video_size','1920x1080','-framerate','50','-i','pipe:0','-ss',str(first/12800.),'-i',source,
             '-map','0:v:0','-map','1:a:0?',*args,'-pix_fmt','yuv420p','-c:a','aac','-b:a','192k',
             '-t',str(duration),'-video_track_timescale','12800','-movflags','+faststart',str(video)]
    capture={round(value*50)*256 for value in (82.,87.06,90.,92.60)}
    overlays=[];count=0
    with log_path.open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log,creationflags=0x08000000)
        try:
            for stamp,native in decode(source,first,last):
                i=lookup[int(stamp)]
                image=native.copy()
                contact=p['contact_uv'][i]
                inferred=bool(p['inferred'][i])
                recovered=bool(p['component_recovered_from_ankle_roi'][i])
                colour=(0,180,255) if inferred else (255,240,40)
                center=tuple(np.rint(contact).astype(int))
                cv2.circle(image,center,8,(0,0,0),3,cv2.LINE_AA)
                cv2.circle(image,center,6,colour,2,cv2.LINE_AA)
                cv2.circle(image,center,2,colour,-1,cv2.LINE_AA)
                if inferred or recovered:
                    raw=p['raw_largest_component_uv'][i]
                    if np.isfinite(raw).all():
                        point=tuple(np.rint(raw).astype(int))
                        cv2.drawMarker(image,point,(60,60,235),cv2.MARKER_TILTED_CROSS,10,2,cv2.LINE_AA)
                label=('inferred' if inferred else 'ROI recovered mask' if recovered else 'mask candidate')
                cv2.putText(image,f'source {stamp/12800.:.2f}s | {label}',(34,1050),
                            cv2.FONT_HERSHEY_SIMPLEX,.68,(10,10,10),4,cv2.LINE_AA)
                cv2.putText(image,f'source {stamp/12800.:.2f}s | {label}',(34,1050),
                            cv2.FONT_HERSHEY_SIMPLEX,.68,(245,245,245),1,cv2.LINE_AA)
                if int(stamp) in capture:
                    file=output_dir/f'front_contact_{round(stamp/12800.*1000):06d}.jpg'
                    if not cv2.imwrite(str(file),image,[cv2.IMWRITE_JPEG_QUALITY,96]):raise IOError(file)
                    overlays.append(str(file))
                process.stdin.write(image.tobytes());count+=1
        finally:
            process.stdin.close();return_code=process.wait()
        if return_code:raise RuntimeError(log_path.read_text(errors='replace')[-3000:])
    if count!=len(rows):raise ValueError('Contact review source count changed')
    report=dict(video=ref(video),frames=count,fps=FPS,duration_seconds=duration,
                official_first_pts=first,official_last_pts=last,time_base='1/12800',
                source_seconds=[first/12800.,last/12800.],overlays=overlays,
                markers={'cyan':'mask candidate, including explicitly recovered connected component',
                         'orange':'inferred replacement','red_cross':'original largest-component candidate'},
                contact_packet=ref(packet_file),encoder=name,
                diagnostic_only=True,full_programme_rendered=False)
    (output_dir/'front_spin_contact_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--windows', default=json.dumps(DEFAULT_WINDOWS), help='JSON list of grounded official-source-second intervals')
    parser.add_argument('--full', action='store_true', help='Append reviewed front spin to immutable end-spin baseline')
    parser.add_argument('--source-packet', type=Path, default=DEFAULT_OUTPUT/'source_packet.npz')
    parser.add_argument('--baseline-contact', type=Path, default=DEFAULT_OUTPUT/'contact_packet.npz')
    parser.add_argument('--front-window', default=json.dumps(FRONT_WINDOW))
    parser.add_argument('--review', action='store_true', help='Only with--full: render the front-spin blade-marker review')
    args = parser.parse_args()
    if args.full:
        target=args.output or FULL_OUTPUT
        report=build_full_packet(target,source_packet_path=args.source_packet,
                                 baseline_contact_path=args.baseline_contact,
                                 front_window=json.loads(args.front_window))
        summary={k:report[k] for k in ('output','per_spin','new_spin_valid_frames','new_spin_inferred_frames',
                                       'component_recovered_frames','existing_end_spins_all_fields_bitexact')}
        if args.review:summary['review']=render_front_contact_review(target)
    else:
        if args.review:parser.error('--review requires--full')
        report=build_packet(args.output or DEFAULT_OUTPUT,json.loads(args.windows))
        summary={'per_spin':report['per_spin'],'output':report['output'],'inferred_frames':report['inferred_frames']}
    print(json.dumps(summary,ensure_ascii=False,indent=2))
