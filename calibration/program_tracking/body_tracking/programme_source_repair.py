"""Repair absent programme body rows with exact-source RTMW observations.

Preview is read-only with respect to the prepared input and saved candidate.
ROI interpolation and model COCO projections locate crops only; neither is a
measurement. Inference and preparation are separate, explicit CLI actions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .rebuild_prepare import digest, load
from .programme_contract import required_clock


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf8')


def _json_array(value):
    a = np.asarray(value, float)
    return np.where(np.isfinite(a), a, None).tolist()


def _output(path, data_root):
    path, data_root = Path(path).resolve(), Path(data_root).resolve()
    if path == data_root or not path.is_relative_to(data_root):
        raise ValueError('New output must be strictly inside the Data root')
    if path.exists():
        raise FileExistsError('Preserve existing output: ' + str(path))
    return path


def _box_valid(boxes):
    boxes = np.asarray(boxes, float)
    return np.isfinite(boxes).all(-1) & (boxes[..., 2:] > boxes[..., :2]).all(-1)


def plan_rois(packet, observations, coco_uv, source_size=(1920, 1080)):
    """Conservative navigation only, bounded by each contiguous true shot.

    Inferred/interpolated boxes include the nearest qualified real bbox on
    either side in the same segment, including one-sided cut/tail anchors.
    Expand each side by 30% (20% for a same-frame real bbox), then clip to the
    source image. No upper crop-size limit can truncate the athlete again.
    """
    pts, shots = packet['pts'], packet['shot_ids']
    n = len(pts)
    if observations['use'].shape != (n, 17) or coco_uv.shape != (n, 17, 2):
        raise ValueError('Expected matching COCO17 arrays')
    boxes = np.asarray(packet['source_bbox'], float)
    valid = _box_valid(boxes)
    suspect = np.asarray(packet['source_identity_suspect'], bool)
    real = valid & ~suspect
    qualified = real & (packet['source_qualified'].any(1) | observations['use'].any(1))
    seconds_per_tick = int(packet['time_base'][0]) / int(packet['time_base'][1])
    limit = np.asarray([0, 0, *source_size], float)
    cuts = np.r_[0, np.flatnonzero(shots[1:] != shots[:-1]) + 1, n]
    rows = []
    for start, stop in zip(cuts[:-1], cuts[1:]):
        known = np.flatnonzero(qualified[start:stop]) + start
        missing = np.flatnonzero(~observations['use'][start:stop].any(1)) + start
        for i in missing:
            anchors, alpha = [], None
            model = np.asarray(coco_uv[i], float)
            good = np.isfinite(model).all(1)
            model_box = (np.r_[model[good].min(0), model[good].max(0)]
                         if good.sum() >= 2 else np.full(4, np.nan))
            left, right = known[known < i], known[known > i]
            neighbours = ([int(left[-1])] if len(left) else []) + ([int(right[0])] if len(right) else [])
            if real[i]:
                base, kind, anchors = boxes[i].copy(), 'same_frame_source_bbox', [int(i)]
            elif len(left) and len(right):
                a, b = int(left[-1]), int(right[0])
                alpha = float((int(pts[i]) - int(pts[a])) / (int(pts[b]) - int(pts[a])))
                base = (1 - alpha) * boxes[a] + alpha * boxes[b]
                kind, anchors = 'same_shot_bbox_interpolation', [a, b]
            elif _box_valid(model_box):
                base, kind = model_box, 'same_frame_COCO_uv_inferred_navigation'
            elif neighbours:
                base = boxes[neighbours[0]].copy()
                kind = 'same_shot_source_bbox_union_navigation'
            else:
                base, kind = np.full(4, np.nan), 'unresolved_no_navigation_roi'
            inferred = kind != 'same_frame_source_bbox'
            if inferred:
                anchors = sorted(set(anchors + neighbours))
            components = ([base] if _box_valid(base) else []) + [boxes[a] for a in anchors]
            if components:
                components = np.asarray(components)
                union = np.r_[components[:, :2].min(0), components[:, 2:].max(0)]
                margin = .30 if inferred else .20
                extent = union[2:] - union[:2]
                expanded = union + np.r_[-extent * margin, extent * margin]
                roi = np.r_[np.maximum(expanded[:2], limit[:2]), np.minimum(expanded[2:], limit[2:])]
            else:
                union = expanded = roi = np.full(4, np.nan)
                margin = .30 if inferred else .20
            rows.append(dict(programme_row=int(i), source_pts=int(pts[i]),
                shot_id=str(shots[i]), shot_segment_rows=[int(start), int(stop)],
                source_seconds=float(int(pts[i]) * seconds_per_tick),
                roi_native_xyxy=_json_array(roi), roi_source=kind, roi_inferred=inferred,
                roi_policy='same-shot-qualified-neighbour-union; per-side margin; native-image clip',
                roi_base_native_xyxy=_json_array(base), roi_union_native_xyxy=_json_array(union),
                roi_expanded_before_clip_native_xyxy=_json_array(expanded),
                roi_margin_fraction_per_side=margin,
                roi_source_image_clip=not np.array_equal(roi, expanded, equal_nan=True),
                roi_has_no_upper_size_limit=True,
                anchor_programme_rows=anchors,
                anchor_source_pts=[int(pts[a]) for a in anchors],
                anchor_boxes_native_xyxy=[boxes[a].tolist() for a in anchors],
                anchor_time_delta_seconds=[float((int(pts[a])-int(pts[i]))*seconds_per_tick) for a in anchors],
                anchor_source=[dict(programme_row=a, source_pts=int(pts[a]),
                    side='same_frame' if a == i else 'preceding' if a < i else 'following',
                    delta_pts=int(pts[a])-int(pts[i]),
                    delta_seconds=float((int(pts[a])-int(pts[i]))*seconds_per_tick),
                    source_bbox=boxes[a].tolist(), source_identity_suspect=bool(suspect[a]),
                    source_qualified_points=int(packet['source_qualified'][a].sum()),
                    observation_use_points=int(observations['use'][a].sum())) for a in anchors],
                interpolation_alpha=alpha,
                original_source_bbox=_json_array(boxes[i]),
                original_source_identity_suspect=bool(suspect[i]),
                original_lower_body_inferred=bool(packet['lower_body_inferred'][i]),
                original_observation_use=observations['use'][i].tolist(),
                original_source_qualified=packet['source_qualified'][i].tolist(),
                candidate_COCO_bbox_navigation_only=_json_array(model_box),
                source_warning_preserved=True, human_reviewed=False,
                left_right_identity='network_labels_not_human_confirmed',
                contact_status='unknown_not_inferred_from_network_feet'))
    return rows


def source_frames(video, wanted, time_base, size):
    """Decode actual integer PTS; never replace a missing exposure by a neighbour."""
    import av
    wanted = list(map(int, wanted))
    if not wanted or wanted != sorted(set(wanted)):
        raise ValueError('Require sorted unique real source PTS')
    index = 0
    tb = Fraction(*map(int, time_base))
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        if stream.time_base != tb or (stream.width, stream.height) != tuple(size):
            raise ValueError('Source native dimensions/time base differ')
        preroll = int(Fraction(2) / tb)
        container.seek(max(0, wanted[0] - preroll), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None or frame.pts < wanted[index]:
                continue
            if frame.pts != wanted[index]:
                raise ValueError(f'Missing requested PTS {wanted[index]}; next {frame.pts}')
            yield int(frame.pts), frame.to_ndarray(format='bgr24')
            index += 1
            if index == len(wanted):
                return
    raise ValueError(f'Source ended before PTS {wanted[index]}')


def _tile(image, row):
    import cv2
    tile = np.full((300, 400, 3), 242, np.uint8)
    roi = np.asarray(row['roi_native_xyxy'], float)
    label = row['roi_source'].replace('same_frame_', '').replace('same_shot_', '')
    labels = [f"row {row['programme_row']}  {row['shot_id']}  {row['source_seconds']:.3f}s",
              label, f"old identity warning: {row['original_source_identity_suspect']}",
              f"PTS {row['source_pts']}  ROI navigation only"]
    for j, line in enumerate(labels):
        cv2.putText(tile, line, (8, 20 + 19*j), cv2.FONT_HERSHEY_SIMPLEX,
                    .43, (25, 25, 25), 1, cv2.LINE_AA)
    if not _box_valid(roi):
        return tile
    center, wh = (roi[:2] + roi[2:]) / 2, roi[2:] - roi[:2]
    lo = np.maximum(np.floor(center - wh * .85), [0, 0]).astype(int)
    hi = np.minimum(np.ceil(center + wh * .85), image.shape[1::-1]).astype(int)
    if np.any(hi <= lo):
        return tile
    crop = image[lo[1]:hi[1], lo[0]:hi[0]].copy()
    color = (40, 145, 30) if not row['roi_inferred'] else (0, 115, 225)
    cv2.rectangle(crop, tuple(np.rint(roi[:2] - lo).astype(int)),
                  tuple(np.rint(roi[2:] - lo).astype(int)), color, 2)
    factor = min(384 / crop.shape[1], 204 / crop.shape[0])
    thumb = cv2.resize(crop, None, fx=factor, fy=factor,
                       interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
    y, x = 90 + (204 - thumb.shape[0]) // 2, (400 - thumb.shape[1]) // 2
    tile[y:y+thumb.shape[0], x:x+thumb.shape[1]] = thumb
    return tile


def preview(prepared, candidate, output, data_root, parent_preview=None):
    import cv2
    prepared, candidate = Path(prepared).resolve(), Path(candidate).resolve()
    out = _output(output, data_root)
    p, obs = load(prepared/'input.npz'), load(prepared/'observations.npz')
    meta = json.loads((prepared/'preparation.json').read_text(encoding='utf8'))
    scope = json.loads(Path(meta['scope']).read_text(encoding='utf8'))
    expected, shots, tb = required_clock(scope)
    for a in (p['pts'], obs['pts']):
        np.testing.assert_array_equal(a, expected)
    np.testing.assert_array_equal(p['shot_ids'], shots)
    np.testing.assert_array_equal(p['time_base'], tb)
    with np.load(candidate, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['pts'], expected)
        coco = z['COCO_uv'].copy()
    video = Path(meta['source_video'])
    if digest(video) != meta['source_sha256']:
        raise ValueError('Registered source video hash differs')
    original = {f.name:digest(f) for f in prepared.iterdir() if f.is_file()}
    rows = plan_rois(p, obs, coco, source_size=(scope['source']['width'], scope['source']['height']))
    out.mkdir(parents=True)
    cv2.setNumThreads(2)
    tiles = []
    for row, (stamp, image) in zip(rows, source_frames(video,
            [r['source_pts'] for r in rows], tb, (scope['source']['width'], scope['source']['height']))):
        if stamp != row['source_pts']:
            raise ValueError('PTS order changed')
        row['source_frame_bgr_sha256'] = hashlib.sha256(image.tobytes()).hexdigest()
        tiles.append(_tile(image, row))
    sheets = []
    # Eight pages cover all absent rows, including the full cut boundaries.
    groups = np.array_split(np.arange(len(tiles)), min(8, len(tiles)))
    for page, group in enumerate(groups, 1):
        sheet = np.full((int(np.ceil(len(group)/4))*300, 1600, 3), 242, np.uint8)
        for j, index in enumerate(group):
            y, x = (j//4)*300, (j%4)*400
            sheet[y:y+300, x:x+400] = tiles[int(index)]
            rows[int(index)]['contact_sheet'] = f'roi_contact_{page:02d}.jpg'
            rows[int(index)]['contact_sheet_tile'] = j
        target = out/f'roi_contact_{page:02d}.jpg'
        if not cv2.imwrite(str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError('Could not write contact sheet')
        sheets.append(str(target))
    document = dict(schema='rinkspace.programme-source-repair-preview.v2',
        roi_policy=dict(inferred_union_nearest_qualified_same_shot_bboxes=True, one_sided_navigation_anchors=True, inferred_margin_per_side=.30, real_same_frame_margin_per_side=.20, clip_to_native_source_image=True, upper_size_limit=False),
        parent_preview=None if parent_preview is None else dict(path=str(Path(parent_preview).resolve()),sha256=digest(parent_preview)),
        data_root=str(Path(data_root).resolve()), prepared=str(prepared), candidate=str(candidate),
        candidate_sha256=digest(candidate), source_files_sha256=original,
        source_video=str(video), source_sha256=meta['source_sha256'], time_base=list(tb),
        source_size=[scope['source']['width'], scope['source']['height']],
        frames=len(expected), absent_body_rows=len(rows),
        original_source_identity_warning_count=int(p['source_identity_suspect'].sum()),
        roi_source_counts=dict(Counter(r['roi_source'] for r in rows)),
        contact_sheets=sheets, rows=rows,
        network_executed=False, camera_changed=False, observations_changed=False,
        original_warning_and_missing_lineage_preserved=True,
        left_right_identity='network labels remain unreviewed',
        foot_semantics='anatomical network landmarks; never blade contacts')
    if {f.name:digest(f) for f in prepared.iterdir() if f.is_file()} != original:
        raise ValueError('Original prepared changed during preview')
    write_json(out/'preview.json', document)
    with (out/'missing_rows.csv').open('w', encoding='utf-8-sig', newline='') as f:
        import csv
        fields = ['programme_row','source_pts','source_seconds','shot_id','roi_source',
                  'original_source_identity_suspect','original_lower_body_inferred','roi_native_xyxy']
        writer = csv.DictWriter(f, fields, extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(event='source_repair_preview', path=str(out),
        absent_body_rows=len(rows), roi_source_counts=document['roi_source_counts'],
        contact_sheets=sheets, network_executed=False)), flush=True)
    return document


def _checked_preview(path):
    path = Path(path).resolve()
    d = json.loads(path.read_text(encoding='utf8'))
    root = Path(d['prepared'])
    for name, sha in d['source_files_sha256'].items():
        if digest(root/name) != sha:
            raise ValueError('Parent prepared changed: ' + name)
    if digest(d['candidate']) != d['candidate_sha256']:
        raise ValueError('Navigation candidate changed')
    return d


def infer(preview_path, output, model, batch_size=8):
    """Explicit second stage: the existing RTMW-x crop/decode path, nothing else."""
    d = _checked_preview(preview_path)
    out = _output(output, d['data_root'])
    model = Path(model).resolve()
    if not model.is_file():
        raise FileNotFoundError('Existing local RTMW checkpoint required: ' + str(model))
    if digest(d['source_video']) != d['source_sha256']:
        raise ValueError('Source video changed')
    rows = d['rows']
    if not rows or any(not _box_valid(np.asarray(r['roi_native_xyxy'], float)) for r in rows):
        raise ValueError('Resolve absent/invalid ROI before inference')
    if batch_size < 1:
        raise ValueError('Positive batch size required')
    local_runtime=Path(d['data_root'])/'runtime_tools/body'
    sys.path[:0]=[str(local_runtime/'exmr_packages'),str(local_runtime/'ExMR')]
    # Reuse the same RTMW preprocessing/decoding as exmr_prior.keypoints. Do
    # not call exmr_prior.runtime, which loads unrelated model repositories.
    from exmr.preprocessing.keypoints import rtmw, VISIBILITY_THRESHOLD
    import cv2
    import onnxruntime as ort
    cv2.setNumThreads(2)
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(model),sess_options=options,
        providers=[('CUDAExecutionProvider',{'device_id':0}),'CPUExecutionProvider'])
    if session.get_providers()[0] != 'CUDAExecutionProvider':
        raise RuntimeError('Existing RTMW CUDA provider unavailable')
    inp=session.get_inputs()[0];names=[x.name for x in session.get_outputs()]
    if inp.shape[-2:] not in [[384,288],['height','width']]:
        raise ValueError('Selected checkpoint is not the existing 384x288 RTMW')
    out.mkdir(parents=True)
    raw=np.empty((len(rows),133,3),np.float32)
    input_centers=np.empty((len(rows),2),np.float64)
    input_scales=np.empty((len(rows),2),np.float64)
    stream=source_frames(d['source_video'],[r['source_pts'] for r in rows],d['time_base'],d['source_size'])
    start=time.perf_counter()
    with (out/'progress.jsonl').open('w',encoding='utf8',buffering=1) as log:
        for first in range(0,len(rows),batch_size):
            tensors=[];centers=[];scales=[]
            count=min(batch_size,len(rows)-first)
            for row in rows[first:first+count]:
                stamp,image=next(stream)
                if stamp!=row['source_pts'] or hashlib.sha256(image.tobytes()).hexdigest()!=row['source_frame_bgr_sha256']:
                    raise ValueError('Source image differs from preview')
                center,scale=rtmw._bbox_xyxy2cs(np.asarray(row['roi_native_xyxy']),rtmw.BOX_PADDING)
                crop,scale=rtmw._top_down_affine(rtmw.INPUT_SIZE,scale,center,image)
                normalized=(crop-np.array(rtmw.MEAN))/np.array(rtmw.STD)
                tensors.append(np.ascontiguousarray(normalized.transpose(2,0,1),dtype=np.float32))
                centers.append(center);scales.append(scale)
            images=np.pad(np.stack(tensors),[(0,batch_size-count),(0,0),(0,0),(0,0)],mode='edge')
            sx,sy=session.run(names,{inp.name:images})
            xy,score=rtmw._decode(sx[:count],sy[:count],np.asarray(centers),np.asarray(scales))
            if xy.shape!=(count,133,2):raise ValueError('Expected native 133-point RTMW output')
            raw[first:first+count]=np.concatenate([xy,score[...,None]],axis=-1)
            input_centers[first:first+count]=np.asarray(centers)
            input_scales[first:first+count]=np.asarray(scales)
            event=dict(event='rtmw_source_2d',completed=first+count,total=len(rows),seconds=time.perf_counter()-start)
            log.write(json.dumps(event)+'\n');print(json.dumps(event),flush=True)
    input_regions=np.c_[input_centers-input_scales/2,input_centers+input_scales/2]
    clipped_regions=np.c_[np.maximum(input_regions[:,:2],0),np.minimum(input_regions[:,2:],d['source_size'])]
    source_fraction=np.prod(np.maximum(clipped_regions[:,2:]-clipped_regions[:,:2],0),axis=1)/np.prod(input_scales,axis=1)
    write_json(out/'input_regions.json',dict(schema='rinkspace.rtmw-actual-input-regions.v1',
        pixel_space='native source image',BOX_PADDING=float(rtmw.BOX_PADDING),input_size=list(rtmw.INPUT_SIZE),
        semantics='actual post-aspect top_down_affine center/scale; out-of-image area uses OpenCV constant border',
        rows=[dict(programme_row=row['programme_row'],source_pts=row['source_pts'],source_seconds=row['source_seconds'],
            navigation_roi_native_xyxy=row['roi_native_xyxy'],center=input_centers[i].tolist(),scale=input_scales[i].tolist(),
            actual_input_region_native_xyxy=input_regions[i].tolist(),native_image_intersection_xyxy=clipped_regions[i].tolist(),
            native_image_area_fraction=float(source_fraction[i]))for i,row in enumerate(rows)]))
    np.savez_compressed(out/'network_2d.npz',pts=np.asarray([r['source_pts'] for r in rows],np.int64),
        programme_rows=np.asarray([r['programme_row'] for r in rows],np.int64),
        source_time_base=np.asarray(d['time_base']),raw_wholebody133=raw,
        rtmw_input_centers=input_centers,rtmw_input_scales=input_scales,
        rtmw_actual_input_region_native_xyxy=input_regions,
        rtmw_native_image_intersection_xyxy=clipped_regions,rtmw_native_image_area_fraction=source_fraction,
        source_bbox_model=np.asarray([r['roi_native_xyxy'] for r in rows]),
        source_bbox_inferred=np.asarray([r['roi_inferred'] for r in rows]),
        human_reviewed=np.zeros(len(rows),bool),left_right_identity_confirmed=np.zeros(len(rows),bool),
        contact_allowed=np.zeros(len(rows),bool))
    report=dict(schema='rinkspace.programme-source-repair-inference.v1',preview=str(Path(preview_path).resolve()),
        preview_sha256=digest(preview_path),model=str(model),model_sha256=digest(model),
        rtmw_code=str(Path(rtmw.__file__).resolve()),rtmw_code_sha256=digest(rtmw.__file__),
        backend='existing RTMW-x 384x288 fp16 ONNX; exmr_prior crop/decode interface',
        runner_code_sha256=digest(__file__),actual_input_regions_sha256=digest(out/'input_regions.json'),
        actual_input_region_semantics='post-aspect top_down_affine scale; navigation padding is not source visibility',
        onnxruntime=ort.__version__,onnxruntime_file=str(Path(ort.__file__).resolve()),providers=session.get_providers(),
        visibility_raw_score_threshold=float(VISIBILITY_THRESHOLD['rtmw']),score_semantics='raw SimCC, not probability',
        output_sha256=digest(out/'network_2d.npz'),frames=len(rows),model_executed=True,
        GVHMR_ExMR_SAM_HaMeR_executed=False,training=False,download=False,foot_contact=False)
    write_json(out/'inference.json',report)
    return report


def prepare(preview_path, inference_dir, output, sigma_px):
    """Version only absent observation rows; preserve the nominal initial priors."""
    from .body_prior import original_observation_mask
    d=_checked_preview(preview_path);inference_dir=Path(inference_dir).resolve()
    out=_output(output,d['data_root']);source=Path(d['prepared'])
    info=json.loads((inference_dir/'inference.json').read_text(encoding='utf8'))
    if info['preview_sha256']!=digest(preview_path) or digest(inference_dir/'network_2d.npz')!=info['output_sha256']:
        raise ValueError('Inference is bound to another preview/cache')
    if not np.isfinite(sigma_px) or sigma_px<=0:
        raise ValueError('Supply an explicit positive engineering sigma in source pixels')
    net=load(inference_dir/'network_2d.npz');p=load(source/'input.npz');obs=load(source/'observations.npz')
    flags=load(source/'export_source.npz');ix=net['programme_rows'];n=len(p['pts'])
    np.testing.assert_array_equal(p['pts'][ix],net['pts'])
    if obs['use'][ix].any():raise ValueError('Refuse replacement of existing valid body observations')
    raw=net['raw_wholebody133'][:,:17].copy()
    # An expanded/clipped navigation ROI is not a measured person box and
    # cannot establish actual body truncation. Preserve the existing real-box
    # gate, finite source pixels and raw RTMW threshold, joint by joint.
    use,truncated=original_observation_mask(dict(coco17=raw,bbox_native_xyxy=p['source_bbox'][ix]),tuple(d['source_size']))
    use &= raw[:,:,2]>info['visibility_raw_score_threshold']
    missing_before=~obs['use'].any(1);old_obs={k:v.copy()for k,v in obs.items()}
    obs['uv'][ix]=raw[:,:,:2];obs['use'][ix]=use;obs['sigma_px'][ix]=float(sigma_px)
    p['source_uv'][ix]=raw[:,:,:2];p['source_qualified'][ix]=use
    # Retain original source_score: its historical probability/validity scale
    # cannot be replaced by raw SimCC. Keep the actual new scores separately.
    score=np.full((n,17),np.nan,np.float32);score[ix]=raw[:,:,2]
    added=np.zeros(n,bool);added[ix]=True
    p.update(source_repair_present=added,source_repair_raw_score=score,
        source_repair_score_semantics=np.array('RTMW raw SimCC; source_score retained from parent'),
        source_score_semantics=np.array('parent source_score unchanged; source_repair_present rows use source_repair_raw_score for RTMW raw SimCC'),
        source_repair_source_bbox_navigation_only=np.array(True))
    flags.update(source_observation_use=obs['use'],source_repair_present=added,
        source_repair_original_missing=missing_before,source_repair_left_right_identity_confirmed=np.zeros(n,bool),
        source_repair_contact_allowed=np.zeros(n,bool))
    np.testing.assert_array_equal(p['source_identity_suspect'],load(source/'input.npz')['source_identity_suspect'])
    for key in old_obs:np.testing.assert_array_equal(obs[key][~missing_before],old_obs[key][~missing_before])
    out.mkdir(parents=True);links=[]
    changed={'input.npz','observations.npz','export_source.npz','preparation.json'}
    for old in source.iterdir():
        if not old.is_file() or old.name in changed:continue
        os.link(old,out/old.name);links.append(old.name)
    for name,arrays in [('input.npz',p),('observations.npz',obs),('export_source.npz',flags)]:
        np.savez_compressed(out/name,**arrays)
    parent=out/'source_repair_parent';parent.mkdir()
    for name in changed:
        if (source/name).is_file():os.link(source/name,parent/name)
    meta=json.loads((source/'preparation.json').read_text(encoding='utf8'))
    meta.update(no_new_network=False, source_repair=dict(preview=str(Path(preview_path).resolve()),preview_sha256=digest(preview_path),
        inference=str(inference_dir),inference_sha256=digest(inference_dir/'inference.json'),
        network_rows=len(ix),new_rows_with_usable_points=int(use.any(1).sum()),new_usable_body_points=int(use.sum()),
        sigma_px=float(sigma_px),sigma_semantics='explicit uncalibrated engineering scale, not model confidence',
        truncation_gate_bbox='unchanged original source_bbox; expanded navigation ROI is never treated as measured source truncation',
        missing_before_pts=p['pts'][missing_before].tolist(),missing_after_pts=p['pts'][~obs['use'].any(1)].tolist(),
        original_identity_warnings_preserved=int(p['source_identity_suspect'].sum()),
        hardlinked_unchanged_files=links,original_source_score_preserved=True,original_source_bbox_preserved=True,
        initial_numerical_state='parent initial.npz hardlinked unchanged; main session rebuilds a separate v6 warm checkpoint',
        left_right_identity='network labels unconfirmed',contact='unknown; anatomical foot points are not contact truth'))
    write_json(out/'preparation.json',meta)
    write_json(out/'source_repair.json',dict(schema='rinkspace.programme-source-repair-prepared.v1',
        source_files_sha256=d['source_files_sha256'],parent=str(source),output=str(out),frames=n,
        initial_required=False,**meta['source_repair']))
    _checked_preview(preview_path)
    print(json.dumps(dict(event='source_repair_prepared',output=str(out),frames=n,
        replaced_absent_rows=len(ix),remaining_missing=int((~obs['use'].any(1)).sum()),initial_required=False)),flush=True)
    return out


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='action',required=True)
    p=subs.add_parser('preview')
    for name in ('prepared','candidate','output','data-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--parent-preview',type=Path)
    p=subs.add_parser('infer')
    for name in ('preview','output','model'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=8)
    p=subs.add_parser('prepare')
    for name in ('preview','inference','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sigma-px',type=float,required=True)
    a=parser.parse_args(argv)
    if a.action=='preview':preview(a.prepared,a.candidate,a.output,a.data_root,a.parent_preview)
    elif a.action=='infer':infer(a.preview,a.output,a.model,a.batch_size)
    else:prepare(a.preview,a.inference,a.output,a.sigma_px)


if __name__=='__main__':
    main()
