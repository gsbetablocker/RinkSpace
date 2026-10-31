"""Image-only registration graph for a complete, uncropped shot.

Edges contain measured forward/backward optical-flow correspondences. Surface
maps navigate searches; they never manufacture observations or entity labels.
Both past and future clear frames are retained as reference candidates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import time

import cv2
import numpy as np

from rinkspace.geometry import fit_homography, project


@dataclass(frozen=True)
class RegistrationOptions:
    scale: float = .5
    keyframe_window: int = 25
    max_features: int = 1600
    forward_backward_px: float = 1.5
    fit_threshold_px: float = 2.5
    check_median_px: float = 2.
    check_p90_px: float = 6.
    minimum_inliers: int = 16


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', 'utf-8')
    pending.replace(path)


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def frame_features(image, surface_masks, options):
    """Cache source photometry and spatially distributed corner candidates."""
    scale = options.scale
    size = (round(image.shape[1]*scale), round(image.shape[0]*scale))
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), size,
                      interpolation=cv2.INTER_AREA)
    # A fixed local contrast transform exposes pale paint without emitting UV.
    enhanced = cv2.createCLAHE(clipLimit=2., tileGridSize=(12, 8)).apply(gray)
    result = dict(gray=gray, enhanced=enhanced)
    for name, mask in surface_masks.items():
        small = cv2.resize(np.asarray(mask, np.uint8)*255, size,
                           interpolation=cv2.INTER_NEAREST)
        small = cv2.erode(small, np.ones((5, 5), np.uint8))
        corners = cv2.goodFeaturesToTrack(enhanced, options.max_features,
            .002, 7, mask=small, blockSize=5)
        xy = np.zeros((0, 2), np.float32) if corners is None else corners[:, 0]
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        valid = small > 0
        sharpness = float(np.mean(lap[valid]**2)) if valid.any() else 0.
        result[name+'_mask'] = small
        result[name+'_corners'] = xy
        result[name+'_quality'] = np.array([sharpness, len(xy), valid.mean()])
    return result


def track_pair(reference, current, surface, options, initial_h=None):
    """Return actual two-image coordinates, in full source pixel units."""
    a_gray, b_gray = reference['enhanced'], current['enhanced']
    x = reference[surface+'_corners'].reshape(-1, 1, 2)
    empty = np.zeros((0, 2), np.float32)
    base = dict(status='unavailable', surface=surface, physical_identity=False,
                observations_created=0, source_pixels_per_work_pixel=1/options.scale)
    if len(x) < options.minimum_inliers:
        return base | dict(reason='too_few_reference_corners'), empty, empty, np.zeros(0, bool)
    kwargs = dict(winSize=(25, 25), maxLevel=4,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .005))
    initial = None
    if initial_h is not None:
        initial = np.ascontiguousarray(project(initial_h, x[:, 0]/options.scale)*options.scale,
                                       dtype=np.float32).reshape(-1, 1, 2)
    y, sy, _ = cv2.calcOpticalFlowPyrLK(a_gray, b_gray, x, initial,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW if initial is not None else 0, **kwargs)
    if y is None:
        return base | dict(reason='forward_flow_failed'), empty, empty, np.zeros(0, bool)
    back, sb, _ = cv2.calcOpticalFlowPyrLK(b_gray, a_gray, y, None, **kwargs)
    if back is None:
        return base | dict(reason='backward_flow_failed'), empty, empty, np.zeros(0, bool)
    a, b, c = x[:, 0], y[:, 0], back[:, 0]
    height, width = a_gray.shape
    finite = np.isfinite(b).all(1) & np.isfinite(c).all(1)
    ij = np.rint(np.nan_to_num(b, nan=-1., posinf=-1., neginf=-1.)).astype(int)
    inside = (ij[:, 0] >= 0) & (ij[:, 0] < width) & (ij[:, 1] >= 0) & (ij[:, 1] < height)
    ij[:, 0] = np.clip(ij[:, 0], 0, width-1)
    ij[:, 1] = np.clip(ij[:, 1], 0, height-1)
    fb = np.linalg.norm(a-c, axis=1)/options.scale
    valid = finite & inside & (sy[:, 0] > 0) & (sb[:, 0] > 0)
    valid &= fb <= options.forward_backward_px
    valid &= current[surface+'_mask'][ij[:, 1], ij[:, 0]] > 0
    a, b, fb = a[valid]/options.scale, b[valid]/options.scale, fb[valid]
    note = base | dict(matches=len(a), forward_backward_p95_px=float(np.percentile(fb, 95)) if len(fb) else None,
                       initialized_search=initial is not None, backward_search_independent_initialization=True)
    inliers = np.zeros(len(a), bool)
    if len(a) < options.minimum_inliers:
        return note | dict(reason='too_few_bidirectional_tracks'), a, b, inliers
    # Separate spatial cells test transfer beyond the cells used to estimate H.
    cells = (a[:, 0]//120).astype(int) + 17*(a[:, 1]//120).astype(int)
    unique = np.unique(cells)
    selected = np.random.default_rng(9182).permutation(unique)[::4]
    check = np.isin(cells, selected)
    if check.sum() < 4 or (~check).sum() < 12:
        return note | dict(reason='insufficient_spatial_support'), a, b, inliers
    try:
        h, train_inliers = fit_homography(a[~check], b[~check], options.fit_threshold_px)
        errors = np.linalg.norm(project(h, a)-b, axis=1)
        inliers = errors <= options.fit_threshold_px
        inverse_errors = np.linalg.norm(project(np.linalg.inv(h), b)-a, axis=1)
    except ValueError as error:
        return note | dict(reason=str(error)), a, b, inliers
    hull = cv2.convexHull(a[inliers].astype(np.float32)) if inliers.any() else None
    med = float(np.median(errors[check])); p90 = float(np.percentile(errors[check], 90))
    span = np.ptp(a[inliers], axis=0) if inliers.any() else np.zeros(2)
    usable = (inliers.sum() >= options.minimum_inliers and med <= options.check_median_px
              and p90 <= options.check_p90_px and span[0] >= 160 and span[1] >= 40)
    note.update(status='navigation' if usable else 'ambiguous',
        reason='source_transfer_supported' if usable else 'source_transfer_not_supported',
        H_current_from_reference=h.tolist(), inliers=int(inliers.sum()),
        check_count=int(check.sum()), check_median_px=med, check_p90_px=p90,
        inverse_inlier_p95_px=float(np.percentile(inverse_errors[inliers], 95)) if inliers.any() else None,
        support_span_px=span.tolist(),
        support_polygon=[] if hull is None else hull[:, 0].tolist(),
        diagnostic_is_not_independent_physical_validation=True)
    return note, a, b, inliers


class SourceRegistration:
    """One source cache, clear-frame selection and bidirectional shot graph."""

    def __init__(self, root, frames, source_sha256, options=None):
        self.root = Path(root)
        self.frames = frames
        self.source_sha256 = source_sha256
        self.options = options or RegistrationOptions()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root/'features').mkdir(exist_ok=True)
        (self.root/'edges').mkdir(exist_ok=True)
        self.cache_identity = dict(schema='rinkspace.source_registration.v1',
            source_video_sha256=source_sha256, options=asdict(self.options),
            frames=[dict(pts=f['pts'], source_BGR_sha256=f['source_BGR_sha256'],
                         surface_input_revision=f['surface_input_revision']) for f in frames])
        identity = self.root/'identity.json'
        if identity.exists():
            if json.loads(identity.read_text('utf-8')) != self.cache_identity:
                raise ValueError('Source graph cache identity differs')
        else:
            write_json(identity, self.cache_identity)

    def features_path(self, frame):
        return self.root/'features'/f'{frame["pts"]}.npz'

    def prepare(self, surface_loader, emit=print):
        started = time.monotonic()
        quality = []
        for i, frame in enumerate(self.frames):
            path = self.features_path(frame)
            if not path.exists():
                image = cv2.imread(frame['image'])
                if image is None or array_sha(image) != frame['source_BGR_sha256']:
                    raise ValueError('Decoded source pixels changed: '+frame['image'])
                masks, mask_identity = surface_loader(frame, image)
                features = frame_features(image, masks, self.options)
                features['mask_identity_json'] = np.array(json.dumps(mask_identity, sort_keys=True))
                np.savez_compressed(path, **features)
            with np.load(path) as cached:
                quality.append(dict(pts=frame['pts'], **{name:cached[name+'_quality'].tolist() for name in ('ice', 'board')}))
            if i % 50 == 0 or i == len(self.frames)-1:
                emit(dict(event='source_features', frames=i+1, total=len(self.frames), elapsed_s=time.monotonic()-started))
        write_json(self.root/'frame_quality.json', quality)
        return quality

    def select_keyframes(self, quality):
        result = {}
        width = self.options.keyframe_window
        for surface in ('ice', 'board'):
            chosen = []
            for begin in range(0, len(quality), width):
                candidates = range(begin, min(begin+width, len(quality)))
                # Sharpness and usable area select image references, not labels.
                score = lambda i: np.log1p(quality[i][surface][0])*np.sqrt(quality[i][surface][1])*quality[i][surface][2]**.25
                chosen.append(max(candidates, key=score))
            result[surface] = sorted(set([0, *chosen, len(quality)-1]))
        write_json(self.root/'keyframes.json', {k:[self.frames[i]['pts'] for i in v] for k,v in result.items()})
        return result

    def build(self, keyframes, emit=print):
        from functools import lru_cache
        @lru_cache(maxsize=12)
        def load(i):
            with np.load(self.features_path(self.frames[i])) as values:
                return {k:values[k] for k in values.files}
        started = time.monotonic(); records = []; pairs = set()
        for surface, keys in keyframes.items():
            for i in range(len(self.frames)):
                if i: pairs.add((surface, i-1, i))
                earlier = [k for k in keys if k < i]
                later = [k for k in keys if k > i]
                if earlier: pairs.add((surface, earlier[-1], i))
                if later: pairs.add((surface, later[0], i))
        ordered = sorted(pairs, key=lambda item:(item[2], item[0], item[1]))
        for ni, (surface, a, b) in enumerate(ordered):
            stem = f'{surface}_{self.frames[a]["pts"]}_{self.frames[b]["pts"]}'
            path = self.root/'edges'/f'{stem}.json'
            if path.exists():
                note = json.loads(path.read_text('utf-8'))
            else:
                note, xy_a, xy_b, inliers = track_pair(load(a), load(b), surface, self.options)
                note.update(reference_index=a, current_index=b,
                            reference_pts=self.frames[a]['pts'], current_pts=self.frames[b]['pts'])
                np.savez_compressed(path.with_suffix('.npz'), reference_uv=xy_a, current_uv=xy_b, inliers=inliers)
                write_json(path, note)
            records.append(note)
            if ni % 100 == 0 or ni == len(ordered)-1:
                emit(dict(event='source_edges', edges=ni+1, total=len(ordered), elapsed_s=time.monotonic()-started))
        coverage = []
        for i, frame in enumerate(self.frames):
            row = dict(pts=frame['pts'])
            for surface in keyframes:
                edges = [e for e in records if e['surface']==surface and e['status']=='navigation'
                         and i in (e['reference_index'],e['current_index'])]
                row[surface] = dict(usable_edges=len(edges),
                    past=any(min(e['reference_index'],e['current_index'])<i for e in edges),
                    future=any(max(e['reference_index'],e['current_index'])>i for e in edges))
            coverage.append(row)
        summary = dict(frames=len(self.frames), edges=len(records),
            surfaces={s:dict(navigation_edges=sum(e['surface']==s and e['status']=='navigation' for e in records),
                            frames_with_edges=sum(c[s]['usable_edges']>0 for c in coverage),
                            frames_with_past_and_future=sum(c[s]['past'] and c[s]['future'] for c in coverage),
                            uncovered_pts=[c['pts'] for c in coverage if not c[s]['usable_edges']]) for s in keyframes},
            scene_or_camera_projection_used=False, physical_identity_assigned=False,
            measurements_emitted_as_fit_observations=0, coverage=coverage)
        write_json(self.root/'graph.json', dict(summary=summary, edges=records))
        emit(dict(event='source_graph_ready', **{k:v for k,v in summary.items() if k!='coverage'}))
        return summary
