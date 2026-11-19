"""Screen the registered programme on the exact source clock; emit review images."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from rinkspace.storage import AssetStore
from rinkspace.viewer3d.source import SourceVideo


PROGRAMME = Path(__file__).resolve().parents[2]
PROJECT = PROGRAMME.parents[3]
BASELINE = PROGRAMME / 'fit_v11/body_first_spin_1113.json'
NAMES = ('nose', 'L eye', 'R eye', 'L ear', 'R ear', 'L shoulder', 'R shoulder',
         'L elbow', 'R elbow', 'L wrist', 'R wrist', 'L hip', 'R hip',
         'L knee', 'R knee', 'L ankle', 'R ankle')
PAIRS = np.arange(1, 17).reshape(-1, 2)
OPPOSITE = np.r_[0, PAIRS[:, ::-1].ravel()]
LINKS = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def whole_skate_observations(programme=PROGRAMME):
    return AssetStore(PROJECT).resolve(programme / 'fit_v11/body_skate_whole_1115.json', must_exist=True)


def read(path):
    return Path(path).read_text(encoding='utf-8')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def baseline(data, registry=BASELINE, *, result_key=None):
    doc = json.loads(read(registry))
    scope = json.loads(read(PROGRAMME / 'rebuild_scope.json'))
    paths = {key: data / item['path'] for key, item in doc['data_root_artifacts'].items()}
    if result_key is None:
        result_key = next(key for key in ('fitted_state', 'result', 'N_result') if key in paths)
    with np.load(paths[result_key], allow_pickle=False) as z:
        keys = ('source_pts', 'time_base', 'source_shot_ids', 'COCO_uv', 'source_uv_native',
                'source_sigma_px', 'source_observation_use', 'source_observation_initial_use',
                'source_observation_objective_use', 'source_body_model_joint_ids',
                'source_observation_ledger_sha256')
        old = {k: z[k] for k in keys}
    pts = np.concatenate([np.arange(s['processing']['first_pts'], s['processing']['stop_pts_exclusive'],
                                    256, dtype=np.int64) for s in scope['shots']])
    np.testing.assert_array_equal(old['source_pts'], pts)
    np.testing.assert_array_equal(old['time_base'], doc['source_time_base'])
    np.testing.assert_array_equal(old['source_shot_ids'], np.repeat(
        [s['id'] for s in scope['shots']], [s['processing']['native_frame_count'] for s in scope['shots']]))
    return scope, paths, old


def crop_bounds(uv, shape, margin):
    lo = np.maximum(0, np.floor(np.min(uv, axis=0) - margin)).astype(int)
    hi = np.minimum(shape[1::-1], np.ceil(np.max(uv, axis=0) + margin)).astype(int)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def tracks(first, second, points, bounds):
    """Local patch motion and forward/backward error measured in native pixels."""
    x, y, x1, y1 = bounds
    origin = np.array([x, y], dtype=np.float32)
    coordinates = (points.astype(np.float32) - origin).reshape(-1, 1, 2)
    opt = dict(winSize=(25, 25), maxLevel=3,
               criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 23, .025))
    a, b = first[y:y1, x:x1], second[y:y1, x:x1]
    predicted, good, photometric = cv2.calcOpticalFlowPyrLK(a, b, coordinates, None, **opt)
    back, back_good, _ = cv2.calcOpticalFlowPyrLK(b, a, predicted, None, **opt)
    valid = good[:, 0].astype(bool) & back_good[:, 0].astype(bool)
    fb = np.linalg.norm((back - coordinates)[:, 0], axis=1)
    return np.where(valid[:, None], predicted[:, 0] + origin, np.nan), np.where(valid, fb, np.inf), np.where(valid, photometric[:, 0], np.inf)


def source_flow(scope, pts, shot, uv):
    """Decode every exposure and measure swappable tracks without a body model."""
    clock = scope['source_clock']['time_base_num'], scope['source_clock']['time_base_den']
    reader = SourceVideo(scope['source']['registered_path'], clock)
    if reader.stream.time_base != reader.time_base:
        raise ValueError('Expected original source stream time base')
    forward = np.full(uv.shape, np.nan, np.float32)
    backward = forward.copy()
    forward_fb = np.full(uv.shape[:2], np.inf, np.float32)
    backward_fb = forward_fb.copy()
    forward_photo = forward_fb.copy()
    backward_photo = forward_fb.copy()
    try:
        for i, stamp in enumerate(pts):
            rgb = reader.read(stamp)
            if rgb.shape != (scope['source']['height'], scope['source']['width'], 3):
                raise ValueError('Original pixel domain changed')
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if i and shot[i] == shot[i-1]:
                bounds = crop_bounds(uv[i-1:i+1].reshape(-1, 2), gray.shape, 96)
                forward[i-1], forward_fb[i-1], forward_photo[i-1] = tracks(previous, gray, uv[i-1], bounds)
                backward[i], backward_fb[i], backward_photo[i] = tracks(gray, previous, uv[i], bounds)
            previous = gray
            if i % 1200 == 0:
                print(f'decoded {i+1}/{len(pts)} pts={stamp}', flush=True)
    finally:
        reader.close()
    return dict(forward=forward, backward=backward, forward_fb=forward_fb, backward_fb=backward_fb,
                forward_photo=forward_photo, backward_photo=backward_photo)


def candidates(old, flow):
    uv, sigma = old['source_uv_native'], old['source_sigma_px']
    pts, shot = old['source_pts'], old['source_shot_ids']
    ids, admitted = old['source_body_model_joint_ids'], old['source_observation_objective_use']
    fit_uv = old['COCO_uv']
    indices = np.arange(len(pts))[:, None]
    a = np.linalg.norm(fit_uv[indices, ids] - uv, axis=-1)
    b = np.linalg.norm(fit_uv[indices, OPPOSITE[ids]] - uv, axis=-1)
    torso = np.linalg.norm(.5*(uv[:, 5]+uv[:, 6])- .5*(uv[:, 11]+uv[:, 12]), axis=-1)
    separation = np.linalg.norm(uv[:, PAIRS[:, 0]] - uv[:, PAIRS[:, 1]], axis=-1)
    # Use model proximity only to find frames to look at. The LK cue has no model input.
    locator = admitted & (a > np.maximum(36, 3*sigma)) & (a-b > np.maximum(18, 1.8*sigma)) & (b < np.maximum(38, 2*sigma))
    legacy = admitted & (a > 100) & (b < 20)
    identity_uv = np.take_along_axis(uv, np.argsort(ids, axis=1)[..., None], axis=1)
    mapped = np.argsort(ids, axis=1)
    swapped = np.zeros_like(admitted)
    single = np.zeros_like(admitted)
    forward_uses = (flow['forward_fb'] < 3.5) & (flow['forward_photo'] < 32)
    backward_uses = (flow['backward_fb'] < 3.5) & (flow['backward_photo'] < 32)
    for prev, nex, prediction, valid in ((slice(None, -1), slice(1, None), flow['forward'], forward_uses),
                                         (slice(1, None), slice(None, -1), flow['backward'], backward_uses)):
        tracking = np.take_along_axis(prediction[prev], mapped[prev, :, None], axis=1)
        valid = np.take_along_axis(valid[prev], mapped[prev], axis=1)
        direct = np.linalg.norm(tracking - identity_uv[nex], axis=-1)
        cross = np.linalg.norm(tracking - identity_uv[nex][:, OPPOSITE], axis=-1)
        gap = np.maximum(8, .12*torso[nex, None])
        closer = valid & (cross < gap) & (direct-cross > gap)
        paired = closer & closer[:, OPPOSITE]
        paired[:, :5] = False
        swapped[nex] |= paired
        swapped[prev] |= paired
        # Unilateral jumps are meaningful when the opposite source region is
        # simultaneously tracked. The review distinguishes overlap and occlusion.
        unilateral = closer & valid[:, OPPOSITE] & (direct[:, OPPOSITE] < gap)
        unilateral[:, :5] = False
        single[nex] |= unilateral
    close = separation < np.maximum(4., .035*torso[:, None])
    parents = np.array([[5, 6], [7, 8], [11, 12], [13, 14]])
    duplicate = np.zeros_like(close)
    # Crossings alone are not conflicts. Use separated parent links, plus an
    # isolated/asymmetric endpoint discontinuity or the separate pixel locator.
    for pair_no, parents_pair in zip((3, 4, 6, 7), parents, strict=True):
        pair = PAIRS[pair_no]
        parent_gap = np.linalg.norm(uv[:, parents_pair[0]]-uv[:, parents_pair[1]], axis=-1)
        duplicate[:, pair_no] = (close[:, pair_no] & (parent_gap > .32*torso)
                                & (single[:, pair].any(1) | locator[:, pair].any(1)))
    return dict(error=a, opposite_error=b, legacy=legacy, locator=locator, source_swap_boundary=swapped,
                source_unilateral_boundary=single, possible_duplicate=duplicate, separation=separation,
                torso_px=torso, pts=pts, shot=shot)


def review_bounds(old, i, shape):
    indices = np.arange(max(0, i-3), min(len(old['source_pts']), i+4))
    same_shot = old['source_shot_ids'][indices] == old['source_shot_ids'][i]
    points = old['source_uv_native'][indices[same_shot]]
    torso = np.linalg.norm(points[:, 5:7].mean(axis=1)-points[:, 11:13].mean(axis=1), axis=1)
    finite = points[np.isfinite(points).all(axis=-1)]
    if not len(finite):
        return (0, 0, shape[1], shape[0])
    valid_torso = torso[np.isfinite(torso)]
    scale = np.median(valid_torso) if len(valid_torso) else 0.
    return crop_bounds(finite, shape, max(28., .4*scale))


def panel(reader, old, i, labels, *, model=False, source_overlay=True):
    pts = old['source_pts']
    uv = old['source_uv_native'][i]
    rgb = reader.read(pts[i])
    usable = old['source_observation_use'][i]
    ids = old['source_body_model_joint_ids'][i]
    x, y, x1, y1 = review_bounds(old, i, rgb.shape)
    full = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    image = full[y:y1, x:x1]
    scale = min(280 / image.shape[1], 267 / image.shape[0])
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    target = np.full((485, 286, 3), 23, dtype=np.uint8)
    ox = (target.shape[1] - image.shape[1]) // 2
    target[23:23+image.shape[0], ox:ox+image.shape[1]] = image
    finite = np.isfinite(uv).all(axis=-1)
    p = np.zeros(uv.shape, dtype=int)
    p[finite] = np.rint((uv[finite]-[x, y])*scale+[ox, 23]).astype(int)
    colors = ((255, 210, 56), (54, 85, 255))
    if source_overlay:
        for a, b in LINKS:
            if not (finite[a] and finite[b]):
                continue
            end = b if a % 2 == b % 2 else a
            cv2.line(target, tuple(p[a]), tuple(p[b]), colors[(end+1) % 2], 1, cv2.LINE_AA)
    if model:
        projected = np.rint((old['COCO_uv'][i]-[x, y])*scale+[ox, 23]).astype(int)
        for a,b in LINKS:
            cv2.line(target, tuple(projected[a]), tuple(projected[b]), (230, 85, 200), 2, cv2.LINE_AA)
    for j in labels if source_overlay else ():
        if not finite[j]:
            continue
        c = colors[(j+1) % 2]
        cv2.circle(target, tuple(p[j]), 5, (15, 15, 15), 2, cv2.LINE_AA)
        cv2.circle(target, tuple(p[j]), 4, c, 1 if not usable[j] else -1, cv2.LINE_AA)
        pos = p[j]+([-18, -5] if j % 2 else [6, 9])
        name = f'{j}' if ids[j] == j else f'{j}>{ids[j]}'
        cv2.putText(target, name, tuple(pos), cv2.FONT_HERSHEY_SIMPLEX, .43, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.putText(target, name, tuple(pos), cv2.FONT_HERSHEY_SIMPLEX, .43, c, 1, cv2.LINE_AA)
    tb = old.get('time_base', (1, 12800))
    cv2.putText(target, f'{old["source_shot_ids"][i].upper()} {pts[i]*tb[0]/tb[1]:.2f}s   {pts[i]}',
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, .53, (240, 240, 240), 1, cv2.LINE_AA)
    overview_scale = min(280 / full.shape[1], 162 / full.shape[0])
    overview = cv2.resize(full, None, fx=overview_scale, fy=overview_scale, interpolation=cv2.INTER_AREA)
    ox = (target.shape[1]-overview.shape[1])//2
    target[320:320+overview.shape[0], ox:ox+overview.shape[1]] = overview
    lo, hi = np.rint(np.array([[x, y], [x1-1, y1-1]])*overview_scale + [ox, 320]).astype(int)
    cv2.rectangle(target, tuple(lo), tuple(hi), (80, 237, 230), 1)
    cv2.putText(target, f'Full frame  ROI {x},{y} to {x1},{y1}', (6, 312),
                cv2.FONT_HERSHEY_SIMPLEX, .44, (240, 240, 240), 1, cv2.LINE_AA)
    return target


def sheets(scope, old, times, output, *, width, labels, model=False, source_overlay=True):
    indices = np.searchsorted(old['source_pts'], times)
    np.testing.assert_array_equal(old['source_pts'][indices], times)
    reader = SourceVideo(scope['source']['registered_path'], old['time_base'])
    try:
        panels = [panel(reader, old, i, labels, model=model, source_overlay=source_overlay) for i in indices]
    finally:
        reader.close()
    while len(panels) % width:
        panels.append(np.zeros_like(panels[0]))
    sheet = np.concatenate([np.concatenate(panels[i:i+width], axis=1) for i in range(0, len(panels), width)])
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    tm = parser.add_subparsers(dest='command', required=True)
    scan = tm.add_parser('scan')
    scan.add_argument('output', type=Path)
    select = tm.add_parser('sheet')
    select.add_argument('output', type=Path)
    select.add_argument('--seconds', type=float, nargs='+', required=True)
    select.add_argument('--width', type=int, default=4)
    select.add_argument('--model', action='store_true')
    select.add_argument('--clean', action='store_true', help='Clean source image and ROI boundary')
    select.add_argument('--registry', type=Path, default=BASELINE)
    select.add_argument('--labels', type=int, nargs='+', default=list(range(5, 17)))
    args = parser.parse_args()
    scope, paths, old = baseline(args.data, getattr(args, 'registry', BASELINE))
    if args.command == 'sheet':
        stamps = [round(t*12800) for t in args.seconds]
        sheets(scope, old, stamps, args.output, width=args.width, labels=args.labels,
               model=args.model, source_overlay=not args.clean)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    flow = source_flow(scope, old['source_pts'], old['source_shot_ids'], old['source_uv_native'])
    result = candidates(old, flow)
    np.savez_compressed(args.output/'scan.npz', **flow, **result)
    print(json.dumps({key: [int(v.sum()), int(v.any(axis=1).sum())]
                      for key, v in result.items() if v.dtype == bool}), flush=True)


if __name__ == '__main__':
    main()
