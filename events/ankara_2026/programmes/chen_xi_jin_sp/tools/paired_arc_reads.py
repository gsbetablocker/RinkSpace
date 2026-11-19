"""Read both visible runs of a painted rink curve and the kickplate contact."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import sys
import warnings

import cv2
import numpy as np
from scipy.signal import find_peaks

from common import HEIGHT, JOINT, OUTPUT, WIDTH, observation_packet, save, scope, source_frames, visibility_packet, write
from ice_evidence import ICE_EVIDENCE_FIELDS, MIDPOINT, assert_ice_evidence
from ice_width_reads import WidthReader, paired_flanks
from measure import DT, FIRST, fold_map, load_camera, motion_clock
from physical_reads import CHANNELS, coherent, subpixel
from precision_observations import exclude
from review import restore_source_graphics
from stability_observations import ROW_FIELDS, read_arrays


def projected_crossings(px, stations, x):
    left, right = px[:-1], px[1:]
    valid = np.isfinite(left).all(1) & np.isfinite(right).all(1)
    denominator = right[:, 0] - left[:, 0]
    indexes = np.flatnonzero(valid & (abs(denominator) > 1.e-7)
                             & (x >= np.minimum(left[:, 0], right[:, 0]))
                             & (x < np.maximum(left[:, 0], right[:, 0])))
    fraction = (x - left[indexes, 0]) / denominator[indexes]
    tangent = (right[indexes] - left[indexes]) * np.sign(denominator[indexes, None])
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    y = (1. - fraction) * left[indexes, 1] + fraction * right[indexes, 1]
    st = (1. - fraction) * stations[indexes] + fraction * stations[indexes + 1]
    return sorted(zip(y, tangent, st, strict=True), key=lambda q: q[0])


def traces(px, stations, *, step=37.):
    sides = ([], [])
    for x in np.arange(42., WIDTH - 41., step):
        roots = projected_crossings(px, stations, x)
        if len(roots) != 2 or roots[1][0] - roots[0][0] < 78.:
            continue
        for side, (y, tangent, station) in zip(sides, roots, strict=True):
            if not 349. < y < 1060. or tangent[0] < .86:
                continue
            side.append((x, y, tangent[0], tangent[1], station))
    return tuple(np.asarray(s, dtype=float).reshape(-1, 5) for s in sides)


def read_profiles(signal, mask, points, normal, tangent, radius, along):
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    coords = (points[:, None, None] + offsets[None, :, None, None] * normal[:, None, None]
              + along[None, None, :, None] * tangent[:, None, None])
    mx, my = np.moveaxis(coords.reshape(len(points), -1, 2).astype(np.float32), -1, 0)
    raw = cv2.remap(signal, mx, my, cv2.INTER_LINEAR).reshape(len(points), len(offsets), len(along))
    hidden = cv2.remap(mask, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1)
    hidden = hidden.reshape(raw.shape).astype(bool)
    # Both edges use a symmetric set of same-PTS source profiles.
    paired_samples = ~(hidden | np.flip(hidden, axis=2))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)
        profile = np.nanmedian(np.where(paired_samples, raw, np.nan), axis=2)
    unavailable = paired_samples.sum(2) < max(3, len(along) // 2)
    profile = np.nan_to_num(profile, nan=-100.)
    profile = cv2.GaussianBlur(profile, (9, 1), 1.40, borderType=cv2.BORDER_REPLICATE)
    return profile, unavailable


def arc_parts(red, mask, px, station, *, refine=False):
    for side, samples in enumerate(traces(px, station, step=28. if refine else 37.)):
        if len(samples) < 7:
            continue
        points, tangent, st = samples[:, :2], samples[:, 2:4], samples[:, 4]
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        radius = 44 if refine else 79
        half = 26 if refine else 34
        profiles, hidden = read_profiles(red, mask, points, normal, tangent, radius,
                                         np.linspace(-half,half,13))
        why = []
        for row, omit in zip(profiles, hidden, strict=True):
            why.append(paired_flanks(row, omit, radius, 'red', (7., 39.),
                      minimum_contrast=1.58 if refine else 1.18,
                      minimum_inside_chroma=2.12 if refine else 1.65))
        # Run the same coherent detector separately on unoccluded portions,
        # since the skater can hide much of the middle of the arc.
        by_half = []
        for ids in np.split(np.arange(len(points)), np.flatnonzero(np.diff(points[:, 0]) > 85.) + 1):
            for k, q in coherent([why[i] for i in ids], points[ids], stripe=True):
                by_half.append((ids[k], q))
        if len(by_half) < 5:
            continue
        j = np.asarray([a for a, b in by_half])
        offsets = np.asarray([b[2:4] for a, b in by_half])
        sides = points[j, None] + offsets[:, :, None] * normal[j, None]
        uv = sides.mean(1)
        quality = np.asarray([b[5] for a, b in by_half])
        spread = np.ptp(uv[:, 0])
        if spread < 160. or np.quantile(np.linalg.norm(sides[:, 0] - sides[:, 1], axis=1), .9) > 35.:
            continue
        yield side, uv, st[j], normal[j], sides, quality, np.linalg.norm(uv - points[j], axis=1)


def toe_parts(yellow, mask, px, station):
    samples = []
    for x in np.arange(48., WIDTH - 45., 47.):
        options = [(y, t, st) for y, t, st in projected_crossings(px, station, x)
                   if 225. < y < 650. and t[0] > .90]
        if len(options) != 1:
            continue
        y, tangent, s = options[0]
        samples.append((x, y, *tangent, s))
    samples = np.asarray(samples).reshape(-1, 5)
    if len(samples) < 12:
        return
    points, tangents, st = samples[:, :2], samples[:, 2:4], samples[:, 4]
    normal = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    radius = 34
    profiles, hidden = read_profiles(yellow, mask, points, normal, tangents, radius, np.arange(-14., 15., 4.))
    candidates = []
    for g, omit in zip(profiles, hidden, strict=True):
        gradient = -np.gradient(g)
        options = []
        for k in find_peaks(gradient, height=.24, prominence=.27)[0]:
            if k < 10 or k >= len(g) - 10 or np.any(omit[k - 10:k + 10]):
                continue
            high, low = np.median(g[k - 9:k - 3]), np.median(g[k + 3:k + 9])
            if high < 9. or high - low < 3.1 or low > .83 * high:
                continue
            offset = k + subpixel(gradient, k) - radius
            strength = float((high - low) * np.sqrt(gradient[k]))
            options.append((strength - .045 * abs(offset), offset, high - low))
        candidates.append(sorted(options, reverse=True)[:4])
    # A broad isolated shadow or the white-to-yellow top can fail this vote.
    best = []
    for k, options in enumerate(candidates):
        if not options:
            continue
        def match(item):
            offset = item[1]
            neighbor = sum(bool(other) and any(abs(t[1] - offset) < 5. for t in other)
                           for other in candidates[max(0, k - 3):k + 4])
            return neighbor, item[0]
        choice = max(options, key=match)
        if match(choice)[0] >= 3:
            best.append((k, choice))
    if len(best) < 12:
        return
    j = np.array([i for i, v in best])
    offset, contrast = np.array([[v[1], v[2]] for i, v in best]).T
    uv = points[j] + offset[:, None] * normal[j]
    return uv, st[j], normal[j], contrast, abs(offset)


def store_rows(rows, *, path, family, index, p, fold, uv, station, normal, contrast, error, kind,
               edges=None, quality=None, which=-1):
    n = len(uv)
    item = dict(path=np.repeat(path, n), family=np.repeat(family, n), idx=np.repeat(index, n),
                pts=np.repeat(p, n), fold=np.broadcast_to(fold,(n,)).copy(), uv=uv, station=station, normal=normal,
                contrast=contrast, guide_error=error, material=np.repeat(kind, n),
                eligible=np.ones(n, bool), eligible_before_quality_review=np.ones(n, bool),
                excluded_reason=np.repeat('', n), source_bracket=edges if edges is not None else np.full((n, 2, 2), np.nan),
                source_side_contrast=quality if quality is not None else np.full((n, 2), np.nan),
                ice_measurement_role=np.repeat(MIDPOINT if edges is not None else '', n),
                side=np.full(n, which))
    for field, value in item.items():
        rows[field].append(value)


def measure(guide, baseline, output, *, refine=False):
    record = scope()
    shot = record['shots'][-1]
    reader = WidthReader(record)
    native = reader.native
    paths = ('demo_initial/A2:outer', 'board/continuous/bottom')
    pts, controls, rig = load_camera(shot, guide)
    times = motion_clock(shot, guide, pts)
    folds = fold_map(shot, len(pts))
    if str(JOINT) not in sys.path:
        sys.path.insert(0, str(JOINT))
    from joint_inputs import native_paths
    wk = native_paths(native, paths, count=2401)
    both = np.vstack([wk[k][0] for k in paths])
    n_arc = len(wk[paths[0]][0])
    original = read_arrays(observation_packet(baseline, 's20'))
    hashes = dict(zip(original['source_frame_pts'], original['source_frame_BGR_sha256'], strict=True))
    with np.load(visibility_packet(OUTPUT/'visibility_fast_checked', 's20')) as packet:
        np.testing.assert_array_equal(packet['pts'], pts)
        masks = packet['packed']
    rows = defaultdict(list)
    for i, (p, frame) in enumerate(source_frames(record['source']['registered_path'], pts)):
        raw = frame.to_ndarray(format='bgr24')
        if hashlib.sha256(raw.tobytes()).hexdigest() != hashes[p]:
            raise ValueError(f"Source pixels differ at PTS {p}")
        mask = np.unpackbits(masks[i], axis=1, count=WIDTH)
        mask = restore_source_graphics(raw, p, cv2.dilate(mask, np.ones((13, 13), np.uint8)))
        coordinates = native.core.project_motion(rig, controls, FIRST, DT, HEIGHT, times[i], both)
        v = coordinates['uv']
        v[coordinates['status'] != 0] = np.nan
        b, g, r = np.moveaxis(raw.astype(np.float32), 2, 0)
        red = raw.astype(np.float32) @ np.asarray(CHANNELS['red'], np.float32)
        yellow = .55 * r + .45 * g - b
        reads = arc_parts(red, mask, v[:n_arc], wk[paths[0]][1],refine=refine)
        for side, uv, st, normal, edges, quality, error in reads:
            family = ('A2.upper','A2.lower')[side] if refine else 'A2'
            fold = np.where(np.floor(st*52).astype(int)%4==2,3,0).astype(np.int8) if refine else folds[i]
            store_rows(rows, path=paths[0], family=family, index=i, p=p, fold=fold, uv=uv, station=st,
                       normal=normal, contrast=quality.min(1), error=error, kind='source_verified_width_centre',
                       edges=edges, quality=quality, which=side)
        toe = toe_parts(yellow, mask, v[n_arc:], wk[paths[1]][1])
        if toe is not None:
            uv, st, normal, contrast, error = toe
            store_rows(rows, path=paths[1], family='board.bottom', index=i, p=p, fold=folds[i], uv=uv,
                       station=st, normal=normal, contrast=contrast, error=error, kind='source_signed_kickplate_lower_edge')
    new = {k: np.concatenate(v) for k, v in rows.items()}
    # A solitary color fluctuation does not establish a physical paint run.
    active = new['eligible'].copy()
    for side in (0, 1):
        group = new['side'] == side
        ids = np.unique(new['idx'][group])
        near = np.diff(ids) <= 2
        if len(ids):
            adjacent = ids[np.r_[False, near] | np.r_[near, False]]
            active[group] &= np.isin(new['idx'][group], adjacent)
    exclude(new, ~active, 'isolated_same_exposure_profile')
    roles = reader.roles
    assert_ice_evidence(new, new['eligible'], roles)
    output.mkdir(exist_ok=True)
    save(output/'s20_paired_source.npz', **new, source_sha256=original['source_sha256'],
         geometry_sha256=original['geometry_sha256'], source_frame_pts=pts,
         source_frame_BGR_sha256=np.array([hashes[x] for x in pts]))
    for family in np.unique(new['family']):
        nset = new['eligible'] & (new['family'] == family)
        previous = original['family'] == family
        if not refine:
            previous = original['path'] == new['path'][nset][0] if nset.any() else previous
        replace = previous & np.isin(original['pts'], new['pts'][nset])
        exclude(original, replace, 'new_signed_source_read_of_same_material')
    z = {k: np.concatenate([original[k], new[k]]) for k in (*ROW_FIELDS, *ICE_EVIDENCE_FIELDS)}
    save(output/'s20.npz', **z, source_sha256=original['source_sha256'],
         geometry_sha256=original['geometry_sha256'], source_frame_pts=original['source_frame_pts'],
         source_frame_BGR_sha256=original['source_frame_BGR_sha256'])
    all_packets = {s['id']: str(observation_packet(baseline, s['id'])) for s in record['shots']}
    all_packets['s20'] = str(output/'s20.npz')
    write(output/'shot_packets.json', all_packets)
    summary = {}
    for side, label in ((0, 'upper_arc'), (1, 'lower_arc'), (-1, 'board_ice_contact')):
        allow = new['eligible'] & (new['side'] == side)
        summary[label] = dict(exposures=int(np.unique(new['pts'][allow]).size), source_rows=int(np.sum(allow)))
    write(output/'s20_receipt.json', summary)
    print(summary, flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--guide', required=True)
    cli.add_argument('--baseline', default='fast_inputs_panel06')
    cli.add_argument('--out', required=True)
    cli.add_argument('--refine',action='store_true',help='Read tighter signed profiles for both identified A2 runs')
    a = cli.parse_args()
    cv2.setNumThreads(1)
    measure(OUTPUT/a.guide, OUTPUT/a.baseline, OUTPUT/a.out, refine=a.refine)


if __name__ == '__main__':
    main()
