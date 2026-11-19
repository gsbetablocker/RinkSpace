"""Locate the named printed word, then read signed raw letter contours."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib

import cv2
import numpy as np

from common import HEIGHT, OUTPUT, WIDTH, observation_packet, read, save, scope, source_frames, visibility_packet, write
from fast_print_reads import PrintReader, raw_edges
from measure import DT, FIRST, load_camera, localize, motion_clock
from merge_source_constraints import observation_join
from review import restore_source_graphics
from stability_observations import read_arrays
from stability_tracks import ji


ANKARA = 'print_instance/Ad4::Ad4.ANKARA.native440.boundary_'


def registered_translation(masked, polygons, stroke_signal, speed):
    # Clip to the original panel image, then search a blurred full-word
    # template to obtain independent windows for reading its actual flanks.
    uv = np.concatenate(polygons)
    lo = np.maximum([5, 5], np.floor(uv.min(0)).astype(int) - [2, 2])
    hi = np.minimum([WIDTH - 5, HEIGHT - 5], np.ceil(uv.max(0)).astype(int) + [2, 2])
    if min(hi - lo) < 21 or (hi[0] - lo[0]) < 76:
        return
    origin = np.maximum([0, 0], lo - [64, 19])
    end = np.minimum([WIDTH, HEIGHT], hi + [64, 19])
    size = end - origin
    canvas = np.zeros((size[1], size[0]), np.uint8)
    for poly in polygons:
        cv2.fillPoly(canvas, [np.rint((poly - origin) * 4.).astype(np.int32)], 255, cv2.LINE_8, shift=2)
    expected = canvas.astype(np.float32) / 255.
    x0, y0 = lo - origin
    x1, y1 = hi - origin
    anchor = np.array([x0, y0])
    a, b = origin
    c, d = end
    image = stroke_signal[b:d, a:c].copy()
    occluded = masked[b:d, a:c] != 0
    # Dark purple and its colored graphics contribute little to this signal.
    # Unknown foreground pixels are excluded from both the mean and dot product.
    has = (~occluded).astype(np.float32)
    image[occluded] = 0.
    judged = []
    blur_widths = np.unique(np.rint(np.array([1., max(1., .40*speed), max(2., .7*speed),
                                               max(3., speed), max(4., 1.2*speed)])).astype(int))
    blur_widths = np.minimum(49, 2 * (blur_widths // 2) + 1)
    region = (slice(y0, y1), slice(x0, x1))
    for smear in blur_widths:
        template = cv2.blur(expected, (int(smear), 3), borderType=cv2.BORDER_CONSTANT)[region].copy()
        weight = (template > .03).astype(np.float32)
        if np.sum(weight) < 150:
            continue
        # The full bounding box includes the gaps between the glyphs.
        norm = np.ones(template.shape, dtype=np.float32)
        dot = cv2.matchTemplate(image, template, cv2.TM_CCORR)
        i_sum = cv2.matchTemplate(image, norm, cv2.TM_CCORR)
        t_sum = cv2.matchTemplate(has, template, cv2.TM_CCORR)
        i2 = cv2.matchTemplate(image**2, norm, cv2.TM_CCORR)
        t2 = cv2.matchTemplate(has, template**2, cv2.TM_CCORR)
        n = cv2.matchTemplate(has, norm, cv2.TM_CCORR)
        tt = np.maximum(0., t2 - t_sum * t_sum / np.maximum(1., n))
        ii = np.maximum(0., i2 - i_sum * i_sum / np.maximum(1., n))
        correlation = (dot - t_sum * i_sum / np.maximum(1., n)) / np.maximum(.01, np.sqrt(tt * ii))
        correlation[(n < .71 * template.size) | (t_sum < .74*template.sum())] = -2.
        yy, xx = np.indices(correlation.shape)
        rank = correlation - .0007 * np.sqrt((xx-x0)**2 + (yy-y0)**2)
        y, x = np.unravel_index(np.argmax(rank), rank.shape)
        rivals = (abs(xx-x) > 15.) | (abs(yy-y) > 8.)
        gap = float(correlation[y, x] - np.max(correlation[rivals], initial=-2.))
        if correlation[y, x] < .46 or gap < .040:
            continue
        judged.append((float(rank[y,x]), np.array([x, y], float) - anchor, int(smear), gap))
    return max(judged, key=lambda v:v[0]) if judged else None


def glyph_rows(reader, keys, projected, raw, mask, tolerance):
    status = projected['status']
    uv = projected['uv'].copy()
    uv[status != 0] = np.nan
    visible = {k: uv[reader.slices[k]] for k in keys}
    polygons = [v for v in visible.values() if np.isfinite(v).all()
                and np.ptp(v[:,0]) > 9. and np.ptp(v[:,1]) > 18.
                and v[:,0].max()>10. and v[:,0].min()<WIDTH-10.]
    if len(polygons) < 3:
        return
    f = raw.astype(np.float32)
    # Sample a white/colored material opponent, not a screen-space outline.
    white = f[:, :, 1] - .37 * np.ptp(f, axis=2)
    shifted = registered_translation(mask, polygons, white, tolerance)
    if shifted is None:
        return
    rank, shift, blur, gap = shifted
    output = []
    for k in keys:
        sl = reader.slices[k]
        p, st = visible[k], status[sl]
        q, normal = reader.P.projected_queries(p, st, 10., 1.)
        if len(q) < 3:
            continue
        take = raw_edges(reader.P, white, mask, q, normal, shift, 'white')
        if len(take) < 3:
            continue
        j = np.array([item[0] for item in take])
        observed = np.array([item[1] for item in take])
        contrast = np.array([item[2] for item in take])
        width = np.array([item[3] for item in take])
        station, distance = localize(p + shift, reader.samples[k][1], observed)
        good = np.isfinite(station) & (distance < 7.)
        if np.count_nonzero(good) < 3:
            continue
        output.append((k, observed[good], station[good], normal[j][good], contrast[good],
                       width[good], q[j][good], rank))
    if len(output) < 2:
        return
    nv = np.vstack([item[3] for item in output])
    positions = np.vstack([item[1] for item in output])
    if np.count_nonzero(abs(nv[:,0])>.57) < 4 or np.ptp(positions[:,0]) < 55.:
        return
    return output, dict(guide_shift_px=shift.tolist(), smear_width_px=blur, match_score=rank,
                        competing_maximum_margin=gap)


def read_word(record, shot, case, guide, source, destination):
    sid = shot['id']
    reader = PrintReader(record)
    prefixes = tuple(case['path_prefixes'])
    keys = tuple(k for k in reader.config if k.startswith(prefixes)
                 and '.source_arc_' not in k and '_inner' not in k)
    if len(keys) < 3:
        raise ValueError(f'Insufficient named outline paths for {sid}')
    pts, c, g = load_camera(shot, guide)
    t = motion_clock(shot, guide, pts)
    orig = read_arrays(observation_packet(source, shot['id']))
    hashes = dict(zip(orig['source_frame_pts'], orig['source_frame_BGR_sha256'], strict=True))
    windows = [tuple(round(s * 12800) for s in w) for w in case['windows_seconds']]
    exposures = np.zeros(len(pts),bool)
    for lo, hi in windows:
        exposures |= (pts >= lo) & (pts <= hi)
    with np.load(visibility_packet(OUTPUT/'visibility_fast_checked', sid)) as f:
        np.testing.assert_array_equal(f['pts'], pts)
        mask_by_frame = f['packed']
    with np.load(OUTPUT/'fast_tracks02'/f'{sid}.npz') as tracks:
        speed = np.full(len(pts), 11.)
        selected = (tracks['lag'] == 1) & (tracks['family'] == 0)
        source_speed = abs(tracks['uv'][:, 0] - tracks['left_uv'][:, 0])
        for i in np.unique(tracks['index'][selected]):
            speed[i] = np.median(source_speed[selected & (tracks['index'] == i)])
    rows = defaultdict(list)
    phases = {}
    all_ids = [i for k in keys for i in range(reader.slices[k].start, reader.slices[k].stop)]
    compact = reader.world[all_ids]
    # Narrow the native projected sample buffer to the named printed contours.
    reader.config = {k: reader.config[k] for k in keys}
    starts = np.r_[0, np.cumsum([len(reader.samples[k][0]) for k in keys])]
    reader.slices = {k: slice(a, b) for k, a, b in zip(keys, starts[:-1], starts[1:], strict=True)}
    for p, frame in source_frames(record['source']['registered_path'], pts[exposures]):
        i = int(np.searchsorted(pts, p))
        raw = frame.to_ndarray(format='bgr24')
        if hashlib.sha256(raw.tobytes()).hexdigest() != hashes[p]:
            raise ValueError(f'Source frame bytes differ at {p}')
        person = np.unpackbits(mask_by_frame[i], axis=1, count=WIDTH)
        mask = restore_source_graphics(raw, p, cv2.dilate(person, np.ones((11,11), np.uint8)))
        projected = reader.native.core.project_motion(g, c, FIRST, DT, HEIGHT, t[i], compact)
        found = glyph_rows(reader, keys, projected, raw, mask, speed[i])
        if found is None:
            continue
        data, reg = found
        phases[p] = reg
        for k, uv, station, normal, contrast, width, hint, score in data:
            size = len(station)
            # Hold fixed, spaced pieces of the actual closed outline in every
            # frame, including the fast-pan frames in the previous time hold.
            phase = np.floor(station * 24).astype(np.int32) % 4
            held = phase == 2
            family = k.split('::', 1)[0].rsplit('/', 1)[-1]
            item = dict(path=np.repeat(k, size), family=np.repeat(family, size), pts=np.repeat(p, size),
                        idx=np.repeat(i, size), fold=np.where(held, 3, 0).astype(np.int8),
                        uv=uv, station=station,
                        normal=normal, contrast=contrast, guide_error=np.linalg.norm(uv-hint, axis=1),
                        material=np.repeat('source_board_print_contour', size),
                        source_transition_fwhm_px=width, source_registration_score=np.repeat(score, size),
                        eligible=np.ones(size, bool), eligible_before_quality_review=np.ones(size, bool),
                        excluded_reason=np.repeat('', size))
            for key, value in item.items():
                rows[key].append(value)
    if not rows:
        print(sid, 'No original signed named letter contours were readable', flush=True)
        return False
    result = {k: np.concatenate(v) for k, v in rows.items()}
    destination.mkdir(exist_ok=True)
    out = destination/f'{sid}_word_original.npz'
    save(out, **result, source_sha256=orig['source_sha256'], geometry_sha256=orig['geometry_sha256'],
         source_frame_pts=orig['source_frame_pts'], source_frame_BGR_sha256=orig['source_frame_BGR_sha256'])
    # Original and derived searches can use the same path, but their active
    # points cannot vote twice on the same path and exposure.
    from precision_observations import replace_group
    replace_group(orig, result)
    ref = destination/f'{sid}_base_without_same_path.npz'
    save(ref, **orig)
    report = observation_join(shot, ref, out, destination/f'{sid}.npz', per_path=True)
    write(destination/f'{sid}_word_receipt.json', {'points': int(len(result['pts'])),
          'holdout_definition': 'Closed outline station bands, held for each measured exposure',
          'source_exposures': len(phases), 'profiles': {int(p):v for p,v in phases.items()}, 'qualified': report})
    print(sid, 'actual named contours:', len(result['pts']), 'in', len(phases), 'exposures', flush=True)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--guide', required=True)
    p.add_argument('--observations', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--cases', help='JSON shot IDs with path_prefixes and windows_seconds')
    a = p.parse_args()
    cv2.setNumThreads(1)
    ji.enable_historical_source_readers()
    record = scope()
    cases = read(a.cases) if a.cases else {'s09': {'path_prefixes':[ANKARA], 'windows_seconds':[[77.60,79.34]]}}
    source, destination = OUTPUT/a.observations, OUTPUT/a.out
    destination.mkdir(exist_ok=True)
    aliases = {s['id']:str(observation_packet(source,s['id'])) for s in record['shots']}
    for shot in record['shots']:
        sid = shot['id']
        if sid in cases and read_word(record,shot,cases[sid],OUTPUT/a.guide,source,destination):
            aliases[sid] = str(destination/f'{sid}.npz')
    write(destination/'shot_packets.json',aliases)


if __name__ == '__main__':
    main()
