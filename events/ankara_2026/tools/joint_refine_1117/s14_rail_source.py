"""Audit the thin board face's two source boundaries over Jin S14."""
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from lane_end_evidence import foreground
from review import camera_state, ji, localize, native_for, source_frames
from shared import OUT, arrays, read, save, sha, write
from side_repair import native_table


SHOT = 'jin_s14'
GUIDE = 'lane4_side_1118/verified'
PATHS = ('board/continuous/top', 'board/continuous/bottom')
FAMILIES = ('board.top', 'board.bottom')
MATERIALS = {
    1: 'vertical_fascia_upper_junction',
    2: 'ice_facing_cap_lip',
    3: 'kickplate_ice_contact',
}
REPORT_PTS = (1568000, 1583360)
X = np.arange(12., 1913., 6.)
OFFSETS = np.arange(-36., 47.)


def projection(state, shot, out):
    pts = state[SHOT + '_pts']
    native = native_for(GUIDE, shot['program'])
    table = native_table(native, PATHS, 4097)
    world = {key: table[key][0] for key in PATHS}
    projected = []
    status = []
    times = state[SHOT + '_times']
    for start in range(0, len(pts), 20):
        jobs = [(i, key, times[i]) for i in range(start, min(len(pts), start + 20))
                for key in PATHS]
        batch = ji.project_groups(native, state[SHOT + '_rigs'], state[SHOT + '_controls'],
                                  shot['first'], shot['dt'], jobs, world)
        projected.extend(uv for uv, _ in batch)
        status.extend(code for _, code in batch)
    data = dict(uv=np.asarray(projected).reshape(len(pts), len(PATHS), -1, 2),
                status=np.asarray(status).reshape(len(pts), len(PATHS), -1),
                station=np.asarray([table[key][1] for key in PATHS]), pts=pts,
                paths=np.asarray(PATHS), state_sha256=sha(OUT / GUIDE / 'state.npz'))
    save(out, **data)
    return data


def image_curve(uv, status):
    # S14 sees a single monotone image branch. Reject invalid projection roots
    # near the horizon before constructing the search neighborhood.
    valid = (status == 0) & (uv[:, 0] > -180) & (uv[:, 0] < 2100)
    valid &= (uv[:, 1] > 210) & (uv[:, 1] < 910)
    poly = uv[valid]
    if len(poly) < 10:
        return np.full(len(X), np.nan)
    poly = poly[np.argsort(poly[:, 0])]
    return np.interp(X, poly[:, 0], poly[:, 1], left=np.nan, right=np.nan)


def normals(y):
    slope = np.gradient(y, X)
    normal = np.column_stack((-slope, np.ones(len(y))))
    return normal / np.linalg.norm(normal, axis=1, keepdims=True)


def profiles(raw, mask, uv, normal):
    tangent = np.column_stack((normal[:, 1], -normal[:, 0]))
    points = uv[:, None, :] + OFFSETS[None, :, None] * normal[:, None, :]
    grid = points[:, :, None, :] + np.arange(-6., 7., 3.)[None, None, :, None] * tangent[:, None, None, :]
    visible = (
        (grid[..., 0] > 1) & (grid[..., 0] < raw.shape[1] - 2)
        & (grid[..., 1] > 1) & (grid[..., 1] < raw.shape[0] - 2)
    )
    maps = np.nan_to_num(grid, nan=-5.).astype('f4').reshape(len(uv), -1, 2)
    colors = cv2.remap(raw, maps, None, cv2.INTER_LINEAR).reshape(*grid.shape[:-1], 3)
    occluded = cv2.remap(mask, maps, None, cv2.INTER_NEAREST).reshape(grid.shape[:-1]) > 0
    return np.median(colors, axis=2), (visible & ~occluded).all(axis=2)


def refined_peak(slope, k):
    curv = 2 * slope[k] - slope[k - 1] - slope[k + 1]
    return OFFSETS[k] + np.clip(
        (slope[k + 1] - slope[k - 1]) / max(2 * curv, 1.e-6), -.5, .5)


def fascia_edge(light, blue, visible, lower, depth):
    # On the head-on far side, the navy strip with fasteners belongs to
    # the vertical face. The navy-to-white step locates only its lower
    # margin; use it to inspect the separate upper margin.
    contrast_slope = -np.gradient(gaussian_filter1d(light, .78))
    rise = np.gradient(gaussian_filter1d(blue, .85))
    upper_strength = contrast_slope + .46 * rise
    peaks = (upper_strength > np.roll(upper_strength, 1)) & (upper_strength >= np.roll(upper_strength, -1))
    width = lower - np.arange(len(light))
    options = []
    for j in np.flatnonzero(peaks & (width >= 3) & (width <= min(22., max(4., depth * .083)))):
        if not visible[j - 5:lower + 2].all():
            continue
        above = slice(j - 5, j - 2)
        strip = slice(j + 1, lower)
        dark = light[strip].mean()
        blue_in, blue_out = blue[strip].mean(), blue[above].mean()
        step = light[above].mean() - dark
        if (dark > 163 or light[above].mean() < 134 or step < 19
                or blue_in < 9 or blue_in - blue_out < 6 or abs(blue_out) > 17
                or upper_strength[j] < 5.5 or np.ptp(light[above]) > 20):
            continue
        options.append((upper_strength[j], j, min(step, 75.)))
    if not options:
        return None
    strength, j, step = max(options)
    if any(score > .62 * strength and abs(alt - j) > 1.5 for score, alt, _ in options):
        return None
    return refined_peak(upper_strength, j), step


def read_edge(raw, mask, y, depth, top, wide_pan):
    uv = np.column_stack((X, y))
    normal = normals(y)
    colors, visible = profiles(raw, mask, uv, normal)
    b, g, r = np.moveaxis(colors, -1, 0)
    light = .114 * b + .587 * g + .299 * r
    signal = b - r if top else .5 * (r + g) - b
    smooth = gaussian_filter1d(light if top else -signal, .85, axis=1)
    slope = np.gradient(smooth, axis=1)
    profile_y = np.full_like(y, np.nan)
    strength = np.zeros(len(y))
    material = np.zeros(len(y), dtype='u1')
    for i in range(len(y)):
        if not np.isfinite(y[i] + depth[i]) or depth[i] < 18:
            continue
        span = int(np.clip(depth[i] * .074, 3, 9))
        if top:
            limit = (OFFSETS >= -8.) & (OFFSETS <= (19. if depth[i] > 60 else 12.))
        else:
            limit = (OFFSETS >= -9.) & (OFFSETS <= (24. if depth[i] > 70 else 13.))
        peaks = (slope[i] > np.roll(slope[i], 1)) & (slope[i] >= np.roll(slope[i], -1))
        options = []
        for k in np.flatnonzero(peaks & limit):
            if not visible[i, k - span - 3:k + span + 4].all():
                continue
            if top:
                # Blue/dark cap on the inward cut and white immediately on
                # the board face. The back silhouette has the opposite order.
                before = slice(k - 4, k - 1)
                after = slice(k + 1, k + 4)
                dark, white = light[i, before].mean(), light[i, after].mean()
                blue_in, blue_out = signal[i, before].mean(), signal[i, after].mean()
                contrast = white - dark
                if (dark > 178 or white < 139 or contrast < 29 or blue_in < 6.
                        or blue_out > max(16., blue_in * .84) or slope[i, k] < 7.):
                    continue
            else:
                # Warm kickplate above a fall to the neutral ice. Require a
                # warm cap on the actual plate, plus a neutral ice shoulder.
                before = slice(k - span - 2, k - 1)
                after = slice(k + 2, k + span + 3)
                warm, cold = signal[i, before].mean(), signal[i, after].mean()
                contrast = warm - cold
                if (warm < 8.3 or abs(cold) > 10 or contrast < 5.4
                        or slope[i, k] < .95 or light[i, after].mean() < 106):
                    continue
                if np.max(signal[i, before]) < 10.4:
                    continue
            # Score the material step using the camera window only as a
            # weak spatial guide. Persist the actual location from RGB.
            score = min(contrast, 72.) / (1. + .06 * abs(OFFSETS[k]))
            options.append((score, k, contrast))
        if options:
            best, k, contrast = max(options)
            if any(score > .73 * best and abs(alt - k) > 2.5 for score, alt, _ in options):
                continue
            # The source pan reveals the exposed top surface along the
            # diagonal judges' board, with the lip at the lower end of that
            # surface. The thin horizontal fascia requires its upper edge.
            near_lip = wide_pan and normal[i, 0] < -.097 and depth[i] > 69.
            if top and not near_lip:
                if abs(normal[i, 0]) > .072:
                    continue
                fascia = fascia_edge(light[i], signal[i], visible[i], k, depth[i])
                if fascia is None:
                    continue
                profile_y[i], contrast = fascia
                material[i] = 1
            else:
                profile_y[i] = refined_peak(slope[i], k)
                material[i] = 2 if top else 3
            strength[i] = contrast
    return uv + profile_y[:, None] * normal, strength, material


def coherent(hits, guide, material):
    # Keep local measurements that agree with other edges on their own
    # exposure and have neighboring source exposures supporting the same run.
    # Never impute missing UVs from the neighbors or guide.
    valid = np.isfinite(hits[..., 1])
    delta = hits[..., 1] - guide
    agree = np.zeros_like(delta, dtype=np.int16)
    for dx in (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6):
        threshold = 1.35 + abs(dx) * .22
        match = np.abs(delta - np.roll(delta, dx, axis=-1)) < threshold
        match &= material == np.roll(material, dx, axis=-1)
        match[..., :dx if dx > 0 else 0] = False
        if dx < 0:
            match[..., dx:] = False
        agree += match
    valid &= agree >= 3
    spatial = np.where(valid, delta, np.nan)
    support = np.zeros_like(agree)
    for dt in (-2, -1, 1, 2):
        for dx in (-1, 0, 1):
            match = np.abs(spatial - np.roll(spatial, (dt, dx), axis=(0, 2))) < 3.4
            match &= material == np.roll(material, (dt, dx), axis=(0, 2))
            if dt > 0:
                match[:dt] = False
            else:
                match[dt:] = False
            if dx > 0:
                match[..., :dx] = False
            elif dx < 0:
                match[..., dx:] = False
            support += match
    # Source corner changes rapidly during the wide pan. Permit some nearby
    # tangential drift but require at least two supporting cuts.
    return valid & (support >= 2)


def sweep(shot, guide_data, out):
    pts = guide_data['pts']
    guide = np.array([[image_curve(uv, status) for uv, status in zip(uvs, codes)]
                      for uvs, codes in zip(guide_data['uv'], guide_data['status'])])
    hits = np.full((*guide.shape, 2), np.nan)
    quality = np.zeros_like(guide)
    material = np.zeros_like(guide, dtype='u1')
    for i, (pt, raw) in enumerate(source_frames(shot['source'], pts)):
        mask = foreground(shot, None, i, pt)
        if mask is None:
            continue
        mask = cv2.dilate(mask, np.ones((19, 19), 'u1'))
        mask[:270, :411] = 255
        mask[936:, 1620:] = 255
        depth = guide[i, 1] - guide[i, 0]
        for k in range(len(PATHS)):
            hits[i, k], quality[i, k], material[i, k] = read_edge(
                raw, mask, guide[i, k], depth, k == 0, pt >= 1580544)
    support = coherent(hits, guide, material)
    save(out / 'profile_sweep.npz', hits=hits, strength=quality, accepted=support, material=material, guide_y=guide,
         pts=pts, paths=np.asarray(PATHS), source_sha256=shot['source_sha256'])
    return hits, quality, support, material


def source_distance(points, source):
    distance = np.full(len(points), np.nan)
    if len(source) < 2:
        return distance
    source = source[np.argsort(source[:, 0])]
    after = np.searchsorted(source[:, 0], points[:, 0]).clip(1, len(source) - 1)
    lo, hi = source[after - 1], source[after]
    delta = hi - lo
    gap = delta[:, 0]
    local = (gap > 0.) & (gap < 21.)
    local &= (points[:, 0] > lo[:, 0] - 3.5) & (points[:, 0] < hi[:, 0] + 3.5)
    distance[local] = (
        delta[local, 0] * (points[local, 1] - lo[local, 1])
        - delta[local, 1] * (points[local, 0] - lo[local, 0])
    ) / np.linalg.norm(delta[local], axis=1)
    return distance


def audit_packet(path, out, pts, hits, accepted):
    packet = arrays(path)
    is_rail = np.isin(packet['path'], PATHS) & (packet['fold'] < 9)
    old_rows = np.flatnonzero(is_rail)
    np.testing.assert_array_equal(packet['pts'][old_rows], pts[packet['idx'][old_rows]])
    distance = np.full(len(is_rail), np.nan)
    confirmed = np.zeros_like(is_rail)
    for k, key in enumerate(PATHS):
        rows = old_rows[packet['path'][old_rows] == key]
        for i in np.unique(packet['idx'][rows]):
            selected = rows[packet['idx'][rows] == i]
            d = source_distance(packet['uv'][selected], hits[i, k, accepted[i, k]])
            distance[selected] = d
            confirmed[selected] = np.isfinite(d) & (abs(d) <= (1.35 if k == 0 else 2.05))
    retire = old_rows[~confirmed[old_rows]]
    reason = np.full(len(retire), 'board_boundary_not_resolved_in_original', dtype='<U55')
    supported = np.isfinite(distance[retire])
    reason[supported] = 'different_source_edge_than_ice_facing_board_boundary'
    save(out, rows=retire, previous=packet['fold'][retire],
         current=np.full(len(retire), 9, 'i1'), reason=reason,
         signed_distance=distance[retire], kept_rows=old_rows[confirmed[old_rows]],
         old_path=str(path), old_sha256=sha(path), pts=packet['pts'][retire], uv=packet['uv'][retire],
         entity=packet['path'][retire], source_packet=packet['source_packet'][retire],
         source_row=packet['source_row'][retire])
    return packet, confirmed, distance


def assemble(shot, guide, hits, quality, accepted, material, old, confirmed):
    data = {key: [] for key in (
        'idx', 'pts', 'uv', 'path', 'station', 'station_fixed', 'family', 'material',
        'fold', 'normal', 'contrast', 'correlation_group', 'admission_reason', 'uncertainty_px'
    )}
    for i, pt in enumerate(guide['pts']):
        # Exact reported frames and recurring time blocks are developed checks.
        hold = pt in REPORT_PTS or (i // 16) % 7 == 4
        for k, path in enumerate(PATHS):
            indices = np.flatnonzero(accepted[i, k])
            if not len(indices):
                continue
            uv = hits[i, k, indices]
            old_row = (old['pts'] == pt) & (old['path'] == path) & (old['fold'] < 5) & confirmed
            if old_row.any() and not hold:
                unused = np.min(abs(uv[:, None, 0] - old['uv'][None, old_row, 0]), axis=1) >= 17.
                indices, uv = indices[unused], uv[unused]
            if not len(uv):
                continue
            # Spatially thin the new correlated samples. Source sweeps keep
            # the dense read separately for reviewing original observations.
            bins = np.floor(uv[:, 0] / 32.).astype(int)
            picks = [np.flatnonzero(bins == key)[np.argmax(quality[i, k, indices[bins == key]])]
                     for key in np.unique(bins)]
            indices, uv = indices[picks], uv[picks]
            proj = guide['uv'][i, k].copy()
            proj[guide['status'][i, k] != 0] = np.nan
            station, dist = localize(proj, guide['station'][k], uv)
            normals_at_source = normals(image_curve(guide['uv'][i, k], guide['status'][i, k]))
            for j in np.flatnonzero(np.isfinite(station) & (dist < 37.)):
                code = material[i, k, indices[j]]
                row = dict(idx=i, pts=pt, uv=uv[j], path=path, station=station[j], station_fixed=False,
                           family=FAMILIES[k], material=MATERIALS[code], fold=6 if hold else 0,
                           normal=normals_at_source[indices[j]], contrast=quality[i, k, indices[j]],
                           uncertainty_px={1: 1.7, 2: 1.5, 3: 2.2}[code],
                           correlation_group=f'{SHOT}:{FAMILIES[k]}:{i // 16}',
                           admission_reason='source_verified_ice_facing_board_boundary')
                for key, value in row.items():
                    data[key].append(value)
    packet = {key: np.asarray(rows) for key, rows in data.items()}
    packet['fold'] = packet['fold'].astype('i1')
    packet['source_fold'] = packet['fold'].copy()
    packet['source_row'] = np.arange(len(packet['pts']), dtype=np.int64)
    packet['source_sha256'] = shot['source_sha256']
    packet['measurement_source'] = shot['source']
    packet['time_base'] = np.array([1, 12800], dtype=np.int64)
    return packet


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT / 's14_boards_1118' / 'rail')
    parser.add_argument('--reuse-profiles', action='store_true')
    args = parser.parse_args()
    output = args.output
    cv2.setNumThreads(2)
    shot = read(OUT / GUIDE / 'setup.json')['shots'][SHOT]
    state = camera_state(GUIDE)
    if sha(shot['source']) != shot['source_sha256']:
        raise ValueError('Original movie checksum mismatch')
    path = output / 'projection_guide.npz'
    guide = arrays(path) if path.is_file() else projection(state, shot, path)
    if guide['state_sha256'].item() != sha(OUT / GUIDE / 'state.npz'):
        raise ValueError('Projection guide checkpoint mismatch')
    pts = guide['pts']
    np.testing.assert_array_equal(pts, state[SHOT + '_pts'])
    if args.reuse_profiles:
        cached = arrays(output / 'profile_sweep.npz')
        np.testing.assert_array_equal(cached['pts'], pts)
        if cached['source_sha256'].item() != shot['source_sha256']:
            raise ValueError('Profile sweep source mismatch')
        hits, quality, accepted, material = (cached[key] for key in ('hits', 'strength', 'accepted', 'material'))
    else:
        hits, quality, accepted, material = sweep(shot, guide, output)
    for name in ('verified_inputs', 'verified'):
        parent = OUT / 'lane4_side_1118' / name / f'{SHOT}.npz'
        old, confirmed, distance = audit_packet(parent, output / f'retire_{name}.npz', pts, hits, accepted)
    packet = assemble(shot, guide, hits, quality, accepted, material, old, confirmed)
    save(output / f'{SHOT}.npz', **packet)
    report = dict(shot=SHOT, source=shot['source'], source_sha256=shot['source_sha256'],
                  pts_range=[int(pts[0]), int(pts[-1])], inspected_exposures=len(pts),
                  source_points=dict(), reported_exposures=[], old_sources=[])
    report['physical_target'] = {
        'top_far': 'Upper margin of the navy vertical fascia where distinctly resolved, judged from the fascia and fasteners on the source and its separate lower navy-to-white margin. Omit optically ambiguous upper bands, background contamination and isolated hardware.',
        'top_near': 'Front termination of the exposed top cap along the sloped judges-side run during the wide pan, at the junction with the ice-facing vertical face. Exclude the cap rear, glass, tables and hardware.',
        'bottom': 'Ice-facing base of the yellow kickplate at the ice contact. Require warm plate and neutral ice shoulders that survive spatial and temporal support. Omit where those shoulders merge with reflections, print or foreground.',
        'hold': 'Developed source checks. Neither adjacent training exposures nor older samples are independent of these checks.',
    }
    for k, family in enumerate(FAMILIES):
        q = packet['family'] == family
        report['source_points'][family] = dict(
            fit=int(np.count_nonzero(q & (packet['fold'] == 0))),
            check=int(np.count_nonzero(q & (packet['fold'] == 6))),
            any_reads_exposures=int(np.count_nonzero(accepted[:, k].any(axis=1))),
            no_reads_exposures=int(np.count_nonzero(~accepted[:, k].any(axis=1))))
    for pt in REPORT_PTS:
        obs = old['pts'] == pt
        item = dict(pts=pt, seconds=pt / 12800, families={})
        for family, key in zip(FAMILIES, PATHS):
            select = obs & (old['path'] == key) & (old['fold'] < 9)
            q = (packet['pts'] == pt) & (packet['family'] == family)
            item['families'][family] = dict(old=int(select.sum()),
                kept=int((select & confirmed).sum()), old_unresolved=int((select & ~np.isfinite(distance)).sum()),
                fit=int(np.count_nonzero(q & (packet['fold'] == 0))),
                check=int(np.count_nonzero(q & (packet['fold'] == 6))))
        report['reported_exposures'].append(item)
    rail = np.isin(old['path'], PATHS) & (old['fold'] < 9)
    for code in np.unique(old['source_packet'][rail]):
        for material in np.unique(old['material'][rail & (old['source_packet'] == code)]):
            q = rail & (old['source_packet'] == code) & (old['material'] == material)
            report['old_sources'].append(dict(code=int(code), material=str(material),
                source_binding=shot['source_packets'][str(code)],
                checked=int(q.sum()), kept=int((q & confirmed).sum()),
                different_edge=int((q & ~confirmed & np.isfinite(distance)).sum()),
                unresolved=int((q & ~np.isfinite(distance)).sum())))
    write(output / 'rail_audit.json', report)
    print(report['source_points'], report['reported_exposures'], flush=True)


if __name__ == '__main__':
    main()
