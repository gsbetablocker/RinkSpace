"""Audit all source exposures and compare native motion to actual static texture."""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np

from board_displacement_1118 import TARGET
from flow_depths import bind
from lane_side_1118 import BASE
from motion_tracks import packet
from repair_motion import frame_median, project, stat
from review import native_for
from shared import OUT, PREPARED, PROGRAMMES, arrays, read, save, sha, write


COORDINATES = ('pan', 'tilt', 'log_f', 'fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2')
FLOW_METRICS = tuple(k+'_'+m for k in ('board', 'ice') for m in ('adjacent', 'lag8', 'jitter'))
CASES = {'jin_s04': (42.16, 42.18, 42.20),
         'jin_s08': (71.78, 71.82, 71.84, 71.86, 71.9, 72.04),
         'jin_s09': (72.80, 72.82, 72.84, 72.86, 72.88),
         'jin_s10': (84.14, 84.16, 84.18),
         'jin_s14': (122.98, 123., 123.02, 123.04, 123.06),
         'jin_s15': (127.32, 127.34, 127.36, 127.38, 127.54, 127.56, 127.6),
         'jin_s16': (129.90, 129.92, 129.94),
         'jin_s19': (143.24, 143.26, 143.28),
         'jin_s22': (155.04, 155.06, 155.08),
         'chen_s12': (113.28, 113.3, 113.32, 113.34, 113.36),
         'chen_s20': (197.00, 197.02, 197.08, 197.14, 197.16, 197.82, 197.90, 197.92, 197.94, 198.12),
         'nayeon_s03': (52.18, 52.22, 52.32, 52.38, 52.46, 52.58, 52.60, 52.62, 52.64, 52.70, 52.80, 52.90),
         'nayeon_s06': (122.08, 122.1, 122.12, 122.14, 122.16),
         'okada_s07': (113.9, 113.92, 113.94, 113.96, 114.12)}
JIN_TEXTURE_PEAKS = ('jin_s04', 'jin_s09', 'jin_s10', 'jin_s16', 'jin_s19')


def point(name, pts, i, value):
    return dict(shot=name, pts=int(pts[i]), seconds=float(pts[i]/12800), value=float(value))


def ranks(name, pts, values, order='max', count=4):
    rows = np.flatnonzero(np.isfinite(values))
    rows = rows[np.argsort(values[rows] if order == 'min' else -values[rows], kind='stable')[:count]]
    return [point(name, pts, i, values[i]) for i in rows]


def gaps(pts, invalid):
    rows = np.flatnonzero(invalid)
    out = []
    for group in np.split(rows, np.flatnonzero(np.diff(rows) != 1)+1):
        if len(group):
            pair = pts[group[[0, -1]]]
            out.append(dict(exposures=len(group), pts=pair.tolist(), seconds=(pair/12800.).tolist()))
    return out


def state_bound(state, name, pts):
    required = [name+'_'+k for k in ('pts', 'rigs', 'controls', 'times')]
    if not all(k in state for k in required):
        return False
    times = state[name+'_times']
    return (np.array_equal(state[name+'_pts'], pts)
            and times.shape == pts.shape
            and np.isfinite(times).all() and (np.diff(times) > 0.).all()
            and len(state[name+'_rigs']) == len(pts))


def trajectory(native, shot, name, state, pts):
    data = {k: np.full(len(pts), np.nan) for k in COORDINATES}
    rotation = np.full((len(pts), 3, 3), np.nan)
    centre = np.full((len(pts), 3), np.nan)
    status = np.full(len(pts), -1, dtype='i4')
    exposure = np.full(len(pts), np.nan)
    if state_bound(state, name, pts):
        exposure = state[name+'_times']
        unique, inv = np.unique(state[name+'_rigs'], axis=0, return_inverse=True)
        for i, rig in enumerate(unique):
            use = inv == i
            result = native.core.evaluate_motion_camera(rig, state[name+'_controls'],
                shot['first'], shot['dt'], exposure[use])
            status[use] = result['status']
            for k in COORDINATES:
                data[k][use] = result[k]
            rotation[use] = result['rotation']
            centre[use] = result['centre']
    finite = np.isfinite(np.column_stack(tuple(data.values()))).all(1)
    valid = ((status == 0) & finite & (data['fx'] > 0.) & (data['fy'] > 0.)
        & np.isfinite(rotation).all((1, 2)) & np.isfinite(centre).all(1))
    for k in COORDINATES:
        data[k][~valid] = np.nan
    rotation[~valid] = np.nan
    centre[~valid] = np.nan
    dt = np.diff(exposure)
    data['rotation_deg_s'] = np.r_[np.nan, np.degrees(np.arccos(
        np.clip((np.sum(rotation[1:]*rotation[:-1], axis=(1, 2))-1.)/2., -1., 1.)))/dt]
    data['centre_L_s'] = np.r_[np.nan, np.linalg.norm(np.diff(centre, axis=0), axis=1)/dt]
    for key in ('pan', 'tilt', 'log_f'):
        delta = np.diff(data[key])
        if key == 'pan':
            delta = (delta+np.pi) % (2*np.pi)-np.pi
        speed = delta/dt
        data[key+'_rate'] = np.r_[np.nan, abs(speed)]
        data[key+'_acceleration'] = np.r_[[np.nan]*min(2, len(pts)), abs(np.diff(speed))/((dt[1:]+dt[:-1])/2)]
    data.update(valid_camera=valid, camera_status=status, centre_L=centre, local_exposure_seconds=exposure)
    return data


def active_counts(name, shot, state, pts, bindings, prepared):
    if name in bindings:
        ref = bindings[name]['packet']
        path = Path(ref['path'])
        source_hash = sha(path)
        if ref.get('sha256', source_hash) != source_hash:
            raise ValueError(('Changed bound observation packet', name, path))
        origin = 'state source bindings'
    elif prepared is not None:
        path = Path(prepared[name]['observations'])
        source_hash = sha(path)
        origin = 'explicit candidate prepared setup'
    else:
        return np.full(len(pts), -1), {'reason': 'No candidate observation binding'}
    fold = state.get(name+'_active_folds')
    if fold is None:
        return np.full(len(pts), -1), {'reason': 'Active observation folds unavailable'}
    with np.load(path) as a:
        index = a['idx']
        if not np.array_equal(a['pts'], pts[index]):
            raise ValueError(('Observation PTS mismatch', name, path))
        if 'source_sha256' in a and a['source_sha256'].item() != shot['source_sha256']:
            raise ValueError(('Observation source mismatch', name, path))
        active = np.isin(a['fold'], fold)
    return np.bincount(index[active], minlength=len(pts)), dict(path=str(path), sha256=source_hash, origin=origin)


def score(native, shot, state, name, pts, source, rows):
    end = source['index'][rows]
    left = end-source['lag'][rows]
    uv = source['left_uv'][rows]
    disp = np.full((len(rows), 2), np.nan)
    curve = disp.copy()
    depth = np.full(len(rows), np.nan)
    if not state_bound(state, name, pts):
        return disp, curve, depth
    world, _ = bind(native, shot, state[name+'_rigs'], state[name+'_controls'],
        state[name+'_times'], left, uv, source['family'][rows])
    valid = np.flatnonzero(np.isfinite(world).all(1))
    prev = project(native, shot, state, name, left[valid], world[valid])
    new = project(native, shot, state, name, end[valid], world[valid])
    disp[valid] = (new-prev)-(source['uv'][rows[valid]]-uv[valid])
    if 'prev2_uv' in source:
        earlier_uv = source['prev2_uv'][rows[valid]]
        triple = ((source['lag'][rows[valid]] == 1) & (end[valid] >= 2)
            & np.isfinite(earlier_uv).all(1))
        ids = valid[triple]
        earlier = project(native, shot, state, name, end[ids]-2, world[ids])
        curve[ids] = (new[triple]-2*prev[triple]+earlier)-(
            source['uv'][rows[ids]]-2*uv[ids]+earlier_uv[triple])
    rig, inv = np.unique(state[name+'_rigs'][left[valid]], axis=0, return_inverse=True)
    for i, value in enumerate(rig):
        ix = valid[inv == i]
        row_time = state[name+'_times'][left[ix]]+value[37]*(uv[ix, 1]/1079.-.5)
        camera = native.core.evaluate_motion_camera(value, state[name+'_controls'],
            shot['first'], shot['dt'], row_time)
        depth[ix] = np.linalg.norm(world[ix]-camera['centre'], axis=1)
    return disp, curve, depth


def unchanged(board_equal, baseline, state, name):
    return board_equal and all(
        name+'_'+k in state and np.array_equal(baseline[name+'_'+k], state[name+'_'+k])
        for k in ('pts', 'times', 'rigs', 'controls'))


def summaries(name, pts, data):
    metrics = ('fx', 'rotation_deg_s', 'centre_L_s', 'pan_rate', 'pan_acceleration', 'tilt_rate',
        'tilt_acceleration', 'log_f_rate', 'log_f_acceleration', 'source_visible_step')+FLOW_METRICS
    out = {m: dict(**stat(data[m]), worst=ranks(name, pts, data[m])) for m in metrics}
    out['fx']['lowest'] = ranks(name, pts, data['fx'], 'min')
    known = data['fit_rows'] >= 0
    out['coverage'] = dict(exposures=len(pts), camera_valid=int(data['valid_camera'].sum()),
        invalid_camera_ranges=gaps(pts, ~data['valid_camera']),
        missing_local_observations=int((data['fit_rows'] == 0).sum()),
        observation_binding_unknown=int((~known).sum()),
        missing_local_observation_ranges=gaps(pts, data['fit_rows'] == 0),
        adjacent_source_with_board_or_ice=int((data['source_rows'] > 0).sum()),
        adjacent_intervals_without_source=int((data['source_rows'][1:] == 0).sum()),
        adjacent_intervals_without_source_ranges=gaps(pts, (data['source_rows'] == 0) & (np.arange(len(pts)) != 0)),
        adjacent_intervals_without_finite_comparison=int((~np.isfinite(data['board_adjacent'][1:])
            & ~np.isfinite(data['ice_adjacent'][1:])).sum()))
    return out


def flow_frames(data, pts, index, family, lag, measured, displacement, curve):
    n = len(pts)
    data['source_visible_step'] = frame_median(index, np.linalg.norm(measured, axis=1), lag == 1, n, 3)
    data['source_rows'] = np.bincount(index[lag == 1], minlength=n)
    for surface, material, minimum in [(0, 'board', 4), (1, 'ice', 3)]:
        for suffix, delta, vectors in [('adjacent', 1, displacement), ('lag8', 8, displacement), ('jitter', 1, curve)]:
            data[material+'_'+suffix] = frame_median(index, np.linalg.norm(vectors, axis=1),
                (family == surface) & (lag == delta), n, minimum)


def window_metrics(name, pts, index, lag, scores, candidates, minimum):
    errors = [np.linalg.norm(score[0], axis=1) for score in scores]
    common = np.logical_and.reduce([np.isfinite(error) for error in errors])
    report = {}
    for delta in np.unique(lag):
        select = common & (lag == delta)
        curves = [frame_median(index, error, select, len(pts), minimum) for error in errors]
        result = {c: dict(**stat(z), worst=ranks(name, pts, z))
                  for c, z in zip(candidates, curves)}
        if len(curves) == 2:
            result['largest_increases_px'] = ranks(name, pts, curves[1]-curves[0])
        report[str(delta)] = result
    return report


def focus(shot, name, pts, candidates, states, natives, target):
    report = {}
    review_path = TARGET/'manual_exclusions.json'
    reviews = read(review_path) if review_path.is_file() else {}
    source_root = TARGET if shot['program'] == 'jin' and name not in JIN_TEXTURE_PEAKS else TARGET/'programme_probes'
    sources = [(source_root, 'board', CASES[name]), (source_root/'ice', 'ice', CASES[name])]
    if name == 'jin_s14':
        seams = (122.16, 122.28, 122.32, 122.34, 122.42, 122.5, 122.56, 122.58, 122.6, 122.82, 122.84, 122.86,
                 123.30, 123.32, 123.34, 123.36, 123.38, 123.46, 123.6, 123.98)
        early = (121.22, 121.44, 121.46, 121.5, 121.60, 121.62, 121.64, 121.66,
                 121.90, 121.96, 122.02, 122.04, 122.08, 122.10, 122.12, 122.14)
        bridge = (120.08, 120.10, 120.16, 120.34, 120.70, 121.06, 121.16, 121.20)
        sources.extend([(TARGET/'s14_extended', 'board_extended', seams),
                        (TARGET/'s14_extended/ice', 'ice_extended', seams),
                        (TARGET/'s14_early', 'board_early', early),
                        (TARGET/'s14_early/ice', 'ice_early', early),
                        (TARGET/'s14_bridge', 'board_bridge', bridge),
                        (TARGET/'s14_bridge/ice', 'ice_bridge', bridge)])
    if name == 'jin_s22':
        onset = (154.64, 154.66, 154.68, 154.70, 154.72, 154.74, 154.76)
        sources.extend([(TARGET/'onset_probe', 'board_onset', onset),
                        (TARGET/'onset_probe/ice', 'ice_onset', onset)])
    if name in ('jin_s08','jin_s15','jin_s22'):
        checks = CASES[name] if name != 'jin_s22' else (154.34,154.36,154.38,154.4,154.42,154.44)
        sources.extend([(TARGET/'guard_boundaries', 'board_boundary', checks),
                        (TARGET/'guard_boundaries/ice', 'ice_boundary', checks)])
    if name == 'jin_s08':
        extended = (71.6, 71.62, *CASES[name], 72.12, 72.16)
        sources.extend([(TARGET/'guard_s08_extended', 'board_boundary_extended', extended),
                        (TARGET/'guard_s08_extended/ice', 'ice_boundary_extended', extended)])
    if name == 'jin_s15':
        extended = (126.84, 126.98, 127.06, 127.12, 127.22, 127.28, *CASES[name])
        sources.append((TARGET/'guard_s15_extended', 'board_boundary_extended', extended))
    if name == 'chen_s20':
        sources = [(TARGET/'chen_s20_full', 'board_full', CASES[name])]
        sources.append((TARGET/'chen_s20_registration/qualified', 'board_qualified', CASES[name]))
        sources.append((TARGET/'chen_s20_registration/glyphs', 'board_glyphs', CASES[name]))
    if name == 'okada_s07':
        extended = (113.48, 113.6, 113.78, *CASES[name], 114.24, 114.4, 114.54)
        sources.append((TARGET/'okada_s07_extended', 'board_extended', extended))
    if name == 'nayeon_s03':
        extended = (*CASES[name], 53.04, 53.16, 53.20, 53.24, 53.28, 53.44)
        early_root = TARGET.parent/'nayeon_late/nayeon_s03/opposite_plate_motion'
        early = (51.60, 51.62, 51.68, 51.76, 51.82, 51.86, 51.92, 51.94, 51.96, 52.16, 52.38, 52.54)
        sources.extend([(TARGET/'nayeon_s03_extended', 'board_extended', extended),
                        (TARGET/'nayeon_s03_extended/ice', 'ice_extended', extended),
                        (early_root, 'board_opposite_plate', early),
                        (early_root/'ice', 'ice_opposite_plate', early)])
    if name == 'nayeon_s06':
        late = (124.12, 124.24, 124.36, 124.44, 125.32, 125.34, 125.38, 125.44, 125.56, 125.60, 125.68)
        tail = (125.60, 125.68, 125.80, 125.88, 125.94)
        sources.extend([(TARGET/'nayeon_s06_late', 'board_late', late),
                        (TARGET/'nayeon_s06_late/ice', 'ice_late', late),
                        (TARGET/'n06_local/tail_source', 'board_tail', tail),
                        (TARGET/'n06_local/tail_source/ice', 'ice_tail', tail)])
    for surface_path, surface, timestamps in sources:
        path = surface_path/(name+'.npz')
        if not path.is_file():
            continue
        source = arrays(path)
        assert source['source_sha256'].item() == shot['source_sha256']
        np.testing.assert_array_equal(source['pts'], pts)
        keep = source['keep'].copy()
        review = reviews.get(path.relative_to(TARGET).as_posix()) if path.is_relative_to(TARGET) else None
        if review:
            if review['packet_sha256'] != sha(path):
                raise ValueError(('Changed reviewed motion packet', path))
            keep[review['excluded_rows']] = False
        rows = np.flatnonzero(keep)
        index = source['index'][rows]
        pairs = np.c_[pts[index-source['lag'][rows]], pts[index]]
        np.testing.assert_array_equal(pairs, source['source_pair_pts'][rows])
        left, right = source['left_uv'][rows], source['uv'][rows]
        actual = right-left
        scores = [score(native, shot, state, name, pts, source, rows)
                  for native, state in zip(natives, states)]
        consistent_depth = scores[0][2]
        holds = source['held'][rows]
        result = []
        lags = np.unique(source['lag'][rows])
        for at in timestamps:
            for lag in lags:
                take = (pairs[:, 1] == round(at*12800)) & (source['lag'][rows] == lag)
                if not take.any():
                    continue
                rec = dict(pts=pairs[np.flatnonzero(take)[0]].tolist(), seconds=at, lag=int(lag))
                zones = {'all': take, 'held_strips': take & holds}
                for key, limits in dict(left=(0, 640), middle=(640, 1280), right=(1280, 1920)).items():
                    zones[key] = take & (left[:, 0] >= limits[0]) & (left[:, 0] < limits[1])
                if surface.startswith('ice'):
                    zones.update(deeper_ice=take & (left[:, 1] < 600),
                                 foreground_ice=take & (left[:, 1] >= 730))
                rec['zones'] = {}
                for zone, mask in zones.items():
                    if not mask.any():
                        continue
                    distance = consistent_depth[mask & np.isfinite(consistent_depth)]
                    common = mask & np.logical_and.reduce([np.isfinite(z[0]).all(1) for z in scores])
                    info = dict(source_rows=int(mask.sum()), source_median_xy_px=np.median(actual[mask], axis=0).tolist(),
                        x_range_px=[float(left[mask, 0].min()), float(left[mask, 0].max())],
                        conditional_depth_L=float(np.median(distance)) if len(distance) else None,
                        common_rows=int(common.sum()))
                    for candidate, (err, _, _) in zip(candidates, scores):
                        delta = err[common]
                        error = stat(np.linalg.norm(delta, axis=1))
                        error.pop('frames')
                        info[candidate] = dict(error_px=error,
                            predicted_median_xy_px=np.median(actual[common]+delta, axis=0).tolist() if len(delta) else None)
                    rec['zones'][zone] = info
                result.append(rec)
        entries = {'source_rows': rows, 'pts': pts, 'source_pair_pts': pairs, 'actual': actual,
            'left_uv': left, 'uv': right, 'held': holds, 'lag': source['lag'][rows]}
        for i, (error, _, depth) in enumerate(scores):
            entries['displacement_error_'+str(i)] = error
            entries['conditional_depth_'+str(i)] = depth
        save(target/(name+'_'+surface+'_focus.npz'), **entries)
        adjacent = source['lag'][rows] == 1
        minimum = 3 if surface.startswith('ice') else 4
        finite_adjacent = {}
        for candidate, (error, _, _) in zip(candidates, scores):
            valid = adjacent & np.isfinite(error).all(1)
            support = np.bincount(index[valid], minlength=len(pts))
            finite_adjacent[candidate] = pts[support >= minimum].tolist()
        report[surface] = dict(packet=str(path), packet_sha256=sha(path),
            source_right_pts=np.unique(pairs[adjacent, 1]).tolist(), source_review=review, comparisons=result,
            finite_adjacent_right_pts=finite_adjacent,
            window_frame_medians=window_metrics(name, pts, index, source['lag'][rows], scores,
                candidates, minimum))
    return report


def pool(programme, candidate, report, frames, metadata):
    shots = [(k, pts, results[candidate]) for k, (pts, results) in frames.items()
             if metadata[k]['program'] == programme]
    if not shots:
        return {}
    result = {}
    for key in report[shots[0][0]][candidate]:
        if key in ('shared_track_rows_with_invalid_projection', 'fit_observation_binding', 'source_clock'):
            continue
        if key == 'coverage':
            coverage = [report[name][candidate][key] for name, _, _ in shots]
            result[key] = {k: sum(v[k] for v in coverage) for k in coverage[0] if isinstance(coverage[0][k], int)}
            continue
        result[key] = dict(**stat(np.concatenate([a[key] for _, _, a in shots])),
            worst=sorted((q for name, pts, a in shots for q in ranks(name, pts, a[key])),
                         key=lambda k: -k['value'])[:6])
        if key == 'fx':
            result[key]['lowest'] = sorted((q for name, pts, a in shots for q in ranks(name, pts, a[key], 'min')),
                                         key=lambda k: k['value'])[:6]
    return result


def common_flow_report(names, pairs):
    result = {}
    for key in FLOW_METRICS:
        before, after, increases = [], [], []
        for name in names:
            pts, old, new = pairs[name]
            before.append(old[key])
            after.append(new[key])
            increases.extend(ranks(name, pts, new[key]-old[key]))
        result[key] = dict(before=stat(np.concatenate(before)), after=stat(np.concatenate(after)),
            largest_changes=sorted(increases, key=lambda q: -q['value'])[:6])
    return result


def focal_probe(name, shot, pts, candidates, states, natives, manifest):
    path = TARGET/'peak_focal'/(name+'.npz')
    source = arrays(path)
    np.testing.assert_array_equal(source['pts'], pts)
    np.testing.assert_array_equal(source['source_pair_pts'], manifest['pts'])
    assert source['source_sha256'].item() == shot['source_sha256']
    assert manifest['source_sha256'] == shot['source_sha256']
    rows = np.arange(len(source['index']))
    scores = [score(native, shot, state, name, pts, source, rows)
              for native, state in zip(natives, states)]
    common = np.logical_and.reduce([np.isfinite(s[0]).all(1) for s in scores])
    actual = source['uv']-source['left_uv']
    result = {}
    for candidate, (error, _, _) in zip(candidates, scores):
        s = stat(np.linalg.norm(error[common], axis=1))
        s.pop('frames')
        result[candidate] = dict(error_px=s, invalid_projection_rows=int((~np.isfinite(error).all(1)).sum()))
    return dict(path=str(path), sha256=sha(path), pts=manifest['pts'], roi_xyxy=manifest['roi_xyxy'],
                source=shot['source'], source_sha256=shot['source_sha256'],
                source_rows=len(rows), paired_rows=int(common.sum()),
                source_median_xy_px=np.median(actual[common], axis=0).tolist() if common.any() else None,
                comparison=result)


def write_overview(target, candidates, hashes, programmes, shots, comparisons):
    before, after = candidates
    names = dict(jin='金书贤', okada='冈田芽依', chen='陈熙锦', nayeon='高娜妍')
    gaps_key = 'adjacent_intervals_without_finite_comparison_including_probes'
    report = ['# Ankara 2026 四节目相机逐曝光运动核查', '',
              f'旧版 `{before}`，状态 SHA-256 `{hashes[0]}`。',
              f'核查版 `{after}`，状态 SHA-256 `{hashes[1]}`。', '',
              '遍历所有登记的原片曝光。表中源时以秒计，镜内相邻比较从每个镜头的第二次曝光开始。', '',
              '| 节目 | 镜头 | 有效相机曝光 / 总数 | 无活跃局部观测 旧→新 | 无可比静物间隔 旧→新 / 镜内间隔 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for key, name in names.items():
        count = sum(row['programme'] == key for row in shots.values())
        a, b = (programmes[key][c]['coverage'] for c in candidates)
        frames = b['exposures']
        oa, ob = (x['missing_local_observations'] for x in (a, b))
        ua, ub = (x['observation_binding_unknown'] for x in (a, b))
        obs = f'{oa}→{ob}' if not (ua or ub) else f'{oa}→{ob}（另有未绑定 {ua}→{ub}）'
        report.append(f"| {name} | {count} | {b['camera_valid']:,} / {frames:,} | {obs} | {a[gaps_key]}→{b[gaps_key]} / {frames-count:,} |")
    report.extend(['', '| 节目 | 新版最高 fx（像素） | 新版最快转速（°/s） | log(f) 最大加速度 旧→新（s⁻²） |',
                   '| --- | --- | --- | --- |'])

    def maximum(r, metric):
        value = r[metric]['worst'][0]
        return f"{value['value']:.2f}（{value['shot'].split('_')[-1].upper()} {value['seconds']:.3f}）"

    for key, name in names.items():
        a, b = (programmes[key][c] for c in candidates)
        f, r = maximum(b, 'fx'), maximum(b, 'rotation_deg_s')
        report.append(f"| {name} | {f} | {r} | {maximum(a, 'log_f_acceleration')} → {maximum(b, 'log_f_acceleration')} |")
    report.extend(['', '下表统计相同可投影原片轨迹覆盖的曝光。每次曝光取图像位移残差中位数，再求节目内 P95。', '',
                   '| 节目 | 围挡相邻 | 围挡跨八曝光 | 冰面相邻 | 冰面跨八曝光 |',
                   '| --- | --- | --- | --- | --- |'])

    def motion(r):
        a, b = r['before'], r['after']
        if not b['frames']:
            return '无足够样本'
        return f"{a['p95']:.2f}→{b['p95']:.2f}（{b['frames']:,}）"

    for key, name in names.items():
        row = comparisons[key]
        fields = (motion(row[k]) for k in ('board_adjacent', 'board_lag8', 'ice_adjacent', 'ice_lag8'))
        report.append('| ' + ' | '.join((name, *fields)) + ' |')
    report.extend(['', '残差单位为原片像素，括号为参与比较的曝光数。光流以同一原片左像点在各自场地模型上求条件深度。',
                   '原片运动缺失的间隔、当前解缺局部观测的曝光和所有极值均留在 `audit.json`。重点窗口和逐曝光数组随核查保存。'])
    (target/'summary_zh.md').write_text('\n'.join(report)+'\n', encoding='utf-8')


def audit(candidate, prepared, shot_filter, focus_only):
    parents = [BASE] if candidate is None else [BASE, candidate]
    target = TARGET/'baseline_audit' if candidate is None else TARGET/'comparison_audit'/candidate
    paths = [OUT/c/'state.npz' for c in parents]
    state_shas = [sha(path) for path in paths]
    states = [arrays(path) for path in paths]
    sources = [OUT/c/'source_bindings.json' for c in parents]
    bindings = [read(path) if path.is_file() else {} for path in sources]
    bindings_setup = [None] if candidate is None else [None, read(prepared/'setup.json')['shots'] if prepared else None]
    natives = [native_for(c, 'jin') for c in parents]
    board_equal = np.array_equal(natives[0].scene['board'], natives[-1].scene['board'])
    registered = arrays(PREPARED/'baseline.npz')
    setup = read(PREPARED/'setup.json')['shots']
    ledger = OUT/BASE/'motion_all_surfaces.json'
    inherited = read(ledger)['shot_files']
    focal_manifest_path = TARGET/'peak_focal/source_motion.json'
    focal_manifest = read(focal_manifest_path) if focal_manifest_path.is_file() else {}
    report, frames, detailed, pair_frames, focal = {}, {}, {}, {}, {}
    for name, shot in setup.items():
        if (shot_filter and name not in shot_filter) or (focus_only and name not in CASES and name not in focal_manifest):
            continue
        pts = registered[name+'_pts']
        if name in CASES:
            detailed[name] = focus(shot, name, pts, parents, states, natives, target)
        if name in focal_manifest:
            focal[name] = focal_probe(name, shot, pts, parents, states, natives, focal_manifest[name])
        if focus_only:
            continue
        program = shot['program']
        sourcefile = packet(name, shot)
        source = arrays(sourcefile)
        old = arrays(inherited[name])
        np.testing.assert_array_equal(pts, source['pts'])
        np.testing.assert_array_equal(pts, old['pts'])
        if 'source_sha256' in source:
            assert source['source_sha256'].item() == shot['source_sha256']
        retained = old['source_row']
        index, family, lag = (source[k][retained] for k in ('index', 'family', 'lag'))
        for k, a in [('index', index), ('family', family), ('lag', lag)]:
            np.testing.assert_array_equal(old[k], a)
        measured = source['uv'][retained]-source['left_uv'][retained]
        records, bundle, row_scores = {}, {'pts': pts}, []
        report[name] = dict(source=shot['source'], source_sha256=shot['source_sha256'], programme=program,
            legacy_motion_input=dict(path=str(sourcefile), sha256=sha(sourcefile), paired_rows=len(retained),
                original_rows=len(source['index']), baseline_motion_record=inherited[name],
                baseline_motion_sha256=sha(inherited[name])))
        for i, (candidate, state, native) in enumerate(zip(parents, states, natives)):
            values = trajectory(native, shot, name, state, pts)
            values['fit_rows'], fit_binding = active_counts(name, shot, state, pts, bindings[i], bindings_setup[i])
            reuse = i == 0 or unchanged(board_equal, states[0], state, name)
            if reuse:
                displacement, curve = old['displacement'], old['curvature']
            else:
                displacement, curve, _ = score(native, shot, state, name, pts, source, retained)
            row_scores.append((displacement, curve))
            flow_frames(values, pts, index, family, lag, measured, displacement, curve)
            for k, a in values.items():
                bundle[str(i)+'_'+k] = a
            results = summaries(name, pts, values)
            clock = values['local_exposure_seconds']
            registered_clock = registered[name+'_times']
            delta = (clock-clock[:1])-(pts-pts[0])/12800.
            results['source_clock'] = dict(registered_exposure_times_equal=bool(np.array_equal(registered_clock, clock)),
                max_local_exposure_offset_from_media_pts_s=finite_float(np.max(abs(delta))))
            results['shared_track_rows_with_invalid_projection'] = int((~np.isfinite(displacement).all(1)).sum())
            results['fit_observation_binding'] = fit_binding
            known = values['source_rows'] > 0
            compared = np.isfinite(values['board_adjacent']) | np.isfinite(values['ice_adjacent'])
            for local_tracks in detailed.get(name, {}).values():
                known |= np.isin(pts, local_tracks['source_right_pts'])
                compared |= np.isin(pts, local_tracks['finite_adjacent_right_pts'][candidate])
            if name in focal and focal[name]['source_rows'] > 0:
                right = pts == focal[name]['pts'][1]
                known |= right
                if focal[name]['source_rows']-focal[name]['comparison'][candidate]['invalid_projection_rows'] >= 4:
                    compared |= right
            results['coverage']['adjacent_intervals_without_source_including_probes'] = int((~known[1:]).sum())
            results['coverage']['adjacent_intervals_without_finite_comparison_including_probes'] = int((~compared[1:]).sum())
            report[name][candidate] = results
            records[candidate] = values
        if len(row_scores) == 2:
            first, second = row_scores
            common = [np.isfinite(a).all(1) & np.isfinite(b).all(1) for a, b in zip(first, second)]
            matched = []
            for i, row_score in enumerate(row_scores):
                values = {}
                ds, cs = (np.where(mask[:, None], vec, np.nan) for mask, vec in zip(common, row_score))
                flow_frames(values, pts, index, family, lag, measured, ds, cs)
                matched.append(values)
                for k in FLOW_METRICS:
                    bundle[str(i)+'_common_'+k] = values[k]
            pair_frames[name] = (pts, *matched)
        frames[name] = (pts, records)
        expected = ranks(name, pts, records[parents[-1]]['rotation_deg_s'], count=1)
        if expected:
            i = np.searchsorted(pts, expected[0]['pts'])
            expected[0].update(fx=float(records[parents[-1]]['fx'][i]),
                source_rows=int(records[parents[-1]]['source_rows'][i]),
                fit_rows=int(records[parents[-1]]['fit_rows'][i]),
                source_median_step_px=finite_float(records[parents[-1]]['source_visible_step'][i]),
                board_motion_error_px=finite_float(records[parents[-1]]['board_adjacent'][i]),
                ice_motion_error_px=finite_float(records[parents[-1]]['ice_adjacent'][i]))
        report[name]['fastest_turn'] = expected
        save(target/'frames'/(name+'.npz'), **bundle)
        print(name, len(pts), {c: {k: report[name][c]['coverage'][k]
                                  for k in ('missing_local_observations', 'observation_binding_unknown')}
                              for c in parents}, flush=True)
    results = {p: {c: pool(p, c, report, frames, setup) for c in parents} for p in PROGRAMMES}
    comparisons = {}
    if pair_frames:
        comparisons = {p: common_flow_report([k for k in pair_frames if setup[k]['program'] == p], pair_frames)
                       for p in PROGRAMMES if any(setup[k]['program'] == p for k in pair_frames)}
        for name in pair_frames:
            report[name]['matched_flow'] = common_flow_report([name], pair_frames)
    if state_shas != [sha(path) for path in paths]:
        raise ValueError('A candidate state changed during the audit. Use a completed checkpoint.')
    write(target/('focus.json' if focus_only else 'audit.json'), dict(
        candidates={c: dict(path=str(p), sha256=h) for c, p, h in zip(parents, paths, state_shas)},
        baseline_source_ledger=dict(path=str(ledger), sha256=sha(ledger)),
        timebase='1/12800', sampling='All registered original source exposures inside each selected shot. Adjacent displacements start at the second exposure of each shot.',
        method='Stored development source texture pairs from the released full-motion ledger, conditioned on a valid baseline left inverse. Each changed candidate rebinds the same measured left light ray to its board or ice, then projects both exposures in the native core. Source identity and reporting use integer media PTS. Camera evaluation and normalized speed use the registered local exposure times, which differ slightly from the display clock in some Chen shots. Separate source-only packets cover named outliers.',
        limitations='Missing source motion comparisons remain unknown. Fit point count is per current active fold. Median comparisons require 4 board or 3 ice tracks, or 3 tracks for the source-motion-only median. Flow provides development evidence with conditional depth.',
        units=dict(fx='source pixels', rotation_deg_s='degree/s', centre_L_s='L/s',
                   pan_rate='rad/s', tilt_rate='rad/s', log_f_rate='1/s',
                   pan_acceleration='rad/s^2', tilt_acceleration='rad/s^2', log_f_acceleration='1/s^2',
                   flow_errors='pixels in the original source image'),
        shots=report, programmes=results, matched_flow=comparisons,
        jin_source_windows={k: v for k, v in detailed.items() if setup[k]['program'] == 'jin'},
        other_source_windows={k: v for k, v in detailed.items() if setup[k]['program'] != 'jin'},
        peak_focal_nearby_original=focal))
    if candidate is not None and len(report) == len(setup):
        write_overview(target, parents, state_shas, results, report, comparisons)


def finite_float(value):
    return float(value) if np.isfinite(value) else None


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', help='Candidate directory relative to joint_refine_1117')
    parser.add_argument('--candidate-prepared', type=Path, help='Explicit prepared directory when the candidate does not publish source bindings')
    parser.add_argument('--shots', nargs='+', help='Limit diagnostics to named scenes')
    parser.add_argument('--focus-only', action='store_true')
    args = parser.parse_args()
    audit(args.candidate, args.candidate_prepared, args.shots, args.focus_only)
