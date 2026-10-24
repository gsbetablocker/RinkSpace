"""Read-only adjacent-exposure motion review of an existing complete candidate.

Reuses the frozen S01--S04 physical point pairs and the current S05--S06
board-inner motion table. These are reused fitting/development observations,
not an independent accuracy certificate. No source pixels are remeasured.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
from .io_utils import read_json, write_json, file_sha256


def resolve_ref(root, ref):
    path = Path(ref['file'])
    return path if path.is_absolute() else root / path


def load_sources(current_input, review_dir):
    current_input = Path(current_input)
    root = current_input.parent
    manifest = read_json(current_input)
    review_dir = Path(review_dir)
    pairs = read_json(review_dir / 'six_static_point_motion_pairs.json')
    previous = read_json(review_dir / 'six_static_point_motion_review.json')
    status = read_json(review_dir / 'six_motion_review_status.json')
    motion = manifest['motion']
    if motion['surface'] != 'board_inner':
        raise ValueError('Only the current declared board-inner motion surface is supported')
    clock = motion['source_clock']
    if clock['time_base'] != [1, 12800] or clock['frame_step_pts'] != 256:
        raise ValueError('Current source clock is not the original 50 fps native PTS clock')
    ref = motion['table']
    table = np.memmap(resolve_ref(root, ref), dtype=ref['dtype'], shape=tuple(ref['shape']), mode='r')
    if len(table) != motion['selected_matches']:
        raise ValueError('Current motion row count changed')
    for row in pairs:
        if row['shot'] not in ('s01', 's02', 's03', 's04') or row['pts1'] - row['pts0'] != 256:
            raise ValueError('Frozen physical point pair scope/clock changed')
        if not row['label'].startswith(('Ad', 'board.', 'print_component.')):
            raise ValueError('Frozen pair is not an existing board material point')
    if np.any(table[:, 2] - table[:, 1] != 256) or not np.isfinite(table[:, :7]).all():
        raise ValueError('Current motion is not finite adjacent native exposure data')
    pairs, binding = bind_current_point_pairs(pairs, current_input, manifest)
    return manifest, pairs, previous, status, table, binding


def bind_current_point_pairs(pairs, current_input, manifest):
    """Bind each frozen endpoint to exact current static-material physical rows.

    The current physical mask is the already-issued source membership authority.
    This reads only current columns; it does not replay any qualification producer.
    """
    from collections import Counter
    from .joint_review_packet import CurrentReviewPacket, packet_field
    root = Path(current_input).parent
    packet = resolve_ref(root, manifest['packet'])
    arrays = packet_field(packet, 'arrays')
    shot_ids = packet_field(packet, 'shot_ids')
    materials = packet_field(packet, 'material_binding', {}).get('materials', [])
    material_by_label = {m['label']: m for m in materials}
    def column(ref):
        return np.memmap(resolve_ref(root, ref), dtype=ref['dtype'], shape=tuple(ref['shape']), mode='r')
    cols = {name: column(arrays[name]) for name in ('shot_index', 'material_index', 'path', 'native_pts', 'uv')}
    physical = column(manifest['qualification']['physical_mask'])
    requested = set(); endpoint_keys = []; structural = []
    for row in pairs:
        material = material_by_label.get(row['label'])
        paths = {int(p['index']): p for p in material.get('paths', [])} if material else {}
        reason = ('current_material_label_missing' if material is None else
                  'current_path_not_static_point' if not paths.get(int(row['path']), {}).get('physical_point') else None)
        keys = []
        for side in (0, 1):
            key = None if reason else (shot_ids.index(row['shot']), int(material['index']), int(row['path']),
                                      int(row[f'pts{side}']), *map(float, row[f'uv{side}']))
            keys.append(key)
            if key is not None: requested.add(key)
        endpoint_keys.append(keys); structural.append(reason)
    target_materials = np.array(sorted({key[1] for key in requested}), dtype=cols['material_index'].dtype)
    candidates = np.flatnonzero(np.isin(cols['material_index'], target_materials) &
                               np.isin(cols['shot_index'], [shot_ids.index(s) for s in ('s01','s02','s03','s04')]))
    matches = {}
    for raw in candidates:
        uv = cols['uv'][raw]
        key = (int(cols['shot_index'][raw]), int(cols['material_index'][raw]), int(cols['path'][raw]),
               int(cols['native_pts'][raw]), float(uv[0]), float(uv[1]))
        if key in requested: matches.setdefault(key, []).append(int(raw))
    retained = []; ledger = []; pair_reasons = Counter(); endpoint_reasons = Counter(); wanted_ids = set()
    for original_index, (row, keys, structural_reason) in enumerate(zip(pairs, endpoint_keys, structural)):
        endpoints = []
        for key in keys:
            exact = matches.get(key, []) if key is not None else []
            admitted = [raw for raw in exact if physical[raw] != 0]
            reason = structural_reason or ('exact_source_UV_PTS_path_not_found' if not exact else
                                          'exact_rows_not_current_physical' if not admitted else None)
            if reason: endpoint_reasons[reason] += 1
            endpoints.append(dict(exact_current_raw_rows=exact, current_physical_raw_rows=admitted, reason=reason))
        reasons = sorted({e['reason'] for e in endpoints if e['reason']})
        accepted = not reasons
        for reason in reasons: pair_reasons[reason] += 1
        ledger.append(dict(legacy_pair_index=original_index, shot=row['shot'], label=row['label'], path=row['path'],
                           pts0=row['pts0'], pts1=row['pts1'], uv0=row['uv0'], uv1=row['uv1'],
                           retained=accepted, reasons=reasons, endpoints=endpoints))
        if accepted:
            bound = {k: row[k] for k in ('shot','material_index','label','path','pts0','pts1','uv0','uv1')}
            bound.update(legacy_pair_index=original_index, current_material_index=keys[0][1],
                         current_raw_rows0=endpoints[0]['current_physical_raw_rows'],
                         current_raw_rows1=endpoints[1]['current_physical_raw_rows'])
            retained.append(bound)
            wanted_ids.update(bound['current_raw_rows0']); wanted_ids.update(bound['current_raw_rows1'])
    # Hydrate only retained exact raw IDs once, through the public compact reader.
    reader = CurrentReviewPacket(packet, current_input=dict(file=str(Path(current_input).resolve()),
                                  sha256=file_sha256(current_input)), required_rows=wanted_ids) if wanted_ids else None
    ids = {raw: reader.source_id(raw) for raw in sorted(wanted_ids)} if reader else {}
    for row in retained:
        for side in (0, 1): row[f'current_source_row_ids{side}'] = [ids[raw] for raw in row[f'current_raw_rows{side}']]
    report = dict(schema='rinkfit.current_exact_physical_static_point_pair_binding.v1',
                  current_input=dict(file=str(Path(current_input).resolve()), sha256=file_sha256(current_input)),
                  packet=manifest['packet'], physical_mask=manifest['qualification']['physical_mask'],
                  matching='Exact shot/material label/current static point path/native PTS/source float64 UV at both endpoints; no tolerance or nearest point',
                  legacy_pairs=len(pairs), retained_pairs=len(retained), excluded_pairs=len(pairs)-len(retained),
                  excluded_pair_reasons=dict(pair_reasons), excluded_endpoint_reasons=dict(endpoint_reasons),
                  examined_current_target_material_rows=len(candidates), hydrated_current_source_ids=len(ids),
                  rows=ledger, retained_endpoint_raw_ID_bindings=retained)
    return retained, report


def coverage(pairs, status, table):
    result = []
    for shot in status['shots']:
        sid = shot['shot']
        fixed = [r for r in pairs if r['shot'] == sid]
        motion = table[table[:, 0] == int(sid[1:]) - 1]
        measured = {r['pts0'] for r in fixed} | set(motion[:, 1].astype(np.int64).tolist())
        total = (shot['last_pts'] - shot['first_pts']) // 256
        result.append(dict(shot=sid, source_seconds=[shot['first_pts'] / 12800, shot['last_pts'] / 12800],
                           original_point_rows=len(fixed), original_point_exposure_pairs=len({r['pts0'] for r in fixed}),
                           current_motion_rows=len(motion), current_motion_exposure_pairs=len(np.unique(motion[:, 1])),
                           total_native_adjacent_pairs=total, measured_native_adjacent_pairs=len(measured),
                           missing_native_adjacent_pairs=total - len(measured)))
    return result


def stats(value):
    value = np.asarray(value)
    good = np.isfinite(value).all(axis=1)
    x = value[good]
    return dict(rows=len(value), finite_rows=int(good.sum()), invalid_rows=int((~good).sum()),
                median_dx_dy=np.median(x, axis=0).tolist() if len(x) else None,
                rms_norm=float(np.sqrt(np.mean(np.sum(x * x, axis=1)))) if len(x) else None,
                p90_norm=float(np.quantile(np.linalg.norm(x, axis=1), .9)) if len(x) else None)


def summarize(sid, indices, pts0, source_flow, model_flow, window_pairs):
    error = model_flow - source_flow
    by = []
    for pt in np.unique(pts0[indices]):
        ii = indices[pts0[indices] == pt]
        by.append(dict(pts0=int(pt), pts1=int(pt + 256), source_seconds=[pt / 12800, (pt + 256) / 12800],
                       source=stats(source_flow[ii]), model=stats(model_flow[ii]), error=stats(error[ii])))
    # Only truly consecutive measured exposures may form a continuous window.
    runs = []
    for row in by:
        if not runs or row['pts0'] != runs[-1][-1]['pts0'] + 256:
            runs.append([])
        runs[-1].append(row)
    windows = []
    for run in runs:
        width = min(window_pairs, len(run))
        for start in range(len(run) - width + 1):
            piece = run[start:start + width]
            lo, hi = piece[0]['pts0'], piece[-1]['pts1']
            ii = indices[(pts0[indices] >= lo) & (pts0[indices] < hi)]
            windows.append(dict(first_pts=lo, last_pts=hi, adjacent_pairs=width, source_seconds=[lo / 12800, hi / 12800],
                                source=stats(source_flow[ii]), model=stats(model_flow[ii]), error=stats(error[ii])))
    worst = max(windows, key=lambda x: (x['error']['rms_norm'] if x['error']['rms_norm'] is not None else -1)) if windows else None
    return dict(shot=sid, source=stats(source_flow[indices]), model=stats(model_flow[indices]), error=stats(error[indices]),
                measured_exposure_pairs=len(by), continuous_measured_runs=len(runs),
                worst_continuous_window=worst, per_exposure_pair=by)


def complete_candidate(candidate, manifest, input_sha256):
    candidate = Path(candidate)
    checkpoint = read_json(candidate / 'current_checkpoint.json')
    if checkpoint['current_input_v1']['sha256'] != input_sha256:
        raise ValueError('Candidate is bound to a different current input')
    if checkpoint['raw_rows'] != manifest['raw_rows']:
        raise ValueError('Candidate is not a complete current raw-row state')
    model_path = resolve_ref(candidate, checkpoint['model'])
    if file_sha256(model_path) != checkpoint['model']['sha256']:
        raise ValueError('Candidate model does not match its complete checkpoint')
    for ref in checkpoint['raw_auxiliary'].values():
        expected = int(np.prod(ref['shape'])) * np.dtype(ref['dtype']).itemsize
        if resolve_ref(candidate, ref).stat().st_size != expected:
            raise ValueError('Incomplete current raw auxiliary')
    engineering = resolve_ref(candidate, checkpoint['checkpoint'])
    if not engineering.is_file():
        raise ValueError('Complete current engineering state is absent')
    return model_path, checkpoint, engineering


def evaluate(candidate, manifest, pairs, table, input_sha256):
    from .joint_review_camera import CurrentJointCameras
    from .dense_visible_metrics import ReviewScene
    model_path, checkpoint, engineering = complete_candidate(candidate, manifest, input_sha256)
    model = read_json(model_path)
    cams, scene = CurrentJointCameras(model), ReviewScene(model)
    shots = sorted(model['shots'])
    nfixed, nmotion = len(pairs), len(table)
    n = nfixed + nmotion
    world = np.empty((n, 3)); uv0 = np.empty((n, 2)); uv1 = np.empty((n, 2))
    pts0 = np.empty(n, np.int64); pts1 = pts0.copy(); shot_index = np.empty(n, np.int16)
    labels = {m['label']: i for i, m in enumerate(model['materials'])}
    if len(labels) != len(model['materials']):
        raise ValueError('Current material labels are not unique')
    for i, row in enumerate(pairs):
        mi = labels[row['label']]; pi = row['path']; path = model['materials'][mi]['paths'][pi]
        if not path.get('physical_point') or path['kind'] not in (2, 3, 6, 7):
            raise ValueError('Frozen named material point no longer has its current physical point binding')
        world[i] = scene.world(mi, pi, np.array([.5]))[0]
        uv0[i], uv1[i] = row['uv0'], row['uv1']
        pts0[i], pts1[i], shot_index[i] = row['pts0'], row['pts1'], shots.index(row['shot'])
    latents = np.empty((nmotion, 2)); seen = np.zeros(nmotion, bool)
    pattern = re.compile(r'^"static_motion/(\d+)" [^\r\n]+\r?\n2 ([^\r\n]+)', re.M)
    current_state = engineering.read_bytes()
    if hashlib.sha256(current_state).hexdigest() != checkpoint['checkpoint']['sha256']:
        raise ValueError('Engineering state does not match the complete current checkpoint')
    for match in pattern.finditer(current_state.decode('utf8')):
        i = int(match.group(1))
        if i >= nmotion or seen[i]:
            raise ValueError('Current motion latent indices are duplicated or outside the current table')
        latents[i] = np.fromstring(match.group(2), sep=' '); seen[i] = True
    if not seen.all() or not np.isfinite(latents).all():
        raise ValueError('Complete current motion latent state is absent')
    world[nfixed:] = scene.board(latents[:, 0], latents[:, 1] * scene.height)
    shot_index[nfixed:] = table[:, 0].astype(np.int16)
    pts0[nfixed:], pts1[nfixed:] = table[:, 1].astype(np.int64), table[:, 2].astype(np.int64)
    uv0[nfixed:], uv1[nfixed:] = table[:, 3:5], table[:, 5:7]
    pred0 = np.empty((n, 2)); pred1 = pred0.copy(); valid = np.ones(n, bool)
    for si, sid in enumerate(shots):
        ii = np.flatnonzero(shot_index == si); cam = cams.native_camera(sid)
        for pts, source, pred in ((pts0, uv0, pred0), (pts1, uv1, pred1)):
            # Same production observed-row operator through the public camera.
            local = cams.local_seconds(sid, pts[ii]) + cam.tau * (source[ii, 1] / 1079. - .5)
            value = cam.at(world[ii], local); pred[ii] = value['uv']; valid[ii] &= value['valid']
    return model_path, checkpoint, dict(world=world, uv0=uv0, uv1=uv1, pts0=pts0, pts1=pts1,
           shot_index=shot_index, pred0=pred0, pred1=pred1, valid=valid,
           source_kind=np.r_[np.zeros(nfixed, np.uint8), np.ones(nmotion, np.uint8)],
           source_index=np.r_[[r['legacy_pair_index'] for r in pairs], np.arange(nmotion)])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--current-input', type=Path, required=True)
    p.add_argument('--review-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--window-pairs', type=int, default=25, help='Consecutive native pairs per diagnostic window (default 0.5 s)')
    p.add_argument('--scope', type=Path)
    p.add_argument('--render-shots', nargs='+', help='Optionally render only three frames of each selected worst window with every model path/number enabled')
    a = p.parse_args(argv)
    if a.window_pairs < 1 or a.out.exists():
        raise ValueError('Use a positive window and a new immutable output directory')
    manifest, pairs, previous, status, table, binding = load_sources(a.current_input, a.review_dir)
    cov = coverage(pairs, status, table)
    for row in cov:
        original = [r for r in binding['rows'] if r['shot'] == row['shot']]
        row['legacy_original_point_rows'] = len(original)
        row['excluded_original_point_rows'] = sum(not r['retained'] for r in original)
    report = dict(schema='rinkfit.current_existing_static_motion_review.v1', prepare_only=a.prepare_only,
                  candidate=str(a.candidate.resolve()), current_input=str(a.current_input.resolve()),
                  frozen_point_pairs=str((a.review_dir / 'six_static_point_motion_pairs.json').resolve()),
                  current_motion_id=manifest['motion']['logical_motion_id'], coverage=cov,
                  current_point_pair_binding={k:v for k,v in binding.items() if k not in ('rows','retained_endpoint_raw_ID_bindings')},
                  source_clock=status['source'], source_measurements_reused_without_remeasurement=True,
                  source_kind_0='S01-S04 frozen point chains whose two exact UV/PTS endpoints both remain current physical static material points',
                  source_kind_1='Current S05-S06 board-inner motion table and this candidate complete motion latents',
                  independent_validation=False, fitting_and_development_data_reuse_explicit=True,
                  previous_source_overview_conclusion=status['visual_review'],
                  previous_time_shift_and_RS=str((a.review_dir / 'time_shift_and_RS.json').resolve()),
                  evaluation='Same current model, public ReviewScene/CurrentJointCameras, native PTS and measured source row; no new camera or local delta',
                  model_source_clock_weights_or_state_changed=False, fit_started=False, quality_accepted=False)
    if not a.prepare_only:
        model_path, checkpoint, z = evaluate(a.candidate, manifest, pairs, table, file_sha256(a.current_input))
        source_flow = z['uv1'] - z['uv0']; model_flow = z['pred1'] - z['pred0']
        z.update(source_flow=source_flow, model_flow=model_flow, flow_error=model_flow - source_flow)
        summaries = [summarize(c['shot'], np.flatnonzero(z['shot_index'] == si), z['pts0'], source_flow, model_flow, a.window_pairs)
                     for si, c in enumerate(cov)]
        report.update(model_file=str(model_path.resolve()), model_sha256=file_sha256(model_path),
                      state_token=checkpoint['from_state_token'], rows=len(source_flow),
                      invalid_projection_rows=int((~z['valid']).sum()), summary=summaries,
                      diagnostic_window_native_pairs=a.window_pairs)
        a.out.mkdir(parents=True)
        np.savez_compressed(a.out / 'pairs.npz', **z)
        write_json(a.out / 'snapshot_plan.json', [dict(shot=s['shot'], **s['worst_continuous_window'])
                   for s in summaries if s['worst_continuous_window']])
        if a.render_shots:
            if not a.scope:
                raise ValueError('--scope is required for optional same-model snapshots')
            from .joint_replay import render_snapshots
            wanted = []
            for s in summaries:
                w = s['worst_continuous_window']
                if s['shot'] in a.render_shots and w:
                    wanted.extend([w['first_pts'], w['first_pts'] + (w['adjacent_pairs'] // 2) * 256, w['last_pts']])
            packet = resolve_ref(a.current_input.parent, manifest['packet'])
            report['snapshots'] = render_snapshots(model_path, a.scope, a.out / 'snapshots', sorted(set(wanted)),
                         shots=a.render_shots, packet_file=packet, only_pick_pts=True)
    else:
        a.out.mkdir(parents=True)
    write_json(a.out / 'current_point_pair_filter.json', binding)
    write_json(a.out / 'review.json', report)
    print(json.dumps({k: report[k] for k in ('prepare_only', 'candidate', 'coverage', 'independent_validation', 'fit_started')}, ensure_ascii=False))
    return report


if __name__ == '__main__':
    main()
