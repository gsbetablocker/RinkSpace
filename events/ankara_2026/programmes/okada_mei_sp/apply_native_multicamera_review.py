"""Apply reviewed source pixels and identities without changing model parameters.

Call after the fit stops. Source-image evidence authorizes each change; projected
geometry never supplies a measured pixel. Original rows, coordinate corrections
and new observations are recorded outside the seven-field solver input.
"""
from pathlib import Path
import argparse
import copy
import numpy as np
import prepare_native_multicamera as p
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.session.gpu_geometry import NativeSceneGPU, initialize_curve_stations


def _evidence(record):
    if not record.get('reason') or not Path(record['evidence_image']).is_file():
        raise ValueError('Source-image evidence is required')


def _pixel(uv):
    value = np.asarray(uv, dtype=np.float64)
    if value.shape != (2,) or not np.isfinite(value).all() or np.any(value < 0) or np.any(value > [1919, 1079]):
        raise ValueError('Invalid native source pixel')
    return value


def revise_observations(original, numeric_stations, manifest, path_ids, report):
    """Prepare a revision in memory; retain old-row identity across insertions."""
    if set(original) != p.FIELDS:
        raise ValueError('Expected exactly seven plain observation fields')
    decisions = report['decisions']
    additions = report.get('added_observations', [])
    lookup = {str(k): i for i, k in enumerate(original['observation_ids'])}
    if len(lookup) != len(original['pts']):
        raise ValueError('Duplicate current observation IDs')
    if len({d['observation_id'] for d in decisions}) != len(decisions):
        raise ValueError('Duplicate review decisions')
    z = {k: a.copy() for k, a in original.items()}
    keep = np.ones(len(z['pts']), dtype=bool)
    touched = np.zeros(len(z['pts']), dtype=bool)
    records = []
    for d in decisions:
        i = lookup[d['observation_id']]
        if int(z['pts'][i]) != d['pts'] or path_ids[z['path_indices'][i]] != d['expected_path_id']:
            raise ValueError('Point identity changed')
        np.testing.assert_array_equal(original['uv'][i], d['uv'])
        _evidence(d)
        action = d['decision']
        if action == 'rebind_preserving_uv':
            z['path_indices'][i] = path_ids.index(d['new_path_id'])
            touched[i] = True
        elif action == 'correct_uv':
            z['uv'][i] = _pixel(d['new_uv'])
            if d.get('new_path_id'):
                z['path_indices'][i] = path_ids.index(d['new_path_id'])
            touched[i] = True
        elif action == 'exclude_preserving_source':
            keep[i] = False
        else:
            raise ValueError('Unknown explicit review decision: ' + action)
        records.append(dict(d, row=i, source_row=int(z['source_rows'][i])))

    added_records = []
    next_source_row = int(original['source_rows'].max()) + 1
    source_frames = set(map(int, original['pts']))
    ids = set(lookup)
    for number, d in enumerate(additions):
        _evidence(d)
        identity = d['observation_id']
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError('New source observation needs a unique ID')
        if int(d['pts']) != d['pts'] or d['pts'] not in source_frames:
            raise ValueError('Added point has no current source exposure')
        uv = _pixel(d['uv'])
        path_index = path_ids.index(d['path_id'])
        shots = [s for s in manifest['shots'] if s['first_pts'] <= d['pts'] <= s['last_pts']]
        if len(shots) != 1:
            raise ValueError('Added point must belong to exactly one shot')
        # A new contour sample gets a free station. A fixed material point needs
        # its explicitly identified station instead of a guessed one.
        kind = int(d.get('kind', 1))
        if kind not in (0, 1) or (kind == 0 and 'station' not in d):
            raise ValueError('Invalid new material-point definition')
        station = float(d.get('station', .5))
        if not np.isfinite(station):
            raise ValueError('Invalid new station seed')
        same = keep & (z['pts'] == d['pts']) & (z['path_indices'] == path_index)
        if np.any(np.all(z['uv'][same] == uv, axis=1)):
            raise ValueError('New point duplicates an existing measured pixel')
        if any(r['pts'] == d['pts'] and r['path_id'] == d['path_id'] and r['uv'] == uv.tolist() for r in added_records):
            raise ValueError('Duplicate added measured pixel')
        ids.add(identity)
        added_records.append(dict(d, uv=uv.tolist(), kind=kind, station=station,
                                  source_row=next_source_row + number, shot=shots[0]['id']))
    added = {
        'observation_ids': np.asarray([d['observation_id'] for d in added_records], dtype=str),
        'pts': np.asarray([d['pts'] for d in added_records], dtype=original['pts'].dtype),
        'uv': np.asarray([d['uv'] for d in added_records], dtype=original['uv'].dtype).reshape(-1, 2),
        'kind': np.asarray([d['kind'] for d in added_records], dtype=original['kind'].dtype),
        'stations': np.asarray([d['station'] for d in added_records], dtype=original['stations'].dtype),
        'source_rows': np.asarray([d['source_row'] for d in added_records], dtype=original['source_rows'].dtype),
        'path_indices': np.asarray([path_ids.index(d['path_id']) for d in added_records], dtype=original['path_indices'].dtype),
    }
    combined = {k: np.concatenate([z[k], added[k]]) for k in z}
    stations = np.concatenate([numeric_stations.copy(), added['stations']])
    origins = np.r_[np.arange(len(keep)), np.full(len(added_records), -1)]
    touched = np.r_[touched, np.ones(len(added_records), dtype=bool)]
    shots = copy.deepcopy(manifest['shots'])
    order, begin = [], 0
    for shot in shots:
        old_rows = np.arange(shot['row_begin'], shot['row_stop'])
        old_rows = old_rows[keep[old_rows]]
        new_rows = np.asarray([len(keep) + j for j, d in enumerate(added_records) if d['shot'] == shot['id']], dtype=int)
        rows = np.r_[old_rows, new_rows]
        if len(new_rows):
            rows = rows[np.argsort(combined['pts'][rows], kind='stable')]
        order.extend(rows)
        shot['row_begin'], shot['row_stop'] = begin, begin + len(rows)
        begin += len(rows)
    order = np.asarray(order, dtype=int)
    new = {k: a[order] for k, a in combined.items()}
    origins = origins[order]
    old_rows = origins >= 0
    for key in p.FIELDS - {'uv', 'path_indices'}:
        np.testing.assert_array_equal(new[key][old_rows], original[key][origins[old_rows]])
    unchanged = old_rows & ~touched[order]
    for key in ('uv', 'path_indices'):
        np.testing.assert_array_equal(new[key][unchanged], original[key][origins[unchanged]])
    if len(set(new['observation_ids'])) != len(new['pts']):
        raise ValueError('Resulting source IDs collide')
    return new, stations[order], shots, touched[order], records, added_records, origins


def apply(decisions_path, resume, output, receipt_dir):
    m = p.read(p.RUN / 'joint_manifest.json')
    mp = p.read(p.RUN / 'mapping.json')
    original = p.npz(p.RUN / 'point_observations.npz')
    v = p.npz(resume)
    model = SceneParameterModel(p.read(m['scene_path']), Path(m['scene_path']), optimize_scene=True)
    p.validate_snapshot(v, model.initial_state.layout, mp, m, original)
    before = p.digest(p.RUN / 'point_observations.npz')
    if before != m['point_input_sha256']:
        raise ValueError('Current input digest differs')
    if receipt_dir.exists() or output.exists():
        raise FileExistsError('Review receipt or seed already exists')
    report = p.read(decisions_path)
    if report.get('input_sha256', before) != before:
        raise ValueError('Review refers to a different input')
    protected = {k: v[k].copy() for k in ('rig', 'controls', 'scene_values')}
    new, stations, shots, touched, records, additions, origins = revise_observations(
        original, v['stations'], m, mp['path_ids'], report)
    native = NativeSceneGPU(m['scene_path'], NativeCore(m['native_library']),
                           parameter_state=ParameterState(model.initial_state.layout, v['scene_values']))
    init = {}
    for shot in shots:
        rows = np.arange(shot['row_begin'], shot['row_stop'])
        ii = rows[touched[rows]]
        if not len(ii):
            continue
        camera = {**p.read(m['cameras'][shot['camera_index']]['camera_definition']),
                  'rig': v['rig'][shot['camera_index']].tolist(), 'reference_pts': shot['reference_pts'],
                  'control_dt': m['control_dt'], 'first_control_time': shot['first_control_time']}
        st, note = initialize_curve_stations(native, camera=camera,
            controls=v['controls'][shot['control_begin']:shot['control_stop']],
            path_ids=[mp['path_ids'][k] for k in new['path_indices'][ii]], stations=stations[ii],
            pts=new['pts'][ii], observed=new['uv'][ii], kinds=new['kind'][ii], half_seconds=m['max_readout'] / 2)
        if note['unresolved']:
            raise ValueError('Reviewed point has no valid station initialization')
        stations[ii] = st
        init[shot['id']] = note
    receipt_dir.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    p.write(receipt_dir / 'manifest_before_review.json', m)
    selected = [r['row'] for r in records]
    np.savez(receipt_dir / 'original_reviewed_observations.npz', **{k: a[selected] for k, a in original.items()})
    m['shots'] = shots
    np.savez(p.RUN / 'point_observations.pending.npz', **new)
    after = p.digest(p.RUN / 'point_observations.pending.npz')
    m['point_input_sha256'] = after
    m['current_resume_snapshot'] = str(output)
    receipt = receipt_dir / 'source_decisions.json'
    counts = {action: sum(d['decision'] == action for d in records) for action in
              ('rebind_preserving_uv', 'exclude_preserving_source', 'correct_uv')}
    m.setdefault('source_repair_history', []).append(dict(receipt=str(receipt), before_sha256=before,
        after_sha256=after, rebound_observations=counts['rebind_preserving_uv'],
        excluded_observations=counts['exclude_preserving_source'], corrected_observations=counts['correct_uv'],
        added_observations=len(additions), original_source_preserved=True))
    v['stations'], v['source_rows'] = stations, new['source_rows']
    p.bind_snapshot(v, model.initial_state.layout, mp, m)
    p.validate_snapshot(v, model.initial_state.layout, mp, m, new)
    for key, value in protected.items():
        np.testing.assert_array_equal(v[key], value)
    np.savez(output, **v)
    p.write(receipt, dict(before_sha256=before, after_sha256=after, decisions=records,
        added_observations=additions, station_initialization=init, geometry_rigs_controls_unchanged=True,
        unchanged_source_fields_preserved=True, source_coordinate_edits_explicitly_logged=True,
        original_rows_archive=str(receipt_dir / 'original_reviewed_observations.npz'), seven_unweighted_fields=True))
    # Replace, never truncate: existing hardlinks remain valid source archives.
    p.write(p.RUN / 'joint_manifest.pending.json', m)
    (p.RUN / 'point_observations.pending.npz').replace(p.RUN / 'point_observations.npz')
    (p.RUN / 'joint_manifest.pending.json').replace(p.RUN / 'joint_manifest.json')
    print(dict(**counts, added=len(additions), remaining=len(new['pts']), seed=str(output)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('decisions', 'resume', 'output', 'receipt-dir'):
        parser.add_argument('--' + key, type=Path, required=True)
    a = parser.parse_args()
    apply(a.decisions, a.resume, a.output, a.receipt_dir)
