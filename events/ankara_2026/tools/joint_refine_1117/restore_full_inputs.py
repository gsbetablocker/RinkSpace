"""Materialize the registered full-resolution source observations for repair."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json

import numpy as np

from shared import DATA, EVENT, OUT, PROGRAMMES, arrays, observation_path, original_native, read
from shared import registration, safe_path, save, shot_inputs, write


TARGET = OUT / 'repair/prepared'
HISTORY = DATA / 'runs/ankara_joint_1113'
GUARDED = HISTORY / 'prepared_guarded'
OKADA = DATA / (
    'runs/native_programme_20261031/feature_refinement_1101/focus_repair_1pct/'
    'review_camera_solver_refined_1102/frozen_problem'
)
ANCHORS = OUT.parent / 'trapezoid_joint_1116'
TRAPEZOIDS = OUT.parent / 'trapezoid_two_ends_1116/full_audit'
ANCHOR_FILES = {
    'jin_s09': 'jin_s09_fresh_anchors_logos.npz',
    'okada_s02': 'okada_s02_fresh_anchors_logos.npz',
    'okada_s08': 'okada_s08_fresh_anchors.npz',
}
BRANCHES = ('left_lower', 'left_upper', 'right_lower', 'right_upper')
WRONG_A4 = (1434880, 1439488, 1444096, 1440768, 1441024)
STOP_RAIL = {'s03', 's06', 's10', 's12', 's13', 's16', 's19', 's20', 's21', 's22'}
PAINT_MATERIALS = {
    'direct_coloured_paint_flanks', 'paired_blue_annulus_outer_source',
    'direct_red_badge_ink', 'original_s04_confirmed_red_center',
    'red_hockey_arc_two_flanks', 'two_sided_original_paint_band',
}
FIELDS = ('pts', 'idx', 'uv', 'path', 'station', 'fold', 'family', 'material')


def lines(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        yield from map(json.loads, stream)


def family_map(native):
    families = {}
    for path, spec in native.paths.items():
        label = spec['label'].removeprefix('ice.').split(':')[0]
        if spec['layer'] == 'advertisement':
            label = path.split('::', 1)[0].rsplit('/', 1)[-1]
        elif path in ('board/continuous/top', 'board/continuous/bottom'):
            label = 'board.' + path.rsplit('/', 1)[-1]
        elif label.startswith('goal_trapezoid'):
            label = 'trapezoid_left' if 'left' in label else 'trapezoid_right'
        families[path] = label
    return families


def source_packet(raw, rows, source, *, display=False, source_rows=None):
    a = {key: raw[key][rows] for key in FIELDS}
    a['fold'] = a['fold'].astype(np.int8, copy=True)
    a['source_fold'] = a['fold'].copy()
    a['source_packet'] = np.full(len(rows), source, np.uint8)
    a['source_row'] = rows.astype(np.int64) if source_rows is None else source_rows
    a['station_fixed'] = (
        raw['kind'][rows] == 0 if 'kind' in raw else np.zeros(len(rows), bool)
    )
    a['baseline_display'] = np.full(len(rows), display, bool)
    return a


def frame_roles(shot):
    # The saved guarded split depends only on original exposure and shot.
    phase = (np.arange(len(shot['pts'])) + 19 * (int(shot['shot'][1:]) % 6)) % 174
    role = np.zeros(len(phase), np.int8)
    role[((45 <= phase) & (phase < 84)) | ((106 <= phase) & (phase < 143))] = 3
    role[((57 <= phase) & (phase < 70)) | ((117 <= phase) & (phase < 129))] = 1
    role[((70 <= phase) & (phase < 77)) | ((129 <= phase) & (phase < 136))] = 2
    return role


def check_old_display(packet, shot, receipt):
    ledger = list(lines(receipt['source_pts_ledger']['path']))
    np.testing.assert_array_equal([f['pts'] for f in ledger], shot['pts'])
    counts = np.bincount(packet['idx'], minlength=len(shot['pts']))
    np.testing.assert_array_equal(counts, [f['n_points'] for f in ledger])
    assert int(counts.sum()) == receipt['coverage']['active_points']
    assert np.count_nonzero(counts) == receipt['coverage']['frames_with_points']
    return {'points': int(counts.sum()), 'frames': int(np.count_nonzero(counts)),
            'ledger': receipt['source_pts_ledger']['path'], 'every_frame_count_matches': True}


def jin_rows(shot, evidence, decisions_path, families, receipt):
    sid = shot['shot']
    decisions = {d['pts']: d for d in lines(decisions_path)}
    ledger = lines(receipt['source_pts_ledger']['path'])
    rows = defaultdict(list)
    provenance = []
    fresh = []
    raw_offset = 0
    fresh_offset = 0
    roles = frame_roles(shot)

    def append(p, idx, path, material, uv, source, ordinal):
        if not (0 <= uv[0] < 1920 and 0 <= uv[1] < 1080):
            return
        values = (p, idx, uv, path, np.nan, roles[idx], families[path], material)
        for key, value in zip(FIELDS, values, strict=True):
            rows[key].append(value)
        provenance.append(source)
        fresh.append(ordinal)

    for idx, (rec, rendered) in enumerate(zip(lines(evidence), ledger, strict=True)):
        p = rec['pts']
        assert p == shot['pts'][idx] == rendered['pts']
        assert rec['source_BGR_sha256'] == rendered['source_BGR_sha256']
        decision = decisions.get(p, {})
        if decision:
            assert decision['fold'] == roles[idx]
            assert decision.get('source_BGR_sha256') in (None, rec['source_BGR_sha256'])
        excluded = defaultdict(set)
        for group in decision.get('excluded', ()):
            if group.get('origin') == 'saved':
                excluded[group['path']].update(map(tuple, group['uv']))
        for group in rec['groups']:
            path = group['path']
            ordinal = raw_offset
            raw_offset += len(group['uv'])
            if path not in families or group.get('check_only', False):
                continue
            family = families[path]
            material = group.get('material', 'unspecified')
            if sid in STOP_RAIL and family.startswith(('goal_line_', 'trapezoid_')):
                if material not in PAINT_MATERIALS:
                    continue
            if sid == 's19' and family == 'F1' and material == 'measured_paired_annulus':
                continue
            if (sid == 's14' and family == 'F1' and path.endswith('146/path0')
                    and 122.94 <= p / 12800 <= 123.42):
                if not decision or any(g['reason'] == 'F1_ambiguous_lower_shadow_arc'
                                       for g in decision['excluded']):
                    continue
            for i, uv in enumerate(group['uv']):
                if tuple(uv) not in excluded[path]:
                    append(p, idx, path, material, uv, 0, ordinal + i)
        for group in decision.get('groups', ()):
            ordinal = fresh_offset
            fresh_offset += len(group['uv'])
            if decision['fold'] == 0 and group['origin'] == 'fresh':
                for i, uv in enumerate(group['uv']):
                    append(p, idx, group['path'], group['material'], uv, 1, ordinal + i)
    raw = {key: np.asarray(rows[key]) for key in FIELDS}
    a = source_packet(raw, np.arange(len(provenance)), 0, display=True,
                      source_rows=np.asarray(fresh, np.int64))
    a['source_packet'] = np.asarray(provenance, np.uint8)
    return a


def okada_rows(shot, source, span, paths, families):
    first, stop = span['row_begin'], span['row_stop']
    piece = slice(first, stop)
    path = paths[source['path_indices'][piece]]
    pts = source['pts'][piece]
    uv = source['uv'][piece]
    accepted = ~((path == 'demo_initial/A4:outer') & np.isin(pts, WRONG_A4))
    accepted &= ((uv >= 0) & (uv < [1920, 1080])).all(axis=1)
    raw = dict(pts=pts, idx=np.searchsorted(shot['pts'], pts), uv=uv, path=path,
               station=source['stations'][piece], kind=source['kind'][piece],
               fold=np.zeros(len(pts), np.int8), family=families[source['path_indices'][piece]],
               material=np.full(len(pts), 'frozen_source_observation'))
    rows = np.flatnonzero(accepted)
    return source_packet(raw, rows, 0, display=True, source_rows=first + rows)


def qualified_trapezoids(name, shot, source, roles, exclusions):
    mask = source['shot'] == shot['shot']
    mask &= np.isfinite(source['source_bracket']).all(axis=(1, 2))
    pts = source['pts']
    for item in exclusions:
        if item['shot'] == shot['shot']:
            mask &= ~((source['branch'] == BRANCHES.index(item['branch']))
                      & np.isin(pts, item['pts']))
    rows = np.flatnonzero(mask)
    fold = np.full(len(rows), -1, np.int8)
    path = np.full(len(rows), '', '<U65')
    for index, branch in enumerate(BRANCHES):
        use = source['branch'][rows] == index
        if not use.any():
            continue
        entry = roles[f'{name}_{branch}']
        assert int(use.sum()) == entry['used_px_rows']
        assert entry['source_sha256'] == shot['source_sha256']
        path[use] = entry['branch']
        for code, role in enumerate(('training', 'check', 'gap')):
            fold[use & np.isin(pts[rows], entry[role + '_exposures'])] = code
    assert (fold >= 0).all()
    np.testing.assert_allclose(source['uv'][rows], source['source_bracket'][rows].mean(axis=1),
                               rtol=0, atol=2e-8)
    raw = {key: source[key][rows] for key in ('pts', 'idx', 'uv', 'station')}
    raw.update(path=path, fold=fold,
               family=np.where(source['branch'][rows] < 2, 'trapezoid_left', 'trapezoid_right'),
               material=np.full(len(rows), 'two_original_paint_flanks'))
    return source_packet(raw, np.arange(len(rows)), 4, source_rows=rows)


def row_keys(packet, *, uv='uv'):
    xy = packet[uv]
    return zip(packet['pts'], packet['path'], xy[:, 0], xy[:, 1], strict=True)


def merge_sources(base, additions):
    # Preserve even repeated historical display dots. A supplemental reading
    # at the same exact pixel and entity can provide an existing station seed.
    by_key = {}
    extra_pts = np.unique(np.concatenate([a['pts'] for a in additions]))
    for i in np.flatnonzero(np.isin(base['pts'], extra_pts)):
        key = (base['pts'][i], base['path'][i], *base['uv'][i])
        by_key.setdefault(key, (base, i))
    chunks = [base]
    for a in additions:
        keep = np.ones(len(a['pts']), bool)
        for i, key in enumerate(row_keys(a)):
            prior = by_key.get(key)
            if prior is None:
                by_key[key] = (a, i)
                continue
            old, j = prior
            if np.isnan(old['station'][j]) and np.isfinite(a['station'][i]):
                old['station'][j] = a['station'][i]
            keep[i] = False
        if keep.any():
            chunks.append({key: v[keep] for key, v in a.items()})
    return {key: np.concatenate([a[key] for a in chunks]) for key in base}


def later_source_decisions(name, recent):
    """Migrate explicit source corrections using exact UV + entity + PTS."""
    red_path = OUT / 'source_admission_red_only' / f'{name}.npz'
    previous_path = Path(recent['previous_packets']) / f'{name}.npz'
    rejected = set()
    replacement = None
    if red_path.exists():
        red = arrays(red_path)
        old_keys = list(row_keys(red, uv='old_uv'))
        new_keys = list(row_keys(red))
        # The separate two-flank packet is added at full density. This row
        # records supersession in the old archive, not a new curve reading.
        authorised = red['reason'] != 'superseded_by_source_reviewed_two_flank_trapezoid'
        for i in range(len(old_keys)):
            if red['fold'][i] == 5 or old_keys[i] != new_keys[i]:
                rejected.add(old_keys[i])
        keep = {(new_keys[i], int(red['origin_row'][i])) for i in np.flatnonzero(
            authorised & (red['fold'] < 5)
        ) if old_keys[i] != new_keys[i]}
        if keep:
            corrected = arrays(previous_path)
            use = [i for i, key in enumerate(row_keys(corrected))
                   if (key, int(corrected['origin_row'][i])) in keep]
            assert len(use) == len(keep)
            replacement = source_packet(corrected, np.asarray(use, np.int64), 5)
    for entry in recent['changes']:
        if entry['shot'] != name or entry['new_fold'] != 5:
            continue
        previous = arrays(previous_path)
        pts, uv = entry['pts'], entry['original_uv']
        row = np.flatnonzero(
            (previous['pts'] == pts) & (previous['origin_row'] == entry['origin_row'])
            & (previous['uv'] == uv).all(axis=1)
        )
        assert len(row) == 1
        rejected.add((pts, previous['path'][row[0]], *uv))
    return rejected, replacement, red_path, previous_path


def validate(packet, shot, families=None):
    idx = packet['idx']
    assert idx.dtype.kind in 'iu' and (idx >= 0).all() and (idx < len(shot['pts'])).all()
    np.testing.assert_array_equal(packet['pts'], shot['pts'][idx])
    assert np.isfinite(packet['uv']).all()
    assert np.isfinite(packet['station'][packet['station_fixed']]).all()
    if families is not None:
        assert all(p in families for p in np.unique(packet['path']))


def restore():
    safe_path(Path(__file__))
    if (TARGET / 'setup.json').exists():
        raise FileExistsError('Full repair packets already exist: ' + str(TARGET))
    shots = shot_inputs()
    ids = {p: registration(p) for p in PROGRAMMES}
    receipts = {
        p: {s['shot']: s for s in read(ids[p]['full_review']['receipt']['path'])['shots']}
        for p in ('jin', 'okada')
    }
    jin_registration = EVENT / 'programmes/jin_shuxian_sp/joint_review_1113.json'
    jin_index = Path(read(jin_registration)['previous_binding']['path'])
    jin = read(jin_index)
    assert jin['revision'] == 'q874'
    raw_file = OKADA / 'point_observations.npz'
    with np.load(raw_file, allow_pickle=False) as source:
        okada = {k: source[k] for k in ('pts', 'uv', 'path_indices', 'stations', 'kind')}
    okada_paths = np.asarray(read(OKADA / 'mapping.json')['path_ids'])
    okada_spans = {s['id']: s for s in read(OKADA / 'joint_manifest.json')['shots']}
    role_path = ANCHORS / 'source_roles.json'
    role_receipt = read(role_path)
    trap_audit = TRAPEZOIDS / 'trapezoid_full_audit.json'
    trap_details = read(trap_audit)['programmes']
    trapezoids = {p: arrays(TRAPEZOIDS / f'{p}_source_trials.npz') for p in ('jin', 'okada')}
    for p, source in trapezoids.items():
        assert source['source_sha256'].item() == ids[p]['source']['sha256']
    recent_file = OUT / 'source_admission_final/audit.json'
    recent = read(recent_file)
    families = family_map(original_native('jin'))
    okada_families = np.asarray([families[p] for p in okada_paths])
    data, meta, coverage = {}, {}, {}
    totals = defaultdict(Counter)
    for name, shot in shots.items():
        p, sid = shot['program'], shot['shot']
        refs = {}
        source_comparison = None
        if p in ('jin', 'okada'):
            guarded_path = GUARDED / f'{name}.npz'
            guarded = arrays(guarded_path)
            refs['2'] = {'path': str(guarded_path), 'source_row': 'NPZ row'}
            if p == 'jin':
                evidence = Path(jin['shots'][sid]['evidence']['path'])
                decisions = GUARDED / f'{name}_decisions.jsonl'
                a = jin_rows(shot, evidence, decisions, families, receipts[p][sid])
                refs['0'] = {'path': str(evidence), 'source_row': 'flattened JSONL groups/uv ordinal'}
                refs['1'] = {'path': str(decisions), 'source_row': 'flattened JSONL groups/uv ordinal'}
            else:
                a = okada_rows(shot, okada, okada_spans[sid], okada_paths, okada_families)
                refs['0'] = {'path': str(raw_file), 'source_row': 'NPZ row',
                             'mapping': str(OKADA / 'mapping.json')}
            source_comparison = check_old_display(a, shot, receipts[p][sid])
            guarded['material'] = np.full(len(guarded['pts']), 'guarded_source_observation')
            additions = [source_packet(guarded, np.arange(len(guarded['pts'])), 2)]
            if name in ANCHOR_FILES:
                anchor_path = ANCHORS / ANCHOR_FILES[name]
                u = arrays(anchor_path)
                assert u['source_sha256'].item() == shot['source_sha256']
                u['material'] = np.full(len(u['pts']), 'local_source_reread_camera_anchor')
                additions.append(source_packet(u, np.arange(len(u['pts'])), 3))
                refs['3'] = {'path': str(anchor_path), 'source_row': 'NPZ row'}
            if name in ANCHOR_FILES:
                additions.append(qualified_trapezoids(
                    name, shot, trapezoids[p], role_receipt, trap_details[p]['ambiguous_proposals']
                ))
                refs['4'] = {'path': str(TRAPEZOIDS / f'{p}_source_trials.npz'),
                             'source_row': 'NPZ row', 'roles': str(role_path),
                             'admission': str(trap_audit)}
            rejected, corrected, red_path, previous_path = later_source_decisions(name, recent)
            if corrected is not None:
                additions.append(corrected)
                refs['5'] = {'path': str(previous_path), 'source_row': 'NPZ row',
                             'source_decisions': str(red_path)}
            a = merge_sources(a, additions)
            if rejected:
                use = np.isin(a['pts'], list({key[0] for key in rejected}))
                exact = np.zeros(len(use), bool)
                for i in np.flatnonzero(use):
                    exact[i] = (a['pts'][i], a['path'][i], *a['uv'][i]) in rejected
                a['fold'][exact] = 5
            else:
                exact = np.zeros(len(a['pts']), bool)
            unconfirmed_centre = np.zeros(len(a['pts']), bool)
            if p == 'okada':
                # The 1102 package identifies these as centres but contains
                # no direct small painted disk measurement for these rows.
                unconfirmed_centre = np.char.endswith(a['path'], ':center')
                a['fold'][unconfirmed_centre] = 5
            a['admission_reason'] = np.where(
                exact, 'exact_later_source_exclusion',
                np.where(unconfirmed_centre, 'okada_centre_disk_identity_unconfirmed', '')
            )
            a['origin_row'] = np.arange(len(a['pts']), dtype=np.int64)
        else:
            path = observation_path(p, sid)
            raw = arrays(path)
            assert raw['source_sha256'].item() == shot['source_sha256']
            rows = np.flatnonzero(raw['eligible'])
            a = source_packet(raw, rows, 0, display=True)
            a['origin_row'] = rows
            refs['0'] = {'path': str(path), 'source_row': 'NPZ row',
                         'complete_original_fields_and_eligibility': True}
        validate(a, shot, families if p in ('jin', 'okada') else None)
        active = a['fold'] < 5
        counts = dict(native_frames=len(shot['pts']), rows=len(a['pts']),
                      frames=len(np.unique(a['pts'])), active_rows=int(active.sum()),
                      active_frames=len(np.unique(a['pts'][active])),
                      fit_rows=int((a['fold'] == 0).sum()),
                      stations_to_initialize=int(np.count_nonzero(~np.isfinite(a['station']))),
                      fixed_stations=int(a['station_fixed'].sum()),
                      rows_by_source={str(k): int(v) for k, v in
                                      zip(*np.unique(a['source_packet'], return_counts=True))})
        totals[p].update({k: v for k, v in counts.items() if isinstance(v, int)})
        if source_comparison:
            coverage[name] = {**counts, 'historical_render': source_comparison}
            totals[p].update(historical_render_points=source_comparison['points'],
                             historical_render_frames=source_comparison['frames'])
        else:
            coverage[name] = counts
        packet = TARGET / f'{name}.npz'
        save(packet, **a)
        for k in ('pts', 'times', 'controls', 'rigs'):
            data[f'{name}_{k}'] = shot[k]
        meta[name] = {k: v for k, v in shot.items() if k not in ('pts', 'times', 'controls', 'rigs')}
        meta[name].update(observations=str(packet), source_packets=refs)
        print(name, counts['active_rows'], '/', counts['rows'],
              'rows', counts['active_frames'], '/', counts['native_frames'], 'frames', flush=True)
    save(TARGET / 'baseline.npz', **data)
    write(TARGET / 'coverage.json', {'programmes': totals, 'shots': coverage})
    write(TARGET / 'setup.json', dict(
        schema='rinkspace.four_programme_prepared.v1', shots=meta,
        source_clock='original integer PTS, time base 1/12800',
        sampling='complete registered source packets and admitted local overlays',
        fold_policy='retained source roles are reused development evidence, not independent checks',
        source_decisions=str(recent_file), jin_reference_index=str(jin_index),
        jin_station_policy='NaN requires a same-entity native station initialization; source UV remains fixed',
        okada_station_policy='station_fixed denotes original kind=0, a source fixed station on a print path',
        okada_centre_policy='old identity retained for display; source of the small disk unconfirmed in this packet'
    ))
    print(json.dumps(totals, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    restore()
