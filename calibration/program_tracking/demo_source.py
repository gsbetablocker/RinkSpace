"""Bridge an immutable legacy demo state onto the official native 50 fps clock.

This is a presentation adapter, never a source-observation or solver update.
Explicit historical pixel correspondences are reused. Missing official frames
interpolate only inside their verified source shot, with end values held.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROGRAMME = ROOT / 'events/ankara_2026/programmes/okada_mei_sp'
DATA_PROGRAMME = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp')
LEGACY_EXPORT = DATA_PROGRAMME / 'results/full_8a28afed_user_requested_20261012'
DEFAULT_OUTPUT = DATA_PROGRAMME / 'results/demo_last_shot_20261016'
FIRST_PTS, LAST_PTS, PTS_STEP, TIME_BASE_DEN = 256256, 2415104, 256, 12800


def path(value):
    """Normalize historically double-escaped Windows paths without relocation."""
    return Path(str(value).replace('\\', '/').replace('//', '/'))


def read(p):
    return json.loads(path(p).read_text(encoding='utf8'))


def sha(p):
    h = hashlib.sha256()
    with path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ref(p):
    return {'path': str(path(p)), 'sha256': sha(p)}


def check_output_path(p):
    p = path(p).resolve()
    if len(str(p).encode('utf-16-le')) // 2 >= 260:
        raise ValueError('Output absolute Windows path must have fewer than 260 UTF-16 units')
    if any(len(part.encode('utf-16-le')) // 2 > 255 for part in p.parts):
        raise ValueError('Output path component exceeds 255 UTF-16 units')
    return p


def build_packet(output_dir=DEFAULT_OUTPUT):
    output_dir = path(output_dir)
    npz_path = check_output_path(output_dir / 'source_packet.npz')
    json_path = check_output_path(output_dir / 'source_packet.json')
    command_path = LEGACY_EXPORT / 'export.command.json'
    command = read(command_path)
    camera_path = path(command[command.index('--camera') + 1])
    body_path = path(command[command.index('--body') + 1])
    scene_path = path(command[command.index('--scene') + 1])
    selection_path = PROGRAMME / 'primary_source_selection.json'
    selection = read(selection_path)
    primary = selection['primary_source']
    source_path = path(primary['path'])
    if primary['native_fps'] != '50/1' or primary['time_base'] != '1/12800':
        raise ValueError('The official native 50 fps / 1:12800 source is required')
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    ledger_path = path(selection['historical_alignment']['path'])
    if sha(ledger_path) != selection['historical_alignment']['sha256']:
        raise ValueError('Historical alignment ledger identity changed')
    with np.load(camera_path, allow_pickle=False) as z:
        camera = {k: z[k].copy() for k in ('pts', 'camera25', 'exposure_representative_pts')}
    with np.load(body_path, allow_pickle=False) as z:
        body = {k: z[k].copy() for k in (
            'actual_pts', 'native_pts', 'native_actual_row', 'shot_index',
            'G_world_m', 'valid', 'inferred', 'metres_per_scene_unit')}
    native_pts = np.arange(FIRST_PTS, LAST_PTS + PTS_STEP, PTS_STEP, dtype=np.int64)
    shots = np.ones(len(native_pts), dtype=np.int16)
    scope_refs = []
    for shot in range(2, 10):
        p = DATA_PROGRAMME / f'results/shot{shot}_review_20261014/selection.json'
        scope = read(p)['shot_scope']
        if int(scope['shot_index']) != shot or scope['time_base'] != '1/12800':
            raise ValueError('Unexpected official source shot scope')
        keep = ((native_pts >= int(scope['first_pts']))
                & (native_pts < int(scope['stop_pts_exclusive'])))
        shots[keep] = shot
        scope_refs.append({'source': ref(p), 'shot_index': shot,
                           'first_pts': int(scope['first_pts']),
                           'stop_pts_exclusive': int(scope['stop_pts_exclusive'])})
    camera_rows = {int(p): i for i, p in enumerate(camera['pts'])}
    body_native_rows = {int(p): int(i) for p, i in
                        zip(body['native_pts'], body['native_actual_row'])}
    body_representatives = set(map(int, body['actual_pts']))
    official_rows = {int(p): i for i, p in enumerate(native_pts)}
    candidates = {}
    rejected = 0
    for line in ledger_path.read_text(encoding='utf8').splitlines():
        entry = json.loads(line)
        official = entry.get('accepted_official_pts')
        legacy = int(entry['full_pts'])
        if official is None or int(official) not in official_rows:
            continue
        official = int(official)
        i = official_rows[official]
        if legacy not in camera_rows or legacy not in body_native_rows:
            continue
        c, b = camera_rows[legacy], body_native_rows[legacy]
        if int(entry['shot_index']) != int(shots[i]) or int(body['shot_index'][b]) != int(shots[i]):
            rejected += 1
            continue
        # Prefer the original representative exposure over a legacy alias of it.
        score = (legacy not in body_representatives,
                 int(camera['exposure_representative_pts'][c]) != legacy,
                 float(entry.get('best_roi_gray_mae', float('inf'))), legacy)
        if official not in candidates or score < candidates[official][0]:
            candidates[official] = (score, legacy, c, b)
    # The last-shot ledger's unaccepted best candidates cover the remaining
    # 44 native frames, including the changing final zoom. Preserve their
    # original states for the demo, explicitly marked as unconfirmed.
    explicit_pts = set(candidates)
    unconfirmed = {}
    for line in ledger_path.read_text(encoding='utf8').splitlines():
        entry = json.loads(line)
        official = entry.get('best_official_pts_candidate')
        if (entry.get('accepted_official_pts') is not None or official is None
                or int(official) not in official_rows or int(official) in explicit_pts
                or int(entry['shot_index']) != 9):
            continue
        official, legacy = int(official), int(entry['full_pts'])
        if int(shots[official_rows[official]]) != 9 or legacy not in camera_rows or legacy not in body_native_rows:
            continue
        c, b = camera_rows[legacy], body_native_rows[legacy]
        if int(body['shot_index'][b]) != 9:
            continue
        score = (legacy not in body_representatives,
                 int(camera['exposure_representative_pts'][c]) != legacy,
                 float(entry.get('best_roi_gray_mae', float('inf'))), legacy)
        if official not in unconfirmed or score < unconfirmed[official][0]:
            unconfirmed[official] = (score, legacy, c, b)
    candidates.update(unconfirmed)
    n = len(native_pts)
    cameras = np.empty((n, 25), dtype=np.float64)
    xy = np.empty((n, 2), dtype=np.float64)
    mask_pts = np.empty(n, dtype=np.int64)
    direct = np.zeros(n, dtype=np.bool_)
    mode = np.empty(n, dtype=np.uint8)  # 0=direct, 1=interpolate, 2/3=hold-start/end, 4=unconfirmed candidate
    left_official = np.empty(n, dtype=np.int64)
    right_official = np.empty(n, dtype=np.int64)
    left_legacy = np.empty(n, dtype=np.int64)
    right_legacy = np.empty(n, dtype=np.int64)
    valid = np.empty(n, dtype=np.bool_)
    inferred = np.empty(n, dtype=np.bool_)
    stats = {}
    for shot in range(1, 10):
        target_rows = np.flatnonzero(shots == shot)
        anchor_pts = np.asarray(sorted(p for p in candidates
                                       if int(shots[official_rows[p]]) == shot), dtype=np.int64)
        if not len(anchor_pts):
            raise ValueError(f'No explicit same-shot bridge anchors for shot {shot}')
        for i in target_rows:
            p = int(native_pts[i])
            upper = int(np.searchsorted(anchor_pts, p, side='left'))
            if upper < len(anchor_pts) and int(anchor_pts[upper]) == p:
                lo = hi = upper
                direct[i] = p in explicit_pts
                mode[i] = 0 if direct[i] else 4
            elif upper == 0:
                lo = hi = 0
                mode[i] = 2
            elif upper == len(anchor_pts):
                lo = hi = len(anchor_pts) - 1
                mode[i] = 3
            else:
                lo, hi = upper - 1, upper
                mode[i] = 1
            lp, rp = int(anchor_pts[lo]), int(anchor_pts[hi])
            _, ll, lc, lb = candidates[lp]
            _, rl, rc, rb = candidates[rp]
            alpha = 0.0 if lp == rp else (p - lp) / (rp - lp)
            cameras[i] = camera['camera25'][lc] * (1.0 - alpha) + camera['camera25'][rc] * alpha
            xy[i] = body['G_world_m'][lb, :2] * (1.0 - alpha) + body['G_world_m'][rb, :2] * alpha
            mask_pts[i] = ll if abs(p - lp) <= abs(rp - p) else rl
            left_official[i], right_official[i] = lp, rp
            left_legacy[i], right_legacy[i] = ll, rl
            valid[i] = bool(body['valid'][lb] and body['valid'][rb])
            inferred[i] = bool(mode[i] or body['inferred'][lb] or body['inferred'][rb])
        counts = {label: int(np.sum(mode[target_rows] == k)) for k, label in
                  enumerate(('direct', 'interpolated', 'held_start', 'held_end', 'candidate_unconfirmed'))}
        stats[str(shot)] = dict(native_frames=len(target_rows), **counts,
                               first_pts=int(native_pts[target_rows[0]]),
                               last_pts=int(native_pts[target_rows[-1]]),
                               first_direct_pts=min(p for p in explicit_pts if int(shots[official_rows[p]]) == shot),
                               last_direct_pts=max(p for p in explicit_pts if int(shots[official_rows[p]]) == shot),
                               first_anchor_pts=int(anchor_pts[0]), last_anchor_pts=int(anchor_pts[-1]))
    if len(native_pts) != 8434 or int(np.sum(shots == 9)) != 2255:
        raise ValueError('Unexpected official programme or last-shot frame count')
    if not np.isfinite(cameras).all() or not np.isfinite(xy).all():
        raise ValueError('Bridge states must remain finite')
    unit = float(body['metres_per_scene_unit'])
    packet = dict(pts=native_pts, native_pts=native_pts,
                  time_seconds=native_pts.astype(np.float64) / TIME_BASE_DEN,
                  camera25=cameras, xy_m=xy, legacy_mask_pts=mask_pts,
                  direct_mapping=direct, bridge_mode=mode, shot_index=shots,
                  valid=valid, inferred=inferred,
                  anchor_left_official_pts=left_official, anchor_right_official_pts=right_official,
                  anchor_left_legacy_pts=left_legacy, anchor_right_legacy_pts=right_legacy,
                  unit=np.asarray(unit), metres_per_scene_unit=np.asarray(unit),
                  time_base=np.asarray('1/12800'), native_fps=np.asarray('50/1'),
                  source_path=np.asarray(str(source_path)), scene_path=np.asarray(str(scene_path)))
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **packet)
    report = dict(schema='rinkspace.official50-demo-source-bridge.v1',
                  output=ref(npz_path), source=primary,
                  inputs=dict(export_command=ref(command_path), source_selection=ref(selection_path),
                              historical_alignment=ref(ledger_path), camera=ref(camera_path),
                              body=ref(body_path), scene=ref(scene_path)),
                  official_source_clock_only=True, time_base='1/12800', fps='50/1',
                  first_pts=FIRST_PTS, last_pts=LAST_PTS, native_frames=n,
                  last_shot_native_frames=int(np.sum(shots == 9)), per_shot=stats,
                  official_source_shot_scopes=scope_refs, rejected_cross_shot_ledger_rows=rejected,
                  bridge_mode={'0': 'explicit historical accepted pixel correspondence',
                               '1': 'same-shot state interpolation for presentation',
                               '2': 'same-shot first state held for presentation',
                               '3': 'same-shot last state held for presentation',
                               '4': 'same-shot historical best candidate, unconfirmed presentation bridge'},
                  direct_mapping_means='Explicit accepted_official_pts; not a new pixel verification',
                  interpolation='Linear camera25 and world XY between same-shot official PTS bridge anchors; endpoints held; no fixed offset correspondence asserted',
                  masks='Nearest same-shot bridge anchor legacy frame; original mask unchanged',
                  unconfirmed_candidate_scope='Only shot9 official PTS without an explicit accepted mapping; candidate identity is not confirmed',
                  candidate_unconfirmed_frames=int(np.sum(mode == 4)),
                  xy_semantics='Original saved pelvis world vertical projection G.xy; no blade replacement in this packet',
                  unit=unit, metric_scale_is_conditional=True, verified_metre_scale=False,
                  legacy_state_unmodified=True, solver_run=False, body_fit_run=False,
                  source_observations_modified=False, accuracy_acceptance_implied=False)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_packet(args.output)
    print(json.dumps({'output': result['output'], 'native_frames': result['native_frames'],
                      'last_shot': result['per_shot']['9']}, ensure_ascii=False, indent=2))
