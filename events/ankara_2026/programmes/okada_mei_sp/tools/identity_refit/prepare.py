"""Bind confirmed source review to the existing continuous programme solver."""
import argparse
import json
from pathlib import Path

import numpy as np

from calibration.program_tracking.body_tracking.body_source_revision import (
    reassess_lower_limb_quality_population, revise_body_observations, revised_quality_population,
)
from calibration.program_tracking.body_tracking.programme_action_factors import active_episode_rows
from calibration.program_tracking.body_tracking.programme_skate_evidence import programme_observation_path
from calibration.program_tracking.body_tracking.rebuild_prepare import load
from source_review import BASELINE, PROGRAMME, baseline, digest, read


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')


def observation_edits(events, old):
    pts = old['pts']
    assignments, exclusions, row_records = [], [], []
    for e in events:
        stamp_list = e['observation_pts']
        checks = e['evidence_pts']
        if (not stamp_list or sorted(set(stamp_list)) != stamp_list or not np.isin(stamp_list, pts).all()
                or not np.isin(checks, pts).all() or min(checks) >= min(stamp_list) or max(checks) <= max(stamp_list)):
            raise ValueError('Require native PTS and observed neighbours on both sides of '+e['id'])
        for stamp in stamp_list:
            i = int(np.searchsorted(pts, stamp))
            detail = dict(source_pts=stamp, event_id=e['id'], reason=e['reason'], evidence_pts=checks)
            if e.get('swap_pairs'):
                assignments.append(dict(detail, joint_pairs=e['swap_pairs']))
            retire = set(e.get('retire_slots', [])) | {row['slot'] for row in e.get('retire_rows', []) if row['pts']==stamp}
            for slot in sorted(retire):
                if old['use'][i, slot]:
                    exclusions.append(dict(detail, joint_id=slot))
            for slot in sorted({j for pair in e.get('swap_pairs', []) for j in pair} | retire):
                row_records.append(dict(detail, slot=slot, before_model_id=int(old['model_joint_ids'][i, slot]),
                    before_use=bool(old['use'][i, slot]), uv_native=old['uv'][i, slot].tolist(),
                    sigma_px=float(old['sigma_px'][i, slot])))
    return assignments, exclusions, row_records


def solver_windows(events, scope):
    # Give the fit reliable approach/recovery rows and keep all programme
    # factors active. The engine selects complete cubic supports of these spans.
    pts_track = []
    for e in events:
        if not (e.get('swap_pairs') or e.get('retire_slots') or e.get('retire_rows') or e['status'] == 'model_pose'):
            continue
        stamps = e['observation_pts']
        lo, hi = next((s['processing']['first_pts'], s['processing']['last_pts']) for s in scope['shots']
                     if s['processing']['first_pts'] <= stamps[0] <= s['processing']['last_pts'])
        if stamps[-1] > hi:
            raise ValueError('An identity event must not extend through a cut')
        pts_track.append((max(lo, stamps[0]-8*256), min(hi, stamps[-1]+8*256)))
    windows = []
    for lo, hi in sorted(pts_track):
        if windows and lo <= windows[-1][1]:
            windows[-1][1] = max(hi, windows[-1][1])
        else:
            windows.append([lo, hi])
    return windows


def prepare(data, prepared, revision, output, *, registry=BASELINE):
    scope, paths, result = baseline(data, registry)
    registered = json.loads(read(registry))
    fit_key = next(key for key in ('fitted_state', 'result', 'N_result') if key in paths)
    solve_key = 'solve_state' if 'solve_state' in paths else 'N_parameters'
    for key in (solve_key, fit_key, 'source_evidence'):
        if digest(paths[key]) != registered['data_root_artifacts'][key]['sha256']:
            raise ValueError('Registered baseline changed: '+key)
    spec = json.loads(read(revision))
    if spec['source_sha256'] != scope['source']['sha256'] or spec['time_base'] != result['time_base'].tolist():
        raise ValueError('Review belongs to another source')
    events = spec['events']
    doc = json.loads(read(paths['source_evidence']))
    obs_path = programme_observation_path(prepared, doc)
    parent_hash = digest(obs_path)
    if parent_hash != str(result['source_observation_ledger_sha256']):
        raise ValueError('Review baseline ledger differs from actual baseline fit')
    old = load(obs_path)
    for src, actual in (('pts', 'source_pts'), ('uv', 'source_uv_native'), ('sigma_px', 'source_sigma_px'),
                        ('model_joint_ids', 'source_body_model_joint_ids'), ('use', 'source_observation_objective_use')):
        np.testing.assert_array_equal(old[src], result[actual])
    assignments, exclusions, rows = observation_edits(events, old)
    provenance = dict(review=str(revision), review_sha256=digest(revision), source_sha256=scope['source']['sha256'],
                      source_time_base=spec['time_base'], parent_ledger_sha256=parent_hash)
    new, record = revise_body_observations(old, assignments, exclusions, provenance)
    output.mkdir(parents=True, exist_ok=False)
    obs = output / 'observations.npz'
    np.savez_compressed(obs, **new)
    ref = dict(path=str(obs), sha256=digest(obs))
    for row in rows:
        i = np.searchsorted(new['pts'], row['source_pts']); j = row['slot']
        row['after_model_id'] = int(new['model_joint_ids'][i, j])
        row['after_use'] = bool(new['use'][i, j])
    write(output/'revision.json', dict(**record, old=dict(path=str(obs_path), sha256=parent_hash), new=ref,
                                      source_review_events=events, edited_observation_rows=rows))

    stage = paths[solve_key].parent.parent
    original = json.loads(read(stage/'primary_quality_baseline.json'))
    original.setdefault('body_observation_reference', dict(path=str(obs_path), sha256=parent_hash))
    if any(row['joint_id'] >= 13 for row in exclusions):
        turn_pts = {e['episode_id']:old['pts'][active_episode_rows(old['pts'], e)]
                    for e in doc['source_episode_review']['episodes'] if e.get('turn_direction_sign') in (-1, 0, 1)}
        quality = reassess_lower_limb_quality_population(original, old, new, ref, record,
            reference_projection=result['COCO_uv'], reference_pts=result['source_pts'], turn_source_pts=turn_pts,
            reference_identity=dict(path=str(paths[fit_key]), sha256=registered['data_root_artifacts'][fit_key]['sha256']))
    else:
        quality = revised_quality_population(original, old, new, ref, record)
    write(output/'quality.json', quality)

    surface = json.loads(read(stage/'retained_surface.json'))
    repair = doc['scoped_candidate_import']
    windows = spec.get('focus_pts') or solver_windows(events, scope)
    tracks = spec.get('tracks_by_window') or [list(range(-1,22)) for _ in windows]
    repair.update(parent_sha256=registered['data_root_artifacts'][solve_key]['sha256'],
        parent_evidence=str(paths['source_evidence']), parent_evidence_sha256=registered['data_root_artifacts']['source_evidence']['sha256'],
        parent_result=str(paths[fit_key]), parent_result_sha256=registered['data_root_artifacts'][fit_key]['sha256'],
        retain_parent_objective_context=True, focus_pts=windows, tracks_by_window=tracks,
        parent_surface=dict(path=surface['surface_path'], sha256=surface['surface_sha256']))
    if com := spec.get('com_motion_policy'):
        parent_rows = stage/'weighted_registry.json'
        repair['phase_objective_replay_registry'] = dict(path=str(parent_rows), sha256=digest(parent_rows))
        doc['com_motion_policy'] = dict(doc['com_motion_policy'], **com)
    old_revision = doc['programme_body_observation_revision']
    doc['programme_body_observation_revision'] = dict(**ref, parent_observations_sha256=parent_hash,
        source_video_sha256=scope['source']['sha256'], time_base=spec['time_base'], previous_revision=old_revision,
        review_path=str(output/'revision.json'), review_sha256=digest(output/'revision.json'))
    policy = doc['programme_structure']['solver_step']
    for key in ('archived_source_direction', 'numerical_trust_seed'):
        policy.pop(key, None)
    write(output/'source.json', doc)
    print(json.dumps(dict(events=len(events), mapped_entries=record['changed_correspondence_entries'],
        excluded_rows=len(exclusions), scoped_windows=len(windows), source_evidence=str(output/'source.json')), ensure_ascii=True))


def main():
    p = argparse.ArgumentParser()
    for name in ('data', 'prepared', 'revision', 'output'):
        p.add_argument(name, type=lambda value: Path(value).resolve())
    p.add_argument('--registry', type=Path, default=BASELINE)
    args = p.parse_args()
    prepare(args.data, args.prepared, args.revision, args.output, registry=args.registry.resolve())


if __name__ == '__main__':
    main()
