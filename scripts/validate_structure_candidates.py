"""Compare saved programme structure candidates on the original source clock.

Run from the repository with python -B -m scripts.validate_structure_candidates.
Candidate directories use solve/candidate.npz, final/fitted/result.npz (or
fitted/result.npz), and final_action_factors.json. Alternatively quote an
explicit "NAME=PARAMETER|RESULT|ACTION_REPORT" binding.
"""
import argparse
from collections import defaultdict
from copy import copy
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from calibration.program_tracking.body_tracking.absolute_motion_state import absolute_reference, curve_from_packet
from calibration.program_tracking.body_tracking.event_physics import ballistic_residual, flight_momentum_residual
from calibration.program_tracking.body_tracking.programme_action_compare import candidate_metrics, saved_state
from calibration.program_tracking.body_tracking.programme_anatomy import anatomy_geometry
from calibration.program_tracking.body_tracking.programme_anatomy_review import parts_for_sample, triangle_crossings
from calibration.program_tracking.body_tracking.programme_contract import required_clock
from calibration.program_tracking.body_tracking.programme_surface import CollisionSurface
from calibration.program_tracking.body_tracking.rebuild_prepare import digest, load
from calibration.program_tracking.body_tracking.sequence_skate_state import shoe_curve_from_packet
from calibration.program_tracking.body_tracking.skate_physics import fk_state
from calibration.program_tracking.body_tracking.source_helpers.action_1107.refit_axel_axis import axis
from calibration.program_tracking.body_tracking.trajectory_backend import load_reference

PROGRAMME = Path('events/ankara_2026/programmes/okada_mei_sp')
FLIGHTS = ('jump_3f', 'jump_3t', 'jump_2a', 'entry_flight_core', 'jump_3lz')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    return dict(path=str(path.resolve()), sha256=digest(path))


def candidate_paths(value):
    name, sep, spec = value.partition('=')
    if not sep or not name.strip() or not spec:
        raise argparse.ArgumentTypeError('Use NAME=DIR or NAME=PARAMETER|RESULT|ACTION_REPORT')
    if '|' in spec:
        paths = spec.split('|')
        if len(paths) != 3 or not all(paths):
            raise argparse.ArgumentTypeError('Explicit candidates require PARAMETER|RESULT|ACTION_REPORT')
        return name, *(Path(path) for path in paths)
    directory = Path(spec)
    parameter = directory / 'solve/candidate.npz'
    if not parameter.is_file():
        parameter = directory / 'candidate.npz'
    result = directory / 'final/fitted/result.npz'
    if not result.is_file():
        result = directory / 'fitted/result.npz'
    return name, parameter, result, directory / 'final_action_factors.json'


def fixed_sets(fixed_path, dense_path):
    previous = read_json(fixed_path)['evaluated']
    first = next(iter(previous.values()))['samples']
    rows = [dict(seconds=r['seconds'], pair=r['pair']) for r in first]
    for other in previous.values():
        if [dict(seconds=r['seconds'], pair=r['pair']) for r in other['samples']] != rows:
            raise ValueError('Historical fixed comparisons contain different pair/time rows')
    dense = [dict(seconds=r['seconds'], pair=r['pair']) for r in read_json(dense_path)]
    if (len(rows), len(dense)) != (39, 36):
        raise ValueError('Expected the historical 39 and 36 pair/time rows')
    return {'fixed_39': rows, 'dense_36': dense}


def replay_mesh(ref, sets, pts, time_base):
    step = time_base[0] / time_base[1]
    t = pts * step
    origin = t[0] - ref.times[0]
    wanted = defaultdict(set)
    for rows in sets.values():
        for row in rows:
            wanted[row['seconds']].add(tuple(row['pair']))
    geometry = anatomy_geometry(ref)
    counts = {}
    for seconds, pairs in sorted(wanted.items()):
        times = np.array([seconds - origin])
        pose = fk_state(ref, ref.at(times), np.zeros((1, 69)))
        parts = parts_for_sample(ref, pose, *geometry, {part for pair in pairs for part in pair})
        for pair in pairs:
            counts[seconds, pair] = int(triangle_crossings(*(parts[part] for part in pair)))
    report = {}
    for key, rows in sets.items():
        result = []
        for row in rows:
            ticks = round(row['seconds'] / step)
            near = int(np.argmin(np.abs(pts - ticks)))
            count = counts[row['seconds'], tuple(row['pair'])]
            result.append(dict(row, sample_clock_ticks=ticks, nearest_native_pts=int(pts[near]),
                               native_exposure=bool(pts[near] == ticks), intersecting_triangle_pairs=count))
        report[key] = dict(checks=len(result), checks_with_crossings=sum(r['intersecting_triangle_pairs'] > 0
                                                                        for r in result), rows=result)
    return report


def s09_grasp(ref, document, policy_path, policy, episodes, pts, time_base):
    document = dict(document, programme_structure=dict(document.get('programme_structure', {}), policy=str(policy_path)))
    geometry = CollisionSurface(ref, document)
    time_scale = time_base[0] / time_base[1]
    scan = {}
    reports = []
    for rule in policy['source_grasp_review']:
        event = episodes[rule['episode_id']]
        lo, hi = (round(seconds / time_scale) for seconds in rule['source_seconds'])
        entries = []
        foot = ('left', 'right')[rule['foot']]
        palm = rule['hand'] + '_palm'
        opposing = {foot + '_' + part for part in ('boot', 'blade_rear', 'blade_front')}

        def local_pairs(pairs):
            return {tuple(sorted((int(a), int(b)))) for a, b in pairs
                    if {geometry.part_names[geometry.parts[a]], geometry.part_names[geometry.parts[b]]} & opposing
                    and palm in (geometry.part_names[geometry.parts[a]], geometry.part_names[geometry.parts[b]])}

        for native_pts in event['evidence_pts']:
            if not lo <= native_pts <= hi:
                continue
            i = int(np.searchsorted(pts, native_pts))
            if i == len(pts) or pts[i] != native_pts:
                raise ValueError(f'Grasp evidence PTS {native_pts} is outside the native clock')
            if i not in scan:
                time = ref.times[i:i + 1]
                pose = fk_state(ref, ref.at(time), np.zeros((1, 69)))
                _, scan[i] = geometry.scan_sample(geometry.vertices(pose)[0], time[0])
            sample = scan[i]
            local_crossing = local_pairs(sample['pairs'])
            local_tagged = local_pairs(sample['grip_pairs'])
            entries.append(dict(pts=int(native_pts), seconds=float(native_pts * time_scale),
                                tagged_model_crossing_pairs=len(local_tagged),
                                retained_model_crossing_pairs=len(local_crossing),
                                conditionally_allowed_pairs=len(local_tagged - local_crossing)))
        reports.append(dict(policy=rule, source_episode=event, evidence_frames=entries))
    return reports


def flight_diagnostics(result, saved_metrics, episodes, pts, time_base):
    t = pts * (time_base[0] / time_base[1])
    cases = {}
    for name in FLIGHTS:
        source = episodes[name]
        mask = (pts >= source['start_pts']) & (pts <= source['end_pts_inclusive'])
        selected = pts[mask]
        com = result['com_world_mid'][mask]
        residual = ballistic_residual(com, t[mask], [0., 0., -9.81], sigma_position_m=1.)['residual']
        row = dict(source_episode=source, com_pts=selected.tolist(),
                   com_gravity_time_weighted_rms_m=float(np.linalg.norm(residual)))
        metrics = saved_metrics.get(f'episode/{name}/metrics')
        if metrics is None:
            row['momentum_status'] = 'saved H rows missing'
        else:
            np.testing.assert_array_equal(metrics['source_pts'], selected)
            momentum_pts = np.asarray(metrics['momentum_pts'])
            if not np.isin(momentum_pts, selected).all():
                raise ValueError(f'{name} momentum points extend beyond the confirmed flight')
            recorded_com = np.asarray(metrics['com_world_m'])
            np.testing.assert_allclose(recorded_com, com, rtol=0., atol=1e-7)
            H = np.asarray(metrics['specific_H_world_m2ps'])
            motion = flight_momentum_residual(H, momentum_pts * (time_base[0] / time_base[1]),
                                              sigma_specific_H=1., phase='flight')
            row.update(momentum_pts=momentum_pts.tolist(), mean_specific_H_world_m2ps=motion['mean_H'].tolist(),
                       specific_H_time_weighted_rms_m2ps=float(np.linalg.norm(motion['residual'])),
                       full_orbital_and_intrinsic=metrics['full_orbital_and_intrinsic'])
        cases[name] = row
    return cases


def evaluate_case(root, document, prepared, paths, prepared_hashes, pts, shots, time_base,
                  observations, episodes, sets, policy_path, policy):
    name, parameter_path, result_path, report_path = paths
    parameter = load(parameter_path)
    for field in ('prepared_input_sha256', 'prepared_observations_sha256'):
        if str(parameter[field]) != prepared_hashes[field]:
            raise ValueError(f'{name} has different prepared inputs: {field}')
    ref = absolute_reference(root, curve_from_packet(parameter), root=prepared)
    ref.relative_skate_curve = shoe_curve_from_packet(parameter)
    ref.binding = copy(root.binding)
    ref.binding.reference = ref
    ref.binding.x = parameter['x'][:17].copy()
    ref.binding._packet = ref.binding._parameter_key = None

    result = saved_state(result_path)
    np.testing.assert_array_equal(result['pts'], pts)
    np.testing.assert_array_equal(result['time_base'], time_base)
    with np.load(result_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['source_pts'], pts)
        np.testing.assert_array_equal(archive['source_shot_ids'], shots)
        source_evidence_sha = str(archive['source_action_ledger_sha256'])
        if source_evidence_sha != str(parameter['source_evidence_sha256']):
            raise ValueError(f'{name} parameter/result bind different source evidence')
        camera = json.loads(str(archive['camera_binding_json']))
        if json.loads(str(parameter['camera_binding_json'])) != camera:
            raise ValueError(f'{name} parameter/result bind different camera states')
        pose = fk_state(ref, ref.at(ref.times), np.zeros((len(pts), 69)))
        np.testing.assert_allclose(archive['native_body_local_R_mid'], pose['local_R'], rtol=0., atol=2e-8)
    np.testing.assert_allclose(result['P_world_mid'], pose['P'] * ref.binding.ratio, rtol=0., atol=2e-8)
    np.testing.assert_allclose(result['Q_world_mid'], pose['Q'], rtol=0., atol=2e-8)
    np.testing.assert_allclose(result['skate_R_world_mid'], pose['R'], rtol=0., atol=2e-8)
    np.testing.assert_allclose(result['skate_origin_world_mid'], pose['O'] * ref.binding.ratio, rtol=0., atol=2e-8)

    metrics = read_json(report_path)['source_and_physics'] if report_path.is_file() else {}
    if not metrics:
        print(f'{name}: no saved action-factor report at {report_path}', flush=True)
    flights = flight_diagnostics(result, metrics, episodes, pts, time_base)
    rotation = episodes['axel_rotation']
    compared = candidate_metrics(result, pts, shots, observations, {'episodes': [rotation]}, camera)
    t = pts * (time_base[0] / time_base[1])
    trunk = axis({'native_world': result['native22_world_mid']})
    tilt = np.degrees(np.arccos(np.clip(trunk[:, 2], -1., 1.)))
    at = int(np.argmin(abs(t - 78.02)))
    axel = dict(source_episode=rotation, rotation_path=compared['rotation_paths']['axel_rotation'],
                trunk_axis_78_02_pts=int(pts[at]), trunk_axis_78_02_deg=float(tilt[at]),
                trunk_axis_77_80_to_78_14_peak_deg=float(tilt[(t >= 77.80) & (t <= 78.14)].max()))
    print(f'{name}: replaying fixed surface and S09 source frames', flush=True)
    return dict(parameter=identity(parameter_path), result=identity(result_path),
                action_report=identity(report_path) if report_path.is_file() else dict(path=str(report_path), missing=True),
                source_evidence_sha256=source_evidence_sha,
                camera_state_sha256_by_shot={key: row['state_sha256'] for key, row in camera['shots'].items()},
                source_clock_and_pose_replay_verified=True,
                mesh=replay_mesh(ref, sets, pts, time_base), flights=flights, axel_direction=axel,
                s09_grasp=s09_grasp(ref, document, policy_path, policy, episodes, pts, time_base),
                s09_source_skate_regions=metrics.get('new_source_skate_regions'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--fixed-checks', type=Path, required=True, help='Historical mesh_crossings_fixed_checks.json')
    parser.add_argument('--dense-checks', type=Path, required=True, help='Historical collision_timeline_before.json')
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--policy', type=Path, help='Defaults to the programme structure_camera_1108_policy.json')
    parser.add_argument('--candidate', type=candidate_paths, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New comparison JSON path')
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    names = [row[0] for row in args.candidate]
    if len(names) != len(set(names)):
        parser.error('Candidate names must be unique')

    scope_path = args.project / PROGRAMME / 'rebuild_scope.json'
    policy_path = (args.policy or args.project / PROGRAMME / 'structure_camera_1108_policy.json').resolve()
    policy = read_json(policy_path)
    pts, shots, time_base = required_clock(read_json(scope_path))
    ledger = read_json(args.ledger)
    if (len(pts) != 8329 or ledger['exposures'] != len(pts) or tuple(ledger['time_base']) != time_base
            or ledger['scope_pts'] != pts[[0, -1]].tolist()):
        raise ValueError('Action ledger and 8329-exposure source clock disagree')
    ledger_id = identity(args.ledger)
    sets = fixed_sets(args.fixed_checks, args.dense_checks)
    hashes = {'prepared_input_sha256': digest(args.prepared / 'input.npz'),
              'prepared_observations_sha256': digest(args.prepared / 'observations.npz')}
    with threadpool_limits(limits=1):
        root, _, _, document, _ = load_reference(args.project, args.prepared)
        np.testing.assert_array_equal(root.p['pts'], pts)
        np.testing.assert_array_equal(root.p['shot_ids'], shots)
        if document['source_sha256'] != ledger['source_id']:
            raise ValueError('Prepared source video and action ledger differ')
        source_review = document['source_episode_review']
        if source_review['source_action_ledger_sha256'] != ledger_id['sha256']:
            raise ValueError('Supplied action ledger is not the ledger bound by the prepared evidence')
        episodes = {row['episode_id']: row for row in source_review['episodes']}
        observations = load(args.prepared / 'observations.npz')
        np.testing.assert_array_equal(observations['pts'], pts)
        candidates = {case[0]: evaluate_case(root, document, args.prepared, case, hashes, pts, shots, time_base,
                                              observations, episodes, sets, policy_path, policy)
                      for case in args.candidate}
    report = dict(schema='rinkspace.structure-candidate-comparison.v1',
                  source=dict(video_sha256=ledger['source_id'], native_exposures=len(pts), source_pts=pts.tolist(),
                              shot_ids=shots.tolist(), time_base=list(time_base), scope=identity(scope_path)),
                  inputs=dict(ledger=ledger_id, prepared_source_evidence=identity(args.prepared / 'source_evidence.json'),
                              fixed_checks=identity(args.fixed_checks),
                              dense_checks=identity(args.dense_checks), policy=identity(policy_path), **hashes),
                  candidates=candidates, accuracy_accepted=False,
                  limits=['Fixed mesh counts use the historical skin subsets, checking noncoplanar crossings only.',
                          'S09 is sampled on the registered evidence frames. Allowed local model crossings do not establish physical contact.',
                          'COM, specific H and rotation summaries are conditional model diagnostics. Source direction hypotheses retain their uncertainty.'])
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        handle.write(payload)


if __name__ == '__main__':
    main()
