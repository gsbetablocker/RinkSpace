"""Merge measured video batches with all previous fit observations and solve.

The 279-frame ledger is separate from the numerical input: a frame with no
measured feature never gains an invented observation merely to reach a count.
"""
from pathlib import Path
import argparse
import copy
import json
import sys
import time
from collections import Counter, defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rinkspace.calibration_companion import Scene
from rinkspace.calibration_companion.joint_scene import JointSceneProblem
from rinkspace.calibration_companion.camera_identity import installation_records
from fit_joint_scene_video import BASE, apply_scene_parameterization, normalize_observations
from prepare_joint_scene_evidence import load, save, digest, OUT

WORK = OUT / 'joint_279'
OLD_PROBLEM = OUT / 'perimeter_camera_support_20260922/problem_ready_for_observation_merge.json'
OLD_FIT = OUT / 'fits/joint_ads111_start0/fit.json'


def exposure(frame):
    return (frame['media_track_id'], int(frame['source_pts']), tuple(frame['image_size_wh']))


def assemble(paths):
    previous = load(OLD_FIT)
    problem = load(OLD_PROBLEM)
    old_ids = {o['id'] for o in problem['observations']}
    old_frames = {f['id'] for f in previous['frames']}
    old_cameras = {f['id']: f['camera'] for f in previous['frames']}
    installation_specs = {i['id']: copy.deepcopy(i) for i in problem['installations']}
    for i in previous['installations']:
        installation_specs[i['id']]['initial_center_xyz'] = i['center_xyz']
        if i.get('radial_k1_image') is not None:
            installation_specs[i['id']]['radial_distortion']['initial_k1_image'] = i['radial_k1_image']
    frames, aliases, canonical, observations, landmarks = {}, {}, {}, {}, {}
    files = [OLD_PROBLEM, *paths]
    original_frame_ledger = []
    for path in files:
        batch = normalize_observations(load(path))
        for item in batch.get('installations', []):
            if item['id'] not in installation_specs:
                installation_specs[item['id']] = copy.deepcopy(item)
        for frame in batch['frames']:
            frame = copy.deepcopy(frame)
            original_frame_ledger.append(dict(frame_id=frame['id'], pool_ordinal=frame.get('pool_ordinal'),
                source_pts=frame['source_pts'], physical_camera_id=frame['physical_camera_id'],
                source_file=str(path), fit_status=frame.get('fit_status')))
            key = exposure(frame)
            fid = canonical.setdefault(key, frame['id'])
            aliases[frame['id']] = fid
            if fid in frames:
                for field in ('physical_camera_id', 'installation_id', 'source_sha256'):
                    if frames[fid].get(field) != frame.get(field):
                        raise ValueError(f'{fid}: conflicting exposure {field}')
                if frame.get('pool_ordinal') is not None:
                    frames[fid]['pool_ordinal'] = frame['pool_ordinal']
            else:
                frame['id'] = fid
                if fid in old_cameras:
                    cam = old_cameras[fid]
                    frame['initial_rvec_world_to_camera'] = cam['rvec_world_to_camera']
                    frame['initial_focal_px'] = cam['K_px'][0][0]
                frames[fid] = frame
        for point in batch.get('board_landmarks', []):
            lid = point['id']
            if lid in landmarks:
                for field in ('advertisement_instance_id', 'side', 'along_parameterization'):
                    if landmarks[lid].get(field) != point.get(field):
                        raise ValueError(f'{lid}: conflicting physical detail {field}')
                # A batch may explicitly permit the known adjacent corner
                # whereas another only names the same straight side. Broaden
                # this domain without inventing a second physical landmark.
                corners = {v for v in (landmarks[lid].get('allow_corner'), point.get('allow_corner')) if v}
                if len(corners) > 1:
                    raise ValueError(f'{lid}: conflicting adjacent corner identity')
                if corners:
                    landmarks[lid]['allow_corner'] = next(iter(corners))
                landmarks[lid]['bounds_along'] = [
                    min(landmarks[lid]['bounds_along'][0], point['bounds_along'][0]),
                    max(landmarks[lid]['bounds_along'][1], point['bounds_along'][1])]
                landmarks[lid].setdefault('merged_parameter_sources', []).append(dict(
                    source=str(path), initial_along=point['initial_along'],
                    bounds_along=point['bounds_along'], allow_corner=point.get('allow_corner')))
            else:
                landmarks[lid] = copy.deepcopy(point)
        for original in batch['observations']:
            obs = copy.deepcopy(original)
            obs['frame_id'] = aliases[obs['frame_id']]
            if obs['id'] in observations:
                if observations[obs['id']] != obs:
                    raise ValueError(f"conflicting observation ID {obs['id']}")
            else:
                observations[obs['id']] = obs
    # Equal physical exposures are one pose. Preserve all distinct measured
    # curves/details, removing only byte-equivalent duplicate pixel records.
    seen, kept = set(), {}
    for oid, obs in observations.items():
        key = (obs['frame_id'], obs['use'], json.dumps(obs['target'], sort_keys=True),
               json.dumps(obs['pixels_uv'], separators=(',', ':')))
        if key not in seen or oid in old_ids:
            kept[oid] = obs
            seen.add(key)
    observations = kept
    fit_by_frame = defaultdict(list)
    for obs in observations.values():
        if obs['use'] == 'fit':
            fit_by_frame[obs['frame_id']].append(obs)
    # Empty frames stay in the accounting ledger, not as unconstrained poses.
    fit_frames = {fid: f for fid, f in frames.items() if fit_by_frame[fid]}
    observations = {oid: o for oid, o in observations.items() if o['frame_id'] in fit_frames}
    for row in original_frame_ledger:
        fid = aliases[row['frame_id']]
        row['canonical_frame_id'] = fid
        row['fit_observation_groups'] = len(fit_by_frame[fid])
        row['actually_contributes_to_fit'] = bool(fit_by_frame[fid])
    for point in landmarks.values():
        point['evidence_observation_ids'] = [o['id'] for o in observations.values()
            if o['use'] == 'fit' and o['target'].get('landmark_id') == point['id']]
    installations = installation_records(fit_frames.values(), installation_specs.values())
    problem.update(frames=list(fit_frames.values()), observations=list(observations.values()),
        board_landmarks=list(landmarks.values()), installations=installations,
        source_observation_files=[str(p.resolve()) for p in files], heldout_shots=[],
        prior_holdout_promotions=dict(shots=['S074', 'S085', 'S087', 'S089'],
            reason='User explicitly included all camera-identified 279 frames in training',
            independent_validation_after_promotion=False),
        batch_source_hashes={str(p): digest(p) for p in files},
        training_manifest_sha256=digest(WORK / 'selected_279_manifest.json'))
    problem = apply_scene_parameterization(Scene(BASE), problem)
    if not old_ids <= set(observations):
        raise ValueError('an existing observation was lost while merging')
    for frame in fit_frames.values():
        if 'initial_rvec_world_to_camera' not in frame or 'initial_focal_px' not in frame:
            raise ValueError(f"{frame['id']}: measured frame still needs an initializer")
    audit = dict(old_frames=len(old_frames), old_observation_groups=len(old_ids),
        all_old_observation_groups_retained=True, total_fit_frames=len(fit_frames),
        total_observation_groups=len(observations),
        by_camera=dict(Counter(f['physical_camera_id'] for f in fit_frames.values())),
        by_target=dict(Counter(o['target']['kind'] for o in observations.values() if o['use'] == 'fit')),
        selected_frame_ledger=[r for r in original_frame_ledger if r['pool_ordinal'] is not None],
        input_hashes=problem['batch_source_hashes'])
    return problem, audit


def warm_start(p, previous_path=OLD_FIT):
    old = load(previous_path)
    by_name = dict(zip(old['parameter_names'], old['parameter_values']))
    values = p.x0.copy()
    for i, name in enumerate(p.parameter_names):
        if name in by_name:
            values[i] = by_name[name]
    name = 'scene:F_R_shared_outer_radius'
    if name in p.parameter_names and name not in by_name:
        radii = [by_name.get(f'scene:{side}{n}_radius', p.scene.features[f'{side}{n}']['radius'])
                 for side in ('F', 'R') for n in range(1, 5)]
        values[p.parameter_names.index(name)] = float(np.mean(radii))
    if np.any(values < p.bounds[0]) or np.any(values > p.bounds[1]):
        raise ValueError('old solution cannot initialize new parameter bounds')
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', nargs='+', type=Path,
        default=[WORK / role / 'observations.json' for role in ('main', 'secondary', 'low')])
    ap.add_argument('--out', type=Path, default=WORK / 'joint_fit')
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--warm-start-fit', type=Path, default=OLD_FIT)
    ap.add_argument('--max-nfev', type=int, default=120)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        problem = load(args.out / 'problem.json')
        audit = load(args.out / 'merge_audit.json')
    else:
        problem, audit = assemble(args.batches)
        save(args.out / 'problem.json', problem)
        save(args.out / 'merge_audit.json', audit)
    p = JointSceneProblem(Scene(BASE), problem)
    start = warm_start(p, args.warm_start_fit)
    problem_hash = digest(args.out / 'problem.json')
    if args.resume:
        checkpoint = load(args.out / 'checkpoint.json')
        if checkpoint['problem_sha256'] != problem_hash or checkpoint['parameter_names'] != p.parameter_names:
            raise ValueError('checkpoint does not belong to this input')
        start = np.asarray(checkpoint['parameter_values'])
    measured = {r['source_pts'] for r in audit['selected_frame_ledger'] if r['actually_contributes_to_fit']}
    launch = dict(selected_frames=279, selected_frames_actually_measured=len(measured),
        total_fit_frames=len(p.frames), fit_observation_groups=sum(o['use'] == 'fit' for o, _ in p.observations),
        variables=len(p.x0), active_scene_variables=p.scene_count,
        scene_parameters=[s['id'] for s in p.scene_specs],
        unseen_parameters=[s['id'] for s in p.retained_parameters],
        by_camera=audit['by_camera'], by_target=audit['by_target'],
        warm_start_fit=str(args.warm_start_fit), warm_start_sha256=digest(args.warm_start_fit),
        problem_sha256=problem_hash, all_initial_residuals_finite=bool(np.isfinite(p.residual(start)).all()))
    save(args.out / 'launch_summary.json', launch)
    print(json.dumps(launch), flush=True)
    if args.prepare_only:
        save(args.out / 'initial_parameters.json', dict(parameter_names=p.parameter_names,
            parameter_values=start.tolist(), problem_sha256=problem_hash))
        return
    beginning = time.monotonic()
    initial = p.summarize(start)
    save(args.out / 'initial_summary.json', initial)
    save(args.out / 'initial_parameters.json', dict(parameter_names=p.parameter_names,
        parameter_values=start.tolist(), problem_sha256=problem_hash))
    iteration = [0]
    initial_residual = p.residual(start)
    best_valid_cost = [float(np.sum(np.sqrt(1. + initial_residual ** 2) - 1.))
                       if initial['fit_geometry_valid'] else float('inf')]
    if initial['fit_geometry_valid']:
        save(args.out / 'best_valid_checkpoint.json', dict(parameter_names=p.parameter_names,
            parameter_values=start.tolist(), problem_sha256=problem_hash,
            cost=best_valid_cost[0], iteration=0, geometry_valid=True))
    def callback(intermediate_result):
        iteration[0] += 1
        record = dict(parameter_names=p.parameter_names, parameter_values=intermediate_result.x.tolist(),
            problem_sha256=problem_hash, cost=float(intermediate_result.cost),
            iteration=iteration[0], elapsed_seconds=round(time.monotonic() - beginning, 2))
        save(args.out / 'checkpoint.json', record)
        # Retain a feasible iterate without accepting an invalid final result.
        # Selection uses training cost and geometric validity only, never checks.
        if record['cost'] < best_valid_cost[0]:
            diagnostic = p.summarize(intermediate_result.x)
            record['geometry_valid'] = diagnostic['fit_geometry_valid']
            if record['geometry_valid']:
                best_valid_cost[0] = record['cost']
                save(args.out / 'best_valid_checkpoint.json', record)
        print(json.dumps({k: v for k, v in record.items() if k not in ('parameter_names', 'parameter_values')}), flush=True)
    result = p.solve(args.max_nfev, callback=callback, x0=start)
    result['before_actual_warm_start'] = initial
    result['elapsed_seconds'] = time.monotonic() - beginning
    result['problem_sha256'] = problem_hash
    result['source_scripts'] = {str(Path(__file__)): digest(__file__),
        'joint_scene.py': digest(ROOT / 'src/rinkspace/calibration_companion/joint_scene.py')}
    save(args.out / 'fit.json', result)
    try:
        save(args.out / 'scene_semantic.json', p.export_scene_data(p.result.x, 'v0.16-joint-279-candidate'))
    except ValueError as error:
        save(args.out / 'export_rejection.json', dict(reason=str(error), model_exported=False))
    print(json.dumps(dict(optimizer=result['optimizer'], before=initial['overall'],
        after=result['after']['overall'], accepted_geometry_candidate=result['accepted_geometry_candidate'],
        elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
