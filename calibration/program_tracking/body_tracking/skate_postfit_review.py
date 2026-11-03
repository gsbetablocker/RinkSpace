"""Read-only, same-PTS body differences after the full available S01 fit.

Finite source samples describe changes; they do not certify C2 continuity,
physical feasibility, source accuracy, or user acceptance.
"""
from pathlib import Path
import argparse
import json

import numpy as np

from .opening_timing import QUIET_BODY_JOINTS
from .opening_timing_review import snapshot
from .skate_review import _array_identity, _identical, _jsonable, _stats


S01_PTS = np.arange(283136, 575489, 256, dtype=np.int64)
OLD_BOUNDARIES = (28.12, 34.8, 35.2, 36.8, 37.2, 43.8)
SOURCE_FIELDS = ('pts', 'source_pts', 'time_base', 'source_lower_body_inferred',
    'opening_inferred_mask', 'association_corrected_pts', 'contact_state',
    'user_accepted', 'coordinate_convention', 'observation_ledger_id', 'uv',
    'use', 'sigma_px', 'lower_body_inferred', 'source_id', 'source_qualified')


def _angle(left, right):
    """SO(3) distance in degrees, stable both near zero and near pi."""
    relative = np.asarray(right) @ np.swapaxes(np.asarray(left), -1, -2)
    skew = np.stack((relative[..., 2, 1]-relative[..., 1, 2],
                     relative[..., 0, 2]-relative[..., 2, 0],
                     relative[..., 1, 0]-relative[..., 0, 1]), axis=-1)
    sine = np.linalg.norm(skew, axis=-1)/2
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1)
    return np.degrees(np.arctan2(sine, cosine))


def _state(packet, pts):
    keys = ('P_world_mid', 'Q_world_mid', 'local_R_mid')
    shapes = ((len(pts), 3), (len(pts), 3, 3), (len(pts), 21, 3, 3))
    result = {}
    for key, shape in zip(keys, shapes):
        value = np.asarray(packet[key], float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('Invalid or nonfinite body array: '+key)
        result[key] = value
    return result


def _joint_stats(values, pts):
    return {str(j+1): _stats(values[:, j], pts) for j in range(values.shape[1])}


def _pose_difference(old, new, pts):
    delta = new['P_world_mid']-old['P_world_mid']
    return dict(pelvis_XYZ_difference_m=delta,
        pelvis_distance_m=_stats(np.linalg.norm(delta, axis=1), pts),
        pelvis_XY_distance_m=_stats(np.linalg.norm(delta[:, :2], axis=1), pts),
        pelvis_abs_axis_difference_m={axis: _stats(abs(delta[:, j]), pts) for j, axis in enumerate('XYZ')},
        root_Q_difference_deg=_stats(_angle(old['Q_world_mid'], new['Q_world_mid']), pts),
        local_R_difference_deg=_joint_stats(_angle(old['local_R_mid'], new['local_R_mid']), pts))


def _range_summary(state, pts):
    P, Q, local = (state[k] for k in ('P_world_mid', 'Q_world_mid', 'local_R_mid'))
    return dict(exposures=len(pts), first_pts=int(pts[0]), last_pts=int(pts[-1]),
        pelvis_net_XYZ_m=P[-1]-P[0], pelvis_XYZ_span_m=np.ptp(P, axis=0),
        pelvis_XY_path_m=float(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1).sum()),
        pelvis_distance_from_first_m=_stats(np.linalg.norm(P-P[0], axis=1), pts),
        root_Q_from_first_deg=_stats(_angle(Q[0], Q), pts),
        local_R_from_first_deg=_joint_stats(_angle(local[0], local), pts))


def _steps(state, pts):
    dt = np.diff(pts)/12800.
    P, Q, local = (state[k] for k in ('P_world_mid', 'Q_world_mid', 'local_R_mid'))
    dP = np.diff(P, axis=0)
    dQ = Q[1:] @ np.swapaxes(Q[:-1], -1, -2)
    dR = local[1:] @ np.swapaxes(local[:-1], -1, -2)
    root = _angle(np.eye(3), dQ); joints = _angle(np.eye(3), dR)
    velocity = dP/dt[:, None]
    acceleration = np.diff(velocity, axis=0)/((dt[1:]+dt[:-1])/2)[:, None]
    return dict(dP=dP, dQ=dQ, dR=dR, root=root, joints=joints, velocity=velocity,
        summary=dict(interval_count=len(dt), interval_end_pts=pts[1:],
            pelvis_step_m=_stats(np.linalg.norm(dP, axis=1), pts[1:]),
            pelvis_speed_m_per_s=_stats(np.linalg.norm(velocity, axis=1), pts[1:]),
            pelvis_discrete_acceleration_m_per_s2=_stats(np.linalg.norm(acceleration, axis=1), pts[1:-1]),
            root_step_deg=_stats(root, pts[1:]), root_speed_deg_per_s=_stats(root/dt, pts[1:]),
            local_step_deg=_joint_stats(joints, pts[1:])))


def compare_packets(old, new, *, expected_pts=S01_PTS, policy_record=None):
    """Compare aligned source packets without fitting or choosing acceptance limits."""
    pts = np.asarray(old['pts'])
    if pts.ndim != 1 or len(pts) < 3 or pts.dtype.kind not in 'iu' or np.any(np.diff(pts) <= 0):
        raise ValueError('Expected ordered integer source PTS')
    if not np.array_equal(pts, new['pts']):raise ValueError('Different source PTS; positional pairing is unsafe')
    if expected_pts is not None and not np.array_equal(pts, expected_pts):raise ValueError('Incomplete available S01 source batch')
    for packet in (old, new):
        if not np.array_equal(packet['time_base'], [1, 12800]):raise ValueError('Unexpected source time base')
    sources = {}
    for key in SOURCE_FIELDS:
        if key not in old and key not in new:continue
        sources[key] = dict(old_present=key in old, new_present=key in new,
            byte_equal=key in old and key in new and _identical(old[key], new[key]),
            old=None if key not in old else _array_identity(old[key]),
            new=None if key not in new else _array_identity(new[key]))
    before, after = _state(old, pts), _state(new, pts)
    steps_old, steps_new = _steps(before, pts), _steps(after, pts)
    quiet = (pts >= round(22.12*12800)) & (pts <= round(22.90*12800))
    opening = None
    if quiet.any():
        selected = [{k: value[quiet] for k, value in state.items()} for state in (before, after)]
        opening = dict(intended_phase='22.12–22.90 body quiet; hands may move',
            quiet_body_joint_ids=list(QUIET_BODY_JOINTS), old=_range_summary(selected[0], pts[quiet]),
            new=_range_summary(selected[1], pts[quiet]), difference=_pose_difference(*selected, pts[quiet]))
        for label, state in zip(('old', 'new'), selected):
            rotations = state['local_R_mid'][:, np.asarray(QUIET_BODY_JOINTS)-1]
            values = _angle(rotations[0], rotations)
            opening[label]['quiet_body_from_first_deg'] = {str(j): _stats(values[:, k], pts[quiet])
                for k, j in enumerate(QUIET_BODY_JOINTS)}
            opening[label]['quiet_body_maximum_change_deg'] = _stats(values.max(axis=1), pts[quiet])
    boundaries = {}
    for second in OLD_BOUNDARIES:
        target = round(second*12800);indices = np.flatnonzero(pts == target)
        if not len(indices):
            boundaries[str(second)] = dict(target_pts=target, available=False)
            continue
        i = int(indices[0]);a, b = max(0, i-2), min(len(pts), i+3)
        item = dict(target_pts=target, available=True, source_pts=pts[a:b],
            difference=_pose_difference({k: v[a:b] for k, v in before.items()},
                                        {k: v[a:b] for k, v in after.items()}, pts[a:b]))
        for name, step in (('old', steps_old), ('new', steps_new)):
            item[name] = dict(interval_start_pts=pts[a:b-1], interval_end_pts=pts[a+1:b],
                pelvis_step_XYZ_m=step['dP'][a:b-1], root_step_deg=step['root'][a:b-1],
                local_step_deg=step['joints'][a:b-1])
            if 0 < i < len(pts)-1:
                item[name]['velocity_after_minus_before_m_per_s'] = step['velocity'][i]-step['velocity'][i-1]
        boundaries[str(second)] = item
    baseline = (policy_record or {}).get('baseline') or {}
    pose_base = baseline.get('pose_cost')
    dt = np.diff(pts)/12800.
    root_delta = _angle(before['Q_world_mid'], after['Q_world_mid'])
    local_delta = _angle(before['local_R_mid'], after['local_R_mid'])
    report = dict(schema='rinkspace.skate-postfit-body-difference.v1', user_accepted=False,
        source_seconds=[float(pts[0]/12800.), float(pts[-1]/12800.)], source_exposures=len(pts),
        source_time_base=[1, 12800], source_fields=sources,
        all_compared_source_fields_byte_equal=all(v['byte_equal'] for v in sources.values()),
        same_exposure_difference=_pose_difference(before, after, pts),
        whole_range=dict(old=_range_summary(before, pts), new=_range_summary(after, pts)),
        opening_quiet=opening, single_step=dict(old=steps_old['summary'], new=steps_new['summary'],
            pelvis_step_change_m=_stats(np.linalg.norm(steps_new['dP']-steps_old['dP'], axis=1), pts[1:]),
            root_increment_change_deg=_stats(_angle(steps_old['dQ'], steps_new['dQ']), pts[1:]),
            local_increment_change_deg=_joint_stats(_angle(steps_old['dR'], steps_new['dR']), pts[1:])),
        former_window_boundaries=boundaries,
        rotation_matrix_checks={label: {key: dict(maximum_orthogonality_error=float(np.max(abs(
            np.swapaxes(state[key], -1, -2)@state[key]-np.eye(3)))),
            maximum_determinant_error=float(np.max(abs(np.linalg.det(state[key])-1))))
            for key in ('Q_world_mid', 'local_R_mid')} for label, state in (('old', before), ('new', after))},
        policy_pose_guard=dict(baseline_pose_cost=pose_base,
            relative_guard_comparable=pose_base is not None and pose_base > 0,
            meaning='Monitor skips relative pose degradation when baseline is zero or unavailable; these absolute body differences remain reviewable'),
        curves=dict(source_pts=pts, pelvis_difference_XYZ_m=after['P_world_mid']-before['P_world_mid'],
            root_Q_difference_deg=root_delta, local_R_difference_deg=local_delta,
            step_start_pts=pts[:-1], step_end_pts=pts[1:], step_seconds=dt,
            old_pelvis_step_XYZ_m=steps_old['dP'], new_pelvis_step_XYZ_m=steps_new['dP'],
            old_root_step_deg=steps_old['root'], new_root_step_deg=steps_new['root']),
        limitations=['Saved source exposure midpoints only; discrete steps do not prove continuous C2 or bound between-exposure extrema.',
            'Opening quiet-timing and hand-to-waist factors are retained only through the historical base state; they are not live factors in this full-S01 correction, so new body freedoms can alter the accepted opening.',
            'Position is conditional scene metres; rotations are model SO(3) differences, not source-measured motion.',
            'Root increments use world axes; local-joint increments use their parent coordinate convention.',
            'No translation/pose threshold or accuracy acceptance is inferred; same-source pixel and physical reports remain required.'])
    return _jsonable(report)


def review_postfit(root, out):
    """Read immutable snapshots and write only the new run's postfit_audit.json."""
    root, out = Path(root), Path(out)
    old, old_id = snapshot(root/'skate_contact_1103/fitted/result.npz')
    new, new_id = snapshot(out/'fitted/result.npz')
    policy_path = out/'fit/run_policy.json'
    policy = json.loads(policy_path.read_text(encoding='utf8')) if policy_path.is_file() else None
    report = compare_packets(old, new, policy_record=policy)
    report.update(old_result=old_id, new_result=new_id,
                  policy_record_path=str(policy_path) if policy is not None else None)
    checkpoint_path = out/'fit/candidate.npz'
    if checkpoint_path.is_file():
        checkpoint, checkpoint_id = snapshot(checkpoint_path)
        report['candidate_checkpoint'] = dict(identity=checkpoint_id,
            array_identities={key: _array_identity(value) for key, value in checkpoint.items()})
    ledger_path = root/'boot_integration_1102/observations.npz'
    ledgers = {}
    if ledger_path.is_file():
        source, source_id = snapshot(ledger_path)
        for label, folder in (('old', root/'skate_contact_1103'), ('new', out)):
            path = folder/'observations.npz'
            if path.is_file():
                observed, identity = snapshot(path)
                checks = {key: key in observed and _identical(value, observed[key]) for key, value in source.items()}
                ledgers[label] = dict(identity=identity, fields_byte_equal=checks,
                    all_source_fields_byte_equal=all(checks.values()), extra_fields=sorted(set(observed)-set(source)))
            else:ledgers[label] = dict(present=False, all_source_fields_byte_equal=None)
        report['external_observation_ledger'] = dict(source=source_id, comparisons=ledgers)
    else:report['external_observation_ledger'] = dict(present=False, all_source_fields_byte_equal=None)
    (out/'postfit_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    return report


def verify_control_replay(project, root, out, *, native_library=None, chunk_size=32):
    """Replay all source exposures from OneDrive controls; write diagnostics only."""
    from dataclasses import asdict
    import os
    from rinkspace.core.native import NativeCore
    from .rebuild_prepare import digest
    from .rebuild_exposure_body import moving_projection
    from .skate_binding import TEMPLATE_VERSION, BINDING_VERSION
    from .skate_integration import build, restore_checkpoint_layers

    project, root, out = map(Path, (project, root, out))
    if chunk_size < 1:raise ValueError('chunk_size must be positive')
    programme = project/'events/ankara_2026/programmes/okada_mei_sp'
    saved, control_id = snapshot(programme/'skate_s01_1103.npz')
    fitted, fitted_id = snapshot(out/'fitted/result.npz')
    library = str(native_library or root.parent/'body_perf_1103/native_cpu/rinkspace_core.dll')
    NativeCore(library)
    prior_library = os.environ.get('RINKSPACE_BODY_NATIVE_LIBRARY')
    try:
        os.environ['RINKSPACE_BODY_NATIVE_LIBRARY'] = library
        check_out = root/'skate_kernel_checks_1103'
        check_out.mkdir(exist_ok=True)
        ref, camera, info, _ = build(project, root, check_out)
    finally:
        if prior_library is None:os.environ.pop('RINKSPACE_BODY_NATIVE_LIBRARY', None)
        else:os.environ['RINKSPACE_BODY_NATIVE_LIBRARY'] = prior_library
    binding = ref.binding
    saved_camera = json.loads(str(saved['camera_binding_json']))
    current_camera = _jsonable(info)
    path_fields = {'record', 'state_path', 'native_library', 'recorded_native_library'}
    def camera_value(mapping, key):
        value = mapping.get(key)
        return str(Path(value).resolve()).casefold() if key in path_fields and isinstance(value, str) else value
    camera_checks = {key: key in saved_camera and key in current_camera
        and camera_value(saved_camera, key) == camera_value(current_camera, key)
        for key in sorted(set(saved_camera) | set(current_camera))}
    checks = dict(parent_hand_sha256=str(saved['parent_hand_parameters_sha256']) == digest(programme/'opening_hand_1103.npz'),
        parent_timing_sha256=str(saved['parent_timing_parameters_sha256']) == digest(programme/'opening_timing_1103.npz'),
        initial_R_FB_byte_equal=_identical(saved['R_FB_initial'], binding.R0),
        initial_t_FB_byte_equal=_identical(saved['t_FB_initial_solver'], binding.t0),
        base_parameters_equal=json.loads(str(saved['base_skate_parameters_json'])) == asdict(binding.base_parameters),
        shared_x_byte_equal_to_checkpoint=_identical(saved['shared_x'], saved['x'][:17]),
        template_version_equal=str(saved['template_version']) == TEMPLATE_VERSION,
        binding_version_equal=str(saved['binding_version']) == BINDING_VERSION,
        camera_all_fields_equal=all(camera_checks.values()),
        source_pts_byte_equal=_identical(ref.p['pts'], fitted['pts']),
        complete_S01=np.array_equal(fitted['pts'], S01_PTS),
        source_time_base_equal=np.array_equal(fitted['time_base'], [1, 12800]))
    if 'source_pts' in fitted:checks['source_pts_alias_byte_equal'] = _identical(ref.p['pts'], fitted['source_pts'])
    report = dict(schema='rinkspace.skate-all-source-control-replay.v1', controls=control_id,
        fitted=fitted_id, native_library=dict(path=library, sha256=digest(library)),
        binding_checks=checks, camera_fields_equal=camera_checks,
        camera_path_comparison='absolute canonical filenames; all other fields compared unchanged',
        source_exposures=len(ref.times), source_seconds=[float(ref.p['pts'][0]/12800.), float(ref.p['pts'][-1]/12800.)],
        exposure_nodes_per_frame=5, exposure_nodes_total=5*len(ref.times), chunk_size=chunk_size,
        base_layer_count=int(saved.get('base_layer_count', 0)), maximum_absolute_errors={},
        optimization_performed=False, accuracy_accepted=False,
        limitations='Numerical same-state replay only; it certifies neither source accuracy, physical feasibility nor user acceptance.')
    destination = out/'control_replay.json'
    def write():destination.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    if not all(checks.values()):
        report['replay_completed'] = False;write()
        raise ValueError('Control replay binding mismatch: '+', '.join(k for k, equal in checks.items() if not equal))
    binding.x = saved['shared_x'].copy()
    restore_checkpoint_layers(ref, saved, include_primary=True)
    pts = ref.p['pts'];ratio = float(info['length_export_ratio']);layout = binding.layout
    shutter = float(camera.p['rig'][37])
    def compare(key, predicted, frames, unit):
        expected = fitted[key][frames]
        predicted = np.asarray(predicted)
        if predicted.shape != expected.shape:raise ValueError('Replay shape mismatch: '+key)
        if not np.isfinite(predicted).all() or not np.isfinite(expected).all():raise ValueError('Nonfinite replay prediction: '+key)
        difference = abs(predicted-expected)
        where = np.unravel_index(int(np.argmax(difference)), difference.shape)
        maximum = float(difference[where])
        entry = report['maximum_absolute_errors'].setdefault(key, dict(max_abs_error=0., unit=unit,
            elements_compared=0, source_exposures_compared=0, byte_equal=True, worst_pts=None, worst_index=None))
        entry['elements_compared'] += predicted.size
        entry['source_exposures_compared'] += len(frames)
        entry['byte_equal'] &= _identical(predicted, expected)
        if maximum > entry['max_abs_error']:
            entry.update(max_abs_error=maximum, worst_pts=int(pts[frames[where[0]]]),
                         worst_index=[int(frames[where[0]]), *map(int, where[1:])])
    try:
        for first in range(0, len(pts), chunk_size):
            frames = np.arange(first, min(first+chunk_size, len(pts)));times = ref.times[frames]
            mid = ref.world(ref.at(times))
            for key, source in (('P_world_mid', 'P'), ('native22_world_mid', 'native'),
                                ('COCO_world_mid', 'coco'), ('gear_world_mid', 'gear')):
                compare(key, mid[source]*ratio, frames, 'conditional scene metres')
            compare('Q_world_mid', mid['Q'], frames, 'rotation matrix coefficient')
            compare('local_R_mid', mid['local_R'], frames, 'rotation matrix coefficient')
            nodes = ref.world(ref.at((times[:, None]+np.linspace(-.5, .5, 5)*shutter).ravel()))
            uv = moving_projection(camera, nodes['world'].reshape(len(frames), 5, layout.point_count, 3), frames)['uv']
            compare('COCO_uv', uv[:, layout.body_landmarks], frames, 'source pixels')
            compare('equipment_uv', uv[:, :layout.equipment_count].reshape(len(frames), 2, layout.per_foot, 2), frames, 'source pixels')
        report['replay_completed'] = True
        report['all_compared_arrays_byte_equal'] = all(v['byte_equal'] for v in report['maximum_absolute_errors'].values())
    except Exception as error:
        report.update(replay_completed=False, error=str(error));write();raise
    write()
    return _jsonable(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True);parser.add_argument('--output', required=True)
    parser.add_argument('--verify-controls', action='store_true')
    parser.add_argument('--project', default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument('--native-library')
    args = parser.parse_args()
    result = (verify_control_replay(args.project, args.root, args.output, native_library=args.native_library)
              if args.verify_controls else review_postfit(args.root, args.output))
    filename = 'control_replay.json' if args.verify_controls else 'postfit_audit.json'
    print(json.dumps(dict(source_exposures=result['source_exposures'], output=str(Path(args.output)/filename))))
