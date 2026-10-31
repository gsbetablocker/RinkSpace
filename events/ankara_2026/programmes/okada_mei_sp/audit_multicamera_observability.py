"""Read-only data information for the current prepared multicamera problem.

Reuses the production activity/direction configuration and the existing
profile_camera pseudoinverse diagnostic. No optimizer is attached and no
station is refined. Physical working bounds are not measurement information.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

import numpy as np

os.environ.setdefault('CUDA_PATH', r'D:/Program Files/CUDA')
import prepare_native_multicamera as prepared
import fit_native_s01_joint as single
from audit_native_s01_camera_observability import profile_camera
from rinkspace.core.native import NativeCore
from rinkspace.domain.camera_experiment import CAMERA_NAMES, load_camera_experiment, build_camera_directions
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline, continue_exposure_halos
from rinkspace.domain.camera_coupling import read_camera_coupling, build_multicamera_directions
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.path_policy import guard_project_write
from rinkspace.session.gpu_geometry import NativeSceneGPU, scene_coordinate_charts


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def chart_steps():
    return np.r_[np.array([.001]*6+[.001]*3+[.00025]*3+[.00025]*3+
                         [1., 1., .001, 1., 1., .03, .03, .1, .1, .003, .003, .001]),
                 .05, .1, .003, .003]


def camera_records(motion_columns, cameras, indices, steps):
    records = []
    column = motion_columns
    for camera, active in enumerate(indices):
        for chart in active:
            records.append(dict(column=column, camera_index=camera,
                camera_id=cameras[camera]['id'], chart_index=int(chart),
                parameter=CAMERA_NAMES[chart], name=cameras[camera]['id']+'/'+CAMERA_NAMES[chart],
                chart_step=float(steps[chart]),
                group='mechanical' if chart < 15 else 'optical'))
            column += 1
    return records


def project_station_tangents(raw, kind):
    """Same independent-station projection as the existing S01 information audit."""
    q = raw.shape[-1]-1
    columns = np.flatnonzero(np.any(raw[:, :, :q] != 0, axis=(0, 1)))
    small = raw[:, :, columns].copy()
    tangent = raw[:, :, q]
    norm2 = np.einsum('ij,ij->i', tangent, tangent)
    curve = np.asarray(kind) == 1
    free = curve & (norm2 > 0)
    dot = np.einsum('ij,ijk->ik', tangent[free], small[free])
    small[free] -= tangent[free, :, None]*(dot/norm2[free, None])[:, None, :]
    return columns, small, int(np.count_nonzero(curve & (norm2 == 0)))


def summarize_profile(normal, target, names, all_names, records):
    result = profile_camera(normal, np.asarray(target), names)
    by_column = {r['column']: r for r in records}
    modes = []
    for mode in range(min(3, len(target))):
        vector = result['eigenvectors'][:, mode]
        camera = []
        for j in np.argsort(abs(vector))[::-1][:8]:
            col = target[j]
            record = by_column[col]
            norm = result['scale'][col]
            solver = float(vector[j]/norm) if norm else None
            camera.append(dict(parameter=names[j], normalized_column_coefficient=float(vector[j]),
                solver_coordinate_coefficient=solver,
                native_chart_coefficient=None if solver is None else solver*record['chart_step']))
        compensation = -result['solution']@vector
        nuisance = [dict(parameter=all_names[result['nuisance'][j]],
                         normalized_column_coefficient=float(compensation[j]))
                    for j in np.argsort(abs(compensation))[::-1][:8]]
        modes.append(dict(eigenvalue=float(result['eigenvalues'][mode]),
                          target=camera, compensating_nuisance=nuisance))
    return dict(**result['summary'], columns=list(map(int, target)),
                soft_modes=modes,
                normalization='Native solver chart, then L2 column normalization after free-station projection'), result


def main(args):
    output = guard_project_write(args.output)
    normal_path = guard_project_write(output.with_suffix('.npz'))
    log_path = guard_project_write(args.log)
    for path in (output, normal_path, log_path):
        if path.exists():
            raise FileExistsError('Preserve existing diagnostic: '+str(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open('x', encoding='utf-8', buffering=1) as log:
        def emit(event, **values):
            text = json.dumps(dict(event=event, elapsed_seconds=time.monotonic()-started,
                                   **values), ensure_ascii=False, allow_nan=False)
            log.write(text+'\n')
            print(text, flush=True)

        emit('read_only_audit_start', log=str(log_path), optimizer_started=False)
        m = prepared.read(args.run_root/'joint_manifest.json')
        mapping = prepared.read(args.run_root/'mapping.json')
        input_path = args.run_root/'point_observations.npz'
        input_hash = prepared.digest(input_path)
        state_hash = prepared.digest(args.state)
        if input_hash != m['point_input_sha256']:
            raise ValueError('Current point input differs')
        for key, hash_key in [('native_library', 'native_library_sha256'), ('scene_path', 'scene_file_sha256')]:
            if prepared.digest(Path(m[key])) != m[hash_key]:
                raise ValueError('Current runtime/model differs: '+key)
        z = prepared.npz(input_path)
        v = prepared.npz(args.state)
        model = SceneParameterModel(prepared.read(m['scene_path']), Path(m['scene_path']), optimize_scene=True)
        prepared.validate_snapshot(v, model.initial_state.layout, mapping, m, z)
        state = ParameterState(model.initial_state.layout, v['scene_values'])
        original = {key: array_digest(v[key]) for key in ('rig', 'controls', 'stations', 'scene_values')}
        uv_hash = array_digest(z['uv'])
        pts_hash = array_digest(z['pts'])
        core = NativeCore(m['native_library'])
        timeline = ShotMotionTimeline(m['shots'], m['control_dt'], m['first_control_time'])
        times = timeline.row_times(z['pts'])
        timeline.assert_support_separation(z['pts'], m['max_readout'])
        camera_ids = np.empty(len(z['pts']), np.int32)
        for shot in m['shots']:
            camera_ids[shot['row_begin']:shot['row_stop']] = shot['camera_index']
            block = v['controls'][shot['control_begin']:shot['control_stop']]
            interval = [(shot[k]-shot['reference_pts'])*shot['time_base_num']/shot['time_base_den']
                        for k in ('first_pts', 'last_pts')]
            closed = continue_exposure_halos(block, shot['first_control_time'], m['control_dt'], interval)
            if not np.allclose(closed, block, rtol=2e-12, atol=2e-9):
                raise ValueError('State exposure halo differs: '+shot['id'])
        paths = [mapping['path_ids'][i] for i in z['path_indices']]
        native = NativeSceneGPU(m['scene_path'], core, parameter_state=state)
        motion, labels = timeline.directions(continue_endpoint_cubics=True)
        nm = len(labels)
        parameterization = m.get('camera_parameterization', {})
        profile_path = Path(parameterization.get('profile_path') or prepared.HERE/'camera_experiments/joint/square_center_pp_joint.json')
        coupling = read_camera_coupling(parameterization['coupling_path']) if parameterization.get('coupling_path') else None
        rig = v['rig'].copy()
        indices = []
        for i in range(len(m['cameras'])):
            rig[i], ix, experiment = load_camera_experiment(profile_path, rig[i])
            indices.append(ix)
        if not np.array_equal(rig, v['rig']):
            raise ValueError('Audit profile would overwrite supplied rig values')
        dependencies = {b['id'] for b in state.blocks.values() if b['kind'] in ('board_shape', 'shared_geometry')}
        for path in set(paths):
            if '::' in path:
                dependencies.update(model.dependencies(path))
        active = state.active_columns(dependencies)
        steps = chart_steps()
        camera_rd, camera_layout = build_multicamera_directions(rig, indices, steps, coupling)
        nc = camera_rd.shape[1]
        q = nm+nc+len(active)
        directions = np.zeros((len(v['controls']), 3, q))
        directions[:, :, :nm] = .001*motion
        del motion
        rd = np.zeros((31*len(indices), q))
        sd = np.zeros((len(state.values), q))
        charts = scene_coordinate_charts(state)
        steps = chart_steps()
        records = []
        for r in camera_layout['column_records']:
            members = r['camera_indices']; chart = CAMERA_NAMES.index(r['parameter'])
            identity = '+'.join(m['cameras'][i]['id'] for i in members)
            records.append(dict(column=nm+r['column'], camera_index=members[0], camera_indices=members,
                camera_id=identity, chart_index=chart, parameter=r['parameter'],
                name=identity+'/'+r['parameter'], chart_step=float(steps[chart]),
                group='mechanical' if chart<15 else 'optical'))
        rd[:, nm:nm+nc] = camera_rd
        column_names = ['motion/'+str(tuple(label)) for label in labels]+[r['name'] for r in records]
        for j, k in enumerate(active):
            block = next(b for b in state.blocks.values() if b['available'] and b['offset'] <= k < b['offset']+b['size'])
            kind = block['kind']
            step = .01 if charts[k] > 0 or kind == 'print_size' else (.0002 if kind == 'print_placement' and k-block['offset'] < 2 else (.001 if kind == 'print_placement' else (.002 if kind.startswith('print_') else .001)))
            sd[k, nm+nc+j] = step
            column_names.append(block['id']+'/'+str(k-block['offset']))
        activity = prepared.read(args.run_root/'joint_fit/activity.json')
        if (nm, nc, len(active), q) != (activity['motion_columns'], activity.get('unique_camera_columns',sum(activity['camera_columns_per_rig'])), activity['scene_columns'], activity['retained_columns']):
            raise ValueError('Current declared activity differs')
        if [list(ix) for ix in indices] != activity['active_camera_chart_indices_per_rig'] or labels != [tuple(x) for x in activity['motion_labels']]:
            raise ValueError('Current activity column identities differ')
        metres = prepared.read(prepared.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
        bounds = [single.mount_configuration(metres, r) for r in rig]
        lower = np.concatenate([b[0] for b in bounds])
        upper = np.concatenate([b[1] for b in bounds])
        for i in range(len(rig)):
            for k in (26, 27, 28, 29, 30):
                lower[i*43+k] = upper[i*43+k] = rig[i, k]
        curve = np.flatnonzero(z['kind'] == 1)
        local_ids = np.full(len(z['pts']), -1, np.int32)
        local_ids[curve] = np.arange(len(curve))
        camera = {**prepared.read(m['cameras'][0]['camera_definition']),
                  'first_control_time': timeline.first, 'control_dt': timeline.dt}
        normal = np.zeros((q, q))
        zero_tangents = 0
        peak_probe_host_bytes = 0
        endpoint_rows = int(np.count_nonzero((z['kind'] == 1) & ((v['stations'] <= 0) | (v['stations'] >= 1))))
        emit('activity', rows=len(z['pts']), motion_columns=nm, camera_columns=nc,
             scene_columns=len(active), retained_columns=q, normal_bytes=normal.nbytes,
             active_camera_chart_indices=[list(ix) for ix in indices])
        with native.resident(camera=camera, rigs=rig, camera_ids=camera_ids, controls=v['controls'],
                path_ids=paths, stations=v['stations'], pts=z['pts'], observed=z['uv'],
                scene_directions=sd, motion_times=times) as resident:
            resident.configure_camera_parameters(directions=rd, lower=lower, upper=upper, max_readout=m['max_readout'])
            for i in range(len(rig)):
                resident.configure_mount_constraints(side=m['cameras'][i].get('mount_side', 'negative_y'), camera_index=i)
                windows = timeline.frame_windows()[[j for j, s in enumerate(m['shots']) if s['camera_index'] == i]]
                resident.configure_camera_envelope(**bounds[i][2], frame_time_intervals=windows, camera_index=i)
            plan = resident.configure_pixel_linearization(directions=directions, local_ids=local_ids,
                                                          local_scales=np.full(len(curve), .001))
            linear = resident.linearize('cuda')
            if linear['summary']['failed_rows'] or plan['objective'] != 'unweighted_squared_pixel_distance':
                raise ValueError('Invalid current linearization')
            emit('linearized', summary=linear['summary'], native_sparse_jacobian_bytes=plan['jacobian_values']*8)
            order = np.argsort(z['pts'], kind='stable')
            frames, counts = np.unique(z['pts'][order], return_counts=True)
            begin = 0
            for frame_index, count in enumerate(counts):
                rows = order[begin:begin+count]
                begin += count
                raw = resident.probe_rows(rows)
                peak_probe_host_bytes = max(peak_probe_host_bytes, raw.nbytes)
                columns, small, zero = project_station_tangents(raw, z['kind'][rows])
                zero_tangents += zero
                flat = small.reshape(-1, len(columns))
                normal[np.ix_(columns, columns)] += flat.T@flat
                if frame_index % 100 == 0 or frame_index == len(frames)-1:
                    emit('accumulate', frames_done=frame_index+1, frames_total=len(frames),
                         latest_frame_columns=len(columns), source_seconds=float(frames[frame_index]/12800))
            exported = resident.export_model()
            if not np.array_equal(exported['rig'], v['rig']) or not np.array_equal(resident.export_scene_parameters(), v['scene_values']):
                raise RuntimeError('Read-only diagnostic altered model values')
            if hasattr(resident, 'solver_info'):
                raise RuntimeError('Read-only diagnostic unexpectedly attached an optimizer')
        profiles = {}
        for camera_index, camera in enumerate(m['cameras']):
            for group in ('mechanical', 'optical'):
                subset = [r for r in records if camera_index in r['camera_indices'] and r['group'] == group]
                name = camera['id']+'/'+group
                emit('profile_start', name=name, target_columns=len(subset))
                profiles[name], _ = summarize_profile(normal, [r['column'] for r in subset],
                    [r['name'] for r in subset], column_names, records)
                emit('profile_finished', name=name, eigenvalues=profiles[name]['camera_profile_eigenvalues'])
        pair_records = [r for r in records if r['camera_id'] == 'secondary_long_side' and r['parameter'] in ('pivot_z', 'lever_y')]
        if len(pair_records) == 2:
            pair_name = 'secondary_long_side/pivot_z_lever_y'
            emit('profile_start', name=pair_name, target_columns=2)
            profiles[pair_name], pair = summarize_profile(normal, [r['column'] for r in pair_records],
                [r['name'] for r in pair_records], column_names, records)
            if args.baseline:
                previous = prepared.npz(args.baseline)
                delta = v['rig'][1, [17, 19]]-previous['rig'][1, [17, 19]]
                solver_delta = delta/np.array([r['chart_step'] for r in pair_records])
                normalized_delta = pair['scale'][[r['column'] for r in pair_records]]*solver_delta
                profiles[pair_name]['baseline_delta_local_tangent'] = dict(
                    baseline=str(args.baseline), baseline_sha256=prepared.digest(args.baseline),
                    native_delta=delta.tolist(), conditional_metre_delta=(delta*metres).tolist(),
                    profiled_pixel_derivative_norm=float(np.sqrt(max(0., normalized_delta@pair['profiled']@normalized_delta))),
                    separate_pixel_derivative_norms=(abs(normalized_delta)*np.sqrt(np.maximum(0., np.diag(pair['profiled'])))).tolist(),
                    limitation='Finite state difference evaluated only in the candidate local tangent with free nuisance directions; not a nonlinear forward comparison or feasible-bound certificate.')
        if prepared.digest(args.state) != state_hash or prepared.digest(input_path) != input_hash:
            raise RuntimeError('Input files changed during diagnostic')
        if original != {key: array_digest(v[key]) for key in original} or array_digest(z['uv']) != uv_hash:
            raise RuntimeError('Input numeric arrays changed during diagnostic')
        np.savez(normal_path, normal=normal)
        report = dict(schema='rinkspace.native_multicamera_observability.v1',
            state=str(args.state), state_sha256=state_hash, point_input=str(input_path), point_input_sha256=input_hash,
            source_uv_sha256=uv_hash, source_pts_sha256=pts_hash,
            native_library=m['native_library'], native_library_sha256=m['native_library_sha256'],
            scene_path=m['scene_path'], scene_file_sha256=m['scene_file_sha256'],
            rows=len(z['pts']), frames=len(frames), retained_columns=q, motion_columns=nm,
            camera_columns_per_rig=list(map(len, indices)), unique_camera_columns=nc, camera_coupling=coupling, scene_columns=len(active),
            conditional_metres_per_L=metres, column_names=column_names, camera_column_records=records,
            normal_path=str(normal_path), normal_sha256=prepared.digest(normal_path),
            normal_shape=list(normal.shape), normal_units='unweighted squared original pixels per scaled native solver coordinate pair',
            curve_station_endpoint_rows_profiled_as_free=endpoint_rows,
            zero_curve_station_derivative_rows=zero_tangents,
            native_corner_station_tangent_selection_retained=True,
            peak_host_probe_bytes=peak_probe_host_bytes, native_sparse_jacobian_bytes=plan['jacobian_values']*8,
            full_jacobian_or_component_normal_exported=False,
            read_only=dict(optimizer_attached=False, optimizer_steps=0, station_refinement_calls=0,
                           accepted_state_writes=0, input_files_hash_unchanged=True,
                           input_numeric_arrays_bitwise_unchanged=True, exported_rig_scene_unchanged=True),
            data_information=dict(damping=0., observation_weights=False, parameter_priors=False,
                                  physical_inequality_boundaries_used_as_measurements=False),
            method='Existing independent-station native tangent projection, zero-column-pruned frame Gram accumulation, existing profile_camera column-normalized pseudoinverse Schur of all remaining motion/camera/scene columns.',
            limitation='Local first-order data ambiguity at delivered stations and native corner-side choices. No calibrated covariance, hardware plausibility, nonlinear fit or feasible inequality-direction certificate.',
            profiles=profiles, elapsed_seconds=time.monotonic()-started)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        emit('finished', report=str(output), normal=str(normal_path), read_only=report['read_only'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'state', 'output', 'log'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    main(parser.parse_args())
