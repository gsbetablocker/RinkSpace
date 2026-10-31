"""Append independently captured shot packets to the frozen six-shot problem.

Only new curve nuisance stations are initialized. Existing source arrays,
trajectories, physical rigs and the selected scene are never revised here.
"""
from pathlib import Path
import argparse
import copy
import json

import numpy as np

import prepare_native_multicamera as prepared
from rinkspace.core.native import NativeCore
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline
from rinkspace.path_policy import guard_project_write
from rinkspace.session.gpu_geometry import NativeSceneGPU, initialize_curve_stations

HERE = Path(__file__).resolve().parent
BASELINE_RECORD = Path('D:/CodexData/RinkSpace_Data/runs/native_programme_20261031/baseline.json')
SCOPE_PATH = HERE/'rebuild_scope.json'
SEED_FIELDS = {'id', 'camera_index', 'reference_pts', 'time_base_num', 'time_base_den',
               'first_control_time', 'control_dt', 'knot_stride', 'motion_reference_time',
               'controls_ref', 'independent_source_capture', 'source_video_sha256'}


def same_bytes(actual, expected, label):
    if (actual.dtype != expected.dtype or actual.shape != expected.shape
            or actual.tobytes() != expected.tobytes()):
        raise ValueError('Byte preservation failed: '+label)


def lossless_cast(value, dtype, label):
    result = value.astype(dtype)
    if not np.array_equal(result, value):
        raise ValueError('New packet cannot preserve baseline dtype: '+label)
    return result


def integer(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError('Expected integer '+label)
    return int(value)


def validate_points(z, mapping):
    if set(z) != prepared.FIELDS:
        raise ValueError('Only the current seven unweighted point fields are permitted')
    count = len(z['pts'])
    if not count:
        raise ValueError('Each shot packet must contain real source observations')
    for key, value in z.items():
        shape = (count, 2) if key == 'uv' else (count,)
        if value.shape != shape or value.dtype.hasobject:
            raise ValueError('Invalid point array: '+key)
    for key in ('pts', 'kind', 'source_rows', 'path_indices'):
        if z[key].dtype.kind not in 'iu':
            raise ValueError('Expected integer point array: '+key)
    if z['observation_ids'].dtype.kind not in 'US' or len(np.unique(z['observation_ids'])) != count:
        raise ValueError('Observation IDs must be unique strings')
    if not np.isin(z['kind'], [0, 1]).all():
        raise ValueError('Unknown observation kind')
    if (not np.isfinite(z['uv']).all() or np.any(z['uv'] < 0)
            or np.any(z['uv'] > [1919, 1079]) or not np.isfinite(z['stations']).all()):
        raise ValueError('Invalid source UV or station')
    paths = mapping.get('path_ids')
    if (not isinstance(paths, list) or not paths
            or any(not isinstance(p, str) or not p for p in paths) or len(set(paths)) != len(paths)):
        raise ValueError('Mapping needs unique native path IDs')
    if np.any(z['path_indices'] < 0) or np.any(z['path_indices'] >= len(paths)):
        raise ValueError('Point path index outside mapping')


def source_coverage(pts, window, step, allow_sparse):
    expected = np.arange(window['first_pts'], window['stop_pts_exclusive'], step, dtype=np.int64)
    if (not len(expected) or int(expected[-1]) != window['last_pts']
            or len(expected) != window['native_frame_count']):
        raise ValueError('Scope has an inconsistent native exposure clock')
    observed = np.unique(pts)
    if not np.isin(observed, expected).all():
        raise ValueError('Packet PTS outside the registered shot exposure clock')
    missing = expected[~np.isin(expected, observed)]
    if len(missing) and not allow_sparse:
        raise ValueError('Incomplete shot exposure coverage; use --allow-sparse-bootstrap explicitly')
    # Compact contiguous missing exposure ranges; no synthetic point rows.
    runs = []
    for pts_value in missing:
        p = int(pts_value)
        if runs and p == runs[-1]['last_pts']+step:
            runs[-1]['last_pts'] = p
            runs[-1]['count'] += 1
        else:
            runs.append(dict(first_pts=p, last_pts=p, count=1))
    return dict(expected_exposures=len(expected), observed_exposures=len(observed),
                missing_exposures=len(missing), partial_source_coverage=bool(len(missing)),
                missing_ranges=runs, synthetic_rows_added=0)


def dependency_files(manifest, scene_path, library_path, scene):
    """Only current executable model dependencies; provenance stays opaque."""
    records = {str(scene_path): prepared.digest(scene_path),
               str(library_path): prepared.digest(library_path)}
    for ref in scene.get('advertisement_assets', []):
        path = Path(ref['path'])
        path = (path if path.is_absolute() else scene_path.parent/path).resolve(strict=True)
        if prepared.digest(path) != ref['sha256']:
            raise ValueError('Current scene asset digest differs: '+str(path))
        records[str(path)] = ref['sha256']
    parameterization = manifest.get('camera_parameterization', {})
    for key in ('profile', 'coupling'):
        if key+'_path' in parameterization:
            path = Path(parameterization[key+'_path']).resolve(strict=True)
            digest = prepared.digest(path)
            if digest != parameterization[key+'_sha256']:
                raise ValueError('Camera parameterization digest differs: '+key)
            records[str(path)] = digest
    return records


def read_packet(folder, scope, rig_indices, existing_ids, manifest, native, baseline_points, allow_sparse):
    folder = Path(folder).resolve(strict=True)
    seed = prepared.read(folder/'shot_seed.json')
    if set(seed) != SEED_FIELDS or seed.get('independent_source_capture') is not True:
        raise ValueError('Shot seed must contain only the declared independent capture fields')
    shot_id = seed['id']
    if shot_id in existing_ids or shot_id not in scope:
        raise ValueError('Packet shot must extend the baseline and exist in source scope: '+str(shot_id))
    registered = scope[shot_id]
    camera_index = integer(seed['camera_index'], 'camera_index')
    if registered['rig_id'] not in rig_indices or camera_index != rig_indices[registered['rig_id']]:
        raise ValueError('New shot must reuse its registered baseline physical rig')
    for key, expected in [('time_base_num', 1), ('time_base_den', 12800)]:
        if integer(seed[key], key) != expected:
            raise ValueError('Shot seed native time base differs')
    integer(seed['reference_pts'], 'reference_pts')
    if integer(seed['knot_stride'], 'knot_stride') < 1:
        raise ValueError('Invalid shot knot stride')
    if seed['motion_reference_time'] != 0:
        raise ValueError('Independent shot motion reference must be zero')
    if seed['control_dt'] != manifest['control_dt']:
        raise ValueError('New controls must use the inherited storage control_dt without resampling')
    if not np.isfinite([seed['first_control_time'], seed['control_dt']]).all() or seed['control_dt'] <= 0:
        raise ValueError('Invalid shot control clock')
    if seed['controls_ref'] != 'controls.npy':
        raise ValueError('Independent controls must be local controls.npy')
    controls_path = folder/'controls.npy'
    controls = np.load(controls_path, allow_pickle=False)
    if controls.ndim != 2 or controls.shape[1] != 3 or len(controls) < 4 or not np.isfinite(controls).all():
        raise ValueError('Invalid independent pan/tilt/logf controls')
    local = prepared.read(folder/'mapping.json')
    if local.get('pixel_domain') != 'native_distorted' or local.get('image_size') != [1920, 1080]:
        raise ValueError('Packet native source pixel domain differs')
    if (seed['source_video_sha256'] != manifest['_extension_source_sha256']
            or local.get('source_video_sha256') != seed['source_video_sha256']):
        raise ValueError('Independent packet source video SHA256 differs')
    points_path = folder/'point_observations.npz'
    point_digest = prepared.digest(points_path)
    if local.get('point_input_sha256', point_digest) != point_digest:
        raise ValueError('Packet point input digest differs')
    z = prepared.npz(points_path)
    validate_points(z, local)
    if set(local['path_ids'])-native.paths.keys():
        raise ValueError('Packet path absent from the inherited current native scene')
    if local.get('scene_revision', native.scene_revision) != native.scene_revision:
        raise ValueError('Packet current scene revision differs')
    coverage = source_coverage(z['pts'], registered['processing'], 256, allow_sparse)
    # Casting only the new suffix prevents numpy concatenation from widening
    # old fixed-width IDs or changing any old source array representation.
    z = {key: lossless_cast(value, baseline_points[key].dtype, key) for key, value in z.items()}
    info = {key: seed[key] for key in ('id', 'camera_index', 'reference_pts', 'time_base_num',
            'time_base_den', 'first_control_time', 'knot_stride', 'motion_reference_time')}
    info.update({key: registered['processing'][key] for key in ('first_pts', 'last_pts', 'stop_pts_exclusive')})
    info['source_coverage'] = coverage
    paths = [folder/name for name in ('point_observations.npz', 'mapping.json', 'shot_seed.json', 'controls.npy')]
    return dict(folder=folder, seed=seed, points=z, mapping=local, controls=controls,
                info=info, coverage=coverage, hashes={str(p):prepared.digest(p) for p in paths})


def prepare(args):
    root, resume, out = [Path(p).resolve() for p in (args.run_root, args.resume, args.output)]
    out = guard_project_write(out)
    if out.exists():
        raise FileExistsError('Preserve existing prepared output: '+str(out))
    if out.is_relative_to(prepared.ROOT.resolve()):
        raise ValueError('Prepared arrays belong outside the OneDrive source tree')
    authority = prepared.read(BASELINE_RECORD)
    manifest = prepared.read(root/'joint_manifest.json')
    mapping = prepared.read(root/'mapping.json')
    scene_path = Path(manifest['scene_path']).resolve(strict=True)
    library_path = Path(manifest['native_library']).resolve(strict=True)
    current_files = dict(manifest=root/'joint_manifest.json', mapping=root/'mapping.json',
        points=root/'point_observations.npz', snapshot=resume, scene=scene_path, library=library_path)
    protected = {str(path):prepared.digest(path) for path in current_files.values()}
    for key, path in current_files.items():
        if protected[str(path)] != authority['inputs'][key]['sha256']:
            raise ValueError('Input differs from the selected frozen baseline: '+key)
    if manifest.get('camera_parameterization') != authority.get('camera_parameterization'):
        raise ValueError('Frozen camera parameterization differs from baseline record')
    if [s['id'] for s in manifest['shots']] != authority['source_shots']:
        raise ValueError('Frozen baseline shot partition differs')
    if manifest['point_input_sha256'] != protected[str(current_files['points'])]:
        raise ValueError('Baseline point input digest differs')
    if (manifest['scene_file_sha256'] != protected[str(scene_path)]
            or manifest['native_library_sha256'] != protected[str(library_path)]):
        raise ValueError('Baseline scene/runtime dependency differs')
    scene = prepared.read(scene_path)
    protected.update(dependency_files(manifest, scene_path, library_path, scene))
    protected[str(BASELINE_RECORD.resolve())] = prepared.digest(BASELINE_RECORD)
    protected[str(SCOPE_PATH.resolve())] = prepared.digest(SCOPE_PATH)
    model = SceneParameterModel(scene, scene_path, optimize_scene=True)
    points, base = prepared.npz(current_files['points']), prepared.npz(resume)
    validate_points(points, mapping)
    prepared.validate_snapshot(base, model.initial_state.layout, mapping, manifest, points)
    scope_document = prepared.read(SCOPE_PATH)
    if scope_document['source_clock']['native_pts_step'] != 256:
        raise ValueError('Source scope native clock differs')
    scope = {s['id']:s for s in scope_document['shots']}
    rig_indices = {}
    for shot in manifest['shots']:
        rig_id = scope[shot['id']]['rig_id']
        if rig_id in rig_indices and rig_indices[rig_id] != shot['camera_index']:
            raise ValueError('Baseline physical rig has inconsistent camera indices')
        rig_indices[rig_id] = shot['camera_index']
    core = NativeCore(library_path)
    native = NativeSceneGPU(scene_path, core,
        parameter_state=ParameterState(model.initial_state.layout, base['scene_values']))
    if native.scene_revision != manifest['scene_revision']:
        raise ValueError('Inherited native scene revision differs')
    temporary_manifest = dict(manifest, _extension_source_sha256=scope_document['source']['sha256'])
    packets = [read_packet(folder, scope, rig_indices, {s['id'] for s in manifest['shots']},
                          temporary_manifest, native, points, args.allow_sparse_bootstrap)
               for folder in args.shot_packets]
    if not packets or len({p['info']['id'] for p in packets}) != len(packets):
        raise ValueError('Extension needs unique new shot packets')
    packets.sort(key=lambda p: p['info']['first_pts'])
    if packets[0]['info']['first_pts'] < manifest['shots'][-1]['stop_pts_exclusive']:
        raise ValueError('Extension packets must follow the frozen baseline')
    paths = list(mapping['path_ids']); lookup = {p:i for i,p in enumerate(paths)}
    chunks = [points]; controls = [base['controls']]; stations = [base['stations']]
    shots = copy.deepcopy(manifest['shots']); notes = {}; source_inputs = copy.deepcopy(manifest.get('source_inputs', {}))
    row_offset, control_offset = len(points['pts']), len(base['controls'])
    for packet in packets:
        info, z, local = packet['info'], packet['points'], packet['mapping']
        for path in local['path_ids']:
            if path not in lookup:
                lookup[path] = len(paths); paths.append(path)
        z['path_indices'] = np.asarray([lookup[local['path_ids'][i]] for i in z['path_indices']],
                                      dtype=points['path_indices'].dtype)
        cc = lossless_cast(packet['controls'], base['controls'].dtype, 'controls')
        ss = lossless_cast(z['stations'], base['stations'].dtype, 'numeric stations')
        curve = z['kind'] == 1
        station_note = dict(unresolved=[], groups=0, initialized_rows=0)
        if curve.any():
            camera = dict(info, rig=base['rig'][info['camera_index']].tolist(),
                          control_dt=manifest['control_dt'], image_height=1080)
            initialized, station_note = initialize_curve_stations(native, camera=camera, controls=cc,
                path_ids=[paths[i] for i in z['path_indices'][curve]], stations=ss[curve],
                pts=z['pts'][curve], observed=z['uv'][curve], kinds=z['kind'][curve],
                half_seconds=manifest['max_readout']/2)
            if station_note['unresolved']:
                raise ValueError('New native curve station initialization unresolved: '+str(station_note['unresolved'][:10]))
            ss[curve] = initialized
            station_note['initialized_rows'] = int(curve.sum())
        same_bytes(ss[~curve], z['stations'][~curve], 'new fixed-point stations')
        info.update(row_begin=row_offset, row_stop=row_offset+len(z['pts']),
                    control_begin=control_offset, control_stop=control_offset+len(cc))
        row_offset, control_offset = info['row_stop'], info['control_stop']
        if info['first_pts'] < shots[-1]['stop_pts_exclusive']:
            raise ValueError('New shot ranges overlap')
        shots.append(info); chunks.append(z); controls.append(cc); stations.append(ss)
        protected.update(packet['hashes'])
        source_inputs[info['id']] = dict(path=str(packet['folder']/'point_observations.npz'),
            sha256=packet['hashes'][str(packet['folder']/'point_observations.npz')],
            independent_source_capture=True, source_video_sha256=packet['seed']['source_video_sha256'])
        notes[info['id']] = dict(packet=str(packet['folder']), source_coverage=packet['coverage'],
            station_initialization=station_note, camera_index=info['camera_index'],
            controls_imported_only_from_independent_packet=True)
    joined = {key:np.concatenate([chunk[key] for chunk in chunks]) for key in prepared.FIELDS}
    if len(np.unique(joined['observation_ids'])) != len(joined['pts']):
        raise ValueError('Cross-shot observation IDs collide')
    combined_mapping = copy.deepcopy(mapping); combined_mapping['path_ids'] = paths
    timeline = ShotMotionTimeline(shots, manifest['control_dt'], manifest['first_control_time'])
    if timeline.max_readout()+1e-10 < manifest['max_readout']:
        raise ValueError('Independent controls lack full-shot inherited readout support')
    timeline.assert_support_separation(joined['pts'], manifest['max_readout'])
    for key in prepared.FIELDS:
        same_bytes(joined[key][:len(points['pts'])], points[key], 'baseline '+key)
    result = {key:value.copy() for key,value in base.items()}
    result.update(controls=np.concatenate(controls), stations=np.concatenate(stations),
                  source_rows=joined['source_rows'].copy())
    same_bytes(result['controls'][:len(base['controls'])], base['controls'], 'baseline controls')
    same_bytes(result['stations'][:len(base['stations'])], base['stations'], 'baseline stations')
    for name in ('rig', 'scene_values'):
        same_bytes(result[name], base[name], name)
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out/'point_observations.npz', **joined)
    combined = copy.deepcopy(manifest)
    combined.update(shots=shots, point_input_sha256=prepared.digest(out/'point_observations.npz'),
        baseline_snapshot=str(resume), baseline_sha256=protected[str(resume)],
        current_resume_snapshot=str(out/'initial_pixels.npz'), source_inputs=source_inputs,
        partial_source_coverage=any(p['coverage']['partial_source_coverage'] for p in packets),
        extension_preparation=dict(schema='rinkspace.independent_shot_extension.v1',
            baseline_record=str(BASELINE_RECORD), new_shots=[p['info']['id'] for p in packets],
            allow_sparse_bootstrap=bool(args.allow_sparse_bootstrap), shot_notes=notes,
            source_pixels_or_pts_generated=False, inherited_scene_and_rigs_changed=False,
            fit_executed=False, user_visual_acceptance=False))
    if 'point_input_sha256' in combined_mapping:
        combined_mapping['point_input_sha256'] = combined['point_input_sha256']
    prepared.bind_snapshot(result, model.initial_state.layout, combined_mapping, combined)
    prepared.validate_snapshot(result, model.initial_state.layout, combined_mapping, combined, joined)
    np.savez(out/'initial_pixels.npz', **result)
    prepared.write(out/'mapping.json', combined_mapping)
    prepared.write(out/'joint_manifest.json', combined)
    saved = prepared.npz(out/'initial_pixels.npz')
    prepared.validate_snapshot(saved, model.initial_state.layout, combined_mapping, combined, joined)
    for path, digest in protected.items():
        if prepared.digest(path) != digest:
            raise RuntimeError('Protected baseline/model/packet changed: '+path)
    receipt = dict(schema='rinkspace.independent_shot_extension_receipt.v1', output=str(out),
        new_shots=[p['info']['id'] for p in packets], rows=len(joined['pts']),
        partial_source_coverage=combined['partial_source_coverage'],
        baseline_source_fields_byte_preserved=True, baseline_controls_and_stations_byte_preserved=True,
        scene_rigs_and_parameterization_preserved=True, model_dependencies_sha256=protected,
        only_new_kind1_stations_initialized=True, seven_unweighted_fields=True,
        snapshot_bind_and_validate_passed=True, fit_executed=False)
    prepared.write(out/'import_receipt.json', receipt)
    print(json.dumps(dict(output=str(out), initial_state=str(out/'initial_pixels.npz'),
        rows=len(joined['pts']), partial_source_coverage=combined['partial_source_coverage'], fit_executed=False)), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'resume', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--shot-packets', type=Path, nargs='+', required=True)
    parser.add_argument('--allow-sparse-bootstrap', action='store_true')
    prepare(parser.parse_args())
