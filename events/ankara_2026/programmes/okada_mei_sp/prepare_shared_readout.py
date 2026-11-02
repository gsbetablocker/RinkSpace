"""Prepare one shared row-time engineering seed using the existing view as target.

Only pan/tilt controls transfer the previous native projection. Source points,
scene and stations are untouched; this does not start the production fit.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import os

import numpy as np

import prepare_native_multicamera as p
from initialize_camera_repair import evaluate, extent
from prepare_optics_subfamily import identical
from transfer_camera_projection import transfer_projection
from rinkspace.core.native import NativeCore
from rinkspace.domain.camera_coupling import build_multicamera_directions, read_camera_coupling
from rinkspace.domain.camera_experiment import CAMERA_NAMES, load_camera_experiment, read_camera_experiment
from rinkspace.path_policy import guard_project_write


def projection_change(previous, current, shots):
    distances = np.linalg.norm(current['values'][:, 3:5] - previous['values'][:, 3:5], axis=1)
    return dict(all=extent(distances), shots={
        shot['id']: extent(distances[shot['row_begin']:shot['row_stop']]) for shot in shots})


def require_projection(result, label):
    if np.any(result['status']) or not np.isfinite(result['values'][:, 3:5]).all():
        raise ValueError(label + ' projection is invalid')


def main(a):
    parent = a.prepared.resolve()
    state_path = a.state.resolve()
    coupling_path = a.coupling.resolve()
    out = guard_project_write(a.output.resolve())
    if out.exists():
        raise FileExistsError(out)
    manifest_path = parent / 'joint_manifest.json'
    mapping_path = parent / 'mapping.json'
    points_path = parent / 'point_observations.npz'
    m = p.read(manifest_path)
    mapping = p.read(mapping_path)
    parameterization = m['camera_parameterization']
    profile_path = Path(parameterization['profile_path']).resolve()
    previous_coupling = Path(parameterization['coupling_path']).resolve()
    scene_path = Path(m['scene_path']).resolve()
    library_path = Path(m['native_library']).resolve()
    working_envelope = p.camera_working_envelope_path(m)
    protected = {str(path): p.digest(path) for path in (
        state_path, manifest_path, mapping_path, points_path, profile_path,
        previous_coupling, coupling_path, scene_path, library_path, working_envelope)}
    for path, expected in ((points_path, m['point_input_sha256']),
                           (profile_path, parameterization['profile_sha256']),
                           (previous_coupling, parameterization['coupling_sha256']),
                           (scene_path, m['scene_file_sha256']),
                           (library_path, m['native_library_sha256'])):
        if protected[str(path)] != expected:
            raise ValueError('Current input checksum differs: ' + str(path))
    if not np.isfinite(a.readout_seconds) or abs(a.readout_seconds) > m['max_readout']:
        raise ValueError('Readout must be finite and within the current supported time domain')
    profile = read_camera_experiment(profile_path)
    if profile['id'] != 'reference_pupil_joint':
        raise ValueError('Expected the existing reference_pupil_joint profile')
    coupling = read_camera_coupling(coupling_path)
    groups = coupling['groups']
    if (len(groups) != 1 or groups[0]['shared_names'] != ['readout']
            or set(groups[0]['camera_indices']) != {0, 1, 2}):
        raise ValueError('Expected exactly one three-camera readout-only engineering group')
    z = p.npz(points_path)
    if set(z) != p.FIELDS:
        raise ValueError('Expected exactly the seven current plain point fields')
    v = p.npz(state_path)
    if v['rig'].shape != (3, 43):
        raise ValueError('Expected the existing three physical camera rigs')
    names = parameterization['active_names_per_rig']
    if (not isinstance(names, list) or len(names) != 3
            or any(not isinstance(row, list) or len(set(row)) != len(row)
                   or set(row) - set(profile['active_camera_names']) for row in names)):
        raise ValueError('Each physical rig must retain an explicit profile subset')
    selections = []
    for rig, selected in zip(v['rig'], names, strict=True):
        checked, indices, _ = load_camera_experiment(profile_path, rig)
        if not identical(checked, rig):
            raise ValueError('Existing profile would change the parent numeric seed')
        if 'readout' not in selected:
            raise ValueError('All three readout directions must remain active for production fitting')
        selections.append(tuple(k for k in indices if CAMERA_NAMES[k] in selected))
    _, parent_layout = build_multicamera_directions(
        v['rig'], selections, np.ones(31), read_camera_coupling(previous_coupling))
    if parent_layout['unique_camera_columns'] != parameterization['unique_camera_columns']:
        raise ValueError('Parent camera direction count differs from its binding')
    if parent_layout['unique_camera_columns'] != 64:
        raise ValueError('Expected the current 64-column reference-pupil parent')
    print(json.dumps(dict(event='evaluate_parent', parent_state=str(state_path))), flush=True)
    core = NativeCore(library_path)
    old, timeline, model = evaluate(parent, m, v, z, mapping, core)
    p.validate_snapshot(v, model.initial_state.layout, mapping, m, z)
    require_projection(old, 'Parent')
    candidate = {key: value.copy() for key, value in v.items()}
    candidate['rig'][:, 37] = a.readout_seconds
    current, _, _ = evaluate(parent, m, candidate, z, mapping, core)
    require_projection(current, 'Shared-readout initial')
    before_transfer = projection_change(old, current, m['shots'])
    print(json.dumps(dict(event='transfer_previous_view', readout_seconds=a.readout_seconds,
                         projection_change_px=before_transfer)), flush=True)
    candidate, current, transfer = transfer_projection(
        m, z, mapping, model, core, timeline, old, candidate, current,
        working_envelope=working_envelope, channels=('pan', 'tilt'), backend='cuda')
    require_projection(current, 'Transferred shared-readout')
    after_transfer = projection_change(old, current, m['shots'])
    if not identical(candidate['controls'][:, 2], v['controls'][:, 2]):
        raise ValueError('View initialization changed zoom controls')
    if not identical(np.delete(candidate['rig'], 37, axis=1), np.delete(v['rig'], 37, axis=1)):
        raise ValueError('View initialization changed a non-readout camera value')
    for key in v:
        if key not in ('rig', 'controls', 'joint_numeric_dependencies_json'):
            if not identical(v[key], candidate[key]):
                raise ValueError('View initialization changed protected state: ' + key)
    _, layout = build_multicamera_directions(candidate['rig'], selections, np.ones(31), coupling)
    if layout['unique_camera_columns'] != 62:
        raise ValueError('One common row-time must leave exactly 62 camera directions')
    target = deepcopy(m)
    target['camera_parameterization'].update(
        coupling_path=str(coupling_path), coupling_sha256=protected[str(coupling_path)],
        unique_camera_columns=layout['unique_camera_columns'])
    target['current_resume_snapshot'] = str(out / 'initial_pixels.npz')
    target['shared_readout_preparation'] = dict(
        receipt_path=str(out / 'shared_readout_receipt.json'),
        parent_state_sha256=protected[str(state_path)], readout_seconds=a.readout_seconds,
        initializer_target='previous_native_projection_at_same_material_stations',
        source_fit_started=False, source_observations_added=False)
    p.bind_snapshot(candidate, model.initial_state.layout, mapping, target)
    p.validate_snapshot(candidate, model.initial_state.layout, mapping, target, z)
    for path, expected in protected.items():
        if p.digest(path) != expected:
            raise ValueError('Preparation changed its input: ' + path)
    out.mkdir(parents=True)
    os.link(points_path, out / 'point_observations.npz')
    os.link(mapping_path, out / 'mapping.json')
    p.write(out / 'joint_manifest.json', target)
    np.savez_compressed(out / 'initial_pixels.npz', **candidate)
    saved = p.npz(out / 'initial_pixels.npz')
    p.validate_snapshot(saved, model.initial_state.layout, mapping, target, z)
    for key in candidate:
        if not identical(candidate[key], saved[key]):
            raise ValueError('Saved snapshot differs: ' + key)
    if not os.path.samefile(points_path, out / 'point_observations.npz'):
        raise ValueError('Point input must be reused by hardlink')
    receipt = dict(
        schema='rinkspace.shared_readout_preparation.v1', parent_prepared=str(parent),
        parent_state=dict(path=str(state_path), sha256=protected[str(state_path)]),
        parent_manifest_sha256=protected[str(manifest_path)],
        parent_mapping_sha256=protected[str(mapping_path)], protected_input_sha256=protected,
        point_input_sha256=m['point_input_sha256'], point_fields=sorted(z),
        point_input_hardlink=True, all_seven_point_fields_byte_unchanged=True,
        profile=dict(path=str(profile_path), sha256=protected[str(profile_path)]),
        coupling=dict(path=str(coupling_path), sha256=protected[str(coupling_path)]),
        native_library=dict(path=str(library_path), sha256=protected[str(library_path)]),
        initializer_source=dict(path=str(Path(__file__).resolve()), sha256=p.digest(__file__)),
        transfer_helper_sha256=p.digest(Path(__file__).with_name('transfer_camera_projection.py')),
        previous_readout_seconds=v['rig'][:, 37].tolist(),
        initialized_readout_seconds=candidate['rig'][:, 37].tolist(),
        camera_columns=layout['unique_camera_columns'], active_names_per_rig=names,
        local_motion_refinements_preserved={shot['id']: shot['local_motion_refinement']
            for shot in m['shots'] if shot.get('local_motion_refinement')},
        projection_change_before_transfer_px=before_transfer,
        projection_change_after_transfer_px=after_transfer, view_transfer=transfer,
        scene_values_stations_source_rows_byte_unchanged=True,
        nonreadout_rig_byte_unchanged=True, zoom_controls_byte_unchanged=True,
        source_UV_PTS_unchanged=True, source_observations_added=False,
        initializer_targets_not_production_observations=True, fit_started=False,
        snapshot_bound_and_validated=True,
        output_snapshot_sha256=p.digest(out / 'initial_pixels.npz'),
        output_manifest_sha256=p.digest(out / 'joint_manifest.json'))
    p.write(out / 'shared_readout_receipt.json', receipt)
    print(json.dumps(dict(output=str(out), camera_columns=layout['unique_camera_columns'],
        previous_readout_seconds=receipt['previous_readout_seconds'],
        initialized_readout_seconds=receipt['initialized_readout_seconds'],
        projection_change_after_transfer_px=after_transfer,
        receipt=str(out / 'shared_readout_receipt.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'state', 'output', 'coupling'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--readout-seconds', type=float, default=.018)
    main(parser.parse_args())
