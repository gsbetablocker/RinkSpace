"""Prepare an existing radial subfamily warm seed without fitting source data."""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import os

import numpy as np

import prepare_native_multicamera as p
from rinkspace.domain.camera_coupling import build_multicamera_directions, read_camera_coupling
from rinkspace.domain.camera_experiment import apply_camera_experiment, read_camera_experiment
from rinkspace.domain.parameters import SceneParameterModel
from rinkspace.path_policy import guard_project_write


RIG_KEY = 'main_secondary_drop_wide_k2_end_unchanged'
RADIAL_INDICES = (31, 32, 33, 34, 39, 40)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def identical(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def main(a):
    parent = a.prepared.resolve()
    state_path = a.state.resolve()
    rig_path = a.rig_initializer.resolve()
    out = guard_project_write(a.output.resolve())
    if out.exists():
        raise FileExistsError(out)
    manifest_path = parent / 'joint_manifest.json'
    mapping_path = parent / 'mapping.json'
    points_path = parent / 'point_observations.npz'
    m = p.read(manifest_path)
    mapping = p.read(mapping_path)
    state_sha = digest(state_path)
    parameterization = m['camera_parameterization']
    profile_path = Path(parameterization['profile_path']).resolve()
    coupling_path = Path(parameterization['coupling_path']).resolve()
    scene_path = Path(m['scene_path']).resolve()
    protected = {str(path): digest(path) for path in (
        state_path, manifest_path, mapping_path, points_path, profile_path,
        coupling_path, scene_path, rig_path)}
    for path, expected in ((points_path, m['point_input_sha256']),
                           (profile_path, parameterization['profile_sha256']),
                           (coupling_path, parameterization['coupling_sha256']),
                           (scene_path, m['scene_file_sha256'])):
        if protected[str(path)] != expected:
            raise ValueError('Current input checksum differs: ' + str(path))
    v = p.npz(state_path)
    with np.load(rig_path, allow_pickle=False) as warm:
        if not identical(warm['original_rig'], v['rig']):
            raise ValueError('Warm initializer has a different original rig')
        migrated = warm[RIG_KEY].copy()
    if migrated.shape != v['rig'].shape or not np.isfinite(migrated).all():
        raise ValueError('Invalid migrated rig')
    changes = [np.flatnonzero(migrated[i] != rig).tolist()
               for i, rig in enumerate(v['rig'])]
    if len(changes) != 3 or changes[2] or any(set(x) - set(RADIAL_INDICES) for x in changes):
        raise ValueError('Only main/secondary radial values may change')
    if any(40 not in changes[i] or migrated[i, 40] != 0 for i in (0, 1)):
        raise ValueError('Main and secondary wide_k2 must be explicitly removed')
    candidate = {key: value.copy() for key, value in v.items()}
    candidate['rig'] = migrated
    profile = read_camera_experiment(profile_path)
    names = profile['active_camera_names']
    if 'wide_k2' not in names or 'active_names_per_rig' in parameterization:
        raise ValueError('Expected the current global full optical profile')
    active = [[name for name in names if i == 2 or name != 'wide_k2'] for i in range(3)]
    selections = []
    for rig, selected_names in zip(migrated, active, strict=True):
        local_profile = deepcopy(profile)
        local_profile['active_camera_names'] = selected_names
        fixed, indices, _ = apply_camera_experiment(rig, local_profile)
        if not identical(fixed, rig):
            raise ValueError('Global profile would change the migrated numeric seed')
        selections.append(indices)
    coupling = read_camera_coupling(coupling_path)
    _, camera_layout = build_multicamera_directions(migrated, selections, np.ones(31), coupling)
    camera_columns = camera_layout['unique_camera_columns']
    if camera_columns != parameterization['unique_camera_columns'] - 2:
        raise ValueError('Removing two independent wide_k2 coordinates must remove two columns')
    target = deepcopy(m)
    target['camera_parameterization'].update(
        active_names_per_rig=active, unique_camera_columns=camera_columns)
    target['current_resume_snapshot'] = str(out / 'initial_pixels.npz')
    target['optics_subfamily_preparation'] = dict(
        receipt_path=str(out / 'optics_subfamily_receipt.json'),
        parent_state_sha256=state_sha, rig_initializer_sha256=protected[str(rig_path)],
        rig_initializer_key=RIG_KEY, changed_rig_indices=changes,
        warm_mapping_only=True, source_observations_added=False)
    model = SceneParameterModel(p.read(scene_path), scene_path, optimize_scene=True)
    # Validation consumes only the current prepared arrays; no qualification or historical chain.
    with np.load(points_path, allow_pickle=False) as points:
        if set(points.files) != p.FIELDS:
            raise ValueError('Expected exactly the seven current plain point fields')
        p.validate_snapshot(v, model.initial_state.layout, mapping, m, points)
        p.bind_snapshot(candidate, model.initial_state.layout, mapping, target)
        p.validate_snapshot(candidate, model.initial_state.layout, mapping, target, points)
        for key in v:
            if key not in ('rig', 'joint_numeric_dependencies_json') and not identical(v[key], candidate[key]):
                raise ValueError('Non-radial state changed: ' + key)
        out.mkdir(parents=True)
        os.link(points_path, out / 'point_observations.npz')
        os.link(mapping_path, out / 'mapping.json')
        p.write(out / 'joint_manifest.json', target)
        np.savez_compressed(out / 'initial_pixels.npz', **candidate)
        saved = p.npz(out / 'initial_pixels.npz')
        p.validate_snapshot(saved, model.initial_state.layout, mapping, target, points)
        for key in candidate:
            if not identical(candidate[key], saved[key]):
                raise ValueError('Saved snapshot differs: ' + key)
        point_fields = sorted(points.files)
    if not os.path.samefile(points_path, out / 'point_observations.npz'):
        raise ValueError('Point input must be reused by hardlink')
    if digest(out / 'point_observations.npz') != m['point_input_sha256']:
        raise ValueError('Point input bytes changed')
    for path, expected in protected.items():
        if digest(path) != expected:
            raise ValueError('Preparation changed its input: ' + path)
    receipt = dict(
        schema='rinkspace.optics_subfamily_preparation.v1', parent_prepared=str(parent),
        parent_state=dict(path=str(state_path), sha256=state_sha),
        parent_manifest_sha256=protected[str(manifest_path)],
        parent_mapping_sha256=protected[str(mapping_path)],
        point_input_sha256=m['point_input_sha256'], point_fields=point_fields,
        point_input_hardlink=True, all_seven_point_fields_byte_unchanged=True,
        profile=dict(path=str(profile_path), sha256=protected[str(profile_path)]),
        coupling=dict(path=str(coupling_path), sha256=protected[str(coupling_path)]),
        scene=dict(path=str(scene_path), sha256=protected[str(scene_path)]),
        rig_initializer=dict(path=str(rig_path), sha256=protected[str(rig_path)], key=RIG_KEY),
        changed_rig_indices=changes, radial_indices=list(RADIAL_INDICES),
        camera_columns=camera_columns, active_names_per_rig=active,
        scene_controls_stations_byte_unchanged=True, nonradial_rig_byte_unchanged=True,
        end_rig_byte_unchanged=True, original_readout_byte_unchanged=True,
        source_uv_generated=False, source_observations_added=False, fit_started=False,
        snapshot_bound_and_validated=True,
        output_snapshot_sha256=digest(out / 'initial_pixels.npz'),
        output_manifest_sha256=digest(out / 'joint_manifest.json'))
    p.write(out / 'optics_subfamily_receipt.json', receipt)
    print(json.dumps(dict(output=str(out), camera_columns=camera_columns, changed_rig_indices=changes,
                         snapshot_sha256=receipt['output_snapshot_sha256'],
                         receipt_sha256=digest(out / 'optics_subfamily_receipt.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'state', 'rig-initializer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
