"""Prepare zero ice deviations from a current accepted multicamera state.

This copies current data and accepted numeric state only. It never runs solve,
station refinement, source extraction, source qualification or motion fitting.
"""
from pathlib import Path
import argparse, copy, json, os
import numpy as np
import prepare_native_multicamera as prepared
from rinkspace.core.native import NativeCore
from rinkspace.domain.camera_experiment import load_camera_experiment
from rinkspace.domain.camera_coupling import read_camera_coupling, build_multicamera_directions
from rinkspace.domain.ice_tolerance import enable_ice_tolerances, migrate_scene_state, DEVIATION_COUNT
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.path_policy import guard_project_write
from rinkspace.session.gpu_geometry import NativeSceneGPU


def prepare(parent, snapshot, library, output, scene_output, parameters_output):
    parent, snapshot, library = [Path(p).resolve(strict=True) for p in (parent, snapshot, library)]
    output = Path(output).resolve()
    scene_output, parameters_output = [guard_project_write(p) for p in (scene_output, parameters_output)]
    if any(p.exists() for p in (scene_output, parameters_output, output/'joint_manifest.json')):
        raise FileExistsError('Refuse to overwrite an existing prepared scene or state')
    m = prepared.read(parent/'joint_manifest.json')
    if prepared.digest(parent/'point_observations.npz') != m['point_input_sha256']:
        raise ValueError('Current parent input differs')
    if prepared.digest(m['scene_path']) != m['scene_file_sha256']:
        raise ValueError('Current parent scene differs')
    z = prepared.npz(parent/'point_observations.npz')
    mapping = prepared.read(parent/'mapping.json')
    v = prepared.npz(snapshot)
    original_scene = prepared.read(m['scene_path'])
    old_model = SceneParameterModel(original_scene, Path(m['scene_path']), optimize_scene=True)
    prepared.validate_snapshot(v, old_model.initial_state.layout, mapping, m, z)
    old_state = ParameterState(old_model.initial_state.layout, v['scene_values'])
    changed = enable_ice_tolerances(old_state.apply_scene(original_scene))
    changed['ice_tolerance']['numeric_parent'] = dict(path=str(snapshot), sha256=prepared.digest(snapshot))
    prepared.write(scene_output, changed)
    new_model = SceneParameterModel(changed, scene_output, optimize_scene=True)
    new_state = migrate_scene_state(old_state, new_model.initial_state)
    next_manifest = copy.deepcopy(m)
    next_manifest.update(scene_path=str(scene_output), scene_file_sha256=prepared.digest(scene_output),
        scene_revision=new_state.layout['scene_content_revision'], native_library=str(library),
        native_library_sha256=prepared.digest(library), baseline_snapshot=str(snapshot),
        baseline_sha256=prepared.digest(snapshot), current_resume_snapshot=str(output/'initial_pixels.npz'))
    next_manifest['ice_tolerance_initialization'] = dict(schema='rinkspace.ice_tolerance_initialization.v1',
        parent_run=str(parent), parent_snapshot=str(snapshot), parent_snapshot_sha256=prepared.digest(snapshot),
        deviation_count=DEVIATION_COUNT, deviation_initial_value=0., relative_limit=.01,
        source_fields_unchanged=True, numeric_parent_stop_reason='cost_progress_stagnation_without_convergence',
        fit_executed=False, optimizer_internal_state='not present in parent numeric export; no claim of full LM checkpoint restoration')
    next_mapping = dict(mapping, scene_revision=next_manifest['scene_revision'])
    current = {k:a.copy() for k,a in v.items()}
    current.update(board=new_state.values[:6].copy(), layout=new_state.values[6:6+len(changed['shared_layout'])].copy(),
                   scene_values=new_state.values.copy())
    prepared.bind_snapshot(current, new_state.layout, next_mapping, next_manifest)
    prepared.validate_snapshot(current, new_state.layout, next_mapping, next_manifest, z)
    for key in ('rig','controls','stations','source_rows','board'):
        np.testing.assert_array_equal(current[key], v[key])
    np.testing.assert_array_equal(current['layout'][:39], v['layout'])
    np.testing.assert_array_equal(current['layout'][39:], 0.)
    for identity, block in old_state.blocks.items():
        if block['available']:
            np.testing.assert_array_equal(new_state[identity], old_state[identity])
            if new_state[identity].tobytes()!=old_state[identity].tobytes():
                raise ValueError('Existing parameter bytes changed: '+identity)
    def joint_columns(model, state):
        dependencies={b['id'] for b in state.blocks.values() if b['kind'] in ('board_shape','shared_geometry')}
        for index in np.unique(z['path_indices']):
            identity=mapping['path_ids'][int(index)]
            if '::' in identity:dependencies.update(model.dependencies(identity))
        return len(state.active_columns(dependencies))
    profile = next_manifest['camera_parameterization']
    selected = []
    for rig in current['rig']:
        transformed, indices, _ = load_camera_experiment(profile['profile_path'], rig)
        np.testing.assert_array_equal(transformed, rig)
        selected.append(indices)
    _, camera_layout = build_multicamera_directions(current['rig'], selected, np.ones(31),
        read_camera_coupling(profile['coupling_path']))
    if camera_layout['unique_camera_columns'] != 66:
        raise ValueError('Latest simplified camera model was not inherited')
    old_native = NativeSceneGPU(m['scene_path'], NativeCore(m['native_library']), parameter_state=old_state)
    new_native = NativeSceneGPU(scene_output, NativeCore(library), parameter_state=new_state)
    if list(old_native.paths) != list(new_native.paths):raise ValueError('Native inventory changed')
    maximum = 0.
    for identity, old in old_native.paths.items():
        new = new_native.paths[identity]
        if old['kind'] != new['kind']:raise ValueError('Native primitive family changed: '+identity)
        for key in ('parameters','vertices','knots'):
            left, right = np.asarray(old.get(key,[])), np.asarray(new.get(key,[]))
            if left.shape != right.shape:raise ValueError('Native primitive shape changed: '+identity)
            if left.size:
                maximum = max(maximum, float(np.max(np.abs(left-right))))
                np.testing.assert_allclose(right, left, rtol=0., atol=2e-12, err_msg=identity+' '+key)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('point_observations.npz','mapping.json'):
        if name == 'mapping.json':prepared.write(output/name, next_mapping)
        else:os.link(parent/name, output/name)
    prepared.write(output/'joint_manifest.json', next_manifest)
    prepared.write(output/'parameter_layout.json', new_state.layout)
    np.savez_compressed(output/'initial_pixels.npz', **current)
    np.savez_compressed(parameters_output, **{k:a for k,a in current.items() if k not in ('stations','source_rows')})
    report = dict(schema='rinkspace.ice_tolerance_preparation.v1', status='prepared_not_fitted',
        parent_snapshot=str(snapshot), parent_sha256=prepared.digest(snapshot),
        scene=str(scene_output), parameters=str(parameters_output), prepared_run=str(output),
        initial_state=str(output/'initial_pixels.npz'), deviation_count=DEVIATION_COUNT,
        all_new_deviations_exact_zero=True, all_existing_parameter_blocks_byte_preserved=True,
        camera_columns=camera_layout['unique_camera_columns'],
        old_scene_columns=joint_columns(old_model,old_state), new_scene_columns=joint_columns(new_model,new_state),
        original_observation_sha256=m['point_input_sha256'], observations_byte_unchanged=True,
        original_source_pts_rows_stations_and_camera_motion_preserved=True,
        native_paths=len(new_native.paths), maximum_native_parameter_delta_at_zero=maximum,
        native_library=str(library), native_library_sha256=prepared.digest(library),
        optimizer_started=False, full_optimizer_checkpoint_restored=False)
    prepared.write(output/'preparation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-run','snapshot','native-library','output','scene-output','parameters-output'):
        p.add_argument('--'+name, type=Path, required=True)
    a=p.parse_args()
    prepare(a.parent_run,a.snapshot,a.native_library,a.output,a.scene_output,a.parameters_output)


if __name__=='__main__':main()
