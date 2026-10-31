"""Extend the current four-shot state with the two current end-camera packs.

This consumes current seven-field inputs. It does not replay old extraction or
import package geometry, and does not treat a camera seed as a calibration.
"""
from pathlib import Path
import argparse
import json
import numpy as np

import prepare_native_multicamera as prepared
import fit_native_s01_joint as single
from initialize_native_joint_principal_point import constant_principal_initial_state
from rinkspace.core.native import NativeCore
from rinkspace.domain.camera import complete_camera_rig
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.domain.multi_shot_motion import (
    ShotMotionTimeline, refine_uniform_controls, continue_exposure_halos)
from rinkspace.domain.motion_chart import cubic_samples
from rinkspace.session.gpu_geometry import NativeSceneGPU, initialize_curve_stations

DATA = Path('D:/CodexData/RinkSpace_Data')


def feasible_end_seed(rig, board):
    """Move a legacy seed out of the current rounded board at unchanged Y/Z.

    The 1e-6 L margin only avoids floating point boundary rejection. This is
    an initialization, not a measured installation or a changed constraint.
    """
    result = rig.copy()
    half_length, half_width, rx, ry, power = board[:5]
    u = max(0., (abs(rig[15])-half_length+rx)/rx)
    w = max(0., (abs(rig[16])-half_width+ry)/ry)
    if u**power+w**power < 1.:
        boundary = half_length-rx+rx*max(0.,1.-w**power)**(1./power)
        result[15] = -boundary-1e-6
    return result, dict(original_pivot=rig[15:18].tolist(),
        feasible_pivot=result[15:18].tolist(), boundary_margin_L=1e-6,
        seed_adjustment_only=True, installation_measurement=False)


def initialize_new_print_heights(core, scene, scene_path, state, used_paths, old_paths):
    """Put newly observed installation seeds on the finite board surface.

    Existing observed installations and all shared sizes/shapes are protected.
    The native full-path constraints still certify the resulting solver seed.
    """
    from rinkspace.domain.scene_assets import read_scene_assets
    from rinkspace.domain.print_assets import evaluate_print_path, resolve_instance, _print_context
    result = ParameterState(state.layout, state.values.copy())
    height = result.apply_scene(scene)['board'][5]
    changes = []
    for reference, assets in read_scene_assets(scene, scene_path):
        for identity in reference.get('instance_ids', [reference.get('instance_id')]):
            instance, template, size, placement = resolve_instance(assets, identity)
            paths = [p for p in template['paths'] if identity+'::'+p['id'] in used_paths]
            if not paths:continue
            context = _print_context(assets, identity, result)
            values = [evaluate_print_path(assets, identity, p['id'], None, core,
                np.linspace(0.,1.,257), parameter_state=result, jacobian=False,
                _context=context)['sz'][:,1] for p in paths]
            low, high = min(float(v.min()) for v in values), max(float(v.max()) for v in values)
            if low>=-1e-12 and high<=height+1e-12:continue
            if any(p.startswith(identity+'::') for p in old_paths):
                raise ValueError('A previously observed installation needs a separate scene revision: '+identity)
            if high-low > height:
                raise ValueError('New installation cannot fit by height translation: '+identity)
            shift = min(max(0., -low+1e-10), height-high-1e-10)
            offset = result.blocks[placement['id']]['offset']+1
            before = float(result.values[offset])
            updated = result.values.copy();updated[offset] += shift
            result = ParameterState(state.layout, updated)
            changes.append(dict(instance=identity, parameter_block=placement['id'],
                parameter_index=offset, previous_height=before,
                initialized_height=float(result.values[offset]), board_height=height,
                old_source_rows=0, reason='newly_observed_full_path_extended_below_or_above_board',
                seed_only=True, source_pixels_changed=False))
    return result, changes


def prepare(args):
    root, out = args.run_root, args.output
    if (out/'joint_manifest.json').exists():
        raise FileExistsError('Six-shot input already exists; consume its current state')
    out.mkdir(parents=True, exist_ok=True)
    log = (out/'prepare.jsonl').open('a', encoding='utf-8', buffering=1)
    def emit(event, **values):
        record = dict(event=event, **values)
        log.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
    print('MAIN LOG: '+str(out/'prepare.jsonl'), flush=True)
    manifest = prepared.read(root/'joint_manifest.json')
    mapping = prepared.read(root/'mapping.json')
    points = prepared.npz(root/'point_observations.npz')
    base = prepared.npz(args.resume)
    scene_path = Path(manifest['scene_path'])
    model = SceneParameterModel(prepared.read(scene_path), scene_path, optimize_scene=True)
    prepared.validate_snapshot(base, model.initial_state.layout, mapping, manifest, points)
    if prepared.digest(root/'point_observations.npz') != manifest['point_input_sha256']:
        raise ValueError('Current four-shot input differs')
    if [s['id'] for s in manifest['shots']] != ['s01', 's02', 's04', 's06']:
        raise ValueError('Expected the verified four-shot baseline')
    core = NativeCore(manifest['native_library'])
    native = NativeSceneGPU(scene_path, core,
        parameter_state=ParameterState(model.initial_state.layout, base['scene_values']))
    protected = {str(p): prepared.digest(p) for p in
        (root/'point_observations.npz', root/'joint_manifest.json', args.resume, scene_path)}
    scope = prepared.read(prepared.HERE/'rebuild_scope.json')
    new = {}
    path_index = {p:i for i,p in enumerate(mapping['path_ids'])}
    for shot_id, camera_file in [('s03', 'ad1_cameras/s03.json'), ('s05', 'ad_cameras/s05.json')]:
        folder = DATA/'runs'/f'native_points_{shot_id}_20261028'
        z = prepared.npz(folder/'points/point_observations.npz')
        local = prepared.read(folder/'points/mapping.json')
        digest = prepared.digest(folder/'points/point_observations.npz')
        if set(z) != prepared.FIELDS or digest != local['point_input_sha256']:
            raise ValueError('Current seven-field package differs: '+shot_id)
        if local['scene_revision'] != native.scene_revision or set(local['path_ids'])-path_index.keys():
            raise ValueError('Current native scene/path identity differs: '+shot_id)
        if local['pixel_domain'] != 'native_distorted' or local['image_size'] != [1920,1080]:
            raise ValueError('Source pixel domain differs: '+shot_id)
        span = next(s['processing'] for s in scope['shots'] if s['id']==shot_id)
        expected = np.arange(span['first_pts'], span['stop_pts_exclusive'], 256)
        if not np.array_equal(np.unique(z['pts']), expected):
            raise ValueError('Incomplete shot exposure clock: '+shot_id)
        if not np.isfinite(z['uv']).all() or np.any(z['uv']<0) or np.any(z['uv']>[1919,1079]):
            raise ValueError('Invalid source pixels: '+shot_id)
        if not np.isin(z['kind'], [0,1]).all():
            raise ValueError('Unknown source observation type')
        camera_path = prepared.HERE/camera_file
        camera = prepared.read(camera_path)
        controls_path = camera_path.parent/camera['controls_ref']
        controls = np.load(controls_path, allow_pickle=False)
        new[shot_id] = dict(points=z, mapping=local, camera=camera, controls=controls,
            source_path=str(folder/'points/point_observations.npz'), source_sha256=digest,
            camera_path=str(camera_path), controls_path=str(controls_path))
        emit('current_package', shot=shot_id, rows=len(z['pts']), exposures=len(expected),
             source_sha256=digest, package_geometry_imported=False)
    original = complete_camera_rig(new['s03']['camera'])
    if not np.array_equal(original, complete_camera_rig(new['s05']['camera'])):
        raise ValueError('S03 and S05 no longer share the same registered end-camera seed')
    if len(original)==38:
        focal = np.concatenate([n['controls'][1:-1,2] for n in new.values()])
        original = np.r_[original, np.median(focal), np.zeros(4)]
    rig = original.copy()
    source_camera = new['s03']['camera']
    before_centre = core.evaluate_motion_camera(rig, new['s03']['controls'],
        source_camera['first_control_time'], source_camera['control_dt'], [0.])['centre'][0]
    metres = prepared.read(prepared.ROOT/'events/ankara_2026/model_constraints/ankara_2026_metric_scale_assumed_v1.json')['converted_span_xy_m'][0]
    lower, upper, envelope, ranges_path, ranges = single.mount_configuration(metres, rig)
    rig[18:21] = np.asarray(ranges['working_ranges']['complete_tilt_axis_to_projection_centre_camera_m']['initial_reference'])/metres
    rig[21] = np.clip(rig[21], lower[21], upper[21])
    rig[22:24] = 0.
    after_centre = core.evaluate_motion_camera(rig, new['s03']['controls'],
        source_camera['first_control_time'], source_camera['control_dt'], [0.])['centre'][0]
    rig[15:18] += before_centre-after_centre
    rig[17] = np.clip(rig[17], lower[17], upper[17])
    rig[28] = 0.
    rig[37] = 0.
    current_scene = ParameterState(model.initial_state.layout, base['scene_values']).apply_scene(prepared.read(scene_path))
    rig, feasible_note = feasible_end_seed(rig, current_scene['board'])
    chunks, controls, stations, shots, notes = [], [], [], [], {}
    rigs = [r.copy() for r in base['rig']]
    row_offset = control_offset = 0
    old_by_id = {s['id']:s for s in manifest['shots']}
    for shot_id in ['s01','s02','s03','s04','s05','s06']:
        if shot_id in old_by_id:
            old = old_by_id[shot_id]
            rows = slice(old['row_begin'],old['row_stop'])
            cols = slice(old['control_begin'],old['control_stop'])
            z = {k:v[rows].copy() for k,v in points.items()}
            cc = base['controls'][cols].copy()
            ss = base['stations'][rows].copy()
            info = dict(old)
        else:
            data = new[shot_id]
            z = {k:v.copy() for k,v in data['points'].items()}
            z['path_indices'] = np.asarray([path_index[data['mapping']['path_ids'][i]]
                for i in z['path_indices']], np.int32)
            camera = data['camera']
            cc, first = refine_uniform_controls(data['controls'], camera['first_control_time'],
                camera['control_dt'], manifest['control_dt'])
            cc = cc[1:-1]
            first += manifest['control_dt']
            interval = [(int(z['pts'].min())-camera['reference_pts'])/12800,
                        (int(z['pts'].max())-camera['reference_pts'])/12800]
            closed = continue_exposure_halos(cc, first, manifest['control_dt'], interval)
            indices, weights = cubic_samples(len(cc), first, manifest['control_dt'],
                (np.unique(z['pts'])-camera['reference_pts'])/12800)
            interior_delta = np.max(abs(np.einsum('ij,ijk->ik', weights, (cc-closed)[indices])))
            if interior_delta > 1e-10:
                raise ValueError('Endpoint continuation changed an observed exposure')
            moved, transfer = constant_principal_initial_state(core,
                dict(rig=rig, controls=closed), [959.5,539.5])
            cc = continue_exposure_halos(moved['controls'], first, manifest['control_dt'], interval)
            if len(rigs)==2:
                rigs.append(moved['rig'])
            elif not np.array_equal(rigs[2], moved['rig']):
                raise ValueError('End-camera shared installation changed across shots')
            camera = {**camera, 'rig':rigs[2].tolist(), 'first_control_time':first,
                      'control_dt':manifest['control_dt']}
            emit('initialize_stations', shot=shot_id, rows=len(z['pts']))
            ss, station_note = initialize_curve_stations(native, camera=camera, controls=cc,
                path_ids=[mapping['path_ids'][i] for i in z['path_indices']],
                stations=z['stations'], pts=z['pts'], observed=z['uv'], kinds=z['kind'], half_seconds=.02)
            if station_note['unresolved']:
                raise ValueError('New source projection unresolved: '+str(station_note['unresolved'][:10]))
            info = dict(id=shot_id, camera_index=2, reference_pts=camera['reference_pts'],
                time_base_num=1, time_base_den=12800, first_control_time=first,
                knot_stride=10, motion_reference_time=0., first_pts=int(z['pts'].min()),
                last_pts=int(z['pts'].max()), stop_pts_exclusive=int(z['pts'].max()+256))
            notes[shot_id] = dict(principal_transfer=transfer, station_initialization=station_note,
                pre_transfer_exposure_halo_max_channel_delta=float(interior_delta),
                camera_seed_only=True, old_sentinel_controls_not_used_as_observations=True)
        info.update(row_begin=row_offset, row_stop=row_offset+len(z['pts']),
                    control_begin=control_offset, control_stop=control_offset+len(cc))
        row_offset, control_offset = info['row_stop'], info['control_stop']
        chunks.append(z); controls.append(cc); stations.append(ss); shots.append(info)
    z = {key:np.concatenate([chunk[key] for chunk in chunks]) for key in prepared.FIELDS}
    if len(np.unique(z['observation_ids'])) != len(z['pts']):
        raise ValueError('Cross-shot observation IDs collide')
    expected = np.arange(shots[0]['first_pts'], shots[-1]['stop_pts_exclusive'], 256)
    if not np.array_equal(np.unique(z['pts']), expected):
        raise ValueError('Six-shot original exposure coverage is not contiguous')
    np.savez(out/'point_observations.npz', **z)
    prepared.write(out/'mapping.json', mapping)
    timeline = ShotMotionTimeline(shots, manifest['control_dt'], manifest['first_control_time'])
    readout = min(manifest['max_readout'], timeline.max_readout())
    timeline.assert_support_separation(z['pts'], readout)
    combined = {**manifest, 'shots':shots,
        'cameras':manifest['cameras']+[dict(id='end', camera_definition=new['s03']['camera_path'],
            mount_side='negative_x_end', installation_status='registered_end_camera_seed_not_new_calibration')],
        'max_readout':readout, 'point_input_sha256':prepared.digest(out/'point_observations.npz'),
        'baseline_snapshot':str(args.resume), 'baseline_sha256':prepared.digest(args.resume),
        'current_resume_snapshot':str(out/'initial_pixels.npz'),
        'source_inputs':{**manifest['source_inputs'], **{k:dict(path=v['source_path'],sha256=v['source_sha256']) for k,v in new.items()}},
        'six_shot_source_receipt':str(out/'import_receipt.json')}
    result = dict(rig=np.stack(rigs), controls=np.concatenate(controls),
        scene_values=base['scene_values'].copy(), stations=np.concatenate(stations), source_rows=z['source_rows'])
    prepared.bind_snapshot(result, model.initial_state.layout, mapping, combined)
    prepared.validate_snapshot(result, model.initial_state.layout, mapping, combined, z)
    preserved = {}
    for shot_id, old in old_by_id.items():
        current = next(s for s in shots if s['id']==shot_id)
        a,b = slice(old['row_begin'],old['row_stop']),slice(current['row_begin'],current['row_stop'])
        ac,bc = slice(old['control_begin'],old['control_stop']),slice(current['control_begin'],current['control_stop'])
        checks = {k:bool(np.array_equal(points[k][a],z[k][b])) for k in prepared.FIELDS}
        checks.update(controls=bool(np.array_equal(base['controls'][ac], result['controls'][bc])),
            fitted_stations=bool(np.array_equal(base['stations'][a],result['stations'][b])))
        if not all(checks.values()):raise ValueError('Existing four-shot data/state changed: '+shot_id)
        preserved[shot_id] = checks
    if not np.array_equal(result['rig'][:2], base['rig']):raise ValueError('Existing physical rigs changed')
    np.savez(out/'initial_pixels.npz', **result)
    prepared.write(out/'joint_manifest.json', combined)
    used_paths = {mapping['path_ids'][i] for i in np.unique(z['path_indices'])}
    old_paths = {mapping['path_ids'][i] for i in np.unique(points['path_indices'])}
    initialized, installation_notes = initialize_new_print_heights(core, prepared.read(scene_path),
        scene_path, ParameterState(model.initial_state.layout, result['scene_values']), used_paths, old_paths)
    working = {**result, 'scene_values':initialized.values.copy()}
    np.savez(out/'working_initial_pixels.npz', **working)
    prepared.write(out/'new_installation_initialization.json', dict(changes=installation_notes,
        shared_sizes_and_shapes_unchanged=True, old_observed_installations_unchanged=True,
        current_initial_state=str(out/'working_initial_pixels.npz'), measured_installation=False))
    for path, sha in protected.items():
        if prepared.digest(path)!=sha:raise ValueError('Protected baseline file changed: '+path)
    prepared.write(out/'import_receipt.json', dict(rows=len(z['pts']), exposures=len(expected),
        source_start_seconds=float(expected[0]/12800), source_last_seconds=float(expected[-1]/12800),
        preserved_four_shot_arrays=preserved, preserved_four_shot_files=protected,
        scene_values_and_old_rigs_exactly_preserved=True, source_uv_and_pts_unchanged=True,
        packages_geometry_imported=False, seven_fields_only=True, shot_notes=notes,
        end_rig_seed_original=original.tolist(), end_rig_seed_normalized=result['rig'][2].tolist(),
        end_rig_feasibility_initialization=feasible_note,
        mount_working_envelope=str(ranges_path), mount_ranges_are_assistant_estimates=True,
        camera_seed_is_not_accepted_calibration=True, numerical_fit_executed=False,
        user_visual_acceptance=False))
    emit('prepared', rows=len(z['pts']), exposures=len(expected), cameras=len(rigs),
         initial_state=str(out/'initial_pixels.npz'))
    log.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--resume', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    prepare(parser.parse_args())
