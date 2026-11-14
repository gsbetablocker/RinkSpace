"""Prepare an independent programme's qualified packets for native initialization.

This command only exports four prepared files. It never starts a solver.
Scope requires event_id, programme_id, source{id,sha256,width,height},
source_clock{time_base_num,time_base_den,native_pts_step}, shots with
{id,rig_id,processing}, and rig_bindings mapping rig IDs to {camera_index}.
A rig_bindings list with explicit rig_id is also accepted. Camera indices refer
verbatim to seed-state rig rows; unused seed rigs are retained for coupling.
Each packets/<shot_id>/ contains mapping.json, shot_seed.json, controls.npy,
and point_observations.npz in the existing independent seven-field format.

Only camera_initialize and motion consumers are authorized by this preparation.
Both have zero scene directions in the existing consumer; its joint phase is
outside this frozen-geometry contract. Qualification is supplied by the caller,
not inferred from residuals. Missing exposure observations remain explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
LEGACY_TOOLS = PROJECT/'events/ankara_2026/programmes/okada_mei_sp'
sys.path.insert(0, str(PROJECT/'src'))
sys.path.insert(0, str(LEGACY_TOOLS))
import prepare_native_multicamera as prepared
from prepare_native_extension import (dependency_files, initialize_curve_stations,
                                     same_bytes, source_coverage, validate_points)
from rinkspace.core.native import NativeCore
from rinkspace.domain.camera_coupling import read_camera_coupling, build_multicamera_directions
from rinkspace.domain.camera_experiment import load_camera_experiment
from rinkspace.domain.multi_shot_motion import ShotMotionTimeline, continue_exposure_halos
from rinkspace.domain.parameters import SceneParameterModel, ParameterState
from rinkspace.path_policy import guard_project_write
from rinkspace.session.gpu_geometry import NativeSceneGPU

OUTPUTS = ('joint_manifest.json', 'mapping.json', 'point_observations.npz', 'initial_pixels.npz')


def _integer(value, label, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError('Invalid integer '+label)
    return value


def _identity(value, label):
    if not isinstance(value, str) or not value or value in ('.', '..') or any(c in value for c in '/\\:'):
        raise ValueError('Invalid stable ID '+label)
    return value


def _binding_identity(document, scope, source, label):
    for key in ('event_id', 'programme_id'):
        if key in document and document[key] != scope[key]:
            raise ValueError(label+' '+key+' differs from scope')
    if 'source_id' in document and document['source_id'] != source['id']:
        raise ValueError(label+' source_id differs from scope')
    if document.get('source_video_sha256') != source['sha256']:
        raise ValueError(label+' source SHA256 differs from scope')


def scope_contract(scope, rig_count):
    for key in ('event_id', 'programme_id'):
        _identity(scope.get(key), key)
    source, clock = scope['source'], scope['source_clock']
    _identity(source.get('id'), 'source.id')
    sha = source.get('sha256')
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('Scope needs the original source SHA256')
    if (source.get('width'), source.get('height')) != (1920, 1080):
        raise ValueError('Existing native multicamera consumer requires 1920x1080 native pixels')
    for key in ('time_base_num', 'time_base_den', 'native_pts_step'):
        _integer(clock.get(key), 'source_clock.'+key, 1)
    bindings = scope['rig_bindings']
    if isinstance(bindings, list):
        if len({b['rig_id'] for b in bindings}) != len(bindings):
            raise ValueError('Duplicate rig binding')
        bindings = {b['rig_id']:b for b in bindings}
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError('Explicit rig_bindings required')
    indices = {}
    for rig_id, binding in bindings.items():
        _identity(rig_id, 'rig_id')
        index = _integer(binding['camera_index'], 'camera_index', 0)
        if index >= rig_count or index in indices.values():
            raise ValueError('Rig bindings must use distinct existing seed rig rows')
        indices[rig_id] = index
    shots = scope['shots']
    if not isinstance(shots, list) or not shots or len({s['id'] for s in shots}) != len(shots):
        raise ValueError('Unique nonempty shot registry required')
    previous_stop = None
    for shot in shots:
        _identity(shot['id'], 'shot.id')
        if shot['rig_id'] not in indices:
            raise ValueError('Shot has no physical rig binding: '+shot['id'])
        if shot.get('source_id', source['id']) != source['id']:
            raise ValueError('Shot references a different source')
        window = shot['processing']
        for key in ('first_pts', 'last_pts', 'stop_pts_exclusive', 'native_frame_count'):
            _integer(window.get(key), shot['id']+'.'+key)
        if window['native_frame_count'] < 1 or window['stop_pts_exclusive'] != window['last_pts']+clock['native_pts_step']:
            raise ValueError('Inconsistent shot source window')
        if previous_stop is not None and window['first_pts'] < previous_stop:
            raise ValueError('Shots must be in nonoverlapping original source order')
        previous_stop = window['stop_pts_exclusive']
    return source, clock, indices


def read_packets(scope, packets_root, rig_count):
    source, clock, indices = scope_contract(scope, rig_count)
    packets, hashes, control_dt, dtypes = [], {}, None, None
    for shot in scope['shots']:
        sid = shot['id']; folder = (packets_root/sid).resolve(strict=True)
        if not folder.is_relative_to(packets_root):
            raise ValueError('Packet folder escapes root')
        files = [folder/n for n in ('mapping.json','shot_seed.json','controls.npy','point_observations.npz')]
        hashes.update({str(p):prepared.digest(p) for p in files})
        mapping, seed = prepared.read(files[0]), prepared.read(files[1])
        _binding_identity(mapping, scope, source, sid+' mapping')
        _binding_identity(seed, scope, source, sid+' seed')
        if seed.get('independent_source_capture') is not True or mapping.get('independent_source_capture') is not True:
            raise ValueError('Packets must explicitly declare independent source capture')
        _integer(seed.get('camera_index'), 'seed.camera_index', 0)
        if seed.get('id') != sid or seed.get('camera_index') != indices[shot['rig_id']]:
            raise ValueError('Shot seed physical rig binding differs')
        if mapping.get('pixel_domain') != 'native_distorted' or mapping.get('image_size') != [1920,1080]:
            raise ValueError('Packet pixel domain differs from native source')
        for key in ('time_base_num', 'time_base_den'):
            if type(seed.get(key)) is not int or seed[key] != clock[key]:
                raise ValueError('Packet source time base differs')
        _integer(seed.get('reference_pts'), 'reference_pts')
        _integer(seed.get('knot_stride'), 'knot_stride', 1)
        if seed.get('motion_reference_time') != 0 or seed.get('controls_ref') != 'controls.npy':
            raise ValueError('Independent motion needs zero reference time and local controls.npy')
        dt, first = seed.get('control_dt'), seed.get('first_control_time')
        if type(dt) not in (int,float) or type(first) not in (int,float) or not np.isfinite([dt,first]).all() or dt <= 0:
            raise ValueError('Invalid shot control clock')
        if control_dt is not None and dt != control_dt:
            raise ValueError('All shot controls must share storage control_dt; no resampling is performed')
        control_dt = dt
        controls = np.load(files[2], allow_pickle=False)
        if controls.dtype != np.dtype('float64') or controls.ndim != 2 or controls.shape[1] != 3 or len(controls) < 4 or not np.isfinite(controls).all():
            raise ValueError('Controls must be finite float64[M,3]')
        points = prepared.npz(files[3]); validate_points(points, mapping)
        if mapping.get('point_input_sha256') != hashes[str(files[3])]:
            raise ValueError('Qualified point input SHA256 differs')
        if points['pts'].dtype != np.dtype('int64') or points['uv'].dtype != np.dtype('float64') or points['stations'].dtype != np.dtype('float64'):
            raise ValueError('Packets require int64 PTS and float64 UV/stations; no implicit casts')
        current_dtypes = {k:v.dtype for k,v in points.items() if k != 'observation_ids'}
        if dtypes is not None and current_dtypes != dtypes:
            raise ValueError('Packet numerical dtypes differ; no implicit widening/casting')
        dtypes = current_dtypes
        coverage = source_coverage(points['pts'], shot['processing'], clock['native_pts_step'], True)
        info = {k:seed[k] for k in ('id','camera_index','reference_pts','time_base_num','time_base_den',
                                  'first_control_time','knot_stride','motion_reference_time')}
        info.update({k:shot['processing'][k] for k in ('first_pts','last_pts','stop_pts_exclusive')})
        info.update(rig_id=shot['rig_id'], source_id=source['id'], source_video_sha256=source['sha256'], source_coverage=coverage)
        interval = np.array([(info[k]-info['reference_pts'])*clock['time_base_num']/clock['time_base_den'] for k in ('first_pts','last_pts')])
        closed = continue_exposure_halos(controls, first, dt, interval)
        if not np.allclose(closed, controls, rtol=2e-12, atol=2e-9):
            raise ValueError('Controls need explicit endpoint-cubic halo closure: '+sid)
        packets.append(dict(folder=folder, mapping=mapping, seed=seed, points=points, controls=controls, info=info))
    return packets, hashes, control_dt


def assemble_packets(packets):
    """Concatenate original rows, changing only local path-index lookup tables."""
    paths = list(dict.fromkeys(p for packet in packets for p in packet['mapping']['path_ids']))
    lookup = {p:i for i,p in enumerate(paths)}
    chunks = []; shots = []; controls = []; row_offset = control_offset = 0
    for packet in packets:
        points = packet['points']; z = {k:v.copy() for k,v in points.items()}
        remap = [lookup[packet['mapping']['path_ids'][int(i)]] for i in points['path_indices']]
        if max(remap) > np.iinfo(points['path_indices'].dtype).max:
            raise ValueError('Combined native path registry exceeds original integer dtype')
        z['path_indices'] = np.asarray(remap, dtype=points['path_indices'].dtype)
        info = dict(packet['info'], row_begin=row_offset, row_stop=row_offset+len(z['pts']),
                    control_begin=control_offset, control_stop=control_offset+len(packet['controls']))
        row_offset, control_offset = info['row_stop'], info['control_stop']
        chunks.append(z); shots.append(info); controls.append(packet['controls'])
    joined = {key:np.concatenate([z[key] for z in chunks]) for key in prepared.FIELDS}
    if len(set(map(str,joined['observation_ids']))) != len(joined['pts']):
        raise ValueError('Observation IDs collide across shots')
    for packet, shot in zip(packets, shots):
        sl = slice(shot['row_begin'], shot['row_stop'])
        for key in prepared.FIELDS-{'path_indices','observation_ids'}:
            same_bytes(joined[key][sl], packet['points'][key], shot['id']+' '+key)
        if not np.array_equal(joined['observation_ids'][sl], packet['points']['observation_ids']):
            raise ValueError('Observation identity changed during concatenation')
    return joined, paths, shots, np.concatenate(controls)


def prepare(args):
    scope_path, packets_root, seed_path, scene_path, library, profile, coupling_path = [
        Path(p).resolve(strict=True) for p in (args.scope,args.packets,args.seed_state,args.scene,
                                              args.native_library,args.camera_profile,args.camera_coupling)]
    out = guard_project_write(args.output)
    if out.exists():
        raise FileExistsError('Preserve existing prepared output: '+str(out))
    if out.is_relative_to(PROJECT.resolve()):
        raise ValueError('Prepared numerical arrays belong outside OneDrive')
    for name in OUTPUTS:
        guard_project_write(out/name)
    protected = {str(p):prepared.digest(p) for p in (scope_path,seed_path,scene_path,library,profile,coupling_path)}
    scope, seed, scene = prepared.read(scope_path), prepared.npz(seed_path), prepared.read(scene_path)
    rigs, values = seed['rig'], seed['scene_values']
    if rigs.dtype != np.dtype('float64') or rigs.ndim != 2 or rigs.shape[1] != 43 or not np.isfinite(rigs).all():
        raise ValueError('Seed state requires finite float64 physical rigs[N,43]')
    model = SceneParameterModel(scene, scene_path, optimize_scene=True)
    layout = model.initial_state.layout
    if values.dtype != np.dtype('float64') or values.shape != (layout['size'],) or not np.isfinite(values).all():
        raise ValueError('Seed scene_values differs from the current complete numerical layout')
    inherited = json.loads(str(seed['joint_numeric_dependencies_json']))
    if inherited['layout_revision'] != layout['layout_revision'] or inherited['scene_revision'] != layout['scene_content_revision']:
        raise ValueError('Explicit scene is not the seed state scene/layout')
    coupling = read_camera_coupling(coupling_path); indices = []
    for rig in rigs:
        copied, active, metadata = load_camera_experiment(profile, rig)
        if metadata['scene_mode'] != 'joint_current':
            raise ValueError('Existing multicamera consumer requires a joint_current camera profile')
        same_bytes(copied, rig, 'profile-compatible inherited rig')
        indices.append(active)
    build_multicamera_directions(rigs, indices, np.ones(31), coupling)
    packets, packet_hashes, dt = read_packets(scope, packets_root, len(rigs))
    protected.update(packet_hashes)
    joined, paths, shots, controls = assemble_packets(packets)
    timeline = ShotMotionTimeline(shots, dt, 0.)
    max_readout = timeline.max_readout()
    if np.any(np.abs(rigs[:,37]) > max_readout):
        raise ValueError('Shot control halos cannot support inherited readout')
    timeline.assert_support_separation(joined['pts'], max_readout)
    core = NativeCore(library)
    native = NativeSceneGPU(scene_path, core, parameter_state=ParameterState(layout, values.copy()))
    missing = set(paths)-native.paths.keys()
    if missing:
        raise ValueError('Packet paths absent from current frozen scene: '+str(sorted(missing)))
    stations = joined['stations'].copy(); station_notes = {}
    for packet, shot in zip(packets, shots):
        sl = slice(shot['row_begin'],shot['row_stop']); curve = joined['kind'][sl] == 1
        if not curve.any():
            station_notes[shot['id']] = dict(initialized_rows=0, unresolved=[])
            continue
        camera = dict(shot,rig=rigs[shot['camera_index']].tolist(),control_dt=dt,image_height=1080)
        initialized, note = initialize_curve_stations(native, camera=camera, controls=packet['controls'],
            path_ids=[paths[int(i)] for i in joined['path_indices'][sl][curve]],
            stations=stations[sl][curve], pts=joined['pts'][sl][curve], observed=joined['uv'][sl][curve],
            kinds=joined['kind'][sl][curve], half_seconds=max_readout/2)
        if note['unresolved']:
            raise ValueError('Native station initialization unresolved: '+str(note['unresolved'][:10]))
        stations[shot['row_begin']+np.flatnonzero(curve)] = initialized
        station_notes[shot['id']] = dict(note, initialized_rows=int(curve.sum()))
    source, clock, bindings = scope_contract(scope, len(rigs))
    mapping = dict(path_ids=paths, scene_revision=native.scene_revision,
                   event_id=scope['event_id'],programme_id=scope['programme_id'],source_id=source['id'],
                   source_video_sha256=source['sha256'],pixel_domain='native_distorted',image_size=[1920,1080])
    # The consumer reads this shared projection context while taking every rig
    # and shot motion from initial_pixels; no old programme camera is referenced.
    binding_rows = scope['rig_bindings']
    binding_rows = ({b['rig_id']: b for b in binding_rows} if isinstance(binding_rows, list) else binding_rows)
    cameras = []
    for i in range(len(rigs)):
        rid = next((rid for rid, index in bindings.items() if index == i), 'seed_rig_'+str(i))
        side = binding_rows.get(rid, {}).get('mount_side', 'negative_y')
        if side not in ('outside', 'negative_y', 'positive_y', 'negative_x_end', 'positive_x_end'):
            raise ValueError('Unknown physical installation side: '+str(side))
        cameras.append(dict(id=rid,seed_camera_index=i,mount_side=side,
                            camera_definition=str(out/'joint_manifest.json')))
    parameterization = dict(profile_path=str(profile),profile_sha256=protected[str(profile)],
                            coupling_path=str(coupling_path),coupling_sha256=protected[str(coupling_path)])
    manifest = dict(schema='rinkspace.multicamera_disjoint_shots.v1',event_id=scope['event_id'],
        programme_id=scope['programme_id'],source_id=source['id'],source=source,source_clock=clock,
        shots=shots,cameras=cameras,rig_bindings=scope['rig_bindings'],control_dt=dt,first_control_time=0.,
        max_readout=max_readout,reference_pts=shots[0]['reference_pts'],
        time_base_num=clock['time_base_num'],time_base_den=clock['time_base_den'],image_height=1080,
        scene_path=str(scene_path),scene_file_sha256=protected[str(scene_path)],scene_revision=native.scene_revision,
        native_library=str(library),native_library_sha256=protected[str(library)],
        camera_parameterization=parameterization,current_resume_snapshot=str(out/'initial_pixels.npz'),
        source_inputs={s['id']:dict(path=str(p['folder']/'point_observations.npz'),
            sha256=packet_hashes[str(p['folder']/'point_observations.npz')],source_video_sha256=source['sha256'],
            source_id=source['id'],independent_source_capture=True) for p,s in zip(packets,shots)},
        source_pts_unchanged=True,shot_motion_continuity_across_edits=False,
        independent_installations_and_optics=not bool(coupling['groups']),
        objective='0.5*sum((predicted_uv-original_uv)^2); every retained point contributes equally',
        partial_source_coverage=any(s['source_coverage']['partial_source_coverage'] for s in shots),
        frozen_geometry=dict(all_scene_values_byte_inherited=True,active_geometry_coordinates=0,
            seed_state=str(seed_path),seed_state_sha256=protected[str(seed_path)],
            scene_values_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
            compatible_consumer_phases=['camera_initialize','motion'],joint_phase_authorized=False),
        preparation=dict(schema='rinkspace.independent_programme_native.v1',scope=str(scope_path),
            scope_sha256=protected[str(scope_path)],station_initialization=station_notes,
            input_qualification='caller supplied qualified independent source packets; not requalified here',
            source_pixels_or_pts_generated=False,fit_executed=False,user_visual_acceptance=False))
    protected.update(dependency_files(manifest,scene_path,library,scene))
    manifest['preparation']['current_dependency_sha256'] = protected
    result = dict(rig=rigs.copy(),scene_values=values.copy(),controls=controls,stations=stations,
                  source_rows=joined['source_rows'].copy())
    same_bytes(result['scene_values'],values,'all inherited geometry')
    # Validate numerical layout/physical camera constraints before creating output.
    manifest['point_input_sha256'] = '0'*64
    prepared.bind_snapshot(result,layout,mapping,manifest)
    prepared.validate_snapshot(result,layout,mapping,manifest,joined)
    # Validate all current dependencies before the four-file output is created.
    for path, pin in protected.items():
        if prepared.digest(path) != pin:
            raise RuntimeError('Input changed during preparation: '+path)
    out.mkdir(parents=True,exist_ok=False)
    np.savez(out/'point_observations.npz',**joined)
    manifest['point_input_sha256'] = prepared.digest(out/'point_observations.npz')
    mapping['point_input_sha256'] = manifest['point_input_sha256']
    prepared.bind_snapshot(result,layout,mapping,manifest)
    prepared.validate_snapshot(result,layout,mapping,manifest,joined)
    np.savez(out/'initial_pixels.npz',**result)
    prepared.write(out/'mapping.json',mapping)
    prepared.write(out/'joint_manifest.json',manifest)
    saved = prepared.npz(out/'initial_pixels.npz')
    prepared.validate_snapshot(saved,layout,mapping,manifest,joined)
    same_bytes(saved['scene_values'],values,'saved complete geometry')
    same_bytes(saved['rig'],rigs,'saved inherited rigs')
    for path,pin in protected.items():
        if prepared.digest(path) != pin:
            raise RuntimeError('Input changed during output: '+path)
    summary = dict(output=str(out),event_id=scope['event_id'],programme_id=scope['programme_id'],
        source_id=source['id'],shots=len(shots),rows=len(joined['pts']),active_geometry_coordinates=0,
        source_UV_PTS_preserved=True,scene_values_byte_preserved=True,fit_executed=False)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scope','packets','seed-state','scene','native-library','camera-profile','camera-coupling','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
