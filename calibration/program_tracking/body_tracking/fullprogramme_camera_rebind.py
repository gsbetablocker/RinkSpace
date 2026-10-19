"""Prepare the effective v8 saved-body inputs for a replacement physical camera.

The reviewed local geometry, official-clock corrections and source masks are
retained, with optional source-verified reuse of existing direct local slots.
Preparation does not fit or infer a pose.  --fit-root
then fits the continuous pelvis root against the replacement camera; P and
its world vertical projection G remain distinct from blade contact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ..z190_physical_camera import PhysicalCameraModel
from .fullprogramme_body_sequence import _slerp_pairs, packet_metres_per_scene_unit
from .official_body_pose_clock_patch import _slerp
from .scope import PROJECT, load_scope
from rinkspace.storage import AssetStore
from rinkspace.metric_scale import AssumedMetricScale


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


def load(path, keys=None):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in (archive.files if keys is None else keys)}


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def checked_camera(path, packet):
    """Replay physical states; aliases are copied, never resampled to 50 Hz."""
    model, native, aliases, views = PhysicalCameraModel.load_npz(path)
    camera = load(path, ['pts', 'camera25', 'exposure_representative_pts',
                         'installation_id', 'metadata_json'])
    if not same(native, packet['native_pts']):
        raise ValueError('Replacement camera native PTS differs from the frozen body source')
    if not same(camera['exposure_representative_pts'], packet['native_exposure_representative_pts']):
        raise ValueError('Replacement camera changes a confirmed actual exposure or alias')
    actual = np.unique(camera['exposure_representative_pts'])
    if not same(actual, packet['actual_pts']):
        raise ValueError('Replacement camera changes the actual exposure clock')
    rows = np.searchsorted(native, actual)
    if not same(native[rows], actual):
        raise ValueError('An actual exposure lacks its own explicit native camera')
    if not same(np.searchsorted(actual, camera['exposure_representative_pts']), packet['native_actual_row']):
        raise ValueError('Native-to-actual body mapping differs from the physical camera')
    meta = json.loads(str(camera['metadata_json']))
    if meta['source_time_base'] != '1/15360':
        raise ValueError('This frozen source requires its original 1/15360 clock')
    if len(model.calibrations) != 3 or len(views) != len(actual):
        raise ValueError('Expected the existing three installations and every actual view')
    # The model loader has already checked camera25 against XYZ/Rf/lever and
    # copied alias outputs bit for bit; no homography or metric conversion is
    # applied to the body input camera25 (its translations are scene units).
    return camera, rows, meta


def recompose_root_prior(packet, full_camera25, raw_source_pts, incam_orient):
    """Use the exact v8 donor brackets and its corrected official alpha.

    The original preparer used OpenCV Rodrigues; the official patch used
    scipy Rotation.  Replaying each family's own operations preserves the
    existing old-camera schema rather than silently changing its numerics.
    """
    native = packet['native_pts']
    lookup = {int(pts): i for i, pts in enumerate(native)}
    angles = {}
    for pts, angle in zip(raw_source_pts, incam_orient):
        key = int(pts)
        if key in angles and not same(angles[key], angle):
            raise ValueError('Conflicting saved incamera orientation at one source PTS')
        angles[key] = angle
    left, right = packet['raw_root_left_source_pts'], packet['raw_root_right_source_pts']
    alpha = packet['raw_root_alpha']
    patched = np.asarray(packet['official_pose_clock_patch_applied_actual_rows'], int)
    patch_mask = np.zeros(len(left), bool)
    patch_mask[patched] = True
    result = np.empty((len(left), 3, 3), float)
    for patched_family in (False, True):
        rows = np.flatnonzero(patch_mask == patched_family)
        endpoints = []
        for stamps in (left[rows], right[rows]):
            matrices = []
            for pts in stamps:
                key = int(pts)
                if key not in lookup or key not in angles:
                    raise ValueError(f'Saved raw root endpoint missing: {key}')
                cam_angle = full_camera25[lookup[key], 3:6]
                raw_angle = angles[key].astype(float)
                if patched_family:
                    matrices.append(Rotation.from_rotvec(cam_angle).as_matrix().T
                                    @ Rotation.from_rotvec(raw_angle).as_matrix())
                else:
                    matrices.append(cv2.Rodrigues(cam_angle)[0].T
                                    @ cv2.Rodrigues(raw_angle)[0])
            endpoints.append(np.asarray(matrices))
        slerp = _slerp if patched_family else _slerp_pairs
        if len(rows):
            result[rows] = slerp(endpoints[0], endpoints[1], alpha[rows])
    return result


def apply_same_exposure_direct_pose_cache(packet, manifest_path, sample_path):
    """Reuse bound direct local slots after fresh camera and fit-clock binding.

    The small overlay contains local geometry and saved in-camera angles only.
    World root orientation is replayed against the current donor camera, and
    source observations, qualification, masks and native camera aliases stay
    byte-identical. It runs neither a pose network nor forward kinematics.
    """
    path = Path(manifest_path)
    spec = json.loads(path.read_text(encoding='utf-8'))
    if spec.get('schema') != 'rinkspace.source-verified-same-exposure-direct-pose-cache.v1':
        raise ValueError('Unknown same-exposure direct local-pose cache schema')
    if 'body_same_exposure_cache_rows' in packet:
        raise ValueError('Do not apply a same-exposure direct cache twice')
    if digest(sample_path) != spec['base_effective_samples']['sha256']:
        raise ValueError('Direct cache requires its bound effective body-surface samples')
    bindings = [spec[k] for k in ['base_source_packet', 'base_effective_samples',
        'source_prior', 'cache', 'clock_evidence', 'audit']]
    bindings += spec['source_evidence_bindings']
    for row in spec['records']:
        bindings += [row[k] for k in ['raw_file', 'raw_metadata', 'raw_prepared_source']]
    for item in {x['path']: x for x in bindings}.values():
        if digest(item['path']) != item['sha256']:
            raise ValueError('Direct cached-pose authority changed: '+item['path'])
    cache, sample = load(spec['cache']['path']), load(sample_path)
    target, donor = cache['target_actual_rows'], cache['donor_actual_rows']
    actual = packet['actual_pts']
    if (len(np.unique(target)) != len(target) or len(target) != spec['rows']
            or np.any(target < 0) or np.any(donor < 0)
            or np.any(target >= len(actual)) or np.any(donor >= len(actual))):
        raise ValueError('Invalid direct cached-pose target or donor rows')
    if not same(actual[target], cache['target_actual_pts']) or not same(actual[donor], cache['donor_actual_pts']):
        raise ValueError('Direct cached-pose source identity differs from the current exposure ledger')
    if not same(sample['actual_pts'], actual) or not same(packet['body_scale'], cache['body_scale']):
        raise ValueError('Direct cached-pose local scale or sample identity differs')
    if not same(packet['shot_index'][target], packet['shot_index'][donor]):
        raise ValueError('Direct cached-pose reuse cannot cross camera cuts')
    if 'body_fit_clock_evidence_overlay_edge_rows' not in packet or not np.array_equal(
            packet['fit_time_seconds'][target], packet['fit_time_seconds'][donor]):
        raise ValueError('Direct cached-pose reuse needs already verified identical body exposure times')
    if np.any(np.isin(np.r_[target, donor], packet['official_pose_clock_patch_applied_actual_rows'])):
        raise ValueError('Direct cache may not override an existing official cached-pose patch')
    for key in ['local_COCO17_m', 'local_native22_m', 'body_local_pose_inferred',
                'body_local_pose_origin', 'body_local_pose_donor_actual_row',
                'raw_root_left_source_pts', 'raw_root_right_source_pts', 'raw_root_alpha']:
        if not same(packet[key][target], cache['original_'+key]) or not same(packet[key][donor], cache['donor_'+key]):
            raise ValueError('Fresh local cache inputs differ from bound target/donor: '+key)
    sample_keys = ['mesh_envelope_rootlocal_scaled_m', 'local_pose_valid', 'local_pose_origin', 'body_pose_axis_angle']
    for key in sample_keys:
        if not same(sample[key][target], cache['original_sample_'+key]) or not same(sample[key][donor], cache['donor_sample_'+key]):
            raise ValueError('Fresh cached body surface differs: '+key)
    angles = cache['raw_direct_incam_global_orient']
    replay = np.asarray([cv2.Rodrigues(cam[3:6])[0].T @ cv2.Rodrigues(angle.astype(float))[0]
                         for cam, angle in zip(packet['camera25'][donor], angles)])
    error = float(np.max(np.abs(replay-packet['raw_root_prior_Q_world'][donor])))
    if error > 2e-13:
        raise ValueError('Direct donor root prior does not replay from the current camera')
    result = dict(packet)
    changed = ['local_COCO17_m', 'local_native22_m', 'body_local_pose_origin',
               'raw_root_prior_Q_world', 'raw_root_left_source_pts', 'raw_root_right_source_pts', 'raw_root_alpha']
    for key in changed:
        result['body_same_exposure_cache_original_'+key] = packet[key][target].copy()
        result[key] = packet[key].copy()
    for key in ['local_COCO17_m', 'local_native22_m']:
        result[key][target] = cache['donor_'+key]
    result['body_local_pose_origin'][target] = 'source_verified_same_exposure_direct_cached_slot'
    result['raw_root_prior_Q_world'][target] = packet['raw_root_prior_Q_world'][donor]
    for key in ['raw_root_left_source_pts', 'raw_root_right_source_pts']:
        result[key][target] = cache['donor_actual_pts']
    result['raw_root_alpha'][target] = 0.
    for key in sample_keys:
        sample[key][target] = cache['donor_sample_'+key]
    sample['local_pose_origin'][target] = 'source_verified_same_exposure_direct_cached_slot'
    for key in packet:
        if key not in changed and not same(packet[key], result[key]):
            raise ValueError('Direct cached-pose overlay changed source/clock/policy: '+key)
    result['body_same_exposure_cache_rows'] = target.copy()
    result['body_same_exposure_cache_donor_actual_pts'] = cache['donor_actual_pts'].copy()
    result['body_same_exposure_cache_manifest_path'] = np.asarray(str(path.resolve()))
    result['body_same_exposure_cache_manifest_sha256'] = np.asarray(digest(path))
    sample['body_same_exposure_cache_rows'] = target.copy()
    sample['body_same_exposure_cache_manifest_sha256'] = np.asarray(digest(path))
    record = dict(manifest=binding(path), rows=len(target), records=spec['records'],
        direct_root_current_camera_replay_max_absolute=error,
        source_UV_scores_gates_clock_camera_aliases_retained=True,
        actual_cached_local_geometry_and_surface_reused=True,
        world_Q_recomputed_from_replacement_camera_not_imported=True,
        new_pose_or_FK_run=False, preserved_assumption=spec['preserved_assumption'])
    return result, sample, record


def prepare_v8_camera_input(v8_directory, camera_path, *, metres_per_scene_unit=None,
                            scale_source=None):
    """Return (input packet, provenance); no source, body or camera is fitted."""
    directory = Path(v8_directory)
    closeout = json.loads((directory / 'UNIFIED_BODY_V8_CLOSEOUT.json').read_text(encoding='utf-8'))
    original_report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
    packet_path = directory / 'fullprogramme_body_source_input.npz'
    trace_path = directory / 'actual_body_root_fit.npz'
    if digest(trace_path) != closeout['trace']['sha256']:
        raise ValueError('Frozen v8 trajectory warm-start file changed')
    if digest(packet_path) != closeout['packet']['sha256']:
        raise ValueError('Effective v8 source packet changed from its final closeout')
    packet = load(packet_path)
    load_scope().assert_pts(packet['native_pts'])
    old_unit = packet_metres_per_scene_unit(packet)
    scale = AssumedMetricScale.load(scale_source) if scale_source is not None else None
    if (scale is not None and metres_per_scene_unit is not None and not np.isclose(
            float(metres_per_scene_unit), scale.metres_per_scene_unit, rtol=1e-13, atol=0.)):
        raise ValueError('Requested body units differ from the saved uniform model scale')
    new_unit = (scale.metres_per_scene_unit if scale is not None else
                old_unit if metres_per_scene_unit is None else float(metres_per_scene_unit))
    packet_metres_per_scene_unit({'metres_per_scene_unit': np.asarray(new_unit)})
    if new_unit != old_unit and scale_source is None:
        raise ValueError('A changed uniform model scale requires its saved scale-source file')
    if new_unit != old_unit and scale.manifest['schema_version'] != 'rinkspace.assumed-uniform-metric-scale.v2':
        raise ValueError('Replacement model units require the closest-dimensions conversion policy')
    scale_binding = binding(scale_source) if scale_source is not None else None
    old_camera_path = Path(closeout['camera']['path'])
    if digest(old_camera_path) != closeout['camera']['sha256']:
        raise ValueError('The frozen v8 camera changed')
    old_camera, old_actual_rows, old_meta = checked_camera(old_camera_path, packet)
    new_camera, actual_rows, new_meta = checked_camera(camera_path, packet)
    if old_meta['source_sha256'] != new_meta['source_sha256']:
        raise ValueError('Replacement camera belongs to a different original video')
    if not same(old_camera['installation_id'], new_camera['installation_id']):
        raise ValueError('Replacement camera changes a physical installation identity')
    # The replacement package has already replayed every physical state and
    # calibration. Keep pixel/mode identities, while allowing the shared
    # numerical calibration to improve with the common rink geometry.
    fixed_fields = ['Brown14_order', 'output_mode_id', 'focal_domain',
                    'domain_wh', 'final_wh', 'A_source_to_final']
    calibration_changes = {}
    for installation, calibration in old_meta['calibrations'].items():
        replacement = new_meta['calibrations'][installation]
        for field in fixed_fields:
            if calibration[field] != replacement[field]:
                raise ValueError(f'Replacement camera changes source pixel/mode identity: {installation}/{field}')
        changed_fields = [field for field in calibration if calibration[field] != replacement[field]]
        if changed_fields:
            calibration_changes[installation] = {
                field: dict(old=calibration[field], replacement=replacement[field]) for field in changed_fields}
    if not same(packet['camera25'], old_camera['camera25'][old_actual_rows]):
        raise ValueError('Effective v8 body camera is inconsistent with its frozen production camera')
    patch_path = Path(original_report['official_saved_pose_clock_application']['patch']['path'])
    if digest(patch_path) != original_report['official_saved_pose_clock_application']['patch']['sha256']:
        raise ValueError('Official source-clock authority changed')
    raw = load(patch_path, ['raw_orientation_source_pts', 'raw_incam_global_orient'])
    replay = recompose_root_prior(packet, old_camera['camera25'], **{
        'raw_source_pts': raw['raw_orientation_source_pts'],
        'incam_orient': raw['raw_incam_global_orient']})
    replay_error = float(np.max(np.abs(replay - packet['raw_root_prior_Q_world'])))
    if replay_error > 2e-13:
        raise ValueError(f'Frozen v8 raw-root endpoint replay differs ({replay_error})')
    updated = {key: value.copy() for key, value in packet.items()}
    changed = not same(old_camera['camera25'], new_camera['camera25'])
    scale_changed = new_unit != old_unit
    if changed:
        updated['camera25'] = new_camera['camera25'][actual_rows].copy()
        updated['raw_root_prior_Q_world'] = recompose_root_prior(
            packet, new_camera['camera25'], raw['raw_orientation_source_pts'], raw['raw_incam_global_orient'])
    if changed or scale_changed:
        # PnP is only a numerical start: preserve its original camera-local
        # position while moving to the replacement camera's world transform.
        for i, (old, new) in enumerate(zip(packet['camera25'], updated['camera25'])):
            if same(old, new) and not scale_changed:
                continue
            old_r = cv2.Rodrigues(old[3:6])[0]
            new_r = cv2.Rodrigues(new[3:6])[0]
            local = old_r @ (packet['initial_P_m'][i] - old[:3] * old_unit)
            updated['initial_P_m'][i] = new[:3] * new_unit + new_r.T @ local
        updated['metres_per_scene_unit'] = np.asarray(new_unit)
    allowed = {'camera25', 'raw_root_prior_Q_world', 'initial_P_m', 'metres_per_scene_unit'}
    preserved = [key for key in packet if key not in allowed]
    if any(not same(packet[key], updated[key]) for key in preserved):
        raise ValueError('Camera rebinding changed local geometry, source, masks, clock or opening conditions')
    sample_path = directory / 'prepared_body_envelope_samples.npz'
    if digest(sample_path) != closeout['effective_samples']['sha256']:
        raise ValueError('The effective v8 local-body samples changed')
    sample_clock = load(sample_path, ['actual_pts', 'body_scale'])
    # The existing samples serialized the same scale to 15 decimals; retain
    # both stored values without recasting and reuse v8's 1e-12 scale check.
    if not same(sample_clock['actual_pts'], packet['actual_pts']) or not np.isclose(
            float(sample_clock['body_scale']), float(packet['body_scale']), rtol=0, atol=1e-12):
        raise ValueError('Frozen effective local-body samples differ from the body source clock/scale')
    original_binding = json.loads((directory / 'input_binding.json').read_text(encoding='utf-8'))
    source_bindings = original_binding['input_bindings']
    hip_path = next(Path(item['path']) for item in source_bindings if Path(item['path']).name == 'source_hip_joint0_observations.npz')
    clock_path = next(Path(item['path']) for item in source_bindings if Path(item['path']).suffix == '.jsonl')
    base_path = Path(source_bindings[0]['path']).parent.parent
    record = dict(
        schema='rinkspace.body.effective-v8-camera-rebind-prepare.v1',
        prepared_input=binding(packet_path), old_camera=binding(old_camera_path),
        replacement_camera=binding(camera_path), effective_local_body_samples=binding(sample_path),
        numerical_trajectory_warm_start=binding(trace_path),
        official_pose_clock_authority=binding(patch_path),
        native_presentations=len(packet['native_pts']), actual_exposures=len(packet['actual_pts']),
        repeated_native_aliases=len(packet['native_pts'])-len(packet['actual_pts']),
        physical_installations=sorted(np.unique(new_camera['installation_id']).tolist()),
        fixed_source_pixel_and_output_mode_fields_retained=fixed_fields,
        shared_calibration_changes=calibration_changes,
        raw_root_old_camera_replay_max_absolute=replay_error,
        old_camera_identity_all_arrays_bitexact=all(same(packet[key], updated[key]) for key in packet) if not changed else None,
        changed_camera=changed, changed_uniform_model_scale=scale_changed,
        retained_array_keys=preserved,
        official_corrected_actual_count=len(packet['official_pose_clock_patch_applied_actual_rows']),
        reviewed_whole_pose_inferred_actual_count=len(packet['body_local_pose_inference_rows']),
        opening_near_static_end_source_pts=int(packet['opening_near_static_user_source_end_pts']),
        previous_metres_per_scene_unit=old_unit, metres_per_scene_unit=new_unit,
        scale_source=scale_binding,
        metric_scale_status='declared_uniform_working_scale_not_measured_calibration',
        body_scale=float(packet['body_scale']),
        old_world_trajectory_imported_as_measurement=False, body_optimization_run=False,
        trajectory_generated=False, camera_accuracy_accepted=False,
        next_root_fit=dict(module='calibration.program_tracking.body_tracking.fullprogramme_body_sequence',
                           base=str(base_path), camera=str(Path(camera_path).resolve()), hips=str(hip_path),
                           clock=str(clock_path), body_samples=str(sample_path),
                           envelope=original_report['source_envelope_binding']['path'],
                           warm_controls=str(trace_path) if not changed and not scale_changed else None,
                           spacing=original_report['knot_spacing_initial_s'],
                           official_pose_clock_patch=None, pose_inference_packet=None,
                           note='Use this prepared input in a new output. Do not apply already effective pose overlays twice. Refit the continuous pelvis root against the replacement camera; old world controls are not reused when camera or model scale changes.'),
        pelvis_semantics='Future fitted P is the body root/pelvis; G copies its XY with Z=0. Neither is a blade-contact measurement.')
    return updated, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v8-directory', required=True, type=Path)
    parser.add_argument('--camera', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--write-input', action='store_true', help='Only prepare the input for a later root fit; no fit is run')
    parser.add_argument('--metres-per-scene-unit', type=float,
                        help='Uniform scale of the replacement common rink model; local body remains scaled once')
    parser.add_argument('--scale-source', type=Path, help='Saved provenance for the replacement uniform scale')
    parser.add_argument('--fit-root', action='store_true', help='Refit only the continuous pelvis root using retained v8 source observations')
    parser.add_argument('--max-nfev', type=int, default=25)
    parser.add_argument('--opening-planar-binding',choices=['soft-velocity','shared-xy'],default='soft-velocity',
                        help='Soft approximate opening motion; shared-xy only reproduces historical hard-bound runs')
    parser.add_argument('--opening-source-support',choices=['conditional-upper-body','temporal-only'],default='conditional-upper-body',
                        help='Conditional original upper-body pixels for inferred opening; temporal-only reproduces the historical gate')
    parser.add_argument('--body-fit-clock-evidence', type=Path,
                        help='Bound source-pixel evidence for root-fit times; camera/native aliases stay intact')
    parser.add_argument('--same-exposure-direct-pose-cache', type=Path,
                        help='Bound same-exposure direct local-body cache; no new pose or FK')
    args = parser.parse_args()
    storage = AssetStore(PROJECT)
    if not args.output.resolve().is_relative_to(storage.data):
        raise ValueError('New body arrays and logs must stay in the registered RinkSpace data root')
    if args.output.exists():
        raise FileExistsError('Preserve existing body run; choose a fresh output directory')
    packet, record = prepare_v8_camera_input(args.v8_directory, args.camera,
        metres_per_scene_unit=args.metres_per_scene_unit, scale_source=args.scale_source)
    if args.body_fit_clock_evidence:
        from .fullprogramme_body_sequence import apply_body_fit_clock_evidence_overlay
        packet, clock_binding = apply_body_fit_clock_evidence_overlay(
            packet, args.body_fit_clock_evidence, args.camera)
        record['body_fit_clock_evidence_overlay'] = clock_binding
        record['next_root_fit']['warm_controls'] = None
    prepared_samples = None
    if args.same_exposure_direct_pose_cache:
        packet, prepared_samples, cache_binding = apply_same_exposure_direct_pose_cache(
            packet, args.same_exposure_direct_pose_cache, record['next_root_fit']['body_samples'])
        record['same_exposure_direct_pose_cache'] = cache_binding
        record['next_root_fit']['warm_controls'] = None
    record['historical_camera_canonical_rows'] = len(packet['actual_pts'])
    record['distinct_body_fit_exposure_times'] = int(len(np.unique(packet['fit_time_seconds'])))
    record['actual_exposures_field_semantics'] = ('Historical canonical camera rows; '
        'body-fit time evidence may bind multiple retained rows to one exposure time')
    args.output.mkdir(parents=True, exist_ok=True)
    if prepared_samples is not None:
        sample_path = args.output / 'prepared_body_envelope_samples.npz'
        with sample_path.open('xb') as stream:
            np.savez_compressed(stream, **prepared_samples)
        record['next_root_fit']['body_samples'] = str(sample_path)
        record['output_prepared_local_samples'] = binding(sample_path)
    if args.write_input or args.fit_root:
        destination = args.output / 'fullprogramme_body_source_input.npz'
        with destination.open('xb') as stream:
            np.savez_compressed(stream, **packet)
        record['output_prepared_input'] = binding(destination)
    report_path = args.output / 'camera_rebind_prepare.json'
    report_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.fit_root:
        from types import SimpleNamespace
        from .fullprogramme_body_sequence import run
        fit_args = dict(record['next_root_fit'])
        for field in ('module', 'note'):
            fit_args.pop(field)
        run(SimpleNamespace(**fit_args, output=str(args.output), max_nfev=args.max_nfev,
            opening_planar_binding=args.opening_planar_binding,opening_source_support=args.opening_source_support,
            pose_inference_body_samples=None))
    print(json.dumps(dict(report=str(report_path), actual=record['actual_exposures'],
                          native=record['native_presentations'], identity=record['old_camera_identity_all_arrays_bitexact'],
                          root_replay_max=record['raw_root_old_camera_replay_max_absolute'],
                          body_fit_run=args.fit_root)), flush=True)


if __name__ == '__main__':
    main()
