"""Assemble three fitted Z190 installations without changing their states.

Every source PTS and exposure identity must match the existing programme/hip
evidence. This creates a review candidate, not an accuracy acceptance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
if str(ROOT/'src') not in sys.path:sys.path.insert(0,str(ROOT/'src'))

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import numpy as np
import cv2
from z190_physical_camera import PhysicalCameraModel
from camera_hardware_metadata import load_metadata
from three_frame_refit.program_tracking.body_tracking.pelvis_proxy_programme import validate_source_hip_clock
from rinkspace.metric_scale import AssumedMetricScale


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def installation_geometry_summary(calibrations, canonical_views, metres_per_scene_unit):
    """Describe actual pivot/lever/C values, independent of legacy priors.

    A fixed pivot is not a fixed optical centre. Heights refer to ice z=0;
    the user's ground-relative visual estimate is retained separately.
    Only actual exposure states are counted, never repeated presentations.
    """
    result = {}
    for name, calibration in calibrations.items():
        pts = sorted(p for p, state in canonical_views.items()
                     if state.installation_id == name)
        if not pts:
            raise ValueError('An installation has no actual exposure states')
        centres = np.asarray([calibration.camera25(canonical_views[p].view)[:3]
                              for p in pts])*metres_per_scene_unit
        pivot = np.asarray(calibration.pivot_xyz)*metres_per_scene_unit
        lever = np.asarray(calibration.lever_xyz)*metres_per_scene_unit
        result[name] = dict(
            actual_exposures=len(pts),
            pivot_xyz_assumed_m=pivot.tolist(),
            camera_local_lever_xyz_assumed_m=lever.tolist(),
            lever_axes='OpenCV: right, down, forward',
            optical_centre_formula='C = pivot + R_world_to_camera.T @ (lever + axial(q)*ez)',
            zoom_dependent_axial_enabled=bool(any(calibration.axial)),
            fixed_optical_centre_model=bool(not np.any(lever) and not any(calibration.axial)),
            first_actual_exposure_pts=pts[0],
            first_actual_optical_centre_assumed_m=centres[0].tolist(),
            optical_centre_xyz_min_assumed_m=centres.min(axis=0).tolist(),
            optical_centre_xyz_max_assumed_m=centres.max(axis=0).tolist(),
            optical_centre_max_distance_from_first_assumed_m=float(
                np.max(np.linalg.norm(centres-centres[0], axis=1))),
            optical_height_reference='ice z=0; operator ground elevation unmeasured',
            active_motion_model=calibration.field_settings.get('active_motion_model'),
            all_geometry_is_conditional_and_unaccepted=True)
    return result


def assemble(paths, scene_path, hips_path, shots_path, scale_path, output):
    output = Path(output).resolve()
    if not output.is_relative_to(Path('D:/CodexData').resolve()):
        raise ValueError('Numeric programme products must stay outside OneDrive')
    output.mkdir(parents=True, exist_ok=False)
    with np.load(hips_path, allow_pickle=False) as z:
        hips = {name:z[name].copy() for name in z.files}
    shots = json.loads(Path(shots_path).read_text(encoding='utf8'))
    clock = validate_source_hip_clock(hips_path, hips, shots_path)
    scale = AssumedMetricScale.load(scale_path, scene_path=scene_path)
    calibrations, aliases, views, installation_by_pts, bindings = {}, {}, {}, {}, []
    for path in paths:
        model, pts, state_aliases, canonical_views = PhysicalCameraModel.load_npz(path)
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z['metadata_json']))
        if (metadata['source_sha256'] != shots['source_sha256_from_timeline'] or
                metadata['source_time_base'] != shots['source_time_base']):
            raise ValueError('A fitted installation belongs to another source clock')
        if set(calibrations) & set(model.calibrations) or set(aliases) & set(state_aliases):
            raise ValueError('Duplicate installation or native source PTS')
        calibrations.update(model.calibrations)
        aliases.update(state_aliases)
        views.update(canonical_views)
        for p in map(int, pts):
            installation_by_pts[p] = canonical_views[state_aliases[p]].installation_id
        bindings.append(dict(path=str(Path(path).resolve()), sha256=digest(path),
            native_frames=len(pts), exposures=len(canonical_views)))
        component_manifest = Path(path).parent/'input_manifest.json'
        if component_manifest.exists():
            # Preserve the actual component's source rows/packet references,
            # including new direct pixels, without promoting accuracy claims.
            bindings[-1]['source_input_manifest'] = dict(
                path=str(component_manifest.resolve()), sha256=digest(component_manifest))
    if len(calibrations) != 3:
        raise ValueError('Expected three distinct Z190 installations')
    source_authority_context = {}
    for identifier, calibration in calibrations.items():
        admission = calibration.field_settings.get('source_authority_admission')
        if admission is None:
            continue
        admission_path = Path(admission['path'])
        if digest(admission_path) != admission['sha256']:
            raise ValueError('Declared source-authority admission record changed')
        record = json.loads(admission_path.read_text(encoding='utf8'))
        if record['scene_sha256'] != digest(scene_path):
            raise ValueError('Source-authority admission belongs to another rink geometry')
        source_authority_context[identifier] = dict(
            path=str(admission_path.resolve()), sha256=admission['sha256'],
            pending_ice_identity_rows=int(record['pending_ice_rows']),
            unconfirmed_ice_proposals_excluded_from_fit=bool(
                admission['unconfirmed_ice_proposals_excluded_from_fit']),
            rows_selected_by_residual=bool(record['rows_selected_by_residual']))
    hardware=load_metadata(shots_path=shots_path)
    position_facts=hardware.get('user_tripod_position_facts')
    position_context=None
    if position_facts is not None:
        # The latest user annotation supersedes the earlier low installation
        # inside the rink. Check the tripod/pivot, not a displaced lens centre.
        authority=hardware['associations']['latest_user_tripod_positions']
        low=calibrations['okada_sp_installation_2_conditional_z190']
        binding=low.field_settings.get('installation_position_binding',{})
        if (binding.get('user_json_sha256')!=authority['sha256']
                or binding.get('user_png_sha256')!=position_facts['annotated_image']['sha256']):
            raise ValueError('Low installation has not incorporated the latest user tripod position')
        xy=np.asarray(low.pivot_xyz[:2],float)
        nominal=np.asarray(position_facts['low_end']['nominal_pivot_xy_scene_units'],float)
        if (binding.get('pivot_policy')=='fixed_user_approximate_nominal_outside_board'
                and not np.allclose(xy,nominal,rtol=0.,atol=1e-10)):
            raise ValueError('Low installation moved away from its declared fixed user pivot')
        scene=json.loads(Path(scene_path).read_text(encoding='utf8'))
        boundary=np.ascontiguousarray(np.asarray(scene['ice_boundary']['points_xyz'],np.float32)[:,:2])
        signed=cv2.pointPolygonTest(boundary,(float(xy[0]),float(xy[1])),True)
        if signed>=0.:
            raise ValueError('Low tripod/pivot contradicts the user constraint outside the rink')
        position_context=dict(evidence=authority,low_installation_binding=binding,
            user_approximate_position_not_a_survey=True,
            low_pivot_outside_ice_boundary=True,
            signed_boundary_distance_assumed_m=float(signed*scale.metres_per_scene_unit))
    native = np.array(sorted(aliases), np.int64)
    if not np.array_equal(native, hips['pts']):
        raise ValueError('Three fitted installations must cover every programme source PTS exactly')
    expected = dict(zip(map(int, hips['pts']), map(int, hips['source_observation_pts'])))
    if aliases != expected:
        raise ValueError('Three-installation camera aliases differ from confirmed source exposures')
    for p, cut in zip(native, hips['shot_index']):
        group = shots['shots'][int(cut)]['camera_id']
        slot = {'main_long_side':0, 'secondary_long_side':1, 'low_end':2}[group]
        if installation_by_pts[int(p)] != f'okada_sp_installation_{slot}_conditional_z190':
            raise ValueError('Fitted installation changes the classified camera of a source frame')
    model = PhysicalCameraModel(calibrations)
    path = output/'z190_full_programme_camera_candidate.npz'
    model.save_npz(path, native, aliases, views, source_sha256=shots['source_sha256_from_timeline'],
        time_base=shots['source_time_base'], installation_by_pts=installation_by_pts)
    loaded, replay_pts, replay_aliases, replay_views = PhysicalCameraModel.load_npz(path)
    if not np.array_equal(replay_pts, native) or replay_aliases != aliases:
        raise ValueError('Combined Z190 package did not preserve exact source identity')
    geometry_context = installation_geometry_summary(calibrations, views, scale.metres_per_scene_unit)
    latest_facts = hardware.get('user_camera_facts') or {}
    required_mount = latest_facts.get('rigid_mount_offset')
    mount_requirements_met = None
    if required_mount is not None:
        lower = np.asarray(required_mount['lower_camera_xyz_m'], float)
        upper = np.asarray(required_mount['upper_camera_xyz_m'], float)
        mount_requirements_met = all(
            not entry['fixed_optical_centre_model']
            and not entry['zoom_dependent_axial_enabled']
            and np.all(np.asarray(entry['camera_local_lever_xyz_assumed_m']) >= lower-1e-10)
            and np.all(np.asarray(entry['camera_local_lever_xyz_assumed_m']) <= upper+1e-10)
            for entry in geometry_context.values())
    report = dict(schema='rinkspace.z190-complete-programme-binding.v1',
        accuracy_accepted=False, whole_programme_camera_ready=True, interpolation_used=False,
        numeric_camera_states_complete=True, camera_world_alignment_accepted=False,
        camera_ready_semantics='Exact source states available for review; not a whole-programme alignment acceptance',
        source_authority_context=source_authority_context,
        camera_model='Sony PXW-Z190', model_identity_authority='production_team_user_confirmed',
        installation_authority='user_observed_rinkside_fixed_tripods_standing_operators',
        installation_pivot_height_assumed_m={name:float(cal.pivot_xyz[2]*scale.metres_per_scene_unit)
            for name,cal in calibrations.items()},
        installation_geometry=geometry_context,
        latest_user_rigid_mount_assumptions=required_mount,
        all_installations_use_latest_user_nonzero_mount_model=mount_requirements_met,
        installation_height_policy='Independent fixed pivot heights; optical centre heights follow rotation and need not stay constant',
        standing_height_prior_nominal_m=1.60,
        standing_height_prior_policy='Independent reference lens heights above operator ground; not pivot heights, equal heights, or measured ice-relative bounds',
        latest_user_installation_height_policy=latest_facts.get('installation_height_policy'),
        latest_user_lens_and_pivot_height_estimates=latest_facts.get('height_estimate'),
        metric_scale_status='metric_assumed', metres_per_scene_unit=scale.metres_per_scene_unit,
        scene_path=str(Path(scene_path).resolve()), scene_sha256=digest(scene_path),
        source_sha256=shots['source_sha256_from_timeline'], source_time_base=shots['source_time_base'],
        source_time_start=shots['analysis_scope']['source_time_start'],
        source_time_end_exclusive=shots['analysis_scope']['source_time_end_exclusive'],
        native_frames=len(native), actual_exposures=len(views), confirmed_alias_presentations=len(native)-len(views),
        source_hip_exposure_clock=clock, fitted_installation_bindings=bindings,
        latest_user_tripod_position_context=position_context,
        hardware_metadata_input_manifest=hardware['input_manifest'],
        exact_three_installation_states_replayed=True,
        exported_camera_path=str(path), exported_camera_sha256=digest(path),
        rekeyed_actual_source_keys=[])
    path.with_suffix('.binding.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k:report[k] for k in ('whole_programme_camera_ready','native_frames','actual_exposures',
        'confirmed_alias_presentations','exported_camera_path')}, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=Path, action='append', required=True)
    for key in ('scene','source-hips','shots','metric-scale','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    assemble(args.camera, args.scene, args.source_hips, args.shots, args.metric_scale, args.output)
