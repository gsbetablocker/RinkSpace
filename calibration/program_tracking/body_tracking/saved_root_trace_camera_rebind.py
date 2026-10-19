"""Express an existing pelvis trace in a new physical camera without fitting.

Preserve the old model P's native pixel and its conditional metric height.
Neither that pixel nor the retained height becomes a new source observation.
G remains P's world vertical projection, never an ice-blade contact.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# This artifact preparer runs in a fresh process before any renderer imports.
# Keep the original solver import names authenticated when reusing its rays.
from calibration.solver_runtime import install as _install_projection_runtime
_install_projection_runtime()
from .fullprogramme_camera_rebind import binding, checked_camera, digest, load, same


def _projection_functions():
    # Reuse the authenticated full Brown14/skew inverse and forward operators.
    # Importing these small functions prepares no source graph or solver.
    from structured_programme_source_candidate.persistent_ice_long_baseline_source import current_native_rays
    from wide_anchor_candidate.full_brown_projection import project_central_valid_full_brown
    return current_native_rays, project_central_valid_full_brown


def reexpress_root_samples(P, Q, old_camera25, new_camera25, old_unit, new_unit):
    """Return P/Q in the new chart and numerical projection diagnostics."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    old, new = np.asarray(old_camera25, float), np.asarray(new_camera25, float)
    if (P.ndim != 2 or P.shape[1] != 3 or Q.shape != (len(P), 3, 3)
            or old.shape != (len(P), 25) or new.shape != old.shape
            or not all(np.isfinite(x).all() for x in (P, Q, old, new))
            or not np.isfinite(old_unit) or not np.isfinite(new_unit)
            or old_unit <= 0 or new_unit <= 0):
        raise ValueError('Finite saved roots, physical cameras and positive conditional units required')
    rays, project = _projection_functions()
    out_P, out_Q = P.copy(), Q.copy()
    old_pixels = np.empty((len(P), 2), float)
    errors = np.zeros(len(P), float)
    delta = np.tile(np.eye(3), (len(P), 1, 1))
    for i in range(len(P)):
        uv, _, valid = project(old[i], P[i:i+1]/old_unit)
        if not np.all(valid):
            raise ValueError('Saved model P leaves its original central optical domain')
        old_pixels[i] = uv[0]
        if same(old[i], new[i]) and old_unit == new_unit:
            continue  # Preserve the exact identity chart, including matrix bytes.
        ray = rays(new[i], uv)[0]
        if abs(ray[2]) < 1e-12:
            raise ValueError('New model P ray is parallel to its retained height plane')
        distance = (P[i, 2]/new_unit-new[i, 2])/ray[2]
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError('Retained model P height has no positive new-camera ray intersection')
        out_P[i] = (new[i, :3]+distance*ray)*new_unit
        out_P[i, 2] = P[i, 2]  # Exact original metric height; no height estimation.
        delta[i] = cv2.Rodrigues(new[i, 3:6])[0].T @ cv2.Rodrigues(old[i, 3:6])[0]
        out_Q[i] = delta[i] @ Q[i]
        rebound_uv, _, valid = project(new[i], out_P[i:i+1]/new_unit)
        errors[i] = np.linalg.norm(rebound_uv[0]-uv[0])
        if not np.all(valid) or errors[i] > 1e-6:
            raise ValueError('Reexpressed model P does not close its preserved native pixel')
    return out_P, out_Q, delta, old_pixels, errors


_CURRENT_DERIVED = frozenset((
    'P_world_m', 'Q_world', 'camera25', 'G_world_m', 'G_native_uv',
    'COCO17_world_m', 'native22_world_m', 'COCO17_native_uv',
    'native_P_world_m', 'native_Q_world', 'native_G_world_m', 'native_G_native_uv',
    'metres_per_scene_unit'))
_OLD_FIT_HISTORY = frozenset((
    'source_residual_weighted', 'source_envelope_residual', 'raw_root_prior_Q_world',
    'position_controls_world_m', 'shared_opening_xy_control_indices',
    'opening_shared_estimated_xy_world_m', 'conditional_acceleration_world_m_s2',
    'opening_planar_velocity_condition_m_s', 'opening_conditional_velocity_world_m_s',
    'root_rotation_correction_controls', 'position_knots', 'rotation_cut_control_offsets',
    'rotation_knots_concatenated', 'rotation_knot_offsets', 'minimum_body_camera_depth_m'))


def rebind_saved_trace(trace_path, old_camera_path, new_camera_path, *,
                      expected_trace_sha256, expected_camera_sha256, camera_q_sha256,
                      grid_state_identity_path, new_unit=None, scale_source=None):
    """Return new renderer-compatible arrays and provenance; write nothing.

    Paths identify immutable old trace and complete physical camera exports.
    The caller supplies the accepted grid q/camera binding, then writes these
    arrays to a new Data artifact. This output is not a body-solver warm state.
    """
    if digest(trace_path) != expected_trace_sha256 or digest(new_camera_path) != expected_camera_sha256:
        raise ValueError('Immutable trace or intended replacement camera identity changed')
    if len(camera_q_sha256) != 64 or any(x not in '0123456789abcdef' for x in camera_q_sha256):
        raise ValueError('Exact accepted grid q SHA256 required as provenance')
    grid_identity = json.loads(Path(grid_state_identity_path).read_text(encoding='utf8'))
    if (grid_identity['q_sha256'] != camera_q_sha256
            or grid_identity['camera_sha256'] != expected_camera_sha256
            or Path(grid_identity['camera_path']).resolve() != Path(new_camera_path).resolve()):
        raise ValueError('Replacement camera does not belong to the bound accepted grid q')
    trace = load(trace_path)
    if 'trace_reexpression_contract_json' in trace:
        raise ValueError('Rebind the immutable original fitted trace, not a prior reexpression')
    old, old_rows, old_meta = checked_camera(old_camera_path, trace)
    new, rows, new_meta = checked_camera(new_camera_path, trace)
    if (old_meta['source_sha256'] != new_meta['source_sha256']
            or not same(old['installation_id'], new['installation_id'])
            or not same(trace['camera25'], old['camera25'][old_rows])):
        raise ValueError('Old trace/new camera original video or physical installation binding differs')
    for name, calibration in old_meta['calibrations'].items():
        for field in ('Brown14_order', 'output_mode_id', 'focal_domain',
                      'domain_wh', 'final_wh', 'A_source_to_final'):
            if calibration[field] != new_meta['calibrations'][name][field]:
                raise ValueError('Replacement camera changes the source pixel/output mode')
    unit = float(trace['metres_per_scene_unit'])
    target_unit = unit if new_unit is None else float(new_unit)
    if scale_source is not None:
        from rinkspace.metric_scale import AssumedMetricScale
        scale = AssumedMetricScale.load(scale_source)
        if new_unit is None:
            target_unit = scale.metres_per_scene_unit
        elif target_unit != scale.metres_per_scene_unit:
            raise ValueError('Requested display unit differs from its saved scale binding')
    elif target_unit != unit:
        raise ValueError('Changed uniform display scale needs its existing scale-source artifact')
    actual_camera = new['camera25'][rows].copy()
    identity = same(trace['camera25'], actual_camera) and unit == target_unit
    if not np.array_equal(trace['native22_world_m'][:, 0], trace['P_world_m']):
        raise ValueError('Saved native joint0 and model P identity differs')
    result = {key: value.copy() for key, value in trace.items()}
    P, Q, delta, pixels, error = reexpress_root_samples(
        trace['P_world_m'], trace['Q_world'], trace['camera25'], actual_camera, unit, target_unit)
    if not identity:
        for key in _OLD_FIT_HISTORY:
            if key in result:
                result['historical_original_camera__'+key] = result.pop(key)
        G = P.copy(); G[:, 2] = 0.
        _, project = _projection_functions()
        G_uv = np.asarray([project(c, g[None]/target_unit)[0][0] for c, g in zip(actual_camera, G)])
        result.update(P_world_m=P, Q_world=Q, G_world_m=G, G_native_uv=G_uv,
                      camera25=actual_camera, metres_per_scene_unit=np.asarray(target_unit))
        for key in ('COCO17_world_m', 'native22_world_m'):
            result[key] = P[:, None]+np.einsum('nij,nkj->nki', delta,
                trace[key]-trace['P_world_m'][:, None])
        result['native22_world_m'][:, 0] = P
        result['COCO17_native_uv'] = np.asarray([project(c, joints/target_unit)[0]
            for c, joints in zip(actual_camera, result['COCO17_world_m'])])
        native_rows = trace['native_actual_row']
        for key in ('P_world_m', 'Q_world', 'G_world_m', 'G_native_uv'):
            result['native_'+key] = result[key][native_rows]
    preserved = [key for key in trace if key not in _CURRENT_DERIVED|_OLD_FIT_HISTORY]
    if any(not same(trace[key], result[key]) for key in preserved):
        raise ValueError('Reexpression changed original source/clock/mask/status fields')
    if not same(result['camera25'][trace['native_actual_row']], new['camera25']):
        raise ValueError('Renderer actual/native camera25 must match bit for bit')
    if not same(result['P_world_m'][:, 2], trace['P_world_m'][:, 2]):
        raise ValueError('Existing conditional pelvis metric heights changed')
    report = dict(schema='rinkspace.saved-pelvis-trace-camera-reexpression.v1',
        original_trace=binding(trace_path), old_camera=binding(old_camera_path),
        replacement_camera=binding(new_camera_path), grid_q_sha256=camera_q_sha256,
        accepted_grid_state_identity=binding(grid_state_identity_path),
        source_video_sha256=new_meta['source_sha256'], time_base='1/15360',
        actual_exposures=len(P), native_presentations=len(trace['native_pts']),
        old_camera_identity_all_original_arrays_bitexact=all(same(trace[k],result[k]) for k in trace) if identity else None,
        current_body_native_camera25_bitexact=True, preserved_original_fields=preserved,
        previous_metres_per_scene_unit=unit, metres_per_scene_unit=target_unit,
        scale_source=None if scale_source is None else binding(scale_source),
        code=binding(__file__),
        preserved_original_model_P_pixel_max_error_native_px=float(error.max()),
        original_model_P_metric_height_bitexact=True,
        retained_height_is_previous_conditional_fit_not_measured_height=True,
        body_root_orientation='Q_new=R_new.T@R_old@Q_old; saved camera-relative fitted root orientation retained',
        P_pixel_is_old_model_projection_not_new_source_observation=True,
        G_semantics='G=(P.x,P.y,0), world vertical projection of saved native joint0 P, not blade contact',
        original_fit_diagnostics_are_historical_only=True,
        body_fit_or_pose_detection_or_FK_or_filter_or_gap_fill_run=False,
        output_is_body_solver_warm_state=False, trajectory_is_new_world_truth=False,
        accuracy_accepted=False)
    result['trace_reexpression_contract_json'] = np.asarray(json.dumps(report, ensure_ascii=False))
    return result, report
