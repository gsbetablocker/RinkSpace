"""Explicitly restore an authenticated user-adopted rigid mount range.

Historical null constraints remain historical. A revised binding may opt in;
its parameter-only restart replaces the nonphysical mount decomposition, not
the observed pixels or the four camera coordinates of each native exposure.
"""
from __future__ import annotations

import hashlib
import numpy as np

from .official_static_surface_dataset import read_reference


def read_mount_revision(reference, facts_reference, scene_unit_working_m):
    policy = read_reference(reference)
    if (policy.get("schema") != "rinkspace.first-shot-restored-user-mount.v1"
            or policy.get("facts") != facts_reference
            or policy.get("range_policy") != "restore_user_adopted_v4_working_search_range"
            or policy.get("initializer") != "user_lever_and_mean_native_optical_centres"
            or policy.get("working_metres_are_measured") is not False):
        raise ValueError("Explicit authenticated user mount restoration required")
    facts = read_reference(facts_reference)
    mount = facts["rigid_mount_offset"]
    if (mount.get("bounds_are_user_working_search_ranges_not_measurements") is not True
            or mount.get("shared_across_same_installation_exposures") is not True
            or mount.get("per_frame_free_optical_centre_enabled") is not False
            or abs(float(facts["metres_per_original_scene_unit"]) - scene_unit_working_m) > 1e-15):
        raise ValueError("Restored mount must retain its physical chart and working units")
    values = np.asarray([mount[k] for k in ("lower_camera_xyz_m", "initial_camera_xyz_m", "upper_camera_xyz_m")], float)
    if (values.shape != (3, 3) or not np.isfinite(values).all()
            or np.any(values[0] >= values[2])
            or np.any(values[1] < values[0]) or np.any(values[1] > values[2])):
        raise ValueError("Ordered user mount range and admissible initializer required")
    return dict(reference=reference, facts=facts_reference, initial_lever_working_m=values[1].tolist(),
                lower_lever_working_m=values[0].tolist(), upper_lever_working_m=values[2].tolist(),
                range_policy=policy["range_policy"], initializer=policy["initializer"],
                working_metres_are_measured=False, parent_constraint_manifest_changed=False)


def initialize_restored_mount(state, q):
    """Preserve mean native optical C with one P; never create per-frame C."""
    revision = getattr(state, "mount_revision_proof", None)
    if revision is None:
        return q, None
    from .z190_physical_camera import ViewState
    import cv2
    before = np.asarray(q, float).copy()
    after = before.copy()
    old = state.calibration(before)
    scale = state.scene_unit_working_m
    old_lever = np.asarray(old.lever_xyz) * scale
    lower = np.asarray(revision["lower_lever_working_m"], float)
    upper = np.asarray(revision["upper_lever_working_m"], float)
    if np.all(old_lever >= lower-1e-12) and np.all(old_lever <= upper+1e-12):
        return q, dict(policy=revision, already_within_restored_range=True,
                       all_parameter_values_byte_exact=True, mount_reinitialized=False)
    lever = np.asarray(revision["initial_lever_working_m"], float) / scale
    centres, offsets = [], []
    for index in range(1, state.frame_count + 1):
        view = ViewState.from_state4(before[state.view_columns(index)])
        centres.append(old.camera25(view)[:3])
        offsets.append(cv2.Rodrigues(np.asarray(view.rvec_world_to_camera))[0].T @ lever)
    centres, offsets = np.asarray(centres), np.asarray(offsets)
    pivot = np.mean(centres - offsets, axis=0)
    # q's mount and pivot coordinates are deltas in the existing working unit.
    after[118:121] += (pivot - np.asarray(old.pivot_xyz)) * scale
    after[121:124] += (lever - np.asarray(old.lever_xyz)) * scale
    delta = pivot + offsets - centres
    unchanged = np.ones(len(before), bool)
    unchanged[118:124] = False
    if not np.array_equal(before[unchanged], after[unchanged]):
        raise ValueError("Mount restoration changed unrelated parameters")
    return after, dict(policy=revision,
        input_q_sha256=hashlib.sha256(before.tobytes()).hexdigest(),
        output_q_sha256=hashlib.sha256(after.tobytes()).hexdigest(),
        old_pivot_working_m=(np.asarray(old.pivot_xyz)*scale).tolist(),
        old_lever_working_m=(np.asarray(old.lever_xyz)*scale).tolist(),
        new_pivot_working_m=(pivot*scale).tolist(), new_lever_working_m=(lever*scale).tolist(),
        mean_native_optical_centre_change_working_m=(np.mean(delta,axis=0)*scale).tolist(),
        maximum_native_optical_centre_change_working_m=float(np.max(np.linalg.norm(delta,axis=1))*scale),
        native_exposures=state.frame_count, all_RF4_world_optics_and_row_timing_preserved=True,
        fitted_camera_accuracy_not_implied=True, clipped_parameter_values=False)
