"""Rebind explicitly reviewed source exposures without changing rig parameters.

This module only changes programme FrameBinding aliases and spline clocks.
It does not read a source detector, change a reference camera, build a graph,
add factors, or solve.  Caller-supplied PTS maps are the complete authority;
no duplicate-exposure PTS are embedded here.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping

import cv2
import numpy as np

from continuous_camera_model import FrameBinding
from continuous_quadratic_rig_optin import QuadraticRigCameraModel


def _node(value, label: str) -> np.ndarray:
    array = np.asarray(value, np.float64)
    if array.shape != (11,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite Vector11")
    return array


def _exact(a, b) -> bool:
    x, y = np.asarray(a), np.asarray(b)
    return x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()


def _digest_nodes(nodes: Mapping[int, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(nodes):
        value = _node(nodes[key], f"node {key}")
        digest.update(np.asarray([key], np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _clock(row: FrameBinding) -> float:
    return float(row.pts if row.spline_pts is None else row.spline_pts)


def _physical_chain(model: QuadraticRigCameraModel, pts: int,
                    states: Mapping[int, np.ndarray]) -> np.ndarray:
    """Independent C/R/K/Brown15 expression for changed binding QA."""
    row = model.frames[pts]
    group = row.camera_id
    frame = _node(states[row.frame_key], "frame")
    z = float(frame[6])
    dz = z - model.lens_zoom_reference[group]
    half_dz2 = .5 * dz * dz
    rotation = cv2.Rodrigues(frame[3:6])[0]
    residual = np.zeros(3, np.float64)
    for key, weight in model.shot_splines[row.shot_id].basis_pairs(_clock(row)):
        residual += weight * _node(states[key], "spline control")[:3]
    mount = _node(states[model.mount_keys[group]], "mount")[:3]
    offset = (_node(states[model.offset_keys[group]], "offset")[:3]
              + dz * _node(states[model.offset_slope_keys[group]], "offset slope")[:3])
    base = _node(states[model.lens_base_keys[group]], "lens base")
    slope = (_node(states[model.lens_slope_keys[group]], "lens slope")
             if group in model.lens_slope_keys else np.zeros(11))
    lens_q = _node(states[model.lens_quadratic_keys[group]], "lens curvature")
    brown = _node(states[model.brown_keys[group]], "Brown base")
    brown_slope = (_node(states[model.brown_slope_keys[group]], "Brown slope")
                   if group in model.brown_slope_keys else np.zeros(11))
    brown_q = _node(states[model.brown_quadratic_keys[group]], "Brown curvature")
    ratio = base[6] + dz*slope[6] + half_dz2*lens_q[6]
    physical = np.zeros(15, np.float64)
    physical[:3] = mount + rotation.T @ offset + residual
    physical[3:6] = frame[3:6]
    physical[6] = z + .5*ratio
    physical[7] = z - .5*ratio
    physical[8:11] = base[8:11] + dz*slope[8:11] + half_dz2*lens_q[8:11]
    physical[11:15] = brown[:4] + dz*brown_slope[:4] + half_dz2*brown_q[:4]
    return physical


def _role_mappings(model: QuadraticRigCameraModel) -> dict[str, dict]:
    names = ("lens_base_keys", "lens_slope_keys", "lens_zoom_reference",
             "mount_keys", "offset_keys", "offset_slope_keys", "brown_keys",
             "brown_slope_keys", "lens_quadratic_keys", "brown_quadratic_keys")
    return {name: dict(getattr(model, name)) for name in names}


def rebind_source_camera(
    model: QuadraticRigCameraModel,
    warm_states: Mapping[int, np.ndarray],
    prior_states: Mapping[int, np.ndarray],
    pose_aliases: Mapping[int, int],
    pose_clock: Mapping[int, float],
) -> tuple[QuadraticRigCameraModel, dict[int, np.ndarray],
           dict[int, np.ndarray], dict]:
    """Return ``(new_model, new_initial, new_prior, report)``.

    ``pose_aliases`` and ``pose_clock`` are *partial* maps keyed by original
    source PTS, with identical keysets.  Every destination frame key must be
    an existing self-bound canonical source PTS in the same shot and camera.
    After rebinding, every alias group must share exactly one virtual clock.
    The earliest frame of every shot keeps both its key and original clock.

    Removed keys are old programme frame nodes only.  Their old graph dummy
    and R/zoom temporal regularizers must be rebuilt by the caller.  All
    source-UV observations must likewise be rebuilt from source PTS; this
    function never edits an existing factor graph.
    """
    if not isinstance(model, QuadraticRigCameraModel):
        raise TypeError("Rebinding requires an already reconstructed quadratic rig")
    if set(map(int, warm_states)) != set(map(int, prior_states)):
        raise ValueError("Warm and original-prior state keysets differ")
    warm = {int(k): _node(v, f"warm {k}") for k, v in warm_states.items()}
    prior = {int(k): _node(v, f"prior {k}") for k, v in prior_states.items()}
    aliases = {int(k): int(v) for k, v in pose_aliases.items()}
    clocks = {int(k): float(v) for k, v in pose_clock.items()}
    if len(aliases) != len(pose_aliases) or len(clocks) != len(pose_clock):
        raise ValueError("Duplicate or ambiguous source PTS map keys")
    if set(aliases) != set(clocks) or not set(aliases) <= set(model.frames):
        raise ValueError("Alias and virtual-clock maps need the same known source PTS")
    if not np.isfinite(list(clocks.values())).all():
        raise ValueError("New virtual clock contains nonfinite values")
    if any(key not in warm for key in model.frame_keys):
        raise ValueError("Warm state lacks an original frame node")
    original_canonical = {key: model.frames[key] for key in model.frame_keys
                          if key in model.frames and model.frames[key].frame_key == key}
    if set(original_canonical) != model.frame_keys:
        raise ValueError("Every frame key must be a self-bound source PTS")
    for pts, destination in aliases.items():
        source_row = model.frames[pts]
        if destination not in original_canonical:
            raise ValueError(f"Destination {destination} is not an original canonical frame")
        anchor = original_canonical[destination]
        if (source_row.shot_id != anchor.shot_id
                or source_row.camera_id != anchor.camera_id):
            raise ValueError(f"Cross-shot/camera alias {pts}->{destination}")
    rows: list[FrameBinding] = []
    for pts, old in model.frames.items():
        if pts in aliases:
            rows.append(FrameBinding(pts, aliases[pts], old.shot_id,
                                     old.camera_id, old.template25.copy(), clocks[pts]))
        else:
            rows.append(old)
    updated = {row.pts: row for row in rows}
    by_key: dict[int, list[FrameBinding]] = defaultdict(list)
    for row in rows:
        by_key[row.frame_key].append(row)
    for key, members in by_key.items():
        if key not in updated or updated[key].frame_key != key:
            raise ValueError(f"New alias key {key} lacks a self-bound canonical source frame")
        identities = {(r.shot_id, r.camera_id, _clock(r)) for r in members}
        if len(identities) != 1:
            raise ValueError(f"Alias group {key} crosses a shot/camera/virtual clock")
    first_targets = []
    for shot_id in model.shot_splines:
        source_pts = sorted(p for p, row in model.frames.items() if row.shot_id == shot_id)
        if not source_pts:
            raise ValueError(f"Shot {shot_id} has no source frames")
        first_pts = source_pts[0]
        old, new = model.frames[first_pts], updated[first_pts]
        if (new.frame_key != old.frame_key or _clock(new) != _clock(old)
                or not _exact(prior[old.frame_key], prior_states[old.frame_key])):
            raise ValueError(f"Shot {shot_id} first-frame prior target would change")
        first_targets.append(dict(shot_id=shot_id, source_pts=first_pts,
                                  frame_key=old.frame_key,
                                  target_rlogf=prior[old.frame_key][3:7].tolist()))
    old_frame_keys = set(model.frame_keys)
    new_frame_keys = set(by_key)
    removed = sorted(old_frame_keys-new_frame_keys)
    nonframe_keys = set(warm)-old_frame_keys
    if set(removed) & nonframe_keys:
        raise ValueError("A shared or reference state would be removed")
    new_initial = {key: value.copy() for key, value in warm.items() if key not in removed}
    new_prior = {key: value.copy() for key, value in prior.items() if key not in removed}
    if (set(new_initial) != set(new_prior)
            or set(new_initial) != set(warm)-set(removed)
            or any(not _exact(new_prior[key], prior[key]) for key in new_prior)
            or any(not _exact(new_initial[key], warm[key]) for key in new_initial)):
        raise ValueError("A retained warm/prior Vector11 changed")
    role_maps = _role_mappings(model)
    new_model = QuadraticRigCameraModel(
        model.shot_splines, rows, model.lens_base_keys, model.lens_slope_keys,
        model.lens_zoom_reference, mount_keys=model.mount_keys,
        offset_keys=model.offset_keys, offset_slope_keys=model.offset_slope_keys,
        brown_keys=model.brown_keys, brown_slope_keys=model.brown_slope_keys,
        lens_quadratic_keys=model.lens_quadratic_keys,
        brown_quadratic_keys=model.brown_quadratic_keys)
    if (_role_mappings(new_model) != role_maps
            or new_model.shot_splines != model.shot_splines
            or set(new_model.frames) != set(model.frames)
            or set(new_model.frame_keys) != new_frame_keys):
        raise ValueError("Shared rig, shot splines or source PTS changed")
    for pts in new_model.frames:
        if set(new_model.dependencies(pts))-set(new_initial):
            raise ValueError(f"New camera {pts} has an absent graph node")
    change_rows = []
    unchanged_count = 0
    checked_jacobian_blocks = 0
    max_chain_error = 0.
    for pts, old in model.frames.items():
        new = new_model.frames[pts]
        before_camera = model.camera25(pts, warm)
        after_camera = new_model.camera25(pts, new_initial)
        before_physical, before_j = model.physical_frame(pts, warm)
        after_physical, after_j = new_model.physical_frame(pts, new_initial)
        binding_changed = (old.frame_key != new.frame_key or _clock(old) != _clock(new))
        if not binding_changed:
            if (not _exact(before_camera, after_camera)
                    or not _exact(before_physical, after_physical)
                    or set(before_j) != set(after_j)
                    or any(not _exact(before_j[k], after_j[k]) for k in before_j)):
                raise ValueError(f"Unchanged source binding {pts} changed camera/J")
            unchanged_count += 1
            checked_jacobian_blocks += len(before_j)
            continue
        expected = _physical_chain(new_model, pts, new_initial)
        chain_error = float(np.max(np.abs(expected-after_physical)))
        if chain_error > 2e-12:
            raise ValueError(f"New physical C/R/K/Brown chain differs at {pts}: {chain_error}")
        max_chain_error = max(max_chain_error, chain_error)
        canonical_camera = new_model.camera25(new.frame_key, new_initial)
        canonical_physical, canonical_j = new_model.physical_frame(new.frame_key, new_initial)
        if (not _exact(after_camera, canonical_camera)
                or not _exact(after_physical, canonical_physical)
                or set(after_j) != set(canonical_j)
                or any(not _exact(after_j[k], canonical_j[k]) for k in after_j)):
            raise ValueError(f"Aliased camera/J differs from canonical at {pts}")
        delta = np.abs(after_camera-before_camera)
        change_rows.append(dict(
            source_pts=pts, shot_id=old.shot_id,
            old_frame_key=old.frame_key, new_frame_key=new.frame_key,
            old_virtual_pts=_clock(old), new_virtual_pts=_clock(new),
            camera25_max_abs_change=float(np.max(delta)),
            camera25_center_scene_coordinate_l2_change=float(np.linalg.norm(delta[:3])),
            camera25_rvec_max_abs_change=float(np.max(delta[3:6])),
            camera25_logf_max_abs_change=float(np.max(delta[6:8])),
            camera25_intrinsic_px_max_abs_change=float(np.max(delta[8:11])),
            camera25_brown4_max_abs_change=float(np.max(delta[11:15])),
            new_equals_canonical_camera25_and_jacobian_bitwise=True,
            physical_chain_max_abs_error=chain_error))
    group_rows = []
    for key in sorted(new_frame_keys):
        members = sorted(by_key[key], key=lambda row: row.pts)
        if len(members) > 1 or key in aliases.values():
            group_rows.append(dict(canonical_frame_key=key,
                                   source_pts=[row.pts for row in members],
                                   shot_id=members[0].shot_id,
                                   camera_id=members[0].camera_id,
                                   virtual_pts=_clock(members[0])))
    report = dict(
        schema="rinkspace.source-camera-rebinding.v1",
        policy="explicit source-PTS alias/virtual-clock maps only; no solver or source detector",
        source_pts_count=len(model.frames),
        old_frame_node_count=len(old_frame_keys),
        new_frame_node_count=len(new_frame_keys),
        removed_old_frame_nodes=removed,
        removed_old_node_allowed_prior_roles=["frame_dummy", "frame_Rzoom_temporal"],
        removed_old_node_graph_factors_inspected=False,
        retained_shared_and_reference_node_count=len(nonframe_keys),
        retained_warm_states_bitwise_equal=True,
        retained_prior_targets_bitwise_equal=True,
        retained_prior_sha256_before=_digest_nodes({k: prior[k] for k in new_prior}),
        retained_prior_sha256_after=_digest_nodes(new_prior),
        first_frame_targets=first_targets,
        shot_splines_keys_knots_unchanged=True,
        shared_lens_brown_mechanics_keys_and_zoom_reference_unchanged=True,
        reference_model_modified=False,
        unchanged_binding_camera25_and_jacobian_bitwise_count=unchanged_count,
        unchanged_binding_jacobian_blocks_bitwise_count=checked_jacobian_blocks,
        changed_binding_count=len(change_rows),
        max_physical_chain_error=max_chain_error,
        changed_bindings=change_rows,
        new_alias_groups=group_rows,
        source_binding_metadata=dict(
            source_pts=sorted(new_model.frames),
            frame_key_by_source_pts={str(p): new_model.frames[p].frame_key
                                     for p in sorted(new_model.frames)},
            frame_clock_virtual_pts={str(p): _clock(new_model.frames[p])
                                     for p in sorted(new_model.frames)},
            shot_id_by_source_pts={str(p): new_model.frames[p].shot_id
                                   for p in sorted(new_model.frames)}),
        graph_rebuilt=False,
        source_uv_factors_modified=False)
    if report["retained_prior_sha256_before"] != report["retained_prior_sha256_after"]:
        raise ValueError("Retained prior payload hash differs")
    return new_model, new_initial, new_prior, report
