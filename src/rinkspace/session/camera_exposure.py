"""One native exposure of an existing motion camera; no fit or new camera family."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from ..domain.runtime_rigs import _camera_data
from ..domain.camera import complete_camera_rig
from ..path_policy import guard_project_write


FAMILY = {
    "schema": "rinkspace.camera.motion.v1",
    "mount_family": "orthogonal_pan_tilt_pivot",
    "lens_family": "brown_zoom_linear",
    "pupil_model": "zoom_quadratic_camera_z",
    "readout_mode": "predicted_row_implicit_unique",
}


def exposure_controls(camera, controls, pts):
    """Keep original cubic coefficients covering the whole signed readout.

    This slices an already expanded current state. It neither refits the spline
    nor re-applies its historical reference-control elimination.
    """
    if isinstance(pts, bool) or not isinstance(pts, Integral):
        raise ValueError("exposure identity requires native integer PTS")
    if any(camera.get(k) != v for k, v in FAMILY.items()):
        raise ValueError("unsupported complete motion-camera family")
    if not camera["first_pts"] <= pts < camera["stop_pts_exclusive"]:
        raise ValueError("PTS outside camera processing domain")
    t = (int(pts)-int(camera["reference_pts"]))*camera["time_base_num"]/camera["time_base_den"]
    tau = camera["rig"][37]
    interval = np.array([t-abs(tau)/2, t+abs(tau)/2])
    first, dt = camera["first_control_time"], camera["control_dt"]
    x = (interval-first)/dt
    if np.any(x < 1) or np.any(x > len(controls)-2):
        raise ValueError("full readout interval outside camera spline support")
    cells = np.minimum(np.floor(x).astype(int), len(controls)-3)
    start, stop = int(cells[0]-1), int(cells[1]+3)
    # Shifting a floating time origin can put an exact knot just below the
    # first local cell. Retain the adjacent original support at knot endpoints;
    # do not clamp time or alter coefficients to hide the rounding difference.
    if abs(x[0]-round(x[0])) < 1e-9:
        start = max(0, start-1)
    if abs(x[1]-round(x[1])) < 1e-9:
        stop = min(len(controls), stop+1)
    local = np.ascontiguousarray(controls[start:stop], dtype="<f8")
    snapshot = deepcopy(camera)
    snapshot.update(first_control_time=first+start*dt, controls_ref="controls.npy",
                    first_pts=int(pts), stop_pts_exclusive=int(pts)+1)
    dependency = snapshot.pop("reference_dependency", None)
    snapshot["snapshot_origin"] = {
        "purpose": "one_exposure_frozen_replay_not_fit",
        "control_indices": [start, stop], "controls_unchanged": True,
        "original_first_control_time": first,
        "original_reference_dependency_already_expanded": dependency,
        "original_processing_domain": [camera["first_pts"], camera["stop_pts_exclusive"]],
    }
    return snapshot, local, t


def snapshot_exposure(camera_path, pts, core, out_dir):
    """Save a self-contained current camera and evaluated row states.

    The three row states explain the exposure; the saved motion camera remains
    the evaluator for every material point's implicit predicted-row time.
    """
    path = Path(camera_path).resolve(strict=True)
    camera, camera_raw, controls_path, controls_raw, controls = _camera_data(path)
    local_camera, local_controls, t = exposure_controls(camera, controls, pts)
    rows = np.array([0., (camera["image_height"]-1)/2, camera["image_height"]-1.])
    times = t+camera["rig"][37]*(rows/(camera["image_height"]-1)-.5)
    evaluated = core.evaluate_motion_camera(complete_camera_rig(local_camera), local_controls,
        local_camera["first_control_time"], local_camera["control_dt"], times)
    if np.any(evaluated["status"] != 0):
        raise ValueError("native exposure camera evaluation failed: "+str(evaluated["status"].tolist()))
    fields = []
    for row, time, state in zip(rows, times, evaluated["values"]):
        fields.append({"source_row": float(row), "time_from_reference_seconds": float(time),
                      "pan_tilt_logf": state[:3].tolist(), "zoom_z": float(state[3]),
                      "rotation_world_to_camera": state[4:13].reshape(3, 3).tolist(),
                      "centre_world": state[13:16].tolist(),
                      **dict(zip(("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"), state[16:24].tolist()))})
    original = {"camera_path": str(path), "camera_sha256": hashlib.sha256(camera_raw).hexdigest(),
                "controls_path": str(controls_path), "controls_sha256": hashlib.sha256(controls_raw).hexdigest()}
    local_camera["snapshot_origin"].update(original)
    rig = camera["rig"]
    report = {"schema": "rinkspace.camera.exposure.v1", "camera_ref": "camera.json",
              "source_id": camera["source_id"], "shot_id": camera["shot_id"], "pts": int(pts),
              "time_base": {"num": camera["time_base_num"], "den": camera["time_base_den"]},
              "source_seconds": int(pts)*camera["time_base_num"]/camera["time_base_den"],
              "image_domain": {"width": camera["image_width"], "height": camera["image_height"],
                               "coordinates": "original source pixels; u right, v down; no crop or resize"},
              "conventions": {"world": "original d78 L=1 scene gauge; metre scale unverified",
                  "rotation": "world to camera; Rot(tilt,beta) @ R0 @ Rot(pan,alpha)",
                  "centre": "pivot + R.T @ zoom-dependent lever + spacing * unit(pan cross R.T @ tilt)",
                  "row_time": "(PTS-reference_PTS)*time_base + signed_tau*(predicted_v/(height-1)-0.5)",
                  "units": "scene gauge, radians, seconds, original source pixels"},
              "signed_readout_seconds": camera["rig"][37], "row_states": fields,
              "mount_state": {"reference_rotation_world_to_camera": np.asarray(rig[:9]).reshape(3, 3).tolist(),
                  "pan_axis_world": rig[9:12], "tilt_axis_camera": rig[12:15],
                  "pivot_world": rig[15:18], "lever_camera": rig[18:21],
                  "axis_spacing": rig[21], "pupil_linear": rig[22], "pupil_quadratic": rig[23]},
              "lens_state": dict(zip(("f_ref", "radius_max", "cx0", "cy0", "eta", "cx_zoom", "cy_zoom",
                  "k10", "k11", "k20", "k21", "p1", "p2"), rig[24:37])),
              "row_states_are_not_a_single_global_shutter_camera": True,
              "source_state": original, "controls_retained": len(local_controls),
              "unsupported_effective_fields": [], "fit_executed": False, "user_approval": "not_reviewed"}
    out = guard_project_write(out_dir)
    targets = [guard_project_write(out/name) for name in ("camera.json", "controls.npy", "exposure.json")]
    if out.exists():
        raise FileExistsError("exposure snapshot does not overwrite a prior result: "+str(out))
    out.mkdir(parents=True)
    targets[0].write_text(json.dumps(local_camera, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    np.save(targets[1], local_controls, allow_pickle=False)
    targets[2].write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report
