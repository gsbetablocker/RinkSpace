"""Frozen native-PTS window of an existing camera, using its original spline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..domain.runtime_rigs import _camera_data
from ..domain.camera import complete_camera_rig
from ..path_policy import guard_project_write
from .camera_exposure import exposure_controls


def window_controls(camera, controls, pts):
    """Retain original coefficients covering every selected signed readout.

    PTS must come from the source index. Gaps are retained, never filled from
    an assumed frame rate. Historical reference elimination is not replayed.
    """
    identities = np.asarray(pts)
    if (identities.ndim != 1 or not len(identities)
            or identities.dtype.kind not in "iu"
            or any(int(b) <= int(a) for a, b in zip(identities[:-1], identities[1:]))):
        raise ValueError("window requires strictly increasing native integer PTS")
    first, _, _ = exposure_controls(camera, controls, int(identities[0]))
    last, _, _ = exposure_controls(camera, controls, int(identities[-1]))
    start = first["snapshot_origin"]["control_indices"][0]
    stop = last["snapshot_origin"]["control_indices"][1]
    first["stop_pts_exclusive"] = int(identities[-1])+1
    first["snapshot_origin"].update(
        purpose="native_window_frozen_replay_not_fit", control_indices=[start, stop])
    # Subtract integer PTS before conversion, including clocks above 2**53.
    times = np.array([(int(p)-int(camera["reference_pts"]))*camera["time_base_num"]
                      /camera["time_base_den"] for p in identities], dtype=np.float64)
    return first, np.ascontiguousarray(controls[start:stop], dtype="<f8"), times


def snapshot_window(camera_path, pts, core, out_dir):
    """Save a self-contained window camera; evaluate, never optimize or smooth.

    Top/middle/bottom states only check support. Material projections must use
    the saved motion camera's implicit predicted-row forward for every point.
    """
    path = Path(camera_path).resolve(strict=True)
    camera, raw, controls_path, controls_raw, controls = _camera_data(path)
    local, coefficients, times = window_controls(camera, controls, pts)
    rows = np.array([0., (camera["image_height"]-1)/2, camera["image_height"]-1.])
    row_times = times[:, None]+camera["rig"][37]*(rows/(camera["image_height"]-1)-.5)
    evaluated = core.evaluate_motion_camera(complete_camera_rig(local), coefficients,
        local["first_control_time"], local["control_dt"], row_times.ravel())
    if np.any(evaluated["status"] != 0):
        raise ValueError("native window row-state evaluation failed")
    origin = {"camera_path": str(path), "camera_sha256": hashlib.sha256(raw).hexdigest(),
              "controls_path": str(controls_path),
              "controls_sha256": hashlib.sha256(controls_raw).hexdigest()}
    local["snapshot_origin"].update(origin)
    report = {"schema": "rinkspace.camera.window.v1", "camera_ref": "camera.json",
              "source_id": camera["source_id"], "shot_id": camera["shot_id"],
              "native_pts": [int(p) for p in pts], "native_frame_count": len(times),
              "time_base": {"num": camera["time_base_num"], "den": camera["time_base_den"]},
              "image_domain": {"width": camera["image_width"], "height": camera["image_height"],
                               "coordinates": "original source pixels, no crop or resize"},
              "row_time": "(PTS-reference_PTS)*time_base + signed_tau*(predicted_v/(height-1)-0.5)",
              "signed_readout_seconds": camera["rig"][37], "checked_rows": rows.tolist(),
              "checked_row_states": len(row_times.ravel()), "controls_retained": len(coefficients),
              "source_state": origin, "fit_executed": False, "unsupported_effective_fields": [],
              "source_pts_membership": "caller supplies actual decoded/indexed exposures; no FPS synthesis"}
    out = guard_project_write(out_dir)
    targets = [guard_project_write(out/name) for name in ("camera.json", "controls.npy", "window.json")]
    if out.exists():
        raise FileExistsError("window snapshot does not overwrite a prior result: "+str(out))
    out.mkdir(parents=True)
    for target, value in ((targets[0], local), (targets[2], report)):
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    np.save(targets[1], coefficients, allow_pickle=False)
    return report
