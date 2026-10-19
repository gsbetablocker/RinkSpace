"""Native observed-row factors on the same continuous camera used by review.

The observed y coordinate fixes its exposure time. The original finite
material target, raw UV, support interval and full correlated W are retained.
Neighbouring RF4 columns describe camera motion during readout; they are not
temporal penalties or extra observations. Render solves the implicit row of
this same trajectory, so factor and render have the same physical zero set.
Explicit native_line_queries factors additionally solve their own exposure row
and saved profile intersection together; the ordinary factors stay unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from .first_shot_source_kernel import evaluate_projected_targets


def _observations(frame, worlds, derivatives):
    for row in frame["rows"]:
        key = row["_target_key"]
        observed = np.asarray(row["uv"], float)
        world = worlds[key]
        derivative = None if derivatives is None else derivatives[key]
        point = bool(row["point_correspondence"])
        if point and len(world) != len(observed):
            raise ValueError("Point target and native observations must retain their exact order")
        for index in range(len(observed)):
            sample = dict(row, uv=observed[index:index+1])
            if "native_line_queries" in row:
                saved = row["native_line_queries"]
                sample["native_line_queries"] = dict(saved,
                    query_uv=np.asarray(saved["query_uv"])[index:index+1],
                    profile_normal=np.asarray(saved["profile_normal"])[index:index+1],
                    query_ids=list(saved["query_ids"])[index:index+1])
            if "point_support_bounds_xy_px" in row:
                sample["point_support_bounds_xy_px"] = np.asarray(row["point_support_bounds_xy_px"])[index:index+1]
            material = world[index:index+1] if point else world
            dmaterial = (None if derivative is None else
                         derivative[index:index+1] if point else derivative)
            yield sample, {key: material}, None if dmaterial is None else {key: dmaterial}


def _raw_and_local(state, q, frame, worlds, derivatives=None):
    from .first_shot_row_camera import prepare_frame_row_camera
    trajectory = prepare_frame_row_camera(state, q, frame["index"])
    jacobian = derivatives is not None
    values, jacobians = [], []
    columns = None
    cameras = {}
    for sample, material, dmaterial in _observations(frame, worlds, derivatives):
        if "native_line_queries" in sample:
            from .first_shot_native_line_kernel import evaluate_row_native_line_queries
            key = sample["_target_key"]
            raw, local_camera, jworld, local_columns = evaluate_row_native_line_queries(
                trajectory, sample, material[key],
                None if dmaterial is None else dmaterial[key], jacobian=jacobian)
            if columns is None:
                columns = local_columns
            elif not np.array_equal(columns, local_columns):
                raise ValueError("One exposure requires a fixed complete neighbouring camera column inventory")
            values.append(raw)
            if jacobian:
                jacobians.append(np.c_[jworld, local_camera])
            continue
        source_y = float(sample["uv"][0, 1])
        if source_y not in cameras:
            cameras[source_y] = trajectory.camera15_mapping(source_y, jacobian=jacobian)
        camera, local_columns, chain = cameras[source_y]
        local_columns = np.asarray(local_columns, int)
        if columns is None:
            columns = local_columns
        elif not np.array_equal(columns, local_columns):
            raise ValueError("One exposure requires a fixed complete neighbouring camera column inventory")
        try:
            raw, jcam, jworld = evaluate_projected_targets(camera, [sample], material, dmaterial,
                                                         jacobian=jacobian)
        except Exception as error:
            error.add_note(f"Native row camera: frame {frame['index']}, source y {source_y}, target {sample['_target_key']}")
            raise
        values.append(raw)
        if jacobian:
            jacobians.append(np.c_[jworld, jcam @ chain])
    if len(values) != frame["source_point_count"]:
        raise ValueError("Row timing changed the native source point inventory")
    raw = np.concatenate(values)
    return raw, np.vstack(jacobians) if jacobian else None, columns


def row_timed_raw(state, q, frame, worlds):
    return _raw_and_local(state, q, frame, worlds)[0]


def evaluate_row_timed_frame(state, q, frame, whitening, worlds, derivatives):
    raw, local, camera_columns = _raw_and_local(state, q, frame, worlds, derivatives)
    residual = whitening.apply(raw)
    if derivatives is None:
        return residual, None
    columns = np.r_[state.world_parameter_indices, camera_columns]
    if len(np.unique(columns)) != len(columns):
        raise ValueError("Row-time and material Jacobian columns overlap")
    whitened = csr_matrix(whitening.apply(local))
    full = csr_matrix((whitened.data, columns[whitened.indices].astype(np.int32), whitened.indptr),
                      shape=(len(residual), state.count))
    full.sort_indices()
    return residual, full
