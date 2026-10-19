"""Fixed native profile queries on a continuous physical Brown4 world line.

These factors retain the saved query, measured native UV, and noise coordinate.
They never choose a root by its distance to the observation.  Row-timed factors
solve the intersection and its own sensor row together; ordinary factors keep
their existing observed-row evaluator.
"""
from __future__ import annotations

import numpy as np


class NativeLineQueryError(ValueError):
    """A native query has no unique finite forward non-tangent intersection."""


PREDICTION_DOMAIN = dict(
    original_native_UV_and_query_remain_sensor_measurements=True,
    predicted_UV_may_leave_the_recorded_sensor=True,
    predicted_row_law="existing affine geometry_navigation row-time continuation",
    temporal_support="same shot native interpolation and explicit terminal half-interval only",
    off_sensor_predictions_are_not_new_observations=True,
    predicted_coordinates_or_parameters_clamped=False,
)


def _query_arrays(row):
    saved = row.get("native_line_queries")
    if not isinstance(saved, dict):
        raise ValueError("Explicit saved native_line_queries required")
    observed = np.asarray(row["uv"], float)
    query = np.asarray(saved.get("query_uv"), float)
    direction = np.asarray(saved.get("profile_normal"), float)
    ids = saved.get("query_ids")
    if (observed.ndim != 2 or observed.shape[1:] != (2,) or not len(observed)
            or query.shape != observed.shape or direction.shape != observed.shape
            or not all(np.isfinite(x).all() for x in (observed, query, direction))
            or not isinstance(ids, (tuple, list)) or len(ids) != len(observed)
            or len(set(ids)) != len(ids)):
        raise ValueError("Saved native query/UV/identity arrays must retain their exact order")
    if any(np.any(x < 0.) or np.any(x >= [1920., 1080.]) for x in (observed, query)):
        raise ValueError("Original native UV and saved queries must remain inside the recorded sensor")
    norm = np.linalg.norm(direction, axis=1)
    if np.any(norm <= 0.):
        raise ValueError("Saved native profile directions must be nonzero")
    direction = direction / norm[:, None]
    transverse = np.column_stack((direction[:, 1], -direction[:, 0]))
    return observed, query, direction, transverse, tuple(ids)


def _line(world, world_jac, end_columns, start_columns):
    from .first_shot_source_kernel import _line_chart
    world = np.asarray(world, float)
    if (world.ndim != 2 or world.shape[1:] != (3,) or len(world) < 2
            or not np.isfinite(world).all()):
        raise ValueError("A finite ordered physical world line is required")
    derivative = (np.empty(world.shape + (0,)) if world_jac is None
                  else np.asarray(world_jac, float))
    if (derivative.ndim != 3 or derivative.shape[:2] != world.shape
            or not np.isfinite(derivative).all()):
        raise ValueError("Native line world derivatives require the unchanged world chart")
    chart = _line_chart(world, derivative, end_columns, start_columns)
    return world, derivative, chart


def _camera(camera):
    value = np.asarray(camera, float)
    if value.shape != (25,) or not np.isfinite(value).all() or np.any(value[15:] != 0.):
        raise ValueError("Native straight-line queries require the actual Brown4 camera25")
    return value


def _roots(camera, origin, unit, query, transverse):
    """All real Brown polynomial roots, independent of the observed UV."""
    from .first_shot_source_kernel import _line_polynomials, _interfaces
    px, py, start, direction, image_unit = _line_polynomials(camera, origin, unit)
    polynomial = transverse[0] * (px-query[0]) + transverse[1] * (py-query[1])
    coefficients = polynomial.coef
    scale = max(1., np.max(abs(px.coef)), np.max(abs(py.coef)), np.max(abs(query)))
    if np.max(abs(coefficients)) <= 128.*np.finfo(float).eps*scale:
        raise NativeLineQueryError("Native query is coincident with the image line")
    roots = polynomial.roots()
    real = roots.real[abs(roots.imag) <= 8.*np.sqrt(np.finfo(float).eps)*np.maximum(1., abs(roots.real))]
    stations = []
    for image_station in real:
        denominator = image_station*direction[2]-image_unit @ direction[:2]
        arithmetic = 64.*np.finfo(float).eps*max(1., abs(image_station*direction[2]), abs(image_unit @ direction[:2]))
        if abs(denominator) <= arithmetic:
            continue
        station = (image_unit @ start[:2]-image_station*start[2])/denominator
        if np.isfinite(station):
            stations.append(float(station))
    if not stations:
        return []
    stations = np.asarray(stations)
    uv, depth, valid = _interfaces()[0](camera, origin+stations[:, None]*unit, with_jacobian=False)
    return [(float(s), p.copy()) for s, p, good in zip(stations, uv,
        valid & (depth > .05) & np.isfinite(uv).all(axis=1), strict=True) if good]


def _sample(camera, origin, unit, station, origin_jac, unit_jac, *, jacobian):
    from .first_shot_source_kernel import _line_differentials, _interfaces
    project, brown, _, _ = _interfaces()
    points, point_jac, first, _, _ = _line_differentials(camera, origin, unit,
        np.asarray([station]), origin_jac, unit_jac, jacobian=jacobian)
    if jacobian:
        uv, depth, j11, valid = project(camera, points, with_jacobian=True)
    else:
        uv, depth, valid = project(camera, points, with_jacobian=False)
    if not bool(valid[0]) or depth[0] <= .05 or not np.isfinite(uv).all():
        raise NativeLineQueryError("Native query intersection is outside the forward central Brown branch")
    if not jacobian:
        return uv[0], first[0], None, None
    jcam = np.concatenate((np.asarray(j11).reshape(-1, 2, 11), brown(camera, points)), axis=2)[0]
    jworld = (-jcam[:, :3]/100.) @ point_jac[0]
    return uv[0], first[0], jcam, jworld


def _finite_station(station, world, length):
    arithmetic = 128.*np.finfo(float).eps*max(1., length, np.max(abs(world)))
    return -arithmetic <= station <= length+arithmetic


def _unique(candidates, identity):
    if len(candidates) != 1:
        raise NativeLineQueryError(f"Native query {identity} requires one finite forward branch; found {len(candidates)}")
    return candidates[0]


def evaluate_native_line_queries(camera25, row, world, world_jac=None, *, jacobian=True):
    """Return raw 2D intersection-minus-UV and conditioned camera15/world J."""
    if not jacobian and world_jac is not None:
        raise ValueError("Value-only native queries cannot consume geometry derivatives")
    camera = _camera(camera25)
    observed, query, _, transverse, identities = _query_arrays(row)
    world, derivative, chart = _line(world, world_jac,
        row.get("_line_end_extent_geometry_columns", ()), row.get("_line_start_extent_geometry_columns", ()))
    origin, unit, length, _, origin_jac, unit_jac = chart
    values, cameras, geometries = [], [], []
    for uv, source, cross, identity in zip(observed, query, transverse, identities, strict=True):
        candidates = [s for s, _ in _roots(camera, origin, unit, source, cross)
                      if _finite_station(s, world, length)]
        station = _unique(candidates, identity)
        predicted, first, jcam, jworld = _sample(camera, origin, unit, station,
            origin_jac, unit_jac, jacobian=jacobian)
        denominator = float(cross @ first)
        if abs(denominator) <= 128.*np.finfo(float).eps*max(1., np.linalg.norm(first)):
            raise NativeLineQueryError("Native query is tangent to the physical Brown image line: " + identity)
        if abs(cross @ (predicted-source)) > 1e-6:
            raise NativeLineQueryError("Native polynomial root did not reproduce its saved query: " + identity)
        values.append(predicted-uv)
        if jacobian:
            direct = np.c_[jcam, jworld]
            station_jac = -(cross @ direct)/denominator
            total = direct+first[:, None]*station_jac[None]
            cameras.append(total[:, :15]); geometries.append(total[:, 15:])
    n = 2*len(observed)
    return (np.asarray(values).reshape(n),
        np.vstack(cameras) if jacobian else np.empty((n, 0)),
        np.vstack(geometries) if jacobian else np.empty((n, 0)))


def evaluate_row_native_line_queries(trajectory, row, world, world_jac=None, *, jacobian=True):
    """Exact self-row fixed-query factor and its two-variable implicit chain.

    Brown polynomial roots initialize the rolling-row root. Every physical
    central root is retained, and no source residual ranks competing branches.
    An unfitted prediction can leave the recorded sensor. Its row law then uses
    the explicit geometry-navigation continuation, within the same shot's time
    support; original UV/queries remain measured in-sensor source coordinates.
    """
    if not jacobian and world_jac is not None:
        raise ValueError("Value-only native queries cannot consume geometry derivatives")
    observed, query, _, transverse, identities = _query_arrays(row)
    world, derivative, chart = _line(world, world_jac,
        row.get("_line_end_extent_geometry_columns", ()), row.get("_line_start_extent_geometry_columns", ()))
    origin, unit, length, _, origin_jac, unit_jac = chart
    mid = .5*(trajectory.image_height-1.)
    central = _camera(trajectory.camera25_at_row(mid))
    values, cameras, geometries = [], [], []
    columns = np.asarray(trajectory.columns, int)
    for uv, source, cross, identity in zip(observed, query, transverse, identities, strict=True):
        accepted = []; unresolved = []
        for station, seed_uv in _roots(central, origin, unit, source, cross):
            sensor_y = float(seed_uv[1])
            converged = False
            failure = "implicit query root did not converge"
            for _ in range(24):
                from .first_shot_row_camera import RowCameraDomainError
                try:
                    camera, local_columns, chain = trajectory.camera15_mapping(sensor_y,
                        jacobian=True, geometry_navigation=True)
                except RowCameraDomainError:
                    failure = "implicit predicted row leaves the same-shot native/terminal time support"
                    break
                if not np.array_equal(local_columns, columns):
                    raise ValueError("Native self-row camera changed its complete neighbouring column inventory")
                camera = _camera(camera)
                try:
                    predicted, first, jcam, jworld = _sample(camera, origin, unit, station,
                        origin_jac, unit_jac, jacobian=True)
                except NativeLineQueryError:
                    failure = "implicit query root left the forward central Brown branch"
                    break
                velocity = jcam @ trajectory.camera15_row_derivative(sensor_y, geometry_navigation=True)
                system = np.array([[cross @ first, cross @ velocity], [first[1], velocity[1]-1.]])
                singular = np.linalg.svd(system, compute_uv=False)
                if singular[-1] <= 128.*np.finfo(float).eps*max(1., singular[0]):
                    raise NativeLineQueryError("Native self-row query has a tangent or singular row intersection: " + identity)
                equation = np.array([cross @ (predicted-source), predicted[1]-sensor_y])
                if np.max(abs(equation)) <= 1e-8:
                    converged = True
                    break
                step = np.linalg.solve(system, -equation)
                next_y = sensor_y+step[1]
                if not np.isfinite(step).all():
                    failure = "implicit query root has no finite predicted row/station step"
                    break
                station += step[0]; sensor_y = float(next_y)
            if not converged:
                unresolved.append(failure)
                continue
            if not _finite_station(station, world, length):
                continue
            # Distinct polynomial seeds can converge to the same implicit root.
            # Merge only at verified floating-point root precision.
            if any(abs(station-a[0]) <= 256.*np.finfo(float).eps*max(1., length)
                   and abs(sensor_y-a[1]) <= 1e-8 for a in accepted):
                continue
            accepted.append((station, sensor_y, predicted, first, velocity, system, jcam, jworld, chain))
        if unresolved:
            raise NativeLineQueryError("Native query " + identity + ": " + "; ".join(unresolved))
        solved = _unique(accepted, identity)
        _, _, predicted, first, velocity, system, jcam, jworld, chain = solved
        values.append(predicted-uv)
        if jacobian:
            direct = np.c_[jworld, jcam @ chain]
            implicit = -np.linalg.solve(system, np.vstack((cross @ direct, direct[1])))
            total = direct+first[:, None]*implicit[0]+velocity[:, None]*implicit[1]
            geometries.append(total[:, :derivative.shape[2]])
            cameras.append(total[:, derivative.shape[2]:])
    n = 2*len(observed)
    return (np.asarray(values).reshape(n),
        np.vstack(cameras) if jacobian else np.empty((n, 0)),
        np.vstack(geometries) if jacobian else np.empty((n, 0)), columns)
