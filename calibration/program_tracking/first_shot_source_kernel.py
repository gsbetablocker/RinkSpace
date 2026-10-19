"""Batch the existing finite projected-target objective without source weights.

Targets already carry their caller's physical association policy. This module
does not admit observations, change their identity, choose model parameters,
apply sigma/whitening, or optimize. Every target is projected once per call;
each ordinary row retains its original order and nearest-point query. Explicit
native_line_queries rows instead intersect the continuous world line with the
saved source profile, without choosing the intersection by source residual.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _ProjectedTarget:
    uv: np.ndarray
    depth: np.ndarray
    valid: np.ndarray
    camera_jac: np.ndarray | None
    geometry_jac: np.ndarray | None
    segment_valid: np.ndarray | None


def _interfaces():
    # Use the *same* loaded projection, Brown and nearest-segment functions as
    # the original evaluator. Runtime backend restoration remains caller-owned.
    from .framewise_model import _absolute_interface
    namespace = _absolute_interface().__globals__
    return tuple(namespace[name] for name in (
        "project_central_valid", "brown4_world_jac", "nearest_segments",
        "BoardIntersectionError"))


def _checked_row(row):
    if not isinstance(row, Mapping) or "_target_key" not in row:
        raise ValueError("Each original source row requires an explicit _target_key")
    try:
        hash(row["_target_key"])
    except TypeError as error:
        raise ValueError("Target keys must be hashable identities") from error
    point = row.get("point_correspondence")
    if not isinstance(point, (bool, np.bool_)):
        raise ValueError("Each source row requires explicit point_correspondence")
    if "normal_interval" in row:
        raise ValueError("Source normal_interval is not implemented by this raw target kernel")
    if row.get("curve_observation_kind") is not None or row.get("must_preserve_upper_branch"):
        raise ValueError("Special curve observation semantics require their original evaluator")
    observed = np.asarray(row.get("uv"), dtype=np.float64)
    if (observed.ndim != 2 or observed.shape[1:] != (2,) or not len(observed)
            or not np.isfinite(observed).all()):
        raise ValueError("Original source UV must be a nonempty finite N-by-2 array")
    if "point_support_bounds_xy_px" in row and not point:
        raise ValueError("XY support bounds require an explicit point correspondence")
    return bool(point), observed


def _project_target(camera, world, world_jac, point, interfaces, jacobian):
    project, brown, _, _ = interfaces
    count = len(world)
    # Midpoints retain the original finite-curve optical-domain guard. Their
    # presence in this call does not turn them into observations or model knots.
    vertices = world if point else np.vstack((world, .5 * (world[:-1] + world[1:])))
    if jacobian:
        uv, depth, j11, valid = project(camera, vertices, with_jacobian=True)
        camera_jac = np.concatenate((np.asarray(j11).reshape(-1, 2, 11)[:count],
                                     brown(camera, world)), axis=2)
        geometry_jac = np.einsum("nij,njk->nik", -camera_jac[:, :, :3] / 100., world_jac)
    else:
        uv, depth, valid = project(camera, vertices, with_jacobian=False)
        camera_jac = geometry_jac = None
    segment_valid = None
    if not point:
        direction = np.diff(uv[:count], axis=0)
        length2 = np.sum(direction * direction, axis=1)
        segment_valid = (valid[:count - 1] & valid[1:count] & valid[count:]
            & (depth[:count - 1] > 0.) & (depth[1:count] > 0.)
            & np.isfinite(uv[:count - 1]).all(axis=1)
            & np.isfinite(uv[1:count]).all(axis=1) & (length2 > 1e-4))
    return _ProjectedTarget(uv[:count], depth[:count], valid[:count],
                            camera_jac, geometry_jac, segment_valid)


def _nearest_jacobian(observed, projected, vertex_jac, index, selected_t, clipped_t):
    start = projected[index]
    direction = projected[index + 1] - start
    offset = observed - start
    length2 = np.sum(direction * direction, axis=1)
    ja = vertex_jac[index]
    jd = vertex_jac[index + 1] - ja
    numerator = (-np.einsum("nij,ni->nj", ja, direction)
                 + np.einsum("ni,nij->nj", offset, jd))
    denominator = 2. * np.einsum("ni,nij->nj", direction, jd)
    dt = (numerator - selected_t[:, None] * denominator) / length2[:, None]
    dt[(selected_t <= 0.) | (selected_t >= 1.)] = 0.
    return ja + clipped_t[:, None, None] * jd + direction[:, :, None] * dt[:, None, :]


def _line_chart(world, world_jac, extent_columns, start_extent_columns=()):
    """An explicit physical straight line, with terminal-station derivatives."""
    origin, vector = world[0], world[-1] - world[0]
    length = np.linalg.norm(vector)
    if length <= 0.:
        raise ValueError("A continuous world line requires distinct finite endpoints")
    unit = vector / length
    stations = (world - origin) @ unit
    eps = np.finfo(float).eps
    bound = 64. * eps * max(1., np.max(abs(world)), length)
    if (np.max(abs(world - origin - stations[:, None] * unit)) > bound
            or np.any(np.diff(stations) <= 0.)):
        raise ValueError("Explicit continuous world line vertices must be collinear and ordered")
    origin_jac = world_jac[0].copy()
    delta_jac = world_jac[-1] - origin_jac
    length_jac = unit @ delta_jac
    unit_jac = (delta_jac - unit[:, None] * length_jac) / length
    for explicit, endpoint_index in ((extent_columns, -1), (start_extent_columns, 0)):
        columns = np.asarray(explicit)
        if not columns.size or not world_jac.shape[2]:
            continue
        if (columns.ndim != 1 or columns.dtype.kind not in "iu"
                or len(np.unique(columns)) != len(columns)
                or np.any(columns < 0) or np.any(columns >= world_jac.shape[2])):
            raise ValueError("Explicit terminal extent columns must address the supplied geometry Jacobian")
        for column in columns:
            derivative = world_jac[:, :, column]
            endpoint = derivative[endpoint_index]
            fraction = stations / length
            if endpoint_index == 0:
                fraction = 1. - fraction
            arithmetic = 128. * eps * max(1., np.max(abs(derivative)))
            expected = fraction[:, None] * endpoint
            if (np.any(derivative[0 if endpoint_index == -1 else -1] != 0.)
                    or np.max(abs(endpoint - unit * (unit @ endpoint))) > arithmetic
                    or np.max(abs(derivative - expected)) > arithmetic):
                raise ValueError("Declared terminal extent derivative must be an exact straight-line station change")
            # This is a declared physical chart identity, not a small-J test.
            # Floating point normalization must not reopen an unobserved mode.
            origin_jac[:, column] = 0.
            unit_jac[:, column] = 0.
    return origin, unit, length, stations, origin_jac, unit_jac


def _jet_add(a, b):
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _jet_mul(a, b):
    av, ass, ass2, aq, asq = a
    bv, bs, bss, bq, bsq = b
    return (av * bv, ass * bv + av * bs,
            ass2 * bv + 2. * ass * bs + av * bss,
            aq * bv[:, None] + av[:, None] * bq,
            asq * bv[:, None] + ass[:, None] * bq
            + aq * bs[:, None] + av[:, None] * bsq)


def _line_differentials(camera, origin, unit, station, origin_jac, unit_jac, *, jacobian):
    """Brown4 tangent, curvature and mixed derivatives in intrinsic world station.

    Values and the direct camera Jacobian still come from the original native
    projector. These polynomial derivatives supply only the continuous nearest
    station's implicit chain, including residual-times-curvature terms.
    """
    import cv2
    rotation, drot = cv2.Rodrigues(camera[3:6])
    points = origin + station[:, None] * unit
    camera_points = (points - camera[:3]) @ rotation.T
    tangent = rotation @ unit
    count = len(station); ng = origin_jac.shape[1]
    nq = 15 + ng if jacobian else 0
    point_jac = origin_jac[None] + station[:, None, None] * unit_jac[None]
    xyz_q = np.zeros((count, 3, nq)); xyz_sq = np.zeros_like(xyz_q)
    if jacobian:
        xyz_q[:, :, :3] = -100. * rotation
        dr = drot.reshape(3, 3, 3)
        xyz_q[:, :, 3:6] = np.einsum("kij,nj->nik", dr, points - camera[:3])
        xyz_sq[:, :, 3:6] = np.einsum("kij,j->ik", dr, unit)[None]
        xyz_q[:, :, 15:] = np.einsum("ij,njk->nik", rotation, point_jac)
        xyz_sq[:, :, 15:] = (rotation @ unit_jac)[None]
    z = camera_points[:, 2]
    jets = []
    for coordinate in (0, 1):
        value = camera_points[:, coordinate] / z
        first = (tangent[coordinate] - value * tangent[2]) / z
        second = -2. * first * tangent[2] / z
        direct = (xyz_q[:, coordinate] - value[:, None] * xyz_q[:, 2]) / z[:, None]
        mixed = (xyz_sq[:, coordinate] - first[:, None] * xyz_q[:, 2]
                 - direct * tangent[2] - value[:, None] * xyz_sq[:, 2]) / z[:, None]
        jets.append((value, first, second, direct, mixed))
    def constant(value, column=None, multiplier=1.):
        derivative = np.zeros((count, nq))
        if jacobian and column is not None:
            derivative[:, column] = multiplier
        return (np.full(count, value), np.zeros(count), np.zeros(count),
                derivative, np.zeros_like(derivative))
    x, y = jets
    x2, y2, xy = _jet_mul(x, x), _jet_mul(y, y), _jet_mul(x, y)
    radius2 = _jet_add(x2, y2)
    radial = _jet_add(constant(1.), _jet_add(
        _jet_mul(constant(camera[11], 11), radius2),
        _jet_mul(constant(camera[12], 12), _jet_mul(radius2, radius2))))
    p1, p2 = constant(camera[13], 13), constant(camera[14], 14)
    xd = _jet_add(_jet_mul(x, radial), _jet_add(
        _jet_mul(_jet_mul(constant(2.), p1), xy),
        _jet_mul(p2, _jet_add(_jet_mul(constant(3.), x2), y2))))
    yd = _jet_add(_jet_mul(y, radial), _jet_add(
        _jet_mul(p1, _jet_add(x2, _jet_mul(constant(3.), y2))),
        _jet_mul(_jet_mul(constant(2.), p2), xy)))
    fx, fy = np.exp(camera[6:8])
    px = _jet_add(_jet_add(_jet_mul(constant(fx, 6, fx), xd),
                          _jet_mul(constant(camera[10], 10, 1000.), yd)),
                  constant(camera[8], 8, 1000.))
    py = _jet_add(_jet_mul(constant(fy, 7, fy), yd), constant(camera[9], 9, 1000.))
    return (points, point_jac, np.column_stack((px[1], py[1])),
            np.column_stack((px[2], py[2])), np.stack((px[4], py[4]), axis=1))


def _line_polynomials(camera, origin, unit):
    """Pixel Brown4 polynomials on a unit undistorted image line."""
    import cv2
    from numpy.polynomial import Polynomial as P
    rotation = cv2.Rodrigues(camera[3:6])[0]
    start, tangent = rotation @ (origin - camera[:3]), rotation @ unit
    normal = np.cross(start, tangent)
    norm = np.linalg.norm(normal[:2])
    if norm == 0.:
        raise ValueError("A world line through the camera has no finite image curve")
    image_origin = -normal[2] * normal[:2] / norm ** 2
    image_unit = np.array([-normal[1], normal[0]]) / norm
    x, y = P([image_origin[0], image_unit[0]]), P([image_origin[1], image_unit[1]])
    radius2 = x*x + y*y
    radial = 1. + camera[11]*radius2 + camera[12]*radius2*radius2
    xd = x*radial + 2.*camera[13]*x*y + camera[14]*(3.*x*x+y*y)
    yd = y*radial + camera[13]*(x*x+3.*y*y) + 2.*camera[14]*x*y
    px = np.exp(camera[6])*xd + camera[10]*yd + camera[8]
    py = np.exp(camera[7])*yd + camera[9]
    return px, py, start, tangent, image_unit


def _evaluate_continuous_line(camera, observed, world, world_jac, target, extent_columns, start_extent_columns,
                              interfaces, jacobian):
    if np.any(camera[15:] != 0.):
        raise ValueError("Explicit continuous world line currently requires Brown4 with fixed higher terms")
    project, brown, _, domain_error = interfaces
    origin, unit, length, stations, origin_jac, unit_jac = _line_chart(world, world_jac, extent_columns, start_extent_columns)
    if not target.segment_valid.any():
        raise domain_error("ice_curve_no_central_segment", np.arange(len(observed)))
    px, py, start, tangent, image_unit = _line_polynomials(camera, origin, unit)
    # Keep exactly the existing vertices/midpoints segment validity. Candidate
    # feet additionally pass the same central-domain projector themselves.
    valid = target.segment_valid
    boundaries = np.flatnonzero(np.r_[valid[0], valid[1:] != valid[:-1], valid[-1]])
    runs = list(zip(boundaries[::2], boundaries[1::2], strict=True))
    e_stations = ((start[:2] + stations[:, None]*tangent[:2]) @ image_unit
                  / (start[2] + stations*tangent[2]))
    candidates, owners, boundary_indices = [], [], []
    for i, uv in enumerate(observed):
        polynomial = (px-uv[0])*px.deriv() + (py-uv[1])*py.deriv()
        roots = polynomial.roots()
        real = roots.real[abs(roots.imag) <= 8.*np.sqrt(np.finfo(float).eps)*np.maximum(1., abs(roots.real))]
        for e in real:
            denominator = e*tangent[2] - image_unit @ tangent[:2]
            if denominator == 0.:
                continue
            station = (image_unit @ start[:2] - e*start[2]) / denominator
            if any(stations[a] < station < stations[b] for a, b in runs):
                candidates.append(station); owners.append(i); boundary_indices.append(-1)
        for a, b in runs:
            for boundary in (a, b):
                candidates.append(stations[boundary]); owners.append(i); boundary_indices.append(boundary)
    candidate_stations = np.asarray(candidates)
    points = origin + candidate_stations[:, None] * unit
    uv, depth, central = project(camera, points, with_jacobian=False)
    owner = np.asarray(owners); boundary_index = np.asarray(boundary_indices)
    cost = np.sum((uv-observed[owner])**2, axis=1)
    cost[~central | (depth <= .05) | ~np.isfinite(cost)] = np.inf
    chosen = []
    for i in range(len(observed)):
        candidates_i = np.flatnonzero(owner == i)
        index = candidates_i[np.argmin(cost[candidates_i])]
        if not np.isfinite(cost[index]):
            raise domain_error("continuous_ice_line_no_central_foot", np.array([i]))
        chosen.append(index)
    chosen = np.asarray(chosen)
    station = candidate_stations[chosen]; boundary = boundary_index[chosen]
    points, point_jac, first, second, mixed = _line_differentials(
        camera, origin, unit, station, origin_jac, unit_jac, jacobian=jacobian)
    if jacobian:
        uv, depth, j11, central = project(camera, points, with_jacobian=True)
    else:
        uv, depth, central = project(camera, points, with_jacobian=False)
    if not np.all(central & (depth > .05)):
        raise domain_error("continuous_ice_line_foot_not_central", np.flatnonzero(~central | (depth <= .05)))
    residual = uv-observed
    if not jacobian:
        return residual.reshape(-1), np.empty((2*len(observed), 0)), np.empty((2*len(observed), 0))
    camera_jac = np.concatenate((np.asarray(j11).reshape(-1, 2, 11), brown(camera, points)), axis=2)
    # A finite boundary is a material endpoint, not a sliding interior foot.
    clamped = boundary >= 0
    point_jac[clamped] = world_jac[boundary[clamped]]
    geometry_jac = np.einsum("nij,njk->nik", -camera_jac[:, :, :3]/100., point_jac)
    direct = np.concatenate((camera_jac, geometry_jac), axis=2)
    hessian = np.sum(first*first + residual*second, axis=1)
    interior = ~clamped
    if np.any(interior & (hessian <= 0.)):
        raise domain_error("continuous_ice_line_nearest_not_nondegenerate", np.flatnonzero(interior & (hessian <= 0.)))
    station_jac = np.zeros((len(observed), direct.shape[2]))
    station_jac[interior] = -(np.einsum("ni,nij->nj", first, direct)
        + np.einsum("ni,nij->nj", residual, mixed))[interior] / hessian[interior, None]
    derivative = direct + first[:, :, None]*station_jac[:, None, :]
    return residual.reshape(-1), derivative[:, :, :15].reshape(-1, 15), derivative[:, :, 15:].reshape(2*len(observed), world_jac.shape[2])


def evaluate_projected_targets(camera25, rows, world_by_target, world_jac_by_target=None, *, jacobian=True):
    """Return raw ``(r, camera15_J, geometry_J)`` in original source-row order.

    Camera columns are C/100, Rodrigues3, logfx/logfy, cx/cy/skew divided
    by 1000, and Brown k1/k2/p1/p2. Geometry input is one ``(N,3,ng)``
    derivative array per target in scene units. The nearest segment is locally
    fixed, while its projection parameter is differentiated on the interior;
    clamped endpoints have dt=0. Output geometry columns are empty when absent.
    ``jacobian=False`` retains the same raw value/domain algorithm and returns
    empty derivative columns, skipping Brown and nearest-point derivatives.
    Geometry derivatives must be absent in this mode. Explicit rows marked
    ``_continuous_world_line=True`` use the continuous Brown4 image of a finite
    straight world line; all original knots retain their optical-domain guards.
    ``_line_end_extent_geometry_columns`` / ``_line_start_extent_geometry_columns``
    declare exact terminal-station
    coordinates, retaining real endpoint derivatives while interior feet carry
    zero information about their extent. These are caller-proven physical
    semantics, never inferred by checking a small Jacobian. Input UV, world paths,
    source weights and derivative arrays are never edited.
    Rows carrying ``native_line_queries`` use the explicitly saved native
    profile intersection and require exactly one finite forward Brown branch.
    """
    camera = np.asarray(camera25, dtype=np.float64)
    if camera.shape != (25,) or not np.isfinite(camera).all():
        raise ValueError("A finite Brown camera25 is required")
    if not isinstance(world_by_target, Mapping):
        raise ValueError("An explicit target-key-to-world mapping is required")
    if world_jac_by_target is not None and not isinstance(world_jac_by_target, Mapping):
        raise ValueError("Geometry derivatives require an explicit target mapping")
    if not isinstance(jacobian, (bool, np.bool_)):
        raise ValueError("An explicit Boolean jacobian mode is required")
    if not jacobian and world_jac_by_target is not None:
        raise ValueError("Value-only evaluation cannot consume geometry derivatives")
    originals = list(rows)
    checked = [_checked_row(row) for row in originals]
    targets, geometry_count = {}, None
    for row, (point, observed) in zip(originals, checked, strict=True):
        key = row["_target_key"]
        continuous = row.get("_continuous_world_line", False)
        if "native_line_queries" in row and (not continuous or point):
            raise ValueError("Saved native profile queries require an explicit continuous world line")
        if not isinstance(continuous, (bool, np.bool_)) or (continuous and point):
            raise ValueError("Continuous world line requires an explicit curve correspondence")
        extent_columns = tuple(row.get("_line_end_extent_geometry_columns", ()))
        start_extent_columns = tuple(row.get("_line_start_extent_geometry_columns", ()))
        if (extent_columns or start_extent_columns) and not continuous:
            raise ValueError("Terminal station semantics require an explicit continuous world line")
        if key in targets:
            if targets[key][2] != point:
                raise ValueError("One target cannot mix point and curve correspondences")
            if targets[key][3:] != (continuous, extent_columns, start_extent_columns):
                raise ValueError("One physical target cannot mix continuous-line semantics")
            if point and len(targets[key][0]) != len(observed):
                raise ValueError("Point target and source correspondence inventories differ")
            continue
        if key not in world_by_target:
            raise ValueError("A source row has no supplied physical world target: " + str(key))
        world = np.asarray(world_by_target[key], dtype=np.float64)
        if (world.ndim != 2 or world.shape[1:] != (3,) or len(world) < (1 if point else 2)
                or not np.isfinite(world).all() or (point and len(world) != len(observed))):
            raise ValueError("Invalid finite physical world target: " + str(key))
        if world_jac_by_target is None:
            derivative = np.empty((len(world), 3, 0), dtype=np.float64)
        else:
            if key not in world_jac_by_target:
                raise ValueError("A physical target has no supplied geometry derivatives: " + str(key))
            derivative = np.asarray(world_jac_by_target[key], dtype=np.float64)
            if (derivative.ndim != 3 or derivative.shape[:2] != world.shape
                    or not np.isfinite(derivative).all()):
                raise ValueError("Geometry target derivatives must be finite N-by-3-by-ng arrays")
        if geometry_count is None:
            geometry_count = derivative.shape[2]
        elif derivative.shape[2] != geometry_count:
            raise ValueError("Every target must use the same geometry column inventory")
        targets[key] = world, derivative, point, continuous, extent_columns, start_extent_columns
    geometry_count = 0 if geometry_count is None else geometry_count
    if not originals:
        return np.empty(0), np.empty((0, 15 if jacobian else 0)), np.empty((0, geometry_count))
    interfaces = _interfaces()
    projected = {key: _project_target(camera, world, derivative, point, interfaces, jacobian)
                 for key, (world, derivative, point, _, _, _) in targets.items()}
    continuous_results, continuous_offsets = {}, {}
    for key, (world, derivative, _, continuous, extent_columns, start_extent_columns) in targets.items():
        if continuous:
            selected = [uv for row, (_, uv) in zip(originals, checked, strict=True)
                        if row["_target_key"] == key and "native_line_queries" not in row]
            if not selected:
                continue
            observed = np.concatenate(selected)
            continuous_results[key] = _evaluate_continuous_line(camera, observed, world, derivative,
                projected[key], extent_columns, start_extent_columns, interfaces, jacobian)
            continuous_offsets[key] = 0
    nearest, domain_error = interfaces[2:]
    values, cameras, geometries = [], [], []
    for row, (point, observed) in zip(originals, checked, strict=True):
        key = row["_target_key"]
        if "native_line_queries" in row:
            from .first_shot_native_line_kernel import evaluate_native_line_queries
            world, derivative, _, _, _, _ = targets[key]
            raw, jcam, jgeo = evaluate_native_line_queries(camera, row, world,
                derivative if jacobian else None, jacobian=jacobian)
            values.append(raw); cameras.append(jcam); geometries.append(jgeo)
            continue
        if key in continuous_results:
            begin = continuous_offsets[key]; end = begin + 2*len(observed)
            raw, jcam, jgeo = continuous_results[key]
            values.append(raw[begin:end]); cameras.append(jcam[begin:end]); geometries.append(jgeo[begin:end])
            continuous_offsets[key] = end
            continue
        target = projected[row["_target_key"]]
        if point:
            if not np.all(target.valid & (target.depth > .05)):
                raise domain_error("z190_actual_ad_corner_not_central_forward",
                    np.flatnonzero(~target.valid | (target.depth <= .05)))
            residual = target.uv - observed
            camera_jac, geometry_jac = target.camera_jac, target.geometry_jac
            if "point_support_bounds_xy_px" in row:
                bounds = np.asarray(row["point_support_bounds_xy_px"], dtype=np.float64)
                if (bounds.shape != (len(observed), 2, 2) or not np.isfinite(bounds).all()
                        or np.any(bounds[:, :, 0] > bounds[:, :, 1])):
                    raise ValueError("Each point requires finite ordered native XY support intervals")
                low, high = bounds[:, :, 0], bounds[:, :, 1]
                active = (target.uv < low) | (target.uv > high)
                residual = np.minimum(target.uv - low, 0.) + np.maximum(target.uv - high, 0.)
                if jacobian:
                    camera_jac = camera_jac * active[:, :, None]
                    geometry_jac = geometry_jac * active[:, :, None]
        else:
            if not target.segment_valid.any():
                raise domain_error("ice_curve_no_central_segment", np.arange(len(observed)))
            start = target.uv[:-1]
            direction = np.diff(target.uv, axis=0)
            length2 = np.sum(direction * direction, axis=1)
            index, t, clipped, closest = nearest(observed, start, direction, length2,
                                                 target.segment_valid)
            residual = closest - observed
            # One derivative formula covers both camera and world geometry,
            # including the actual sliding nearest point and finite endpoints.
            if jacobian:
                joint = np.concatenate((target.camera_jac, target.geometry_jac), axis=2)
                derivative = _nearest_jacobian(observed, target.uv, joint, index, t, clipped)
                camera_jac, geometry_jac = derivative[:, :, :15], derivative[:, :, 15:]
        values.append(residual.reshape(-1))
        if jacobian:
            cameras.append(camera_jac.reshape(2 * len(observed), 15))
            geometries.append(geometry_jac.reshape(2 * len(observed), geometry_count))
        else:
            cameras.append(np.empty((2 * len(observed), 0)))
            geometries.append(np.empty((2 * len(observed), 0)))
    return np.concatenate(values), np.vstack(cameras), np.vstack(geometries)
