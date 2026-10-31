"""Shared image-shape guidance from finite polylines projected by native.

All paths use one affine registration. This local source-matching diagnostic
does not replace native lens/rolling-shutter geometry or change its parameters.
Nearest curve positions are diagnostics, never source observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np


@dataclass(frozen=True)
class PolylineMatch:
    closest_uv: np.ndarray
    residual_uv: np.ndarray
    distance_px: np.ndarray
    normal_residual_px: np.ndarray
    segment_indices: np.ndarray
    segment_fractions: np.ndarray


@dataclass(frozen=True)
class NativeShapeRegistration:
    observed_uv: np.ndarray
    native_curve_names: np.ndarray
    affine: np.ndarray
    matches: PolylineMatch
    rms_distance_px: float
    initial_rms_distance_px: float
    iterations: int
    stop_reason: str


def _points(value, name, minimum=0):
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.dtype.kind not in "iuf" or len(raw) < minimum:
        raise ValueError(f"{name} must be a numeric (N, 2) array with N >= {minimum}")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite coordinates")
    with np.errstate(over="ignore", invalid="ignore"):
        if len(result) and not np.isfinite(np.ptp(result, axis=0)).all():
            raise ValueError(f"{name} has unrepresentable coordinate spans")
    return result


def point_to_polyline(observed_uv, polyline_uv) -> PolylineMatch:
    """Match original UV to finite line segments, preserving endpoint limits.

    The signed normal points left of the ordered segment tangent. Distance
    includes a finite endpoint's tangential residual as well: a zero normal
    residual alone does not mean that a point lies on the finite path.
    Equal-distance ties retain the first original segment. Duplicate vertices
    are allowed; an entirely zero-length polyline is rejected. Distance work
    uses bounded point/segment blocks, never a full N-by-segment matrix.
    """
    points = _points(observed_uv, "observed_uv")
    line = _points(polyline_uv, "polyline_uv", minimum=2)
    best = np.full(len(points), np.inf)
    closest = np.empty_like(points)
    normal_error = np.empty(len(points))
    segment = np.full(len(points), -1, dtype=np.int64)
    fraction = np.empty(len(points))
    all_directions = np.diff(line, axis=0)
    all_lengths = np.hypot(all_directions[:, 0], all_directions[:, 1])
    if not np.isfinite(all_lengths).all():
        raise ValueError("polyline_uv has a nonfinite segment length")
    valid_segments = np.flatnonzero(all_lengths > 0.)
    if not len(valid_segments):
        raise ValueError("polyline_uv is degenerate: no finite nonzero segment")
    # These block sizes only bound scratch memory, not samples, iterations or
    # source support. Every original point and every finite segment is used.
    for point_begin in range(0, len(points), 512):
        point_rows = np.arange(point_begin, min(point_begin+512, len(points)))
        block = points[point_rows]
        for segment_begin in range(0, len(valid_segments), 64):
            indices = valid_segments[segment_begin:segment_begin+64]
            starts = line[indices]; directions = all_directions[indices]
            lengths = all_lengths[indices]; tangents = directions/lengths[:, None]
            offsets = block[:, None, :]-starts[None, :, :]
            t = np.clip(np.einsum("bsi,si->bs", offsets, tangents)/lengths, 0., 1.)
            targets = starts[None, :, :]+t[:, :, None]*directions[None, :, :]
            errors = block[:, None, :]-targets
            distances = np.hypot(errors[:, :, 0], errors[:, :, 1])
            local = distances.argmin(axis=1)
            rows = np.arange(len(block))
            distance = distances[rows, local]
            better = distance < best[point_rows]
            output = point_rows[better]; local_segments = local[better]
            chosen_errors = errors[rows[better], local_segments]
            chosen_tangents = tangents[local_segments]
            best[output] = distance[better]
            closest[output] = targets[rows[better], local_segments]
            normal_error[output] = (-chosen_errors[:, 0]*chosen_tangents[:, 1]+
                                    chosen_errors[:, 1]*chosen_tangents[:, 0])
            segment[output] = indices[local_segments]
            fraction[output] = t[rows[better], local_segments]
    if not np.isfinite(best).all():
        raise ValueError("point-to-polyline distances are nonfinite")
    return PolylineMatch(closest, points-closest, best, normal_error, segment, fraction)


def point_to_partial_polyline(observed_uv, polyline_uv) -> PolylineMatch:
    """Search finite contiguous pieces without bridging failed projections.

    Returned segment indices still address the original polyline. Invalid
    distant vertices therefore do not remove valid local material support.
    This is a search diagnostic; it does not make failed projections valid.
    """
    points = _points(observed_uv, "observed_uv")
    line = np.asarray(polyline_uv, dtype=np.float64)
    if line.ndim != 2 or line.shape[1] != 2 or len(line) < 2:
        raise ValueError("polyline_uv must have shape (N, 2), N >= 2")
    finite = np.isfinite(line).all(axis=1)
    segments = np.flatnonzero(finite[:-1] & finite[1:])
    if not len(segments):
        raise ValueError("polyline_uv has no supported segment")
    runs = np.split(segments, np.flatnonzero(np.diff(segments) > 1)+1)
    closest = np.empty_like(points); normal = np.empty(len(points))
    distance = np.full(len(points), np.inf)
    indices = np.full(len(points), -1, dtype=np.int64)
    fraction = np.empty(len(points))
    usable = False
    for run in runs:
        begin, stop = int(run[0]), int(run[-1])+2
        piece = line[begin:stop]
        if not np.any(np.linalg.norm(np.diff(piece, axis=0), axis=1) > 0):
            continue
        match = point_to_polyline(points, piece); usable = True
        keep = match.distance_px < distance
        closest[keep] = match.closest_uv[keep]
        normal[keep] = match.normal_residual_px[keep]
        distance[keep] = match.distance_px[keep]
        indices[keep] = begin+match.segment_indices[keep]
        fraction[keep] = match.segment_fractions[keep]
    if not usable:
        raise ValueError("polyline_uv has no supported nonzero segment")
    return PolylineMatch(closest, points-closest, distance, normal, indices, fraction)


def _affine(value, degeneracy_tolerance):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("affine must be a finite homogeneous (3, 3) matrix")
    if not np.array_equal(matrix[2], [0., 0., 1.]):
        raise ValueError("affine must have homogeneous last row [0, 0, 1]")
    singular = np.linalg.svd(matrix[:2, :2], compute_uv=False)
    if singular[-1] <= degeneracy_tolerance*max(1., singular[0]):
        raise ValueError("affine has a degenerate linear part")
    return matrix.copy()


def _apply(points, affine):
    with np.errstate(over="ignore", invalid="ignore"):
        value = points@affine[:2, :2].T+affine[:2, 2]
    if not np.isfinite(value).all():
        raise ValueError("affine produces nonfinite projected coordinates")
    return value


def _match_paths(observed, paths, affine):
    count = len(observed)
    closest = np.empty((count, 2)); residual = np.empty((count, 2))
    distance = np.empty(count); normal = np.empty(count)
    segments = np.empty(count, dtype=np.int64); fractions = np.empty(count)
    native_positions = np.empty_like(observed)
    for name, rows, line in paths:
        match = point_to_polyline(observed[rows], _apply(line, affine))
        closest[rows] = match.closest_uv; residual[rows] = match.residual_uv
        distance[rows] = match.distance_px; normal[rows] = match.normal_residual_px
        segments[rows] = match.segment_indices; fractions[rows] = match.segment_fractions
        a = line[match.segment_indices]; b = line[match.segment_indices+1]
        native_positions[rows] = a+match.segment_fractions[:, None]*(b-a)
    return PolylineMatch(closest, residual, distance, normal, segments, fractions), native_positions


def _normal_affine_increment(paths, affine, match, native_positions, degeneracy_tolerance):
    """Linearize the same curve-distance objective after eliminating stations.

    Interior points constrain the curve normal. Clamped segment endpoints
    retain both residual coordinates. This avoids advancing a very flat oval
    by tiny alternating point/station steps, without changing its source UV.
    """
    center=native_positions.mean(0)
    scales=np.maximum(np.std(native_positions,axis=0),degeneracy_tolerance)
    design=np.c_[(native_positions-center)/scales,np.ones(len(native_positions))]
    normals=np.empty_like(native_positions)
    for _,rows,line in paths:
        segments=match.segment_indices[rows]
        direction=(line[segments+1]-line[segments])@affine[:2,:2].T
        normals[rows]=np.c_[-direction[:,1],direction[:,0]]/np.linalg.norm(direction,axis=1)[:,None]
    interior=(match.segment_fractions>0)&(match.segment_fractions<1)
    normal_design=np.c_[normals[:,0,None]*design,normals[:,1,None]*design]
    endpoints=design[~interior];zeros=np.zeros_like(endpoints)
    jacobian=np.vstack([normal_design[interior],np.c_[endpoints,zeros],np.c_[zeros,endpoints]])
    target=np.r_[np.sum(normals[interior]*match.residual_uv[interior],axis=1),
                 match.residual_uv[~interior,0],match.residual_uv[~interior,1]]
    step=np.linalg.lstsq(jacobian,target,rcond=degeneracy_tolerance)[0].reshape(2,3)
    delta=np.zeros((3,3));delta[:2,:2]=step[:,:2]/scales
    delta[:2,2]=step[:,2]-delta[:2,:2]@center
    return delta


def register_native_shapes(
    observed_uv,
    path_ids,
    native_polylines: Mapping[str, np.ndarray],
    *,
    initial_affine=None,
    transform_model='affine',
    rms_change_tolerance_px=1e-7,
    transform_tolerance=1e-10,
    degeneracy_tolerance=1e-10,
) -> NativeShapeRegistration:
    """ICP-register all named native paths with one shared source-image affine.

    The caller supplies finite polylines from the same native projection,
    exposure and pixel domain; this function does not project scene geometry.
    Closest positions on each bound path are recomputed every iteration. An
    affine normal-distance step competes with the alternating point-distance
    step; a line search checks the original finite-curve objective. Both use
    the same unchanged source UV and one shared transform for every path.
    The optional similarity model preserves the projected shape's aspect and
    allows only positive uniform scale, rotation and translation. No path is independently warped and no row is rejected or
    weighted. Missing paths and numerically degenerate configurations fail.

    Stop when RMS improvement is <= ``rms_change_tolerance_px``, or the
    relative affine change is <= ``transform_tolerance``. These are numerical
    tolerances, not acceptance limits. A stationary wrong binding can retain
    large errors. ICP is local and depends on its initial affine (identity by
    default). Concentric circles alone leave a rotational symmetry: contour
    agreement does not identify a unique material rotation or native camera.
    Returned closest_uv is a projection diagnostic, never replacement UV.
    """
    if transform_model not in ('affine','similarity'):
        raise ValueError('transform_model must be affine or similarity')
    observed = _points(observed_uv, "observed_uv", minimum=3)
    names = np.asarray(path_ids)
    if names.shape != (len(observed),) or names.dtype.kind not in "US":
        raise ValueError("path_ids must contain one native curve name per observed UV")
    names = names.astype(str)
    for name, value in [("rms_change_tolerance_px", rms_change_tolerance_px),
                        ("transform_tolerance", transform_tolerance),
                        ("degeneracy_tolerance", degeneracy_tolerance)]:
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0.:
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(native_polylines, Mapping):
        raise ValueError("native_polylines must map native curve names to finite polylines")
    paths = []
    for name in np.unique(names):
        if name not in native_polylines:
            raise ValueError(f"Missing native projected path: {name}")
        paths.append((name, np.flatnonzero(names == name),
                      _points(native_polylines[name], f"native path {name}", minimum=2)))
    affine = _affine(np.eye(3) if initial_affine is None else initial_affine, degeneracy_tolerance)
    match, native_positions = _match_paths(observed, paths, affine)
    rms = float(np.linalg.norm(match.distance_px)/np.sqrt(len(observed)))
    initial_rms = rms
    iterations = 0
    while True:
        center = native_positions.mean(0)
        destination_center = observed.mean(0)
        design = np.c_[native_positions-center, np.ones(len(observed))]
        coefficients, _, rank, _ = np.linalg.lstsq(
            design, observed-destination_center, rcond=degeneracy_tolerance)
        if rank < 3:
            raise ValueError("Native shape correspondences are degenerate for a shared affine")
        proposed = np.eye(3)
        if transform_model=='affine':
            proposed[:2, :2] = coefficients[:2].T
            proposed[:2, 2] = destination_center+coefficients[2]-proposed[:2, :2]@center
        else:
            # Keep the projected native aspect/shape. A source outline may
            # translate, rotate and change uniform image scale, but cannot
            # independently stretch toward adjacent lettering or reflections.
            x=native_positions-center;y=observed-destination_center
            left,singular,right=np.linalg.svd(x.T@y)
            sign=np.ones(2);sign[-1]=np.sign(np.linalg.det(right.T@left.T))
            rotation=right.T@np.diag(sign)@left.T
            scale=float(singular@sign/np.sum(x*x))
            proposed[:2,:2]=scale*rotation
            proposed[:2,2]=destination_center-proposed[:2,:2]@center
        proposed = _affine(proposed, degeneracy_tolerance)
        next_match, next_native_positions = _match_paths(observed, paths, proposed)
        next_rms = float(np.linalg.norm(next_match.distance_px)/np.sqrt(len(observed)))
        if not np.isfinite(next_rms) or next_rms > rms+1e-10*max(1., rms):
            raise ValueError("Shared affine ICP failed its finite monotone-distance check")
        if transform_model=='affine':
            delta=_normal_affine_increment(paths,affine,match,native_positions,degeneracy_tolerance)
            fraction=1.
            while fraction*np.linalg.norm(delta)>transform_tolerance*(1.+np.linalg.norm(affine)):
                trial=affine+fraction*delta
                try:
                    trial=_affine(trial,degeneracy_tolerance)
                except ValueError:
                    fraction*=.5;continue
                if np.linalg.det(trial[:2,:2])*np.linalg.det(affine[:2,:2])<=0:
                    fraction*=.5;continue
                trial_match,trial_positions=_match_paths(observed,paths,trial)
                trial_rms=float(np.linalg.norm(trial_match.distance_px)/np.sqrt(len(observed)))
                if np.isfinite(trial_rms) and trial_rms<next_rms:
                    proposed,next_match,next_native_positions,next_rms=trial,trial_match,trial_positions,trial_rms
                    break
                if np.isfinite(trial_rms) and trial_rms<rms:
                    break  # The already checked alternating step is better.
                fraction*=.5
        change = float(np.linalg.norm(proposed-affine)/(1.+np.linalg.norm(affine)))
        improvement = rms-next_rms
        affine, match, native_positions, rms = proposed, next_match, next_native_positions, next_rms
        iterations += 1
        if change <= transform_tolerance:
            reason = "relative_transform_change"
            break
        if improvement <= rms_change_tolerance_px:
            reason = "rms_distance_change"
            break
    return NativeShapeRegistration(np.array(observed_uv, copy=True), names.copy(), affine,
                                   match, rms, initial_rms, iterations, reason)
