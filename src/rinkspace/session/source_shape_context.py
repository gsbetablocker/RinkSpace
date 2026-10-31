"""Check measured paint bands in the context of other named native shapes.

A local affine map explains the complete observed context together. It is not
a camera update. Only original RGB transition coordinates may leave this API;
neither this check nor a small distance establishes a physical instance.
"""
from collections.abc import Mapping
import numpy as np
from .native_shape_guidance import register_native_shapes


def _cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def transition_intersects_path(quantiles, path):
    """Does the finite native contour cross each actually measured transition?

    Test the measured q25..q75 segment, not a symmetric radius around q50.
    Thus asymmetric blurred transitions retain their actual source support.
    """
    q = np.asarray(quantiles, float)
    p = np.asarray(path, float)
    if q.ndim != 3 or q.shape[1:] != (3, 2) or not np.isfinite(q).all():
        raise ValueError('Finite measured quantiles must have shape (N,3,2)')
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError('A finite native polyline is required')
    a = q[:, 0, None, :]; r = (q[:, 2]-q[:, 0])[:, None, :]
    b = p[None, :-1, :]; s = np.diff(p, axis=0)[None, :, :]
    denominator = _cross(r, s); difference = b-a
    # Floating arithmetic guard only; no image-distance allowance.
    tolerance = np.finfo(float).eps*32*np.maximum(1., np.linalg.norm(r, axis=2)*np.linalg.norm(s, axis=2))
    usable = np.abs(denominator) > tolerance
    safe = np.where(usable, denominator, 1.)
    t = _cross(difference, s)/safe; u = _cross(difference, r)/safe
    crossed = usable & (t >= 0.) & (t <= 1.) & (u >= 0.) & (u <= 1.)
    return crossed.any(axis=1)


def measured_band_unmasked(quantiles, exclusion):
    """Inspect the entire measured band, including its actual outer transitions."""
    q = np.asarray(quantiles, float)
    mask = np.asarray(exclusion, bool)
    if q.ndim != 4 or q.shape[1:] != (2, 3, 2) or not np.isfinite(q).all():
        raise ValueError('Paired measured quantiles must have shape (N,2,3,2)')
    if mask.ndim != 2:
        raise ValueError('Same-exposure exclusion mask must be two dimensional')
    valid = np.zeros(len(q), bool)
    for i, (start, stop) in enumerate(zip(q[:, 0, 0], q[:, 1, 2])):
        # One native-pixel sampling step; no projected material footprint.
        count = max(1, int(np.ceil(np.linalg.norm(stop-start))))+1
        pixels = np.rint(np.linspace(start, stop, count)).astype(int)
        inside = ((pixels >= 0).all(1) & (pixels < [mask.shape[1], mask.shape[0]]).all(1))
        valid[i] = bool(inside.all() and not mask[pixels[:, 1], pixels[:, 0]].any())
    return valid


def assess_native_band_shape(measurement, candidate_ids, native_curves, *,
                             roles, context_observations: Mapping, exclusion=None):
    """Assess one source-band explanation without changing or admitting UV.

    Both interfaces and all named context paths use one image transform.
    Alternatives must be compared by the caller using their actual material,
    topology and temporal evidence; this API never selects the lowest fit loss.
    """
    ids = np.unique(np.asarray(candidate_ids, int))
    roles = tuple(roles)
    if len(roles) != 2 or len(set(roles)) != 2 or not len(ids):
        raise ValueError('Two different boundary roles and source candidates are required')
    if set(roles) & set(context_observations):
        raise ValueError('Context paths must be distinct from the assessed band roles')
    q = np.asarray(measurement['candidate_side_quantiles_uv'], float)[ids]
    if q.shape != (len(ids), 2, 3, 2) or not np.isfinite(q).all():
        raise ValueError('Original paired RGB transition quantiles are required')
    observed = []; names = []
    for name, values in context_observations.items():
        uv = np.asarray(values, float)
        if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
            raise ValueError('Each named context needs finite original (N,2) UV')
        observed.append(uv); names.extend([name]*len(uv))
    context_rows = len(names)
    for side, role in enumerate(roles):
        observed.append(q[:, side, 1]); names.extend([role]*len(ids))
    observed = np.concatenate(observed)
    fitted = register_native_shapes(observed, names, native_curves)
    support = []
    for side, role in enumerate(roles):
        path = np.asarray(native_curves[role], float)
        transformed = path@fitted.affine[:2, :2].T+fitted.affine[:2, 2]
        support.append(transition_intersects_path(q[:, side], transformed))
    support = np.stack(support, axis=1)
    unmasked = np.ones(len(ids), bool) if exclusion is None else measured_band_unmasked(q, exclusion)
    return dict(candidate_ids=ids, original_source_uv=q[:, :, 1].copy(),
        interface_inside_actual_transition=support, entire_measured_band_unmasked=unmasked,
        source_shape_supported=support.all(1) & unmasked,
        context_rows=context_rows, shared_affine=fitted.affine,
        distance_px=fitted.matches.distance_px[context_rows:].reshape(2, len(ids)).T,
        iterations=fitted.iterations, rms_distance_px=fitted.rms_distance_px,
        fit_eligible=False, physical_identity_confirmed=False,
        source_UV_unchanged=np.array_equal(observed, fitted.observed_uv))


def source_band_conflicts(points, measurement, candidate_ids):
    """Find points in the same measured paint as a competing band explanation.

    Each strip joins consecutive actual source queries only. Missing reads are
    not bridged, and this function neither relabels nor removes any source row.
    It lets a line reader expose an annulus interpretation of a flat arc.
    """
    import cv2
    uv = np.asarray(points, float)
    ids = np.unique(np.asarray(candidate_ids, int))
    query = np.asarray(measurement['candidate_query'], int)[ids]
    sides = np.asarray(measurement['candidate_side_quantiles_uv'], float)[ids, :, 1]
    claims = [set() for _ in uv]
    by_query = {}
    for local, qi in enumerate(query):
        by_query.setdefault(int(qi), []).append(local)
    for qi, left in by_query.items():
        right = by_query.get(qi+1, [])
        if len(left) != 1 or len(right) != 1:
            continue
        if measurement['query_curve'][qi] != measurement['query_curve'][qi+1]:
            continue
        a, b = left[0], right[0]
        quad = np.array([sides[a, 0], sides[b, 0], sides[b, 1], sides[a, 1]], np.float32)
        lo, hi = quad.min(0), quad.max(0)
        for index in np.flatnonzero((uv >= lo).all(1) & (uv <= hi).all(1)):
            if cv2.pointPolygonTest(quad, tuple(map(float, uv[index])), False) >= 0:
                claims[index].add((int(ids[a]), int(ids[b])))
    return [sorted(values) for values in claims]
