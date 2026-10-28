"""Connect observed pixels for review without fitting or moving observations.

Run and order identities belong to the source reader. A connector is a display
aid between two observations, not an additional observation of its interior.
"""
from __future__ import annotations

import colorsys
import hashlib
import numpy as np


def entity_color(path_id):
    """A stable RGB colour, independent of which entities happen to be visible."""
    seed = int.from_bytes(hashlib.sha256(path_id.encode()).digest()[:4], 'big')
    return tuple(round(255*x) for x in colorsys.hsv_to_rgb((seed % 1009)/1009, .70, 1.))


def source_connections(uv, run_ids, order, exclusion_mask, *, closed_runs=(), max_order_gap=None):
    """Return index pairs only, never generated or smoothed pixel positions.

    Separate source spans stay separate even if their endpoints are close.
    The mask is sampled along the entire proposed connection. In particular,
    two visible endpoints do not permit drawing through a person or subtitle.
    """
    uv = np.asarray(uv, float)
    order = np.asarray(order, float)
    runs = np.asarray(run_ids, str)
    mask = np.asarray(exclusion_mask, bool)
    if uv.shape != (len(order), 2) or runs.shape != order.shape:
        raise ValueError('Observed pixels, source runs and order must correspond')
    if not np.isfinite(uv).all() or not np.isfinite(order).all() or mask.ndim != 2:
        raise ValueError('Finite original pixels/order and a source mask are required')
    pairs, breaks = [], []
    closed = set(closed_runs)
    for run in sorted(set(runs)):
        ids = np.flatnonzero(runs == run)
        ids = ids[np.argsort(order[ids], kind='stable')]
        candidates = list(zip(ids[:-1], ids[1:]))
        if run in closed and len(ids) > 2:
            candidates.append((ids[-1], ids[0]))
        for a, b in candidates:
            reason = None
            if order[a] == order[b]:
                reason = 'ambiguous_equal_source_order'
            elif max_order_gap is not None and abs(order[b]-order[a]) > max_order_gap and (a, b) != (ids[-1], ids[0]):
                reason = 'missing_source_span'
            if reason is None:
                n = max(2, int(np.ceil(np.linalg.norm(uv[b]-uv[a])*2))+1)
                pixel = np.rint(np.linspace(uv[a], uv[b], n)).astype(int)
                inside = (pixel >= 0).all(1) & (pixel < [mask.shape[1], mask.shape[0]]).all(1)
                if not inside.all():
                    reason = 'outside_source_image'
                elif mask[pixel[:, 1], pixel[:, 0]].any():
                    reason = 'source_occlusion'
            if reason is None:
                pairs.append((int(a), int(b)))
            else:
                breaks.append(dict(a=int(a), b=int(b), source_run=str(run), reason=reason))
    return pairs, breaks


def polyline_order(points, source_polyline):
    """Closest arclength on an ordered *photographic* contour, for display only."""
    points = np.asarray(points, float)
    p = np.asarray(source_polyline, float)
    d = np.diff(p, axis=0)
    length = np.linalg.norm(d, axis=1)
    valid = length > 1e-8
    if not valid.any():
        raise ValueError('Photographic contour has no finite length')
    start = np.r_[0., np.cumsum(length)][:-1][valid]
    a, d, length = p[:-1][valid], d[valid], length[valid]
    u = np.clip(np.einsum('ijk,jk->ij', points[:, None]-a[None], d)/(length**2), 0., 1.)
    foot = a[None]+u[:, :, None]*d[None]
    distances = np.linalg.norm(points[:, None]-foot, axis=2)
    k = distances.argmin(1)
    return start[k]+u[np.arange(len(points)), k]*length[k], distances[np.arange(len(points)), k]


def whole_entity_connections(uv, run_ids, order, exclusion_mask, *, closed=False):
    """Show the entire declared entity, including unsupported links as dashed.

    Unlike source_connections, this never hides a link at a source-run boundary.
    It is intended for finding wrong observations, not certifying a contour.
    """
    uv = np.asarray(uv, float)
    order = np.asarray(order, float)
    runs = np.asarray(run_ids, str)
    clear, blocked = source_connections(uv, ['entity']*len(uv), order, exclusion_mask,
        closed_runs=['entity'] if closed else [])
    ids = np.argsort(order, kind='stable')
    all_edges = [(int(a),int(b)) for a,b in zip(ids[:-1],ids[1:])]
    if closed and len(ids)>2:all_edges.append((int(ids[-1]),int(ids[0])))
    blocked_by = {(r['a'],r['b']):r['reason'] for r in blocked}
    dashed = []
    for a,b in all_edges:
        reason = blocked_by.get((a,b))
        if reason is None and runs[a] != runs[b]:reason = 'different_source_fragments'
        if closed and (a,b)==(int(ids[-1]),int(ids[0])):
            reason = reason or 'closure_not_independently_observed'
        if reason:dashed.append(dict(a=a,b=b,reason=reason))
    return all_edges,dashed
