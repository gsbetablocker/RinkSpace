"""Organize native projected paths into source-RGB search queries only."""
from __future__ import annotations

import numpy as np


def _visible_runs(vertices, good, height, width):
    # Clip every supported segment together. The scalar arithmetic and its
    # operation order are unchanged; only segment enumeration is batched.
    count = max(0, len(vertices)-1)
    supported = good[:count] & good[1:count+1]
    indices = np.flatnonzero(supported)
    if not len(indices):
        return []
    starts = vertices[indices]
    vectors = vertices[indices+1]-starts
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    if not np.isfinite(lengths).all():
        raise ValueError('Projected segment length is nonfinite')
    nonzero = lengths != 0.
    lo, hi = np.zeros(len(indices)), np.ones(len(indices))
    outside_parallel = np.zeros(len(indices), bool)
    for axis, limit in ((0, width-1), (1, height-1)):
        moving = vectors[:, axis] != 0.
        outside_parallel |= ~moving & ((starts[:, axis] < 0.) | (starts[:, axis] > limit))
        first = np.divide(-starts[:, axis], vectors[:, axis],
                          out=np.zeros(len(indices)), where=moving)
        last = np.divide(limit-starts[:, axis], vectors[:, axis],
                         out=np.zeros(len(indices)), where=moving)
        lo[moving] = np.maximum(lo[moving], np.minimum(first[moving], last[moving]))
        hi[moving] = np.minimum(hi[moving], np.maximum(first[moving], last[moving]))
    visible = nonzero & ~outside_parallel & (hi > lo)
    positions = np.flatnonzero(visible)
    if not len(positions):
        return []
    # Invalid/offscreen segments split runs. Valid zero-length segments are
    # skipped without splitting, exactly as in the original scalar walk.
    breaks = ~supported
    breaks[indices] = nonzero & ~visible
    barriers = np.cumsum(breaks)
    selected = indices[positions]
    lo, hi = lo[positions], hi[positions]
    starts = starts[positions]
    vectors = vectors[positions]
    clipped_starts = starts+lo[:, None]*vectors
    clipped_stops = starts+hi[:, None]*vectors
    disconnected = np.any(np.abs(clipped_stops[:-1]-clipped_starts[1:]) > 1e-9, axis=1)
    split = ((barriers[selected[1:]] != barriers[selected[:-1]]) |
             (hi[:-1] < 1.) | disconnected)
    pieces = [(int(index), start, stop, float(a), float(b))
              for index, start, stop, a, b in zip(selected, clipped_starts, clipped_stops, lo, hi)]
    boundaries = np.r_[0, np.flatnonzero(split)+1, len(pieces)]
    return [pieces[a:b] for a, b in zip(boundaries[:-1], boundaries[1:])]


def projected_curve_queries(uv, valid, image_hw, spacing_px, search_radius_px, closed=False,
                            center_uv=None, preserve_axis_extrema=True):
    """Return search points/normals/intervals without emitting measurements.

    Each contiguous valid, in-image polyline run is sampled by cumulative
    screen arc length, rather than restarting spacing at every vertex.
    ``polyline_stations`` is original segment index plus segment fraction,
    sorted in input path order; it is not a physical material coordinate.
    For closed paths, the last-to-first segment is included when necessary.

    Known closed curves additionally retain their good-vertex x/y extrema
    when those extrema lie in the image. Their normals are exactly outward
    pixel-axis directions. An extremum replaces an ordinary sample at the
    same station, regardless of how close neighbouring queries are. Without
    center_uv, ordinary normals follow the ordered tangent's left normal;
    with a center they are flipped outward. The center is navigation only.

    Search intervals are [-radius,+radius] intersected with the actual image
    interior's 4-pixel halo: [4,w-5] x [4,h-5]. Empty intersections are omitted;
    no minimum search-width or confidence threshold is applied. Radius may
    be one positive scalar or an array matching the generated queries before
    empty-interval removal. All input arrays remain unchanged.
    """
    vertices = np.asarray(uv, dtype=float)
    good = np.asarray(valid, dtype=bool)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or good.shape != (len(vertices),):
        raise ValueError('uv/valid must have shapes (N,2)/(N,)')
    shape = np.asarray(image_hw)
    if shape.shape != (2,) or shape.dtype.kind not in 'iu' or (shape < 9).any():
        raise ValueError('image_hw must leave a 4px source halo on every side')
    height, width = map(int, shape)
    if not np.isscalar(spacing_px) or not np.isfinite(spacing_px) or spacing_px <= 0.:
        raise ValueError('spacing_px must be finite and positive')
    center = None if center_uv is None else np.asarray(center_uv, float)
    if center is not None and (center.shape != (2,) or not np.isfinite(center).all()):
        raise ValueError('center_uv must be a finite navigation point')
    good = good & np.isfinite(vertices).all(1)
    original_vertices, original_good = vertices, good
    if closed and len(vertices) > 1 and not np.allclose(vertices[0], vertices[-1], rtol=0., atol=1e-9):
        vertices = np.r_[vertices, vertices[:1]]
        good = np.r_[good, good[:1]]
    records = []
    for run in _visible_runs(vertices, good, height, width):
        starts = np.array([piece[1] for piece in run]); stops = np.array([piece[2] for piece in run])
        lengths = np.linalg.norm(stops-starts, axis=1)
        cumulative = np.r_[0., np.cumsum(lengths)]
        numerical = 32*np.finfo(float).eps*max(1., cumulative[-1])
        arc = np.arange(0., cumulative[-1]+numerical, float(spacing_px))
        segments = np.minimum(np.searchsorted(cumulative, arc, side='right')-1, len(run)-1)
        for distance, local in zip(arc, segments):
            index, start, stop, lo, hi = run[int(local)]
            fraction = np.clip((distance-cumulative[local])/lengths[local], 0., 1.)
            point = start+fraction*(stop-start)
            # Do not duplicate a complete closed path's seam.
            station = index+lo+fraction*(hi-lo)
            if closed and station > 0. and records and np.allclose(point, vertices[0], rtol=0., atol=1e-9):
                continue
            tangent = (stop-start)/lengths[local]
            normal = np.array([-tangent[1], tangent[0]])
            if center is not None and (point-center)@normal < 0.:normal=-normal
            records.append((float(station), point, normal, False))
    if closed and preserve_axis_extrema and original_good.any():
        ids = np.flatnonzero(original_good)
        points = original_vertices[ids]
        for axis, maximum, normal in ((0, False, [-1.,0.]), (0, True, [1.,0.]),
                                       (1, False, [0.,-1.]), (1, True, [0.,1.])):
            local = int(np.argmax(points[:,axis]) if maximum else np.argmin(points[:,axis]))
            index = int(ids[local]);point = original_vertices[index]
            if not (0. <= point[0] <= width-1 and 0. <= point[1] <= height-1):continue
            station = float(index)
            if index == len(original_vertices)-1 and np.allclose(point, original_vertices[0], rtol=0., atol=1e-9):
                station = 0.
            same = [j for j, item in enumerate(records) if abs(item[0]-station) <= 1e-9]
            # A polygon vertex may own two distinct axis extrema. Keep both
            # directions, but remove its ordinary tangent-based query.
            records = [item for j,item in enumerate(records) if j not in same or item[3]]
            if not any(abs(item[0]-station)<=1e-9 and item[3] and np.array_equal(item[2],normal) for item in records):
                records.append((station, point.copy(), np.asarray(normal), True))
    records.sort(key=lambda item:item[0])
    points = np.array([item[1] for item in records], float).reshape(-1,2)
    normals = np.array([item[2] for item in records], float).reshape(-1,2)
    stations = np.array([item[0] for item in records], float)
    extrema = np.array([item[3] for item in records], bool)
    radius = np.asarray(search_radius_px, float)
    if radius.ndim == 0:
        if not np.isfinite(radius) or radius <= 0.:
            raise ValueError('search_radius_px must be finite and positive')
        radius=np.full(len(points),float(radius))
    elif radius.shape != (len(points),):raise ValueError('Radius must be scalar or match the generated query count')
    if not np.isfinite(radius).all() or (radius <= 0.).any():
        raise ValueError('search_radius_px must be finite and positive')
    intervals = np.c_[-radius, radius]
    keep = np.ones(len(points), bool)
    for axis, limit in ((0,width-1), (1,height-1)):
        moving = normals[:,axis] != 0.
        keep &= moving | ((points[:,axis] >= 4.) & (points[:,axis] <= limit-4.))
        first = np.divide(4.-points[:,axis],normals[:,axis],out=np.full(len(points),-np.inf),where=moving)
        last = np.divide(limit-4.-points[:,axis],normals[:,axis],out=np.full(len(points),np.inf),where=moving)
        intervals[:,0] = np.maximum(intervals[:,0],np.minimum(first,last))
        intervals[:,1] = np.minimum(intervals[:,1],np.maximum(first,last))
    keep &= intervals[:,1] >= intervals[:,0]
    return dict(points_uv=points[keep], normals=normals[keep], intervals=intervals[keep],
                polyline_stations=stations[keep], is_axis_extremum=extrema[keep])
