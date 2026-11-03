"""Explicit open-collar topology for RinkSpace's existing 14-vertex boot proxy.

This is a DISPLAY TOPOLOGY correction, not a shoe-fitting or mechanics model.
It preserves every input vertex and all blade samples. Unlike ConvexHull, it
keeps the concave seven-point side profile and leaves the collar open.

Input ordering must be footwear_geometry.local_vertices()[:14]: indices 0..6
on one side and corresponding indices 7..13 on the other. The collar boundary
is 4 -> 5 -> 12 -> 11 -> 4. Build faces once in the rigid reference geometry;
do not retriangulate each animation frame or infer topology from new positions.

Only NumPy is required. Faces point outward. To inspect the interior in a
viewer, use its two-sided/backface-culling setting; no wall thickness or inner
lining is invented here. The front-upper tag is only a tongue proxy, not a
separately measured tongue, lace opening or hinged boot design.
"""
from __future__ import annotations

import numpy as np


COLLAR_LOOP = np.array([4, 5, 12, 11], dtype=np.int32)


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _signed_area(poly):
    return 0.5 * sum(_cross2(poly[i], poly[(i + 1) % len(poly)])
                     for i in range(len(poly)))


def _validate_simple_polygon(poly, eps):
    def on_segment(a, b, p):
        return (abs(_cross2(b - a, p - a)) <= eps and
                np.dot(p - a, p - b) <= eps)

    def intersects(a, b, c, d):
        aa, bb = _cross2(b - a, c - a), _cross2(b - a, d - a)
        cc, dd = _cross2(d - c, a - c), _cross2(d - c, b - c)
        proper = ((aa > eps and bb < -eps) or (aa < -eps and bb > eps)) and (
            (cc > eps and dd < -eps) or (cc < -eps and dd > eps))
        return proper or on_segment(a, b, c) or on_segment(a, b, d) or \
            on_segment(c, d, a) or on_segment(c, d, b)

    n = len(poly)
    for i in range(n):
        if np.linalg.norm(poly[(i + 1) % n] - poly[i]) ** 2 <= eps:
            raise ValueError("Boot side profile has a degenerate edge")
        for j in range(i + 1, n):
            if j in (i, (i + 1) % n) or (j + 1) % n == i:
                continue
            if intersects(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n]):
                raise ValueError("Boot side profile self-intersects")


def _triangulate(poly, eps):
    """Ear clipping of a small simple polygon, retaining its original winding."""
    area = _signed_area(poly)
    if abs(area) <= eps:
        raise ValueError("Boot side profile has zero area")
    sign = 1.0 if area > 0 else -1.0
    active = list(range(len(poly)))
    faces = []
    while len(active) > 3:
        found = False
        for offset, middle in enumerate(active):
            before, after = active[offset - 1], active[(offset + 1) % len(active)]
            a, b, c = poly[[before, middle, after]]
            if sign * _cross2(b - a, c - b) <= eps:
                continue
            contains = False
            for other in active:
                if other in (before, middle, after):
                    continue
                q = poly[other]
                if min(sign * _cross2(b - a, q - a),
                       sign * _cross2(c - b, q - b),
                       sign * _cross2(a - c, q - c)) >= -eps:
                    contains = True
                    break
            if not contains:
                faces.append((before, middle, after))
                del active[offset]
                found = True
                break
        if not found:
            raise ValueError("Cannot triangulate boot profile without degenerate triangles")
    faces.append(tuple(active))
    return np.asarray(faces, dtype=np.int32)


def open_boot_shell(points):
    """Return (faces, groups) for the existing proxy without changing vertices.

    ``groups`` contains face indices and the original-vertex collar boundary.
    This function accepts one rigidly transformed frame, including world space.
    It validates the paired-side layout; arbitrary 14-point clouds are rejected.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (14, 3) or not np.isfinite(p).all():
        raise ValueError("Expected 14 finite 3D vertices in original boot order")
    centers = (p[:7] + p[7:]) * 0.5
    pairs = p[7:] - p[:7]
    width = pairs.mean(axis=0)
    span = float(np.max(np.linalg.norm(p - p.mean(axis=0), axis=1)))
    if span <= 0 or np.linalg.norm(width) <= 1e-10 * span:
        raise ValueError("Boot has no valid width")
    width_axis = width / np.linalg.norm(width)
    widths = pairs @ width_axis
    if np.any(widths <= 0) or np.max(np.linalg.norm(
            pairs - widths[:, None] * width_axis, axis=1)) > 1e-5 * span:
        raise ValueError("Paired boot sides do not share a rigid extrusion direction")
    c = centers - centers.mean(axis=0)
    if np.max(np.abs(c @ width_axis)) > 1e-5 * span:
        raise ValueError("Boot center profile is not perpendicular to its width")
    normal = np.cross(c, np.roll(c, -1, axis=0)).sum(axis=0)
    if np.linalg.norm(normal) <= 1e-12 * span ** 2:
        raise ValueError("Boot center profile is degenerate")
    normal /= np.linalg.norm(normal)
    u = centers[1] - centers[0]
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    poly = np.column_stack((c @ u, c @ v))
    eps = 1e-12 * span ** 2
    _validate_simple_polygon(poly, eps)
    side = _triangulate(poly, eps)

    # Ring order normally faces the negative-width side. Handle the opposite
    # order explicitly instead of assuming a world-up orientation.
    reverse = float(normal @ width_axis) > 0
    left = side[:, [0, 2, 1]] if reverse else side.copy()
    right = left[:, [0, 2, 1]] + 7
    faces = [*left.tolist(), *right.tolist()]
    groups = {"side_0_faces": np.arange(5, dtype=np.int32),
              "side_1_faces": np.arange(5, 10, dtype=np.int32)}
    bridges = []
    tongue = []
    for i in range(7):
        j = (i + 1) % 7
        if i == 4:  # No bridge across the collar opening 4--5.
            continue
        start = len(faces)
        triangles = np.array([(i, i + 7, j + 7), (i, j + 7, j)], dtype=np.int32)
        if reverse:
            triangles = triangles[:, [0, 2, 1]]
        faces.extend(triangles.tolist())
        bridges.extend((start, start + 1))
        if i == 3:
            tongue.extend((start, start + 1))
    groups.update(bridge_faces=np.asarray(bridges, dtype=np.int32),
                  front_upper_tongue_proxy_faces=np.asarray(tongue, dtype=np.int32),
                  collar_boundary=COLLAR_LOOP.copy())
    return np.asarray(faces, dtype=np.int32), groups


def shell_triangles(points):
    """Drop-in face-array interface for rinkspace.viewer3d.data.shell_triangles."""
    return open_boot_shell(points)[0]
