"""Explicit vertex representation for newly compiled kind-2 board paths.

Kind 2 uses N-1 segments. Closed paths therefore repeat their first vertex.
This helper is for new paths before station assignment, never an automatic
migration of serialized paths or their station/finite-interval coordinates.
"""
import math


def new_board_path_points(points, closed=False):
    result = [[float(v) for v in point] for point in points]
    if not result or any(len(p) != 2 or not all(map(math.isfinite, p)) for p in result):
        raise ValueError("board path requires finite two-dimensional vertices")
    if closed and len(result) > 1 and result[-1] != result[0]:
        result.append(result[0].copy())
    return result
