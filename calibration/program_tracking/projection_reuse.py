"""Bounded, exact-input projection reuse within one residual verification.

The supplied arrays come from the original Python/OpenCV projector, never
from the native factor being checked. No entry survives the context, and a
different camera or any absent world point falls back to the real projector.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np


_active = ContextVar("rinkspace_projection_verification", default=None)


class ProjectionWorld:
    """Immutable coordinate index reusable across cameras for the same curve."""
    def __init__(self, world):
        self.world = np.array(world, dtype=np.float64, order="C", copy=True).reshape(-1, 3)
        self.world.flags.writeable = False
        rows = self.world.view(np.dtype((np.void, 24))).reshape(-1)
        self.rows, self.indices = np.unique(rows, return_index=True)
        # Geometry-only exact lookups. No camera, residual, or projection state.
        self._selection_cache = {}
        self._selection_cache_bytes = 0

    def select_indices(self, points):
        key = (points.shape, points.tobytes())
        indices = self._selection_cache.get(key)
        if indices is not None:
            return indices
        rows = points.view(np.dtype((np.void, 24))).reshape(-1)
        positions = np.searchsorted(self.rows, rows)
        if np.any(positions >= len(self.rows)):
            return None
        if not np.array_equal(self.rows[positions], rows):
            return None
        indices = self.indices[positions]
        # Bound both entry count and payload; oversized/missing queries aren't retained.
        size = len(key[1]) + indices.nbytes
        if size <= 256 * 1024:
            while self._selection_cache and (
                    len(self._selection_cache) >= 8
                    or self._selection_cache_bytes + size > 256 * 1024):
                old_key = next(iter(self._selection_cache))
                old = self._selection_cache.pop(old_key)
                self._selection_cache_bytes -= len(old_key[1]) + old.nbytes
            indices.flags.writeable = False
            self._selection_cache[key] = indices
            self._selection_cache_bytes += size
        return indices


class _Projection:
    def __init__(self, camera, world, uv, depth, central):
        self.camera = np.ascontiguousarray(camera, dtype=np.float64).tobytes()
        self.geometry = world if isinstance(world, ProjectionWorld) else ProjectionWorld(world)
        self.uv = np.asarray(uv, np.float64).reshape(-1, 2)
        self.depth = np.asarray(depth, np.float64).reshape(-1)
        self.central = np.asarray(central, bool).reshape(-1)
        if not (len(self.geometry.world) == len(self.uv) == len(self.depth) == len(self.central)):
            raise ValueError("Projection reuse arrays have different lengths")
        # Sorting 24-byte records handles exact coordinates, including signed
        # zero, without rounding, nearest-neighbour lookup or Python row loops.
        self.rows, self.indices = self.geometry.rows, self.geometry.indices

    def select(self, camera, world):
        if np.ascontiguousarray(camera, dtype=np.float64).tobytes() != self.camera:
            return None
        points = np.ascontiguousarray(world, dtype=np.float64).reshape(-1, 3)
        indices = self.geometry.select_indices(points)
        if indices is None:
            return None
        # Advanced indexing owns each returned array; callers can safely apply
        # skew/validity edits without modifying the stored reference results.
        return self.uv[indices], self.depth[indices], self.central[indices]


def reused_projection(camera, world):
    entry = _active.get()
    return None if entry is None else entry.select(camera, world)


@contextmanager
def reuse_verified_projection(camera, world, uv, depth, central):
    token = _active.set(_Projection(camera, world, uv, depth, central))
    try:
        yield
    finally:
        _active.reset(token)
