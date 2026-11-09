"""Optional native broad phase for the existing sampled body/skate scan."""

import ctypes
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


@lru_cache(maxsize=4)
def _library(path):
    dll = ctypes.CDLL(path)
    pairs = getattr(dll, 'rs_surface_aabb_pairs', None)
    if pairs is None:
        return None
    real = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    integer = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    size = ctypes.c_size_t
    pairs.argtypes = [real, size, integer, size, integer, integer, size, integer, size]
    pairs.restype = ctypes.c_int64
    return pairs


def triangle_spheres(triangles):
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None], axis=-1).max(axis=1)
    buckets = np.floor(np.log2(np.maximum(radii, 1e-12))).astype(int)
    return centers, radii, buckets


def _legacy_pair_order(pairs, triangles):
    """Retain the legacy sphere check and visit order for equal-gap witnesses."""
    if not len(pairs):
        return pairs
    centers, radii, bucket_keys = triangle_spheres(triangles)
    _, buckets = np.unique(bucket_keys, return_inverse=True)
    max_radius = np.zeros(int(buckets.max()) + 1)
    np.maximum.at(max_radius, buckets, radii)
    a, b = pairs.T
    swap = buckets[a] > buckets[b]
    a, b = np.where(swap, b, a), np.where(swap, a, b)
    offset = centers[a] - centers[b]
    radius = max_radius[buckets[a]] + max_radius[buckets[b]]
    inside = np.sum(offset * offset, axis=1) <= radius * radius
    a, b = a[inside], b[inside]

    # cKDTree visits both nodes' left/right branches in alternating depth
    # order, then visits the original leaf permutation. Recreate those keys
    # without materializing the (much larger) sphere-query pair matrices.
    paths = np.empty(len(triangles), dtype=np.uint64)
    ranks = np.empty(len(triangles), dtype=np.int32)
    for group in range(len(max_radius)):
        ids = np.flatnonzero(buckets == group)
        tree = cKDTree(centers[ids])
        permutation = ids[tree.indices]
        ranks[permutation] = np.arange(len(ids), dtype=np.int32)
        pending = [(tree.tree, 0, 62)]
        while pending:
            node, key, shift = pending.pop()
            if node.split_dim == -1:
                paths[permutation[node.start_idx:node.end_idx]] = key
            else:
                if shift < 0:
                    # A rare unbalanced tree needs the original traversal.
                    return None
                pending.append((node.lesser, key, shift - 2))
                pending.append((node.greater, key | (1 << shift), shift - 2))
    leaf_pair = (paths[a] << 1) | paths[b]
    order = np.lexsort((ranks[b], ranks[a], leaf_pair, buckets[b], buckets[a]))
    return np.column_stack((a[order], b[order]))


class SurfaceBroadPhase:
    """Prepare the exact static two-edge neighbourhood once per mesh."""

    @classmethod
    def optional(cls, faces, topology, *, native_library=None):
        explicit = os.environ.get('RINKSPACE_BODY_SURFACE_LIBRARY')
        if explicit:
            pairs = _library(explicit)
            if pairs is None:
                raise ValueError('Configured surface library lacks rs_surface_aabb_pairs')
            return cls(pairs, faces, topology)

        core = os.environ.get('RINKSPACE_BODY_NATIVE_LIBRARY') or native_library
        if core:
            try:
                pairs = _library(str(core))
            except OSError:
                pairs = None
            if pairs is not None:
                return cls(pairs, faces, topology)

        if os.name == 'nt':
            from rinkspace.storage import AssetStore
            project = Path(__file__).resolve().parents[3]
            installed = AssetStore(project).data / 'runtime_tools/program_tracking_native/rinkspace_surface.dll'
            if installed.is_file():
                pairs = _library(str(installed))
                if pairs is None:
                    raise ValueError('Installed surface library lacks rs_surface_aabb_pairs')
                return cls(pairs, faces, topology)
        return None

    def __init__(self, pairs, faces, topology):
        self.pairs = pairs
        self.faces = np.ascontiguousarray(faces, dtype=np.int32)
        incidence = csr_matrix(
            (np.ones(faces.size, dtype=bool),
             (np.repeat(np.arange(len(faces)), 3), self.faces.ravel())),
            shape=(len(faces), topology.shape[0]),
        )
        # A pair is excluded iff ANY of its nine vertex pairs is near.
        neighbours = (incidence @ topology.astype(bool) @ incidence.T).tocsr()
        neighbours.sort_indices()
        self.rows = np.ascontiguousarray(neighbours.indptr, dtype=np.int32)
        self.excluded = np.ascontiguousarray(neighbours.indices, dtype=np.int32)
        self.vertex_count = topology.shape[0]

    def __call__(self, vertices):
        vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        if vertices.shape != (self.vertex_count, 3):
            raise ValueError('Posed vertices do not match the surface topology')
        output = np.empty((2 * len(self.faces), 2), dtype=np.int32)

        def fill(buffer):
            return self.pairs(vertices, self.vertex_count, self.faces, len(self.faces),
                              self.rows, self.excluded, len(self.excluded), buffer, len(buffer))

        count = fill(output)
        if count > len(output):
            output = np.empty((count, 2), dtype=np.int32)
            if fill(output) != count:
                raise RuntimeError('Surface broad phase changed its required pair count')
        if count < 0:
            raise ValueError(f'Native surface broad phase failed ({count})')
        return _legacy_pair_order(output[:count], vertices[self.faces])
