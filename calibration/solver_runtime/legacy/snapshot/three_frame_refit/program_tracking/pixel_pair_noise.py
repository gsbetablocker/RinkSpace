"""Transport uncertainty in real reference pixels, including shared sources.

Covariances are frozen at an explicitly supplied camera linearization. They
cannot shrink or grow as the optimizer changes a camera. A reference pixel
reused by several target frames contributes one shared source error, not one
independent source measurement per target.
"""
from __future__ import annotations

import numpy as np

try:
    from . import acceleration
except ImportError:
    import acceleration


def pair_covariance(source_pixel_jacobian, source_sigma, target_sigma):
    jac = np.asarray(source_pixel_jacobian, float).reshape(-1, 2, 2)
    source = np.broadcast_to(np.asarray(source_sigma, float), (len(jac), 2))
    target = np.broadcast_to(np.asarray(target_sigma, float), (len(jac), 2))
    if (not np.isfinite(jac).all() or not np.isfinite(source).all()
            or not np.isfinite(target).all() or np.any(source <= 0)
            or np.any(target <= 0)):
        raise ValueError('Pixel transfer uncertainty must be finite and positive')
    propagated = jac * source[:, None, :]
    covariance = propagated @ propagated.transpose(0, 2, 1)
    covariance[:, np.arange(2), np.arange(2)] += target**2
    return covariance


def inverse_cholesky(covariance):
    value = np.asarray(covariance, float)
    return np.linalg.inv(np.linalg.cholesky(value))


class SharedSourcePixelNoise:
    """Fixed block covariance for matching packets with identical source UV.

Each record is (source_pts, target_pts, pair, source_sigma[N,2],
target_sigma[N,2]). ``pair.source_pixel_jacobians(states)`` differentiates
the actual source-pixel to target-pixel mapping, including both optics.
"""

    def __init__(self, records, linearization_states):
        self.records = records
        self.keys = sorted({int(k) for row in records for k in row[:2]})
        self.key_index = {key: index for index, key in enumerate(self.keys)}
        self.offsets = np.r_[0, np.cumsum([2*len(row[2].xy0) for row in records])]
        self.dimension = int(self.offsets[-1])
        groups = {}
        gains = []
        for ri, (source_pts, target_pts, pair, source_sigma, target_sigma) in enumerate(records):
            source_sigma = np.broadcast_to(np.asarray(source_sigma, float), (len(pair.xy0), 2))
            target_sigma = np.broadcast_to(np.asarray(target_sigma, float), (len(pair.xy0), 2))
            states = [linearization_states[source_pts], linearization_states[target_pts]]
            jac = pair.source_pixel_jacobians(states)
            pair_covariance(jac, source_sigma, target_sigma)  # Validate inputs.
            gains.extend(np.linalg.svd(jac, compute_uv=False).tolist())
            for i, uv in enumerate(pair.xy0):
                identity = (int(source_pts), float(uv[0]), float(uv[1]))
                groups.setdefault(identity, []).append(
                    (int(self.offsets[ri]+2*i), jac[i], source_sigma[i], target_sigma[i]))
        self.blocks = []
        for rows in groups.values():
            # Conflicting declarations must not make a shared pixel more precise.
            source_sigma = np.max([row[2] for row in rows], axis=0)
            jac = np.concatenate([row[1] for row in rows], axis=0)
            propagated = jac*source_sigma[None, :]
            covariance = propagated @ propagated.T
            target_variance = np.concatenate([row[3]**2 for row in rows])
            covariance[np.diag_indices_from(covariance)] += target_variance
            indices = np.asarray([i for row in rows for i in (row[0], row[0]+1)])
            self.blocks.append((indices, inverse_cholesky(covariance)))
        self._batched_whitener = acceleration.BatchedWhitening(self.blocks) if acceleration.MODE != "baseline" else None
        self.metadata = {
            'model': 'frozen_first_order_source_and_target_pixel_covariance',
            'source_error_reused_across_target_frames': True,
            'actual_pixel_pairs': self.dimension//2,
            'independent_source_pixels': len(groups),
            'shared_source_pixels': sum(len(rows) > 1 for rows in groups.values()),
            'source_to_target_gain_singular_p50': np.median(gains, axis=0).tolist(),
            'source_to_target_gain_singular_p95': np.percentile(gains, 95, axis=0).tolist(),
            'linearization': 'explicit_camera_initialization_before_optimization',
            'source_pixels_modified': False,
            'camera_accuracy_claim': False,
        }

    def _whiten(self, value):
        if self._batched_whitener is not None:
            return self._batched_whitener.whiten(value)
        result = np.empty_like(value)
        for indices, weight in self.blocks:
            result[indices] = weight @ value[indices]
        return result

    def residual(self, states):
        raw = np.concatenate([pair.residual([states[self.key_index[a]], states[self.key_index[b]]])
                              for a, b, pair, _, _ in self.records])
        return self._whiten(raw)

    def jacobians(self, states):
        raw = [np.zeros((self.dimension, 11)) for _ in self.keys]
        for ri, (a, b, pair, _, _) in enumerate(self.records):
            ia, ib = self.key_index[a], self.key_index[b]
            ja, jb = pair.jacobians([states[ia], states[ib]])
            section = slice(self.offsets[ri], self.offsets[ri+1])
            raw[ia][section] += ja
            raw[ib][section] += jb
        return [self._whiten(jac) for jac in raw]
