"""Retain the original nonnegative nuisance depth in camera trial models.

The source residuals remain caller-owned. These moments replace the two
linearized rows of an interior point when its conditional line coordinate
would become negative. Inner profile and outer source IRLS metrics may differ.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix


class BoundedMotionSchurModel:
    def __init__(self, chunks, parameter_count):
        self.parameter_count = int(parameter_count)
        if self.parameter_count < 1 or self.parameter_count != parameter_count:
            raise ValueError('Positive complete parameter count required')
        self.chunks = []
        for supplied in chunks:
            chunk = dict(supplied)
            m = csr_matrix(chunk['projection_factors'], dtype=float)
            v = csr_matrix(chunk['cross_moments'], dtype=float)
            n = m.shape[0]
            if m.shape != (n, self.parameter_count) or v.shape != m.shape:
                raise ValueError('Complete, matching nuisance moment columns required')
            if not np.isfinite(m.data).all() or not np.isfinite(v.data).all():
                raise ValueError('Finite nuisance moments required')
            chunk.update(projection_factors=m, cross_moments=v)
            for name in ('lambda0', 'inner_gradient', 'inner_GN', 'outer_GN', 'outer_gradient'):
                value = np.asarray(chunk[name], dtype=float)
                if value.shape != (n,) or not np.isfinite(value).all():
                    raise ValueError('Finite per-point nuisance moments required: ' + name)
                chunk[name] = value
            if (np.any(chunk['lambda0'] < 0) or np.any(chunk['inner_GN'] <= 0)
                    or np.any(chunk['outer_GN'] < 0)):
                raise ValueError('Original interior depth and positive inner curvature required')
            self.chunks.append(chunk)
        self.chunks = tuple(self.chunks)
        self.point_count = sum(len(c['lambda0']) for c in self.chunks)
        self._last_consistent_active = None

    def _step(self, step):
        d = np.asarray(step, dtype=float)
        if d.shape != (self.parameter_count,) or not np.isfinite(d).all():
            raise ValueError('Finite complete camera displacement required')
        return d

    def active_set(self, step):
        d = self._step(step)
        return tuple(np.asarray(c['projection_factors'] @ d).ravel()
                     + c['inner_gradient'] / c['inner_GN'] > c['lambda0']
                     for c in self.chunks)

    def _activity(self, active):
        if len(active) != len(self.chunks):
            raise ValueError('Every original nuisance chunk must have an activity mask')
        result = []
        for c, supplied in zip(self.chunks, active):
            a = np.asarray(supplied)
            if a.dtype != np.dtype(bool) or a.shape != c['lambda0'].shape:
                raise ValueError('Boolean activity mask must match original points')
            result.append(a)
        return tuple(result)

    def correction(self, active):
        active = self._activity(active)
        h = csc_matrix((self.parameter_count, self.parameter_count), dtype=float)
        g = np.zeros(self.parameter_count)
        constant = 0.
        for c, mask in zip(self.chunks, active):
            if not np.any(mask):
                continue
            m = c['projection_factors'][mask]
            v = c['cross_moments'][mask]
            lam = c['lambda0'][mask]
            gn = c['outer_GN'][mask]
            jr = c['outer_gradient'][mask]
            mv = m.T @ v
            h = h + mv + mv.T + m.T @ m.multiply(gn[:, None])
            g += np.asarray(m.T @ (jr - lam * gn) - v.T @ lam).ravel()
            constant += float(-lam @ jr + .5 * (lam * lam) @ gn)
        return h.tocsc(), g, constant

    def correction_value(self, step):
        d = self._step(step)
        result = 0.
        for c, active in zip(self.chunks, self.active_set(d)):
            if not np.any(active):
                continue
            md = np.asarray(c['projection_factors'][active] @ d).ravel()
            vd = np.asarray(c['cross_moments'][active] @ d).ravel()
            shift = md - c['lambda0'][active]
            result += float(np.sum(shift * (c['outer_gradient'][active] + vd)
                                   + .5 * c['outer_GN'][active] * shift * shift))
        return result

    def model_reduction(self, baseH, baseg, step):
        d = self._step(step)
        return float(-np.asarray(baseg) @ d - .5 * d @ (baseH @ d)
                     - self.correction_value(d))

    def eliminate_fixed(self, indices):
        """Use the same exact zero displacement columns as the source normal."""
        supplied = np.asarray(indices)
        if (supplied.ndim != 1 or (supplied.size and supplied.dtype.kind not in 'iu')
                or len(np.unique(supplied)) != supplied.size):
            raise ValueError('Distinct one-dimensional integer fixed coordinates required')
        indices = supplied.astype(int)
        if np.any(indices < 0) or np.any(indices >= self.parameter_count):
            raise ValueError('Fixed coordinate outside complete source layout')
        self._last_consistent_active = None
        if len(indices):
            keep = np.ones(self.parameter_count)
            keep[indices] = 0.
            for c in self.chunks:
                c['projection_factors'] = c['projection_factors'].multiply(keep).tocsr()
                c['cross_moments'] = c['cross_moments'].multiply(keep).tocsr()
        return self

    def solve(self, baseH, baseg, diagonal, solve_linear, **options):
        """Resolve the same camera QP until its original depth bounds agree.

        Repeated activity is a numerical failure, never permission to return
        the earlier unconstrained-depth prediction. The actual source and
        physical trial acceptance remain outside this routine.
        """
        # This instance belongs to one frozen nuisance model. Reuse only the
        # last full-point-consistent working set, never a step or certificate.
        warmed = self._last_consistent_active is not None
        active = (tuple(a.copy() for a in self._last_consistent_active) if warmed
                  else tuple(np.zeros(len(c['lambda0']), dtype=bool) for c in self.chunks))
        initial_warm_start = warmed
        warm_cycle_cold_restart = False
        seen = set()
        counts = []
        while True:
            token = hashlib.sha256(b''.join(np.packbits(a).tobytes() for a in active)).digest()
            if token in seen:
                if warmed:
                    # A changed SOC target may make the warm route cycle. Give
                    # the original cold route its unchanged opportunity once.
                    active = tuple(np.zeros(len(c['lambda0']), dtype=bool) for c in self.chunks)
                    seen.clear()
                    warmed = False
                    warm_cycle_cold_restart = True
                    continue
                raise ValueError('Original nuisance lower-bound activity cycled; fresh trial model required')
            seen.add(token)
            dh, dg, constant = self.correction(active)
            result = solve_linear((baseH + dh).tocsc(), np.asarray(baseg) + dg,
                                  diagonal, **options)
            actual = self.active_set(result.step)
            counts.append(int(sum(np.count_nonzero(a) for a in active)))
            if all(np.array_equal(a, b) for a, b in zip(active, actual)):
                self._last_consistent_active = tuple(a.copy() for a in actual)
                result.report['bounded_motion_nuisance'] = dict(
                    original_interior_points=self.point_count,
                    active_lower_bound_points=counts[-1],
                    camera_QP_solves=len(counts), active_points_per_solve=counts,
                    initial_activity_warm_started=initial_warm_start,
                    warm_cycle_cold_restart=warm_cycle_cold_restart,
                    all_original_points_rechecked_after_every_camera_QP=True,
                    original_depth_activity_consistent=True,
                    source_model_reduction=self.model_reduction(baseH, baseg, result.step),
                    active_branch_constant=constant,
                    new_source_rows_or_priors=0,
                    inner_and_outer_metrics_kept_separate=True)
                return result
            active = actual
