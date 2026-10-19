"""Numerical view-only correction on the unchanged source normal.

This helper never evaluates or accepts a source state. The caller must retain
all original nonlinear guards, frozen profiles and full source-cost acceptance.
Only columns outside the explicitly declared shared/guard chart are corrected.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class ConditionalViewBacktracking:
    def __init__(self, normal, gradient, source_diagonal, full_step, *,
                 shared_parameter_indices, damping, schur_view_lm_ratio=0.,
                 protected_parameter_indices=(), fixed_parameter_indices=()):
        begin = time.perf_counter()
        normal = normal.tocsc(copy=False)
        gradient = np.asarray(gradient, float)
        diagonal = np.asarray(source_diagonal, float)
        self.full_step = np.asarray(full_step, float).copy()
        count = normal.shape[0]
        shared = np.asarray(shared_parameter_indices, int).reshape(-1)
        protected = np.asarray(protected_parameter_indices, int).reshape(-1)
        fixed = np.asarray(fixed_parameter_indices, int).reshape(-1)
        if (normal.shape != (count, count) or gradient.shape != (count,) or
                diagonal.shape != (count,) or self.full_step.shape != (count,) or
                not np.isfinite(gradient).all() or not np.isfinite(diagonal).all() or
                not np.isfinite(self.full_step).all() or np.any(diagonal <= 0.) or
                len(np.unique(shared)) != len(shared) or np.any(shared < 0) or
                np.any(shared >= count)):
            raise ValueError('Original finite source normal and distinct explicit shared columns required')
        if np.any(protected < 0) or np.any(protected >= count) or not np.isin(protected, shared).all():
            raise ValueError('Every nonlinear truth/guard camera dependency must be in the unchanged shared chart')
        if np.any(fixed < 0) or np.any(fixed >= count):
            raise ValueError('Original fixed source coordinates must be in range')
        ratio = float(schur_view_lm_ratio)
        damping = float(damping)
        if ratio not in (0., 1.) or not np.isfinite(damping) or damping < 0:
            raise ValueError('Existing explicit view LM ratio and damping required')
        is_view = np.ones(count, bool)
        is_view[shared] = False
        self.shared = shared.copy()
        self.views = np.flatnonzero(is_view)
        # Identical view matrix used by the existing global Schur helper:
        # original H + original 1e-8 numerical ridge + explicit view-only LM.
        view_matrix = normal[self.views][:, self.views] + diags(
            (1e-8 + damping * ratio) * diagonal[self.views], format='csc')
        correction = spsolve(view_matrix, -gradient[self.views], permc_spec='MMD_AT_PLUS_A')
        if not np.isfinite(correction).all():
            raise RuntimeError('Nonfinite original-system conditional view step')
        self.zero_shared_step = np.zeros(count)
        self.zero_shared_step[self.views] = correction
        if len(fixed) and (np.any(self.full_step[fixed] != 0.) or
                           np.any(self.zero_shared_step[fixed] != 0.)):
            raise ValueError('Conditional view solve moved an original fixed source coordinate')
        residual = view_matrix @ correction + gradient[self.views]
        self.report = dict(
            schema='rinkspace.original-source-conditional-view-backtracking.v1',
            extra_view_solve_seconds=time.perf_counter()-begin,
            shared_parameter_count=len(shared), view_parameter_count=len(self.views),
            protected_parameter_count=len(protected),
            conditional_view_stationarity_max_absolute=float(np.max(abs(residual))),
            original_source_H_g_profiles_rows_masks_sigma_and_DOF_unchanged=True,
            original_numerical_ridge_and_view_LM_unchanged=True,
            nonlinear_guard_and_actual_full_source_cost_required=True,
            source_state_accepted=False)

    def step(self, alpha):
        alpha = float(alpha)
        if not np.isfinite(alpha) or not 0. <= alpha <= 1.:
            raise ValueError('Numerical shared fraction in [0, 1] required; zero is a diagnostic anchor')
        if alpha == 1.:
            return self.full_step.copy()
        # Exact conditional backsolve for the selected shared displacement:
        # v(alpha) = -A^-1(g_v + B * alpha*u_full).
        value = alpha*self.full_step + (1.-alpha)*self.zero_shared_step
        value[self.shared] = alpha*self.full_step[self.shared]
        return value
