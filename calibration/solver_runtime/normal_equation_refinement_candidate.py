"""Independent diagnostics for refinement of an unchanged full linear system.

This candidate has no installer. It adds no prior, ridge, source rows or
acceptance rule. A production native-pixel consistency gate still owns any
qualification; a small backward error is not a substitute for that gate.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, diags
from scipy.sparse.linalg import splu


def system_identity(matrix):
    value = csc_matrix(matrix, dtype=float, copy=True)
    value.sum_duplicates(); value.sort_indices()
    digest = hashlib.sha256()
    for array in (np.array(value.shape, dtype=np.int64), value.indptr, value.indices, value.data):
        digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()


def compensated_original_residual(matrix, rhs, step):
    """Compute b-Ax using fsum of the original float64 coefficient products.

    Products remain float64. fsum compensates only summation; this makes no
    extended-precision claim on Windows, where longdouble is also float64.
    """
    value = csr_matrix(matrix, dtype=float, copy=False)
    b, x = np.asarray(rhs, float), np.asarray(step, float)
    if value.shape != (len(b), len(b)) or x.shape != b.shape:
        raise ValueError('Same complete square equation and step required')
    if not np.isfinite(value.data).all() or not np.isfinite(b).all() or not np.isfinite(x).all():
        raise ValueError('Finite original equation and candidate required')
    residual = np.empty_like(b)
    for row in range(len(b)):
        lo, hi = value.indptr[row:row+2]
        products = value.data[lo:hi] * x[value.indices[lo:hi]]
        residual[row] = math.fsum([float(b[row]), *(-products).tolist()])
    return residual


def original_equation_error(matrix, rhs, step, residual=None):
    value = csr_matrix(matrix, dtype=float, copy=False)
    b, x = np.asarray(rhs, float), np.asarray(step, float)
    r = compensated_original_residual(value, b, x) if residual is None else np.asarray(residual, float)
    magnitude = abs(value) @ abs(x) + abs(b)
    componentwise = np.divide(abs(r), magnitude, out=np.zeros_like(r), where=magnitude != 0)
    componentwise[(magnitude == 0) & (r != 0)] = np.inf
    diagonal_scale = np.sqrt(np.maximum(abs(value.diagonal()), np.finfo(float).tiny))
    return dict(residual_maximum_absolute=float(abs(r).max()),
        residual_relative_2_norm=float(np.linalg.norm(r))/max(1., float(np.linalg.norm(b))),
        maximum_componentwise_backward_error=float(componentwise.max()),
        original_diagonal_weighted_residual_2_norm=float(np.linalg.norm(r / diagonal_scale)))


class SameEquationLU:
    """A reversible coordinate scaling of the explicitly supplied equation."""
    def __init__(self, matrix, *, equilibrate=False):
        self.matrix = csc_matrix(matrix, dtype=float, copy=True)
        self.matrix.sum_duplicates(); self.matrix.sort_indices()
        if self.matrix.shape[0] != self.matrix.shape[1] or not np.isfinite(self.matrix.data).all():
            raise ValueError('Finite complete square equation required')
        self.identity = system_identity(self.matrix)
        diagonal = self.matrix.diagonal()
        if equilibrate and np.any(diagonal <= 0):
            raise ValueError('Positive original diagonal required for reversible scaling')
        self.coordinate_scale = 1. / np.sqrt(diagonal) if equilibrate else np.ones(len(diagonal))
        scaled = (diags(self.coordinate_scale) @ self.matrix @ diags(self.coordinate_scale)).tocsc()
        self.factor = splu(scaled, permc_spec='MMD_AT_PLUS_A')
        self.equilibrated = bool(equilibrate)

    def solve(self, rhs):
        b = np.asarray(rhs, float)
        if b.shape != (self.matrix.shape[0],) or not np.isfinite(b).all():
            raise ValueError('Same complete finite original right-hand side required')
        return self.coordinate_scale * self.factor.solve(self.coordinate_scale * b)


class SameEquationSchur:
    """Independent view elimination for the same supplied full equation.

    The original reduced-system symmetrization is retained. Corrections are
    independently backsolved through this Schur factor, rather than forcing
    both candidate routes to agree by sharing the full sparse LU factor.
    This unconstrained diagnostic corresponds only to the undamped reference.
    """
    def __init__(self, matrix, global_parameter_indices, *, equilibrate=False):
        self.matrix = csc_matrix(matrix, dtype=float, copy=True)
        self.matrix.sum_duplicates(); self.matrix.sort_indices()
        self.identity = system_identity(self.matrix)
        n = self.matrix.shape[0]
        selected = np.asarray(global_parameter_indices)
        if (self.matrix.shape != (n, n) or selected.ndim != 1
                or selected.dtype.kind not in 'iu' or not 0 < len(selected) < n
                or len(np.unique(selected)) != len(selected)
                or np.any(selected < 0) or np.any(selected >= n)):
            raise ValueError('Original distinct shared indices and complete square equation required')
        diagonal = self.matrix.diagonal()
        if equilibrate and np.any(diagonal <= 0):
            raise ValueError('Positive original diagonal required for reversible scaling')
        self.coordinate_scale = 1. / np.sqrt(diagonal) if equilibrate else np.ones(n)
        scaled = (diags(self.coordinate_scale) @ self.matrix @ diags(self.coordinate_scale)).tocsc()
        mask = np.ones(n, bool); mask[selected] = False
        self.permutation = np.r_[selected, np.flatnonzero(mask)]
        self.global_count = len(selected)
        ordered = scaled[self.permutation, :][:, self.permutation]
        ng = self.global_count
        a = ordered[:ng, :ng].toarray()
        self.b = ordered[:ng, ng:]
        self.view_factor = splu(ordered[ng:, ng:], permc_spec='MMD_AT_PLUS_A')
        self.db = self.view_factor.solve(self.b.T.toarray())
        raw = a - self.b @ self.db
        self.reduced_maximum_asymmetry = float(abs(raw-raw.T).max())
        self.reduced = .5 * (raw + raw.T)
        self.equilibrated = bool(equilibrate)

    def solve(self, rhs):
        b = np.asarray(rhs, float)
        if b.shape != (self.matrix.shape[0],) or not np.isfinite(b).all():
            raise ValueError('Same complete finite original right-hand side required')
        ordered = (b * self.coordinate_scale)[self.permutation]
        ng = self.global_count
        dv0 = self.view_factor.solve(ordered[ng:])
        dg = np.linalg.solve(self.reduced, ordered[:ng] - self.b @ dv0)
        dv = dv0 - self.db @ dg
        step = np.empty_like(b)
        step[self.permutation] = np.r_[dg, dv]
        return self.coordinate_scale * step


def refine_same_equation(matrix, rhs, candidate, correction_solver):
    """Polish x using A*dx=b-A*x without changing A or b.

    Stop when the original diagonally weighted residual stops decreasing, the
    floating step stops changing, or a state repeats. No iteration/time cap or
    native-pixel gate is introduced; this remains an offline candidate.
    """
    value = csr_matrix(matrix, dtype=float, copy=True)
    b, x = np.asarray(rhs, float).copy(), np.asarray(candidate, float).copy()
    identity = system_identity(value)
    rhs_identity = hashlib.sha256(b.tobytes()).hexdigest()
    initial = compensated_original_residual(value, b, x)
    history = [dict(iteration=0, **original_equation_error(value, b, x, initial))]
    residual = initial
    seen = {hashlib.sha256(x.tobytes()).hexdigest()}
    reason = 'exact_original_float64_product_residual_zero'
    while np.any(residual):
        correction = np.asarray(correction_solver(residual), float)
        if correction.shape != x.shape or not np.isfinite(correction).all():
            raise ValueError('Finite complete same-equation correction required')
        updated = x + correction
        if np.array_equal(updated, x):
            reason = 'floating_step_stagnation'; break
        token = hashlib.sha256(updated.tobytes()).hexdigest()
        if token in seen:
            reason = 'repeated_float64_step'; break
        updated_residual = compensated_original_residual(value, b, updated)
        error = original_equation_error(value, b, updated, updated_residual)
        if error['original_diagonal_weighted_residual_2_norm'] >= history[-1]['original_diagonal_weighted_residual_2_norm']:
            reason = 'original_equation_residual_did_not_decrease'; break
        x, residual = updated, updated_residual
        seen.add(token)
        history.append(dict(iteration=len(history), **error))
    if system_identity(value) != identity or hashlib.sha256(b.tobytes()).hexdigest() != rhs_identity:
        raise ValueError('Refinement changed the supplied original equation')
    return x, dict(schema='rinkspace.unchanged-full-normal-equation-refinement-candidate.v1',
        equation_identity=identity, rhs_identity=rhs_identity, history=history,
        accepted_corrections=len(history)-1, termination_reason=reason,
        original_equation_and_rhs_unchanged=True,
        added_ridge_prior_or_source_rows=0,
        residual_arithmetic='float64 coefficient products with math.fsum compensated summation',
        native_pixel_consistency_or_accuracy_qualification_claimed=False,
        production_installer_present=False)
