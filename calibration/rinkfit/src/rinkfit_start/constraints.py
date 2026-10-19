"""v1.1 reference constraint mathematics and full-row native budget audit.

This module is an array reference, not a large-scene optimizer.  Production
optimization is C++/Ceres/SuiteSparse with an external PHR outer loop.  E is a
frozen pixel acceptance region; statistical sigma/W never enters this audit.
Identity/key validation and immutable complete-state persistence belong to the
observation/runner layers.  No row is removed, reweighted, or relabelled here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


def _vector(value, name: str, size: Optional[int] = None) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 1 or (size is not None and a.size != size):
        raise ValueError(f"{name} must be a vector of length {size}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains a non-finite value")
    return a


def _rho(rho: float) -> float:
    rho = float(rho)
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be positive and finite")
    return rho


def _readonly(value, dtype=None) -> np.ndarray:
    a = np.array(value, dtype=dtype, copy=True)
    a.flags.writeable = False
    return a


def phr_inequality_residual(g, multipliers, rho: float) -> np.ndarray:
    """max(0, lambda + rho*g)/sqrt(rho), for the convention g <= 0."""
    g = _vector(g, "g")
    lam = _vector(multipliers, "inequality multipliers", g.size)
    if np.any(lam < 0):
        raise ValueError("inequality multipliers must be nonnegative")
    rho = _rho(rho)
    return np.maximum(0.0, lam + rho * g) / math.sqrt(rho)


def phr_equality_residual(h, multipliers, rho: float) -> np.ndarray:
    """sqrt(rho)*(h + nu/rho); never apply a robust loss to this residual."""
    h = _vector(h, "h")
    nu = _vector(multipliers, "equality multipliers", h.size)
    rho = _rho(rho)
    return math.sqrt(rho) * (h + nu / rho)


def phase_one_budget_residual(b: float) -> float:
    """Ceres half-square is exactly b + 1/2; enforce the closed bound b >= 0."""
    b = float(b)
    if not math.isfinite(b) or b < 0:
        raise ValueError("budget_sq must be finite and nonnegative")
    return math.sqrt(2.0 * b + 1.0)


@dataclass(frozen=True)
class PHRUpdate:
    inequality_multipliers: np.ndarray
    equality_multipliers: np.ndarray
    violation: float
    rho: float
    penalty_limit_reached: bool


def phr_update(g, h, inequality_multipliers, equality_multipliers, rho: float,
               previous_violation: float = math.inf, *, rho_max: float = 1e6,
               improvement_ratio: float = 0.5, penalty_factor: float = 5.0) -> PHRUpdate:
    """One outer update, only after declared inner accuracy is handled.

    The progress test uses *old* lambda and rho.  Hitting the rho cap reports a
    numerical limit; it is neither a successful solve nor an infeasibility proof.
    The caller must perform fresh primal/equality/KKT checks independently.
    """
    g = _vector(g, "g")
    h = _vector(h, "h")
    lam = _vector(inequality_multipliers, "inequality multipliers", g.size)
    nu = _vector(equality_multipliers, "equality multipliers", h.size)
    if np.any(lam < 0):
        raise ValueError("inequality multipliers must be nonnegative")
    rho = _rho(rho)
    rho_max = _rho(rho_max)
    if rho > rho_max:
        raise ValueError("rho already exceeds the frozen cap")
    if not 0 < improvement_ratio < 1 or not math.isfinite(penalty_factor) or penalty_factor <= 1:
        raise ValueError("invalid frozen progress/penalty settings")
    if math.isnan(previous_violation) or previous_violation < 0:
        raise ValueError("previous_violation must be nonnegative or +infinity")
    ineq_progress = float(np.max(np.abs(np.maximum(g, -lam / rho)), initial=0.0))
    eq_progress = float(np.max(np.abs(h), initial=0.0))
    violation = max(ineq_progress, eq_progress)
    next_lam = np.maximum(0.0, lam + rho * g)
    next_nu = nu + rho * h
    grow = violation > improvement_ratio * previous_violation
    limited = grow and rho * penalty_factor > rho_max
    next_rho = rho if limited or not grow else rho * penalty_factor
    return PHRUpdate(_readonly(next_lam), _readonly(next_nu), violation,
                     next_rho, limited)


def scaled_projected_gradient(x, gradient, variable_scale, lower, upper) -> np.ndarray:
    """Euclidean box projected gradient in z=x/scale coordinates.

    Returns z - clip(z - dF/dz, lower/scale, upper/scale).  For manifold blocks
    production adapters must use their tangent/retraction and true active set;
    this Euclidean helper is not a replacement for a manifold KKT derivation.
    """
    x = _vector(x, "x")
    grad = _vector(gradient, "gradient", x.size)
    scale = _vector(variable_scale, "variable_scale", x.size)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if lo.shape != x.shape or hi.shape != x.shape or np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
        raise ValueError("bounds must match x and contain no NaN")
    if np.any(scale <= 0) or np.any(lo > hi) or np.any(x < lo) or np.any(x > hi):
        raise ValueError("invalid scales, bounds, or out-of-bounds current state")
    z = x / scale
    return z - np.clip(z - scale * grad, lo / scale, hi / scale)


def phase_one_budget_stationarity(b: float, pixel_multipliers) -> float:
    """Projected budget KKT residual, including b's lower-bound normal cone."""
    phase_one_budget_residual(b)
    lam = _vector(pixel_multipliers, "pixel multipliers")
    if np.any(lam < 0):
        raise ValueError("pixel multipliers must be nonnegative")
    gradient = 1.0 - float(np.sum(lam))
    return abs(float(b) - max(0.0, float(b) - gradient))


def verified_order_inequalities(stations, ordered_pairs, *, scale: float = 1.0) -> np.ndarray:
    """s_left - s_right <= 0 for explicitly verified pairs; equality is legal.

    No pairs are inferred from array order, repeated IDs, or crop endpoints.
    Scale is frozen independently of pixel budget_sq.
    """
    s = _vector(stations, "stations")
    pairs = np.asarray(ordered_pairs)
    if pairs.size == 0:
        return np.empty(0, dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer):
        raise ValueError("ordered_pairs must be an integer array of shape (K, 2)")
    if np.any(pairs < 0) or np.any(pairs >= s.size):
        raise ValueError("order pair index outside station array")
    scale = _rho(scale)
    return (s[pairs[:, 0]] - s[pairs[:, 1]]) / scale


def ordinary_budget_q(raw_error_uv, E) -> np.ndarray:
    """e.T @ inverse(E) @ e, with E fixed SPD in native pixel squared units."""
    e = np.asarray(raw_error_uv, dtype=np.float64)
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError("raw_error_uv must have shape (N, 2)")
    ellipse = np.asarray(E, dtype=np.float64)
    if ellipse.shape == (2, 2):
        ellipse = np.broadcast_to(ellipse, (len(e), 2, 2))
    if ellipse.shape != (len(e), 2, 2) or not np.all(np.isfinite(ellipse)):
        raise ValueError("E must be a finite (2, 2) or (N, 2, 2) frozen ellipse")
    if not np.allclose(ellipse, np.swapaxes(ellipse, 1, 2), rtol=0, atol=1e-12):
        raise ValueError("E must be symmetric")
    try:
        chol = np.linalg.cholesky(ellipse)
    except np.linalg.LinAlgError as exc:
        raise ValueError("E must be positive definite") from exc
    finite = np.all(np.isfinite(e), axis=1)
    q = np.full(len(e), np.inf)
    if np.any(finite):
        whitened = np.linalg.solve(chol[finite], e[finite, :, None])[:, :, 0]
        q[finite] = np.einsum("ni,ni->n", whitened, whitened)
    return q


def profile_budget_q(native_normal_error_px, radius_px) -> np.ndarray:
    """A genuine profile's budget uses native e_n in px, never e_n/sigma_n."""
    e = np.asarray(native_normal_error_px, dtype=np.float64)
    radius = np.asarray(radius_px, dtype=np.float64)
    if e.ndim != 1:
        raise ValueError("native_normal_error_px must be one-dimensional")
    radius = np.broadcast_to(radius, e.shape)
    if not np.all(np.isfinite(radius)) or np.any(radius <= 0):
        raise ValueError("profile radii must be positive and finite")
    with np.errstate(over="ignore", invalid="ignore"):
        q = (e / radius) ** 2
    return np.where(np.isfinite(q), q, np.inf)


@dataclass(frozen=True)
class NativeAudit:
    raw_error_uv: np.ndarray
    q: np.ndarray
    native_error_px: np.ndarray
    query_error_px: np.ndarray
    projection_valid: np.ndarray
    support_valid: np.ndarray
    root_valid: np.ndarray
    invalid_rows: np.ndarray
    over_budget_rows: np.ndarray
    expected_rows: int
    accounted_rows: int
    t_hat: float
    pixel_pass: bool
    support_pass: bool
    physical_pass: bool

    @property
    def passed(self) -> bool:
        return (self.accounted_rows == self.expected_rows and self.pixel_pass
                and self.support_pass and self.physical_pass)


def audit_native_budget(source_uv, predicted_uv, E, *, expected_rows: int,
                        projection_valid, support_valid, root_valid,
                        physical_valid: bool, profile_mask=None,
                        profile_normal_uv=None, profile_radius_px=None,
                        profile_alpha_bounds_px=None, profile_witness_alpha_px=None,
                        q_tolerance: float = 1e-4,
                        equality_tolerance_px: float = 0.01) -> NativeAudit:
    """Audit *all* native predictions with explicit projection/support statuses.

    Required Boolean status arrays prevent a caller from silently scoring only
    successful projections.  A failed projection or implicit-root replay gets
    q=+inf and remains in the output.  Finite material/order support failures
    remain visible even if their image error is small.  For genuine profiles,
    normals, original query extents and scalar budgets are mandatory; ordinary
    curves always use the full 2D error.  If a lifted alpha witness is supplied,
    it must agree with the native prediction, in addition to true-root validity.

    Callers must verify the original composite-key multiset separately, and
    must independently recompute predictions from a complete frozen state.
    This function accepts neither a fitted common-mode correction nor weights.
    """
    uv = np.asarray(source_uv, dtype=np.float64)
    pred = np.asarray(predicted_uv, dtype=np.float64)
    if (isinstance(expected_rows, bool) or not isinstance(expected_rows, (int, np.integer))
            or expected_rows <= 0):
        raise ValueError("expected_rows must be an explicit positive source cardinality")
    n = int(expected_rows)
    if uv.shape != (n, 2) or pred.shape != (n, 2):
        raise ValueError(f"incomplete native replay: expected {n} rows, got {uv.shape}/{pred.shape}")
    if not np.all(np.isfinite(uv)):
        raise ValueError("original source UV must be finite")
    if (not math.isfinite(q_tolerance) or q_tolerance < 0
            or not math.isfinite(equality_tolerance_px) or equality_tolerance_px < 0):
        raise ValueError("arithmetic tolerances must be finite and nonnegative")

    def flags(value, name):
        a = np.asarray(value)
        if a.shape != (n,) or a.dtype != np.bool_:
            raise ValueError(f"{name} must be an explicit Boolean array of length {n}")
        return a.copy()

    valid = flags(projection_valid, "projection_valid") & np.all(np.isfinite(pred), axis=1)
    support = flags(support_valid, "support_valid")
    roots = flags(root_valid, "root_valid")
    profiles = np.zeros(n, dtype=bool) if profile_mask is None else flags(profile_mask, "profile_mask")
    e = pred - uv
    q = ordinary_budget_q(e, E)
    with np.errstate(over="ignore", invalid="ignore"):
        native_error = np.linalg.norm(e, axis=1)
    query_error = np.zeros(n)
    if np.any(profiles):
        if any(x is None for x in (profile_normal_uv, profile_radius_px, profile_alpha_bounds_px)):
            raise ValueError("true profiles require frozen normal, scalar radius and original query extent")
        normal = np.asarray(profile_normal_uv, dtype=np.float64)
        radius = np.asarray(profile_radius_px, dtype=np.float64)
        bounds = np.asarray(profile_alpha_bounds_px, dtype=np.float64)
        if normal.shape != (n, 2) or radius.shape != (n,) or bounds.shape != (n, 2):
            raise ValueError("profile metadata must retain full original-row indexing")
        normal_p = normal[profiles]
        bounds_p = bounds[profiles]
        if (not np.all(np.isfinite(normal_p))
                or not np.allclose(np.sum(normal_p ** 2, axis=1), 1.0, atol=1e-10, rtol=0)):
            raise ValueError("profile normals must already be finite unit normals")
        if not np.all(np.isfinite(bounds_p)) or np.any(bounds_p[:, 0] > bounds_p[:, 1]):
            raise ValueError("profile query extent must be a finite closed interval")
        ep = e[profiles]
        en = np.einsum("ni,ni->n", ep, normal_p)
        query = np.linalg.norm(ep - en[:, None] * normal_p, axis=1)
        support_p = ((en >= bounds_p[:, 0] - equality_tolerance_px)
                     & (en <= bounds_p[:, 1] + equality_tolerance_px)
                     & np.isfinite(query) & (query <= equality_tolerance_px))
        if profile_witness_alpha_px is not None:
            alpha = np.asarray(profile_witness_alpha_px, dtype=np.float64)
            if alpha.shape != (n,):
                raise ValueError("profile witness must retain full original-row indexing")
            witness_error = np.linalg.norm(ep - alpha[profiles, None] * normal_p, axis=1)
            query = np.maximum(query, witness_error)
            support_p &= (np.isfinite(alpha[profiles]) & np.isfinite(witness_error)
                          & (witness_error <= equality_tolerance_px)
                          & (alpha[profiles] >= bounds_p[:, 0] - equality_tolerance_px)
                          & (alpha[profiles] <= bounds_p[:, 1] + equality_tolerance_px))
        q[profiles] = profile_budget_q(en, radius[profiles])
        native_error[profiles] = np.abs(en)
        query_error[profiles] = query
        support[profiles] &= support_p
    q[~valid | ~roots | ~np.isfinite(q)] = np.inf
    native_error[~valid | ~roots | ~np.isfinite(native_error)] = np.inf
    invalid = ~valid | ~roots | ~support
    over = q > 1.0 + q_tolerance
    t_hat = math.sqrt(float(np.max(q)))
    return NativeAudit(
        *(_readonly(a) for a in (e, q, native_error, query_error, valid, support, roots,
                                np.flatnonzero(invalid), np.flatnonzero(over))),
        expected_rows=n, accounted_rows=n, t_hat=t_hat,
        pixel_pass=bool(np.all(~over)), support_pass=bool(np.all(support & roots & valid)),
        physical_pass=bool(physical_valid),
    )


# Names are intentionally explicit; do not alias a statistical SSE to coverage.
__all__ = ["PHRUpdate", "NativeAudit", "phr_inequality_residual", "phr_equality_residual",
           "phr_update", "phase_one_budget_residual", "phase_one_budget_stationarity",
           "scaled_projected_gradient", "verified_order_inequalities", "ordinary_budget_q",
           "profile_budget_q", "audit_native_budget"]
