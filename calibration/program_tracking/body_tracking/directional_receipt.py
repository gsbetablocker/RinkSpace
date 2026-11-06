"""Read-only, per-factor movement accounting and a source-guided linear probe.

Use the *actual weighted* residuals and Jacobians from one common state. This
module does not choose physical weights, modify observations, start a nonlinear
solve, change contact modes, or publish a fitted candidate. Parameter scales and
step sizes are explicit caller choices, not metric accuracy thresholds.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr


@dataclass(frozen=True)
class Block:
    name: str
    residual: np.ndarray
    jacobian: Any
    association: str = ""

    def checked(self, nparameters: int | None = None) -> "Block":
        r = np.asarray(self.residual, dtype=float)
        J = self.jacobian
        if not self.name or r.ndim != 1 or not np.isfinite(r).all():
            raise ValueError("Named finite 1-D residual required")
        if len(J.shape) != 2 or J.shape[0] != len(r):
            raise ValueError(f"Residual/Jacobian layout mismatch: {self.name}")
        if nparameters is not None and J.shape[1] != nparameters:
            raise ValueError(f"Parameter layout mismatch: {self.name}")
        values = J.data if sparse.issparse(J) else np.asarray(J)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite Jacobian: {self.name}")
        return Block(self.name, r, J, self.association)


def _blocks(blocks: Sequence[Block]) -> list[Block]:
    if not blocks:
        raise ValueError("At least one real residual block is required")
    n = blocks[0].jacobian.shape[1]
    checked = [b.checked(n) for b in blocks]
    if len({b.name for b in checked}) != len(checked):
        raise ValueError("Block names must be unique")
    return checked


def receipt(blocks: Sequence[Block], direction: np.ndarray) -> dict:
    """Positive g_k^T d opposes the requested signed movement, locally.

    A large cost is NOT evidence of a large opposing force. The same direction,
    in the same archived parameter coordinates, is used for every block.
    """
    blocks = _blocks(blocks)
    d = np.asarray(direction, dtype=float)
    if d.shape != (blocks[0].jacobian.shape[1],) or not np.isfinite(d).all():
        raise ValueError("Direction must match the exact parameter layout")
    if not np.any(d):
        raise ValueError("Zero direction cannot diagnose a movement")
    rows = []
    for b in blocks:
        jd = np.asarray(b.jacobian @ d).reshape(-1)
        slope = float(b.residual @ jd)
        rows.append(dict(name=b.name, rows=len(b.residual),
            cost=float(.5 * b.residual @ b.residual),
            directional_slope=slope, directional_sensitivity=float(np.linalg.norm(jd)),
            gauss_newton_curvature=float(jd @ jd),
            response="opposes" if slope > 0 else "supports" if slope < 0 else "first_order_neutral",
            association=b.association))
    return dict(schema="rinkspace.audit.directional-receipt.v1", blocks=rows,
        cost=sum(r["cost"] for r in rows),
        directional_slope=sum(r["directional_slope"] for r in rows),
        gauss_newton_curvature=sum(r["gauss_newton_curvature"] for r in rows),
        direction_is_not_normalized=True, conclusion="local diagnostic, not a fitted result")


def source_guided_direction(blocks: Sequence[Block], source_names: Sequence[str], *,
                            parameter_scales: np.ndarray, damping: float,
                            active_columns: np.ndarray | None = None,
                            maxiter: int = 300) -> tuple[np.ndarray, dict]:
    """A reversible linearized source-only move, regularized in scaled units.

    Solves min ||r_s + J_s D z||^2 + damping^2 ||z||^2, with explicit active
    columns. It does NOT accept the move. Evaluate its effect on EVERY factor
    with probe() and native projection before any initialization is changed.
    """
    blocks = _blocks(blocks)
    n = blocks[0].jacobian.shape[1]
    scales = np.asarray(parameter_scales, dtype=float)
    if scales.shape != (n,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Declare positive per-parameter scales, including angle/length units")
    if not np.isfinite(damping) or damping <= 0 or maxiter < 1:
        raise ValueError("Finite positive damping and bounded iterations required")
    names = set(source_names)
    if not names or not names.issubset({b.name for b in blocks}):
        raise ValueError("Source names must reference actual assembled blocks")
    selected = [b for b in blocks if b.name in names]
    ids = np.arange(n) if active_columns is None else np.asarray(active_columns)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids) or np.any(ids < 0) or np.any(ids >= n) or len(np.unique(ids)) != len(ids):
        raise ValueError("Unique active parameter indices required")
    J = sparse.vstack([sparse.csr_matrix(b.jacobian)[:, ids] for b in selected], format="csr")
    r = np.concatenate([b.residual for b in selected])
    A = J @ sparse.diags(scales[ids], format="csr")
    result = lsmr(A, -r, damp=float(damping), maxiter=int(maxiter), atol=1e-9, btol=1e-9)
    d = np.zeros(n); d[ids] = scales[ids] * result[0]
    return d, dict(source_blocks=sorted(names), active_columns=ids.tolist(), damping=float(damping),
        lsmr_stop=int(result[1]), iterations=int(result[2]),
        original_source_cost=float(.5 * r @ r),
        predicted_source_cost=float(.5 * np.linalg.norm(r + J @ d[ids])**2),
        accepted=False, not_a_nonlinear_fit=True)


def probe(blocks: Sequence[Block], x: np.ndarray, direction: np.ndarray, *,
          evaluate: Callable[[np.ndarray], Mapping[str, Mapping[str, Any]]],
          alphas: Sequence[float], derivative_steps: Sequence[float] = (1e-4, 1e-5, 1e-6)) -> dict:
    """Evaluate real per-block costs and central differences; restore baseline cache.

    evaluate returns {name: {'residual': weighted_r, 'association': stable_id}}.
    Rows must retain identity/order. A changed nearest-edge/contact association
    is reported and prevents interpreting its FD discrepancy as a smooth bug.
    The callback must not write candidates or mutate evidence/AL multipliers.
    """
    bs = _blocks(blocks); x = np.asarray(x, dtype=float).copy()
    d = np.asarray(direction, dtype=float)
    report = receipt(bs, d)
    if x.shape != d.shape or not np.isfinite(x).all():
        raise ValueError("State and direction must have identical finite shape")
    steps = np.asarray(derivative_steps, dtype=float)
    aa = np.asarray(alphas, dtype=float)
    if steps.ndim != 1 or not len(steps) or np.any(steps <= 0) or not np.isfinite(steps).all() or aa.ndim != 1 or not np.isfinite(aa).all():
        raise ValueError("Finite alphas and positive FD step sizes required")
    def call(at):
        raw = evaluate(at.copy())
        if set(raw) != {b.name for b in bs}:
            raise ValueError("Real objective block set changed during probe")
        out = {}
        for b in bs:
            r = np.asarray(raw[b.name]["residual"], dtype=float).copy()
            if r.shape != b.residual.shape or not np.isfinite(r).all():
                raise ValueError("Residual identity/finite-domain changed: " + b.name)
            out[b.name] = dict(residual=r, association=str(raw[b.name].get("association", "")))
        return out
    try:
        base = call(x)
        for b in bs:
            if not np.allclose(base[b.name]["residual"], b.residual, rtol=1e-10, atol=1e-12):
                raise ValueError("Cached Jacobian and baseline residual do not describe the same state")
        report["trials"] = []
        for alpha in aa:
            try:
                values = call(x + float(alpha)*d)
                rows = []
                for b in bs:
                    r = values[b.name]["residual"]
                    cost = float(.5 * r @ r)
                    rows.append(dict(name=b.name, cost=cost, delta_cost=cost-float(.5*b.residual@b.residual),
                        association_changed=values[b.name]["association"] != base[b.name]["association"]))
                report["trials"].append(dict(alpha=float(alpha), valid=True, blocks=rows,
                    total_cost=sum(z["cost"] for z in rows)))
            except ValueError as e:
                report["trials"].append(dict(alpha=float(alpha), valid=False, error=str(e)))
        report["directional_derivative_checks"] = []
        for h in steps:
            plus, minus = call(x+h*d), call(x-h*d)
            for b in bs:
                fd = (plus[b.name]["residual"] - minus[b.name]["residual"])/(2*h)
                jd = np.asarray(b.jacobian @ d).ravel()
                changed = any(values[b.name]["association"] != base[b.name]["association"] for values in (plus, minus))
                report["directional_derivative_checks"].append(dict(name=b.name, step=float(h),
                    fd_norm=float(np.linalg.norm(fd)), predicted_norm=float(np.linalg.norm(jd)),
                    relative_error=float(np.linalg.norm(fd-jd)/max(np.linalg.norm(fd), np.linalg.norm(jd), 1e-30)),
                    maximum_absolute_error=float(np.max(np.abs(fd-jd), initial=0.)),
                    association_changed=changed))
    finally:
        evaluate(x.copy())  # Restore the real evaluate() cache to its baseline.
    return report
