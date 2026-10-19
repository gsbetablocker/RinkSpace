"""Verify a restricted Clarabel infeasibility ray before repeating a full QP.

The numerical ray is extended by exact zero multipliers on every omitted
original row.  This changes neither an objective nor a constraint.  A failed
certificate leaves the existing complete diagonal/affine IPM routes intact.
The candidate installer changes only the live function; registered producer
files and their signed manifests are never rewritten here.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np


class CertifiedSharedQPInfeasible(RuntimeError):
    """Preserve the caller's existing infeasible-SOC/backtracking behavior."""

    def __init__(self, report):
        self.certificate_report = report
        super().__init__(
            "Clarabel shared half-space QP did not return a solved numerical "
            "state: PrimalInfeasible; verified original-row restricted certificate")


def verify_restricted_infeasibility(result, selected, rows, targets, scale,
                                    affine_rows, affine_targets, row_scale,
                                    tolerance, settings):
    """Return a conservative original-unit numerical Farkas check.

    For original inequalities rows @ x >= targets, lambda >= 0,
    rows.T @ lambda = 0, targets @ lambda > 0 is the infeasibility ray.
    Clarabel solves -affine_rows @ y <= -affine_targets, so its z has the
    correct sign and lambda = z / scale under the reversible row transform.
    Both original and affine contractions are checked at the *existing*
    infeasibility precision; the RHS must also exceed the original allowed
    primal slack.  This is a finite-precision numerical certificate, never a
    claim of exact symbolic arithmetic.
    """
    begun = time.perf_counter()
    report = dict(
        schema="rinkspace.original-row-restricted-QP-infeasibility-ray.v1",
        accepted=False, status=str(result.status), selected_row_count=len(selected),
        complete_original_row_count=len(rows),
        omitted_original_rows_have_exact_zero_multiplier=True,
        original_H_g_constraints_bounds_and_feasible_KKT_gate_unchanged=True,
        finite_precision_numerical_certificate=True)
    if str(result.status) != "PrimalInfeasible":
        report["reason"] = "not_strict_primal_infeasible_status"
        return report, None
    selected = np.asarray(selected, int)
    z = np.asarray(result.z, float)
    if (z.shape != selected.shape or not np.isfinite(z).all()
            or np.any(z < 0.) or not np.any(z > 0.)):
        report["reason"] = "invalid_nonnegative_ray"
        return report, None
    original_dual = np.zeros(len(rows), float)
    original_dual[selected] = z / scale[selected]
    mass = float(np.sum(original_dual))
    if not np.isfinite(mass) or mass <= 0.:
        report["reason"] = "invalid_original_ray_mass"
        return report, None
    original_dual /= mass
    # Contract every original column.  Omitted rows are exactly zero; no
    # all-row square Gram or additional solve is needed for this check.
    ray = original_dual[selected]
    original_combined_normal = rows[selected].T @ ray
    original_normal_magnitude = np.abs(rows[selected]).T @ ray
    original_combined_lower = float(targets @ original_dual)
    affine_dual = ray * scale[selected]
    affine_combined_normal = affine_rows[selected].T @ affine_dual
    affine_normal_magnitude = np.abs(affine_rows[selected]).T @ affine_dual
    affine_combined_lower = float(affine_targets[selected] @ affine_dual)
    relative_precision = float(settings.tol_infeas_rel)
    absolute_precision = float(settings.tol_infeas_abs)
    # These are the existing IPM certificate thresholds, with no reduced-
    # accuracy settings.  Cancellation denominators account for each column's
    # original physical units rather than masking a small physical column.
    original_limit = relative_precision * original_normal_magnitude
    affine_limit = relative_precision * affine_normal_magnitude
    original_stationary = bool(np.all(np.abs(original_combined_normal) <= original_limit))
    affine_stationary = bool(np.all(np.abs(affine_combined_normal) <= affine_limit))
    primal_gate_relaxation = float(10 * tolerance * (row_scale @ original_dual))
    positive_limit = max(absolute_precision, primal_gate_relaxation)
    accepted = (original_stationary and affine_stationary
                and original_combined_lower > positive_limit
                and affine_combined_lower > absolute_precision)
    report.update(
        accepted=bool(accepted),
        reason="verified_original_and_affine_nonnegative_infeasibility_ray" if accepted
               else "original_infeasibility_ray_gate_failed_use_complete_IPM",
        minimum_original_multiplier=float(np.min(original_dual)),
        original_dual_mass=float(np.sum(original_dual)),
        original_combined_normal_max_absolute=float(np.max(np.abs(original_combined_normal))),
        original_combined_normal_max_relative=float(np.max(np.divide(
            np.abs(original_combined_normal), original_normal_magnitude,
            out=np.zeros_like(original_combined_normal), where=original_normal_magnitude > 0.))),
        affine_combined_normal_max_absolute=float(np.max(np.abs(affine_combined_normal))),
        affine_combined_normal_max_relative=float(np.max(np.divide(
            np.abs(affine_combined_normal), affine_normal_magnitude,
            out=np.zeros_like(affine_combined_normal), where=affine_normal_magnitude > 0.))),
        original_combined_lower=original_combined_lower,
        affine_combined_lower=affine_combined_lower,
        original_primal_gate_relaxation=primal_gate_relaxation,
        original_positive_combination_threshold=positive_limit,
        existing_tol_infeas_abs=absolute_precision,
        existing_tol_infeas_rel=relative_precision,
        elapsed_seconds=time.perf_counter() - begun)
    return report, original_dual


def install_verified_restricted_infeasibility(output_directory=None):
    """Install an audited candidate hook before calling the original driver.

    Optional output captures only failed restricted QPs.  The original
    H/g/A/b/box and the live ray are retained for independent replay; normal
    feasible QPs execute their identical old code and original KKT checks.
    """
    from calibration.solver_runtime import install
    install()
    import z190_global_schur_linear_step as module
    original = module._linearly_constrained_shared_step
    if getattr(original, "_verified_restricted_infeasibility", False):
        return original._infeasibility_installation_report
    source = inspect.getsource(original)
    tree = ast.parse(source)
    function = tree.body[0]
    nested = next(node for node in function.body
                  if isinstance(node, ast.FunctionDef) and node.name == "separated_affine_step")
    loop = next(node for node in nested.body if isinstance(node, ast.While))
    status_gate = [node for node in loop.body if isinstance(node, ast.If)
                   and ast.unparse(node.test) == "str(result.status) not in ('Solved', 'AlmostSolved')"]
    if len(status_gate) != 1:
        raise RuntimeError("Restricted QP candidate expected exactly one original status gate")
    injected = ast.parse("""
certificate = _verify_failed_restricted_QP(result, selected, rows, targets,
    scale, affine_rows, affine_targets, row_scale, tolerance, settings,
    matrix, gradient, lower, upper, constraint_matrix, constraint_lower)
entry['original_restricted_infeasibility_certificate'] = certificate
if certificate['accepted']:
    raise _CertifiedSharedQPInfeasible(certificate)
""").body
    status_gate[0].body[:0] = injected
    acceleration_try = [node for node in function.body if isinstance(node, ast.Try)
                        and len(node.body) == 1 and isinstance(node.body[0], ast.Assign)
                        and ast.unparse(node.body[0].value) == "separated_affine_step()"]
    if len(acceleration_try) != 1:
        raise RuntimeError("Restricted QP candidate expected exactly one original acceleration try")
    acceleration_try[0].handlers.insert(0, ast.ExceptHandler(
        type=ast.Name(id="_CertifiedSharedQPInfeasible", ctx=ast.Load()),
        name=None, body=[ast.Raise(exc=None, cause=None)]))
    ast.fix_missing_locations(tree)
    destination = None if output_directory is None else Path(output_directory).resolve()
    captured = []

    def check_and_capture(result, selected, rows, targets, scale, affine_rows,
                          affine_targets, row_scale, tolerance, settings,
                          matrix, gradient, lower, upper, constraint_matrix,
                          constraint_lower):
        report, dual = verify_restricted_infeasibility(
            result, selected, rows, targets, scale, affine_rows,
            affine_targets, row_scale, tolerance, settings)
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
            stamp = str(time.time_ns())
            path = destination / ("restricted_QP_infeasibility_" + stamp + ".npz")
            np.savez(path, matrix=matrix, gradient=gradient, lower=lower, upper=upper,
                     constraint_matrix=constraint_matrix, constraint_lower=constraint_lower,
                     selected=selected, ray=np.asarray(result.z, float), scale=scale,
                     affine_rows_selected=affine_rows[selected],
                     affine_targets_selected=affine_targets[selected], row_scale=row_scale,
                     normalized_original_dual=np.empty(0) if dual is None else dual,
                     tolerance=np.asarray(tolerance))
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            report["actual_original_QP_capture"] = dict(path=str(path), sha256=digest)
            (path.with_suffix(".json")).write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf8")
        captured.append(report)
        return report

    namespace = dict(original.__globals__)
    namespace.update(_verify_failed_restricted_QP=check_and_capture,
                     _CertifiedSharedQPInfeasible=CertifiedSharedQPInfeasible)
    exec(compile(tree, str(Path(__file__).resolve()) + ":candidate", "exec"), namespace)
    replacement = namespace[original.__name__]
    for name, value in original.__dict__.items():
        setattr(replacement, name, value)
    replacement._verified_restricted_infeasibility = True
    report = dict(mode="verified_restricted_infeasibility_candidate",
                  original_producer_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  candidate_helper_path=str(Path(__file__).resolve()),
                  candidate_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  output_directory=None if destination is None else str(destination),
                  original_producer_files_or_manifest_rewritten=False,
                  original_feasible_QP_KKT_gate_unchanged=True)
    replacement._infeasibility_installation_report = report
    replacement._infeasibility_capture_reports = captured
    module._linearly_constrained_shared_step = replacement
    return report
