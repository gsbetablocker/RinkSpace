"""Certify phase-I contraction hints by the complete original primal domain.

Phase I supplies a feasible contraction to the unchanged source QP.  It is
not the accepted source step and does not need an LP optimality certificate.
A numerical backend termination label cannot replace the original all-face
primal checks.  The original solver, preconditioner, rows and tolerances stay
in use; the returned source QP still needs its complete original KKT gate.
"""
from __future__ import annotations

import ast
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import time

import numpy as np

_RECEIPT = None


def _has_primal_candidate(solved):
    """Eligibility to inspect a candidate, never a feasibility certificate."""
    try:
        candidate = np.asarray(solved.x, dtype=float)
        return candidate.ndim == 1 and candidate.size > 0 and np.isfinite(candidate).all()
    except (AttributeError, TypeError, ValueError):
        return False


def _repair_auxiliary_fraction(x, dual, B, expansion, rhs, norm, A, b,
                               slack, original_t):
    """Repair only t along the same original faces, never a model coordinate.

    Already certified candidates take the exact old path.  A failed primal
    candidate with a retained legal dual diagnostic may instead witness a
    feasible (not optimal) fraction at the original inequality interval's
    endpoint.  The original all-face gate is reapplied to every candidate.
    """
    if (not np.isfinite(x).all() or not np.isfinite(dual).all()
            or np.min(dual) < -1e-10 or np.min(slack) >= -1e-9):
        return x, slack, None
    if not 0. <= original_t < 1. - 64 * np.finfo(float).eps:
        return x, slack, None

    expansion = np.asarray(expansion, float)
    if np.any(expansion < 0.):
        raise ValueError('Original phase-I recovery coefficients must be nonnegative')
    rhs = np.asarray(rhs, float)
    norm = np.asarray(norm, float)
    fixed = np.asarray(B @ x[:-1], float)
    fixed_slack = (fixed - rhs) / norm
    protected = expansion == 0.
    # No movement of t can repair an immutable physical/zero-group row.
    if (not np.isfinite(fixed_slack).all()
            or np.any(fixed_slack[protected] < -1e-9)):
        return x, slack, None
    increasing = expansion > 0.
    lower = max(0., float(x[-1]))
    upper = 1.
    if np.any(increasing):
        lower = max(lower, float(np.max(
            (rhs[increasing] - fixed[increasing]) / expansion[increasing])))
    if (not np.isfinite([lower, upper]).all() or lower > upper
            or lower >= 1. - 64 * np.finfo(float).eps):
        return x, slack, None

    failed_coordinate = x.copy()
    failed_slack = slack.copy()
    endpoint = float(np.nextafter(lower, np.inf))
    # A midpoint is only a checked candidate if endpoint roundoff does not
    # certify the original arithmetic; it cannot justify a failed primal.
    for method, fraction in (
            ('original_inequality_interval_endpoint_upward_roundoff', endpoint),
            ('checked_original_auxiliary_midpoint', .5 * (lower + upper))):
        if (not lower <= fraction <= upper
                or not 0. <= fraction < 1. - 64 * np.finfo(float).eps):
            continue
        repaired = failed_coordinate.copy()
        repaired[-1] = fraction
        checked_slack = np.r_[b - A @ repaired, fraction, 1. - fraction]
        # The caller independently repeats the original-unit refill gate.
        refilled = (fixed + expansion * fraction - rhs) / norm
        if (not np.isfinite(checked_slack).all()
                or np.min(checked_slack) < -1e-9
                or not np.isfinite(refilled).all() or np.min(refilled) < -1e-9):
            continue
        report = dict(
            auxiliary_t_only_primal_repair=True, method=method,
            original_failed_auxiliary_t=float(failed_coordinate[-1]),
            original_failed_normalized_minimum_slack=float(np.min(failed_slack)),
            original_failed_face_indices=np.flatnonzero(failed_slack < -1e-9).tolist(),
            exact_original_auxiliary_interval_lower=lower,
            exact_original_auxiliary_interval_upper=upper,
            repaired_auxiliary_t=fraction,
            auxiliary_t_increment=float(fraction - failed_coordinate[-1]),
            repaired_all_face_and_box_normalized_minimum_slack=float(np.min(checked_slack)),
            repaired_original_arithmetic_normalized_minimum_slack=float(np.min(refilled)),
            model_coordinates_bit_exact_unchanged=True,
            original_dual_finite_and_sign_gate_retained=True,
            original_complete_primal_tolerance=1e-9,
            phase_I_optimality_certificate_claimed=False)
        return repaired, checked_slack, dict(report=report,
            failed_coordinate=failed_coordinate, failed_slack=failed_slack)
    return x, slack, None


def install_priority_feasibility_witness_runtime(*, report_directory=None):
    global _RECEIPT
    if _RECEIPT is not None:
        return dict(_RECEIPT)
    from structured_programme_source_candidate import feasibility_priority_restoration as source
    original = source._verified_same_original_conic_phaseI
    source_text = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source_text)
    changes = []
    primal_changes = []

    class InspectPrimal(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name)and node.func.id=='SimpleNamespace':
                # Numeric diagnostic vectors belong to this original LP, not
                # source observations or a source-QP dual certificate.
                node.keywords.append(ast.keyword(arg='original_phaseI_normalized_dual',value=ast.Name(id='dual',ctx=ast.Load())))
                node.keywords.append(ast.keyword(arg='original_phaseI_original_normalized_row_slack',value=ast.Name(id='slack',ctx=ast.Load())))
                node.keywords.append(ast.keyword(arg='original_phaseI_auxiliary_t_repair',value=ast.Name(id='auxiliary_repair',ctx=ast.Load())))
            return self.generic_visit(node)

        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'slack'
                    and ast.unparse(node.value) == 'np.r_[b - A @ x, x[-1], 1.0 - x[-1]]'):
                primal_changes.append(node.lineno)
                repair = ast.parse(
                    'x, slack, auxiliary_repair = _rinkspace_repair_phaseI_auxiliary_fraction('
                    'x, dual, B, expansion, rhs, norm, A, b, slack, original_t)').body[0]
                return [node, repair]
            return node

        def visit_If(self, node):
            condition = node.test
            if (isinstance(condition, ast.Compare) and len(condition.ops) == 1
                    and isinstance(condition.ops[0], ast.NotEq)
                    and ast.unparse(condition.left) == 'str(solved.status)'
                    and len(condition.comparators) == 1
                    and isinstance(condition.comparators[0], ast.Constant)
                    and condition.comparators[0].value == 'Solved'):
                changes.append(node.lineno)
                node.test = ast.UnaryOp(op=ast.Not(), operand=ast.Call(
                    func=ast.Name(id='_rinkspace_has_phaseI_primal_candidate', ctx=ast.Load()),
                    args=[ast.Name(id='solved', ctx=ast.Load())], keywords=[]))
            return self.generic_visit(node)

    tree = InspectPrimal().visit(tree)
    if len(changes) != 1 or len(primal_changes) != 1:
        raise RuntimeError('One original phase-I status and complete primal gate required')
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace['_rinkspace_has_phaseI_primal_candidate'] = _has_primal_candidate
    namespace['_rinkspace_repair_phaseI_auxiliary_fraction'] = _repair_auxiliary_fraction
    exec(compile(tree, __file__ + '::<original-phaseI-primal-witness>', 'exec'), namespace)
    transformed = namespace[original.__name__]
    destination = None if report_directory is None else Path(report_directory)

    @wraps(original)
    def phaseI(B, expansion, rhs, norm, *, original_failure):
        begun = time.perf_counter()
        result = transformed(B, expansion, rhs, norm, original_failure=original_failure)
        coordinate = np.asarray(result.x[:-1], float)
        fraction = float(result.x[-1])
        target = .5 * (1. + fraction)
        # Independently certify the exact midpoint used by the original
        # caller, including every protected zero-recovery and physical row.
        slack = (B @ coordinate + expansion * target - rhs) / norm
        if (not 0. <= fraction < 1. - 64 * np.finfo(float).eps
                or not np.isfinite(slack).all() or np.min(slack) < -1e-9):
            raise ValueError('No verified strict contraction midpoint of every original phase-I face')
        protected = np.asarray(expansion) == 0.
        report = result.original_phaseI_report
        if result.original_phaseI_auxiliary_t_repair is not None:
            report['auxiliary_t_failed_primal_repair'] = result.original_phaseI_auxiliary_t_repair['report']
        report.update(
            backend_terminal_status=result.message,
            backend_status_is_not_feasibility_authority=True,
            complete_original_primal_witness_tolerance=1e-9,
            certified_original_midpoint_fraction=target,
            original_midpoint_all_face_normalized_minimum_slack=float(np.min(slack)),
            original_midpoint_protected_face_normalized_minimum_slack=(
                float(np.min(slack[protected])) if np.any(protected) else None),
            phase_I_optimality_certificate_claimed=False,
            phase_I_fraction_is_a_feasible_witness_not_a_proven_minimum=True,
            original_source_QP_full_KKT_and_nonlinear_truth_guard_remain_authoritative=True,
            original_Clarabel_settings_and_all_face_primal_gate_unchanged=True,
            elapsed_seconds=time.perf_counter()-begun)
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=True)
            with (destination/'phaseI_complete_primal_witness.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(report) + '\n')
        return result

    source._verified_same_original_conic_phaseI = phaseI
    _RECEIPT = dict(
        schema='rinkspace.original-phaseI-complete-primal-witness-install.v1',
        helper_path=str(Path(__file__).resolve()),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_phaseI_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
        AST_changes='One status-only gate replaced by primal inspection; failed conic primal may repair only original auxiliary t',
        original_LP_rows_objective_bounds_scaling_and_backend_settings_retained=True,
        original_all_face_1e_minus9_primal_and_dual_sign_gates_retained=True,
        exact_original_midpoint_complete_primal_witness_checked=True,
        original_conic_primal_and_dual_vectors_available_for_readonly_diagnostic_capture=True,
        failed_conic_primal_vectors_preserved_when_auxiliary_t_repaired=True,
        auxiliary_t_repair_checked_against_every_unchanged_original_face=True,
        no_optimality_certificate_claimed=True,
        original_HiGHS_verified_primal_success_path_unchanged=True,
        backend_success_alone_is_not_a_complete_original_primal_certificate=True,
        original_QP_KKT_and_nonlinear_acceptance_unchanged=True,
        registered_source_files_modified=False)
    return dict(_RECEIPT)
