"""Clarabel-first on the complete original normalized priority LP.

The original full A/b/objective and auxiliary box are retained. A successful
backend label is insufficient: the original complete primal/dual KKT gate
must pass. Any unsupported/uncertified attempt falls back to the unchanged
original HiGHS-IPM call, whose numeric exception remains visible.
"""
from __future__ import annotations

import ast
from contextvars import ContextVar
from functools import partial, wraps
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import time

import numpy as np

from .separated_priority_LP_runtime import _full_problem_KKT

_LATEST = ContextVar('rinkspace_complete_priority_Clarabel_trace', default=None)
_RECEIPT = None


def full_original_clarabel_linprog(original, c, *args, **kwargs):
    trace = dict(schema='rinkspace.complete-original-priority-LP-Clarabel-first.v1',
        full_original_problem_unchanged=True, no_row_or_column_elimination=True,
        complete_original_problem_KKT_certified=False,
        original_IPM_fallback_called=False)
    begun = time.perf_counter()
    reason = None
    try:
        if args or set(kwargs) != {'A_ub', 'b_ub', 'bounds', 'method', 'options'}:
            raise TypeError('Original explicit normalized priority LP signature required')
        objective = np.asarray(c, float)
        A = np.asarray(kwargs['A_ub'], float)
        b = np.asarray(kwargs['b_ub'], float)
        bounds, options = kwargs['bounds'], kwargs['options']
        if (objective.ndim != 1 or len(objective) < 2 or objective[-1] != 1.
                or np.any(objective[:-1] != 0.) or b.ndim != 1
                or A.shape != (len(b), len(objective))
                or not np.isfinite(A).all() or not np.isfinite(b).all()
                or kwargs['method'] != 'highs-ipm' or np.any(A[:, -1] > 0.)
                or len(bounds) != len(objective) or tuple(bounds[-1]) != (0., 1.)
                or any(tuple(value) != (None, None) for value in bounds[:-1])):
            raise TypeError('Unsupported original complete priority LP model')
        if (set(options) != {'primal_feasibility_tolerance', 'dual_feasibility_tolerance'}
                or float(options['primal_feasibility_tolerance']) != 1e-9
                or float(options['dual_feasibility_tolerance']) != 1e-9):
            raise TypeError('Original 1e-9 feasibility/dual tolerances required')
        import clarabel
        from scipy.optimize import OptimizeResult
        from scipy.sparse import csc_matrix, vstack
        columns = len(objective)
        box = np.zeros((2, columns))
        box[0, -1], box[1, -1] = -1., 1.
        conic_A = vstack((csc_matrix(A), csc_matrix(box)), format='csc')
        conic_b = np.r_[b, 0., 1.]
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_threads = 1
        # Match the registered same-original-LP Clarabel fallback settings.
        # The independent acceptance threshold remains the original 1e-9.
        for key in ('tol_feas', 'tol_gap_abs', 'tol_gap_rel', 'tol_infeas_abs', 'tol_infeas_rel'):
            setattr(settings, key, 1e-10)
        trace.update(full_original_rows=len(b), original_columns=columns,
            original_backend_tolerances=dict(options), Clarabel_version=clarabel.__version__,
            conic_form='A_ub*x+s=b_ub; -t+s_lower=0; t+s_upper=1',
            LP_P_exact_zero=True, prior_solution_step_or_duals_loaded=False,
            original_source_normalization_coefficients_recomputed=False,
            original_zero_free_columns_retained=True)
        backend_begun = time.perf_counter()
        solved = clarabel.DefaultSolver(csc_matrix((columns, columns)), objective,
            conic_A, conic_b, [clarabel.NonnegativeConeT(len(conic_b))], settings).solve()
        trace.update(Clarabel_backend_seconds=time.perf_counter()-backend_begun,
            Clarabel_status=str(solved.status), Clarabel_iterations=int(solved.iterations))
        if str(solved.status) not in ('Solved', 'AlmostSolved'):
            reason = 'complete_Clarabel_backend_did_not_solve'
        else:
            # Reduced backend termination is only eligibility to inspect the
            # complete original witness. It is never acceptance authority.
            # Both labels face precisely the same original 1e-9 hard gate.
            trace['backend_status_is_not_acceptance_authority'] = True
            x = np.asarray(solved.x, float)
            dual = np.asarray(solved.z, float)
            if x.shape != objective.shape or dual.shape != conic_b.shape:
                raise ValueError('Original complete LP primal/dual shape required')
            # Convert only multiplier signs to SciPy's original convention.
            # Source coordinates and t are neither projected nor clipped.
            lower_dual = np.zeros(columns)
            upper_dual = np.zeros(columns)
            lower_dual[-1], upper_dual[-1] = dual[-2], -dual[-1]
            result = OptimizeResult(success=True, x=x, nit=int(solved.iterations),
                message='Solved: complete original LP via Clarabel and full KKT',
                ineqlin=OptimizeResult(marginals=-dual[:-2]),
                lower=OptimizeResult(marginals=lower_dual),
                upper=OptimizeResult(marginals=upper_dual))
            KKT_begun = time.perf_counter()
            passed, witness = _full_problem_KKT(objective, A, b, bounds, result,
                np.arange(len(b), dtype=int), 1e-9, 1e-9)
            trace.update(complete_original_problem_KKT=witness,
                complete_original_problem_KKT_seconds=time.perf_counter()-KKT_begun)
            if passed and 0. <= x[-1] <= 1.:
                trace.update(route='complete_original_Clarabel_full_KKT',
                    complete_original_problem_KKT_certified=True,
                    complete_Clarabel_attempt_seconds=time.perf_counter()-begun)
                _LATEST.set(trace)
                return result
            reason = 'complete_Clarabel_original_KKT_or_exact_auxiliary_box_not_certified'
    except Exception as error:
        reason = 'complete_Clarabel_attempt_unavailable:' + type(error).__name__ + ': ' + str(error)
    trace.update(route='original_complete_HiGHS_IPM', fallback_reason=reason,
        original_IPM_fallback_called=True, complete_Clarabel_attempt_seconds=time.perf_counter()-begun)
    _LATEST.set(trace)
    fallback_begun = time.perf_counter()
    try:
        return original(c, *args, **kwargs)
    finally:
        trace['original_IPM_fallback_seconds'] = time.perf_counter()-fallback_begun


def _backend_label():
    trace = _LATEST.get()
    if trace and trace['route'] == 'complete_original_Clarabel_full_KKT':
        return 'official Clarabel complete original LP (full primal/dual KKT)'
    return 'scipy HiGHS IPM'


def install_full_priority_clarabel_runtime(*, report_directory=None):
    """Install instead of the old DS/PAMI or subset-LP execution helper."""
    global _RECEIPT
    if _RECEIPT is not None:
        return dict(_RECEIPT)
    from structured_programme_source_candidate import feasibility_priority_restoration as source
    original = source.contract_original_support_bounds
    if hasattr(original, '__wrapped__'):
        raise RuntimeError('Choose the complete Clarabel LP instead of another priority LP installer')
    source_text = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source_text)
    calls, labels = [], []
    class ReplaceBackend(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == 'linprog':
                calls.append(node.lineno)
                node.func.id = '_rinkspace_full_original_clarabel_linprog'
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'phase_I_backend'
                    and isinstance(node.value, ast.Constant) and node.value.value == 'scipy HiGHS IPM'):
                labels.append(node.lineno)
                node.value = ast.Call(func=ast.Name(id='_rinkspace_phaseI_backend_label', ctx=ast.Load()), args=[], keywords=[])
            return self.generic_visit(node)
    tree = ReplaceBackend().visit(tree)
    if len(calls) != 1 or len(labels) != 1:
        raise RuntimeError('One original complete LP call and backend label required')
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace.update(_rinkspace_full_original_clarabel_linprog=partial(
        full_original_clarabel_linprog, source.linprog), _rinkspace_phaseI_backend_label=_backend_label)
    exec(compile(tree, __file__ + '::<complete-original-Clarabel-LP>', 'exec'), namespace)
    transformed = namespace[original.__name__]
    destination = None if report_directory is None else Path(report_directory)

    @wraps(original)
    def contract(*args, **kwargs):
        token = _LATEST.set(None)
        try:
            result = transformed(*args, **kwargs)
            trace = _LATEST.get()
            if trace is not None:
                result['report']['complete_original_Clarabel_LP_backend'] = trace
            return result
        finally:
            trace = _LATEST.get()
            try:
                if trace is not None and destination is not None:
                    destination.mkdir(parents=True, exist_ok=True)
                    with (destination / 'complete_priority_Clarabel_LP.jsonl').open('a', encoding='utf8') as stream:
                        stream.write(json.dumps(trace) + '\n')
            except Exception:
                pass
            _LATEST.reset(token)
    source.contract_original_support_bounds = contract
    _RECEIPT = dict(schema='rinkspace.complete-original-priority-Clarabel-install.v1',
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_contract_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
        AST_changes='One original linprog call and backend report label only',
        full_original_A_b_objective_auxiliary_box_and_normalization_preserved=True,
        all_row_primal_dual_stationarity_complementarity_gap_tolerance=1e-9,
        original_source_Clarabel_fallback_internal_tolerances=1e-10,
        no_row_column_elimination_or_prior_q_H_g_dual_reuse=True,
        original_HiGHS_IPM_on_uncertified_execution=True,
        original_nonlinear_guard_and_QP_KKT_retained=True,
        registered_source_modified=False, actual_full_model_speedup_pending=True)
    return dict(_RECEIPT)


def snapshot_full_priority_clarabel_runtime():
    return dict(installation=None if _RECEIPT is None else dict(_RECEIPT))
