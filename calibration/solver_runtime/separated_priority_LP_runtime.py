"""Separate the same original priority LP, accepting only full-problem KKT.

No source faces, boxes or protected targets are removed from the full model.
The original boxes start active. Original-order worst violated source/physical
rows are added in growing batches; an optimum is accepted only after every original row and all
full-sized dual/KKT conditions pass the original backend tolerances. Otherwise
the unchanged original full linprog call supplies its normal result/error.
"""
from __future__ import annotations

import ast
from contextvars import ContextVar
import functools
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import numpy as np


_LATEST = ContextVar('rinkspace_original_priority_LP_separation_receipt', default=None)
_RECEIPT = None


class _FullOriginalRequired(Exception):
    def __init__(self, result=None):
        self.result = result


def _bounds_vectors(bounds, count):
    if len(bounds) != count:
        raise TypeError('Original explicit LP bounds required')
    lower = np.array([-np.inf if value[0] is None else value[0] for value in bounds], float)
    upper = np.array([np.inf if value[1] is None else value[1] for value in bounds], float)
    return lower, upper


def _full_problem_KKT(c, A, b, bounds, result, selected, primal_tolerance, dual_tolerance):
    """Full primal/dual witness, with exactly zero unselected row duals."""
    x = np.asarray(result.x, float)
    if x.shape != c.shape or not np.isfinite(x).all():
        return False, dict(reason='nonfinite_or_changed_primal_shape')
    try:
        small_dual = np.asarray(result.ineqlin.marginals, float)
        lower_dual = np.asarray(result.lower.marginals, float)
        upper_dual = np.asarray(result.upper.marginals, float)
    except (AttributeError, TypeError, ValueError):
        return False, dict(reason='original_LP_full_dual_witness_unavailable')
    if (small_dual.shape != selected.shape or lower_dual.shape != c.shape
            or upper_dual.shape != c.shape or not np.isfinite(small_dual).all()
            or not np.isfinite(lower_dual).all() or not np.isfinite(upper_dual).all()):
        return False, dict(reason='nonfinite_or_changed_dual_shape')
    y = np.zeros(len(b), float)
    y[selected] = small_dual
    lo, hi = _bounds_vectors(bounds, len(c))
    finite_lo, finite_hi = np.isfinite(lo), np.isfinite(hi)
    slack = b-A@x
    lower_slack, upper_slack = x[finite_lo]-lo[finite_lo], hi[finite_hi]-x[finite_hi]
    stationarity = c-A.T@y-lower_dual-upper_dual
    primal_objective = float(c@x)
    dual_objective = float(b@y + lo[finite_lo]@lower_dual[finite_lo]
                           + hi[finite_hi]@upper_dual[finite_hi])
    max_abs = lambda value: float(np.max(np.abs(value), initial=0.))
    dual_sign_error = max(float(np.max(y, initial=0.)),
        float(np.max(-lower_dual, initial=0.)), float(np.max(upper_dual, initial=0.)))
    # Multipliers on infinite bounds must be exactly inactive in the witness.
    absent_bound_dual = max(max_abs(lower_dual[~finite_lo]), max_abs(upper_dual[~finite_hi]))
    complementarity = max(max_abs(y*slack), max_abs(lower_dual[finite_lo]*lower_slack),
                          max_abs(upper_dual[finite_hi]*upper_slack))
    minimum_slack = min(float(np.min(slack, initial=np.inf)),
                        float(np.min(lower_slack, initial=np.inf)),
                        float(np.min(upper_slack, initial=np.inf)))
    values = dict(full_original_rows=len(b), original_columns=len(c), active_rows=len(selected),
        minimum_full_original_primal_slack=minimum_slack,
        maximum_full_original_dual_sign_error=dual_sign_error,
        maximum_dual_on_absent_bounds=absent_bound_dual,
        maximum_full_original_stationarity_error=max_abs(stationarity),
        maximum_full_original_complementarity_error=complementarity,
        original_primal_objective=primal_objective, original_dual_objective=dual_objective,
        original_primal_dual_gap=primal_objective-dual_objective,
        original_primal_tolerance=primal_tolerance, original_dual_tolerance=dual_tolerance,
        omitted_rows_duals_exactly_zero=True)
    finite = np.isfinite([value for value in values.values() if isinstance(value, (float,int))]).all()
    passed = bool(finite and minimum_slack >= -primal_tolerance
        and dual_sign_error <= dual_tolerance and absent_bound_dual <= dual_tolerance
        and values['maximum_full_original_stationarity_error'] <= dual_tolerance
        and complementarity <= dual_tolerance and abs(primal_objective-dual_objective) <= dual_tolerance)
    values['passed'] = passed
    if passed:
        # Preserve scipy's full-result interface. The original driver uses x,
        # success, message and nit; it does not use duals as source data.
        from scipy.optimize import OptimizeResult
        result.ineqlin = OptimizeResult(residual=slack, marginals=y)
    return passed, values


def separated_original_linprog(original, c, *args, original_box_row_count=None,
                               separate_protected_faces=False, **kwargs):
    """Only the exact source-owned, normalized priority-LP call is eligible."""
    history = []
    trace = dict(schema='rinkspace.original-priority-LP-complete-row-separation.v1',
        full_original_problem_unchanged=True, protected_rows_retained_in_full_problem=True,
        original_boxes_and_auxiliary_bounds_retained=True, batch_history=history,
        complete_original_problem_KKT_certified=False)
    def fallback(reason, already_full_result=None):
        trace.update(route='original_full_linprog', fallback_reason=reason,
                     complete_original_problem_KKT_certified=False)
        _LATEST.set(trace)
        raise _FullOriginalRequired(already_full_result)
    try:
        if args or set(kwargs) != {'A_ub','b_ub','bounds','method','options'}:
            return fallback('unsupported_original_LP_call_signature')
        objective = np.asarray(c, float)
        A, b = np.asarray(kwargs['A_ub'], float), np.asarray(kwargs['b_ub'], float)
        bounds, options = kwargs['bounds'], kwargs['options']
        if (objective.ndim != 1 or len(objective)<2 or objective[-1] != 1.
                or np.any(objective[:-1] != 0.) or A.shape != (len(b),len(objective))
                or b.ndim != 1 or not np.isfinite(A).all() or not np.isfinite(b).all()
                or kwargs['method'] != 'highs-ipm' or np.any(A[:,-1] > 0.)
                or tuple(bounds[-1]) != (0.,1.)
                or any(tuple(value) != (None,None) for value in bounds[:-1])):
            return fallback('unsupported_original_priority_LP_model')
        if set(options) != {'primal_feasibility_tolerance','dual_feasibility_tolerance'}:
            return fallback('unknown_original_backend_tolerance_contract')
        primal_tolerance = float(options['primal_feasibility_tolerance'])
        dual_tolerance = float(options['dual_feasibility_tolerance'])
        if not (0.<primal_tolerance<=1e-9 and 0.<dual_tolerance<=1e-9):
            return fallback('original_strict_tolerances_required')
        protected = np.flatnonzero(A[:,-1] == 0.)
        remaining = np.flatnonzero(A[:,-1] != 0.)
        trace.update(full_original_rows=len(b), original_columns=len(objective),
            protected_rows=len(protected), original_backend_tolerances=dict(options))
        if not len(remaining):
            return fallback('no_separable_original_truth_expansion_rows')
        initial = protected
        if separate_protected_faces:
            if (not isinstance(original_box_row_count,(int,np.integer))
                    or not 0<=original_box_row_count<=len(b)):
                return fallback('exact_original_box_row_inventory_required')
            initial=np.arange(len(b)-original_box_row_count,len(b),dtype=int)
            if np.any(A[initial,-1]!=0.):
                return fallback('original_box_expansion_column_changed')
            remaining=np.arange(len(b)-original_box_row_count,dtype=int)
        trace.update(protected_rows_initially_active=not separate_protected_faces,
            exact_original_box_rows_initially_active=original_box_row_count,
            protected_source_faces_separated_only_under_full_KKT_gate=separate_protected_faces)
        # Batch size follows the original chart dimension rather than a
        # guessed fixed capacity. Ties retain the original full row order.
        batch_size = len(objective)
        ranking = remaining[np.argsort(b[remaining],kind='stable')]
        selected = np.union1d(initial,ranking[:batch_size])
        total_iterations = 0
        while True:
            call = dict(kwargs,A_ub=A[selected],b_ub=b[selected])
            result = original(c,**call)
            total_iterations += int(getattr(result,'nit',0) or 0)
            history.append(dict(active_rows=len(selected), original_LP_iterations=int(getattr(result,'nit',0) or 0),
                                backend_success=bool(result.success)))
            all_rows = len(selected)==len(b)
            if not result.success or result.x is None or not np.isfinite(result.x).all():
                return fallback('original_subset_backend_failed',result if all_rows else None)
            slack = b-A@np.asarray(result.x,float)
            violating = np.flatnonzero(slack < -primal_tolerance)
            history[-1]['full_original_violated_rows']=len(violating)
            if not len(violating):
                passed, KKT = _full_problem_KKT(objective,A,b,bounds,result,selected,
                                              primal_tolerance,dual_tolerance)
                trace['complete_original_problem_KKT']=KKT
                if not passed:
                    return fallback('complete_original_problem_KKT_not_certified',result if all_rows else None)
                result.nit = total_iterations
                trace.update(route='separated_original_LP_full_KKT',
                    complete_original_problem_KKT_certified=True, original_LP_calls=len(history),
                    total_original_LP_iterations=total_iterations)
                _LATEST.set(trace)
                return result
            unseen = violating[~np.isin(violating,selected)]
            if not len(unseen):
                return fallback('original_active_rows_failed_full_primal_tolerance',result if all_rows else None)
            worst = unseen[np.argsort(slack[unseen],kind='stable')]
            selected = np.union1d(selected,worst[:batch_size])
            batch_size *= 2
    except _FullOriginalRequired as request:
        return original(c,*args,**kwargs) if request.result is None else request.result
    except Exception as error:
        # The original full backend still determines source errors/failure.
        # BaseException control cancellation is deliberately not intercepted.
        trace.update(route='original_full_linprog',
            fallback_reason='execution_integration_error:'+type(error).__name__+': '+str(error),
            complete_original_problem_KKT_certified=False)
        _LATEST.set(trace)
        return original(c,*args,**kwargs)


def install_separated_priority_LP_runtime(*, report_directory=None, separate_protected_faces=True):
    """Candidate API; leave disabled until the root explicitly installs it."""
    global _RECEIPT
    if _RECEIPT is not None:
        return _RECEIPT
    from structured_programme_source_candidate import feasibility_priority_restoration as source
    original = source.contract_original_support_bounds
    original_linprog = source.linprog
    text = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(text)
    sites = []
    class OriginalLP(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='linprog':
                sites.append(node.lineno)
                node.func.id='_rinkspace_separated_original_linprog'
                node.keywords.append(ast.keyword(arg='original_box_row_count',value=ast.parse(
                    'len(finite_lo)+len(finite_hi)',mode='eval').body))
            return self.generic_visit(node)
    tree=OriginalLP().visit(tree)
    if len(sites)!=1:
        raise RuntimeError('Exactly one original truth-priority LP call required')
    ast.fix_missing_locations(tree)
    namespace=dict(original.__globals__)
    namespace['_rinkspace_separated_original_linprog']=functools.partial(
        separated_original_linprog,original_linprog,separate_protected_faces=separate_protected_faces)
    exec(compile(tree,__file__+'::<original-priority-LP>','exec'),namespace)
    transformed=namespace[original.__name__]
    destination=None if report_directory is None else Path(report_directory).resolve()
    if destination is not None:destination.mkdir(parents=True,exist_ok=True)
    @functools.wraps(original)
    def contract(*args,**kwargs):
        token=_LATEST.set(None)
        try:
            result=transformed(*args,**kwargs)
            trace=_LATEST.get()
            if trace is not None:
                result['report']['complete_original_LP_separation']=trace
                if destination is not None:
                    with (destination/'original_priority_LP_separation.jsonl').open('a',encoding='utf8') as output:
                        output.write(json.dumps(trace,ensure_ascii=False)+'\n')
            return result
        finally:_LATEST.reset(token)
    source.contract_original_support_bounds=contract
    _RECEIPT=dict(schema='rinkspace.original-priority-LP-complete-row-separation.v1',
        helper_path=str(Path(__file__).resolve()),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_contract_sha256=hashlib.sha256(text.encode()).hexdigest(),
        AST_changes='One original linprog call only',original_normalization_and_targets_unchanged=True,
        protected_rows_retained_in_full_original_problem=True,
        protected_rows_active_from_first_subset=not separate_protected_faces,
        original_exact_appended_box_rows_active_from_first_subset=True,
        original_boxes_and_auxiliary_bounds_retained=True,
        worst_original_violation_stable_order_and_growing_batches=True,
        full_original_primal_dual_stationarity_complementarity_and_gap_required=True,
        original_backend_tolerances_not_relaxed=True,unselected_full_duals_exactly_zero=True,
        original_full_linprog_on_any_uncertified_execution=True,original_return_scalar_usage_retained=True,
        baseline_H_g_or_old_source_J_imported=False,registered_source_files_modified=False,
        actual_complete_problem_performance_pending=True,accuracy_accepted=False)
    if destination is not None:
        (destination/'separated_priority_LP_receipt.json').write_text(json.dumps(_RECEIPT,indent=2)+'\n',encoding='utf8')
    return _RECEIPT
