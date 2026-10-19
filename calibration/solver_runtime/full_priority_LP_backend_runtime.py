"""Full-model default dual-simplex route for the original priority LP.

Only the backend changes. Every original normalized row, objective, bound and
1e-9 tolerance is retained. A complete primal/dual KKT witness is required;
otherwise the unchanged original IPM call supplies its result or error.
Native parallel/scheduler settings remain untouched. Real PAMI configuration
returned NotSet on the complete model and is deliberately no longer used.
Native solver output is enabled for this one call to expose backend failures.
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
import time
import warnings

import numpy as np

from .separated_priority_LP_runtime import _full_problem_KKT


_LATEST = ContextVar('rinkspace_complete_priority_LP_backend_trace', default=None)
_RECEIPT = None


@functools.lru_cache(maxsize=None)
def _default_backend_metadata():
    """Read the installed native defaults without modifying its scheduler."""
    from scipy.optimize._highspy import _core
    highs = _core._Highs()
    return dict(HiGHS_version=highs.version(),
        full_backend_algorithm='HiGHS default serial dual simplex',
        explicit_parallel_thread_strategy_or_concurrency_options_added=False,
        global_scheduler_not_reset=True, native_log_enabled_for_this_call=True)


def full_original_dual_simplex_linprog(original, c, *args, **kwargs):
    """Solve every original row with default dual simplex, then certify KKT."""
    trace = dict(schema='rinkspace.full-original-priority-LP-default-dual-simplex.v2',
        full_original_problem_unchanged=True, original_rows_never_separated=True,
        original_boxes_and_auxiliary_bounds_retained=True,
        complete_original_problem_KKT_certified=False,
        original_IPM_fallback_called=False)
    started = time.perf_counter()
    reason = None
    try:
        if args or set(kwargs) != {'A_ub', 'b_ub', 'bounds', 'method', 'options'}:
            raise TypeError('unsupported_original_LP_call_signature')
        objective = np.asarray(c, float)
        A, b = np.asarray(kwargs['A_ub'], float), np.asarray(kwargs['b_ub'], float)
        bounds, options = kwargs['bounds'], kwargs['options']
        if (objective.ndim != 1 or len(objective) < 2 or objective[-1] != 1.
                or np.any(objective[:-1] != 0.) or b.ndim != 1
                or A.shape != (len(b), len(objective))
                or not np.isfinite(A).all() or not np.isfinite(b).all()
                or kwargs['method'] != 'highs-ipm' or np.any(A[:, -1] > 0.)
                or len(bounds) != len(objective) or tuple(bounds[-1]) != (0., 1.)
                or any(tuple(value) != (None, None) for value in bounds[:-1])):
            raise TypeError('unsupported_original_priority_LP_model')
        if (set(options) != {'primal_feasibility_tolerance', 'dual_feasibility_tolerance'}
                or float(options['primal_feasibility_tolerance']) != 1e-9
                or float(options['dual_feasibility_tolerance']) != 1e-9):
            raise TypeError('original_1e-9_tolerance_contract_required')
        trace.update(full_original_rows=len(b), original_columns=len(objective),
                     original_backend_tolerances=dict(options), **_default_backend_metadata())
        # SciPy's highs-ds route selects its original default serial dual
        # strategy. Add no parallel, threads, strategy or concurrency options.
        # disp enables native log_to_console/output_flag only for this call.
        backend_options = dict(options, disp=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = original(c, **dict(kwargs, method='highs-ds', options=backend_options))
        trace['backend_warnings'] = [str(item.message) for item in caught]
        trace['dual_simplex_backend_seconds'] = time.perf_counter()-started
        trace['dual_simplex_backend_iterations'] = int(getattr(result, 'nit', 0) or 0)
        trace['dual_simplex_backend_message'] = str(getattr(result, 'message', ''))
        if not result.success or result.x is None:
            reason = 'default_dual_simplex_full_backend_failed'
        else:
            passed, KKT = _full_problem_KKT(objective, A, b, bounds, result,
                np.arange(len(b), dtype=int), 1e-9, 1e-9)
            trace['complete_original_problem_KKT'] = KKT
            if passed:
                trace.update(route='full_original_default_dual_simplex',
                             complete_original_problem_KKT_certified=True,
                             full_problem_KKT_seconds=time.perf_counter()-started-trace['dual_simplex_backend_seconds'])
                _LATEST.set(trace)
                return result
            reason = 'default_dual_simplex_full_backend_original_KKT_not_certified'
    except Exception as error:
        # Cancellation/termination BaseExceptions remain visible to the caller.
        reason = 'default_dual_simplex_execution_unavailable:'+type(error).__name__+': '+str(error)
    trace.update(route='original_full_HiGHS_IPM', fallback_reason=reason,
                 original_IPM_fallback_called=True,
                 dual_simplex_attempt_seconds=time.perf_counter()-started)
    _LATEST.set(trace)
    # Deliberately outside the execution catch: original input/backend errors
    # are propagated exactly once rather than masked or retried.
    fallback_started = time.perf_counter()
    try:
        return original(c, *args, **kwargs)
    finally:
        trace['original_IPM_fallback_seconds'] = time.perf_counter()-fallback_started


def _phaseI_backend_label():
    trace = _LATEST.get()
    if trace and trace['route'] == 'full_original_default_dual_simplex':
        return 'scipy HiGHS default serial dual simplex (complete original LP, full KKT)'
    return 'scipy HiGHS IPM'


def install_full_priority_LP_backend_runtime(*, report_directory=None):
    """Explicit unused installer; do not install alongside LP separation."""
    global _RECEIPT
    if _RECEIPT is not None:
        return _RECEIPT
    from structured_programme_source_candidate import feasibility_priority_restoration as source
    original = source.contract_original_support_bounds
    if hasattr(original, '__wrapped__'):
        raise RuntimeError('Original unwrapped priority-LP contract required')
    original_linprog = source.linprog
    source_text = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source_text)
    calls, labels = [], []
    class CompleteLPBackend(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == 'linprog':
                calls.append(node.lineno)
                node.func.id = '_rinkspace_full_original_dual_simplex_linprog'
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'phase_I_backend'
                    and isinstance(node.value, ast.Constant)
                    and node.value.value == 'scipy HiGHS IPM'):
                labels.append(node.lineno)
                node.value = ast.Call(func=ast.Name(id='_rinkspace_phaseI_backend_label', ctx=ast.Load()),
                                      args=[], keywords=[])
            return self.generic_visit(node)
    tree = CompleteLPBackend().visit(tree)
    if len(calls) != 1 or len(labels) != 1:
        raise RuntimeError('Exactly one original LP call and original IPM label required')
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace.update(_rinkspace_full_original_dual_simplex_linprog=functools.partial(
        full_original_dual_simplex_linprog, original_linprog),
        _rinkspace_phaseI_backend_label=_phaseI_backend_label)
    exec(compile(tree, __file__+'::<complete-original-priority-LP>', 'exec'), namespace)
    transformed = namespace[original.__name__]
    destination = None if report_directory is None else Path(report_directory).resolve()
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    @functools.wraps(original)
    def contract(*args, **kwargs):
        token = _LATEST.set(None)
        try:
            result = transformed(*args, **kwargs)
            trace = _LATEST.get()
            if trace is not None:
                result['report']['complete_original_LP_backend'] = trace
            return result
        finally:
            trace = _LATEST.get()
            if trace is not None and destination is not None:
                with (destination/'original_priority_LP_backend.jsonl').open('a', encoding='utf8') as output:
                    output.write(json.dumps(trace, ensure_ascii=False)+'\n')
            _LATEST.reset(token)
    source.contract_original_support_bounds = contract
    _RECEIPT = dict(schema='rinkspace.full-original-priority-LP-default-dual-simplex-install.v2',
        helper_path=str(Path(__file__).resolve()),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_contract_sha256=hashlib.sha256(source_text.encode()).hexdigest(),
        transformed_AST_sha256=hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        AST_changes='One original linprog call and its backend report label only',
        original_normalization_targets_and_return_formula_unchanged=True,
        all_original_A_b_objective_bounds_and_1e_9_tolerances_retained=True,
        full_original_primal_dual_stationarity_complementarity_gap_required=True,
        original_complete_IPM_on_uncertified_default_dual_simplex_execution=True,
        no_source_row_separation_or_old_H_g_reuse=True,
        registered_source_files_modified=False, actual_full_problem_speedup_pending=True,
        no_parallel_thread_strategy_or_concurrency_options_added=True,
        existing_global_scheduler_not_reset=True,
        native_backend_diagnostic_output_enabled=True, PAMI_route_disabled=True,
        accuracy_accepted=False)
    if destination is not None:
        (destination/'full_priority_LP_backend_receipt.json').write_text(
            json.dumps(_RECEIPT, indent=2)+'\n', encoding='utf8')
    return _RECEIPT
