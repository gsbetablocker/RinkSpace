"""Candidate adapter for rejection-only scheduling before expensive trial work.

This adapter changes no source producer, trial order, step, or acceptance rule.
It is separate from the registered source implementation so its first real
comparison can reuse the already authenticated normal at the identical state.
The transformed function and exact adapter identity are recorded by the caller.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap


_PRECHECK = '''
if source_trial_precheck is not None:
    precheck_parent=q.copy();precheck_parent.flags.writeable=False
    precheck_trial=nextq.copy();precheck_trial.flags.writeable=False
    precheck_decision=run_stage('trial_necessary_precheck_total',source_trial_precheck,
        precheck_parent,precheck_trial,trial_context=dict(iteration=len(history),trial=trials,
            line_search_fraction=None if global_schur_count is None else alpha,
            line_search_mode=search_mode))
    if (not isinstance(precheck_decision,dict)
            or precheck_decision.get('action') not in ('reject','continue')
            or precheck_decision.get('acceptance_authority') is not False):
        raise ValueError('Trial precheck must only reject or defer to all original checks')
    print(json.dumps(dict(stage='trial_necessary_condition',iteration=len(history),trial=trials,
        **precheck_decision)),flush=True)
    if precheck_decision['action']=='reject':
        raise ValueError(precheck_decision['reason'])
'''


_CONTROL_HANDLER = '''
try:
    pass
except _SourceControlStop as interruption:
    stop=interruption.stop
    print(json.dumps(dict(stage='source_control_QP_returned_to_original_solver',
        wall_time_ns=time.time_ns(), reason=stop['reason'],
        accepted_iterations=sum(record['accepted'] for record in history),
        unaccepted_trial_discarded=True)),flush=True)
    if stop['control'].get('save_current_linearization_cache') is True:
        early_control_stop()
        if derivative_model is not None:
            derivative_model.refresh('explicit current-cache stop during QP requires fresh full source J')
        repeat_frozen_profiles=True
        stop=None
        continue
    status='stopped_by_control'
    break
'''


def with_trial_precheck(original):
    """Return the original solver with one explicit rejection-only callback."""
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef) or function.name != 'solve_joint_source':
        raise ValueError('Expected the source-owned solve_joint_source function')
    if any(a.arg == 'source_trial_precheck' for a in function.args.kwonlyargs):
        raise ValueError('Trial precheck is already installed')
    function.args.kwonlyargs.append(ast.arg(arg='source_trial_precheck'))
    function.args.kw_defaults.append(ast.Constant(value=None))

    class BeforePhysicalGuard(ast.NodeTransformer):
        count = 0

        def visit_Expr(self, node):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == 'run_stage' and len(call.args) == 4
                    and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value == 'trial_physical_guard_total'
                    and isinstance(call.args[1], ast.Name)
                    and call.args[1].id == '_check_physical'):
                self.count += 1
                return [*ast.parse(_PRECHECK).body, node]
            return node

    transformer = BeforePhysicalGuard()
    tree = transformer.visit(tree)
    if transformer.count != 1:
        raise ValueError('Original trial physical-guard call site changed')
    loops = [node for node in function.body if isinstance(node, ast.While)
             and ast.unparse(node.test) == 'stop is None']
    if len(loops) != 1:
        raise ValueError('Original outer source loop changed')
    handler = ast.parse(_CONTROL_HANDLER).body[0]
    handler.body = loops[0].body
    loops[0].body = [handler]
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    from .solver_control_runtime import SourceControlStop
    namespace['_SourceControlStop'] = SourceControlStop
    # Align the existing stage boundaries with the separate process-tree CPU
    # samples. This records wall time only; source/solver numerical work is
    # untouched and other original print calls keep their exact payloads.
    import builtins
    import time
    original_print = namespace.get('print', builtins.print)
    def print_with_stage_clock(*values, **options):
        if (len(values) == 1 and isinstance(values[0], str)
                and ('"stage": "source_stage_' in values[0]
                     or '"stage":"source_stage_' in values[0])):
            event = json.loads(values[0])
            event['wall_time_ns'] = time.time_ns()
            values = (json.dumps(event),)
        return original_print(*values, **options)
    namespace['print'] = print_with_stage_clock
    code = compile(tree, __file__ + ':rejection_only_solver', 'exec')
    exec(code, namespace)
    candidate = namespace[original.__name__]
    candidate.__dict__.update(original.__dict__)
    candidate.execution_change = dict(
        original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
        transformed_AST_sha256=hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        inserted_check_sites=transformer.count,
        explicit_QP_control_returns_through_original_save_and_result=True,
        explicit_cache_stop_reaches_original_fresh_normal_boundary=True,
        source_stage_wall_clock_recorded_for_process_tree_CPU_alignment=True,
        original_trial_order_and_steps_unchanged=True,
        every_candidate_acceptance_runs_original_guard_cost_and_acceptance=True,
        source_observations_r_J_normal_profiles_unchanged=True)
    return candidate


def restoration_precheck_from_callback(callback, *, blocks=None, ready=None):
    """Bind the current driver's existing cost-independent truth query."""
    from .staged_source_guard import (
        preserve_programme_camera_bindings, restoration_progress_precheck,
    )
    from structured_programme_source_candidate.feasibility_priority_restoration import actual_restoration_trial_decision
    if callback.__name__ != 'source_trial_acceptance':
        raise ValueError('Only the known complete truth-restoration callback is supported')
    closure = inspect.getclosurevars(callback).nonlocals
    required = ('complete_truth_violations', 'priority_restoration_domain')
    if not all(name in closure for name in required):
        raise ValueError('Complete truth callback no longer exposes its original domain')
    violations = closure['complete_truth_violations']
    domain = closure['priority_restoration_domain']
    qtoken = callback.__globals__['qtoken']

    # The original violation query never calls truth_guard and cannot create
    # restoration limits. A matching priority domain is only built after the
    # driver's original limits and constraints exist. An optional ready gate
    # may additionally require those caller-owned limits explicitly.
    if blocks is None:
        query_closure = inspect.getclosurevars(violations).nonlocals
        states = [getattr(factor, 'state', None) for factor in query_closure.values()]
        states = [state for state in states if state is not None and hasattr(state, 'blocks')]
        if states:
            if any(state is not states[0] for state in states):
                raise ValueError('Necessary truth queries must share one original state')
            blocks = states[0].blocks
    blocks = None if blocks is None else tuple(blocks)

    def precheck(parent, trial, *, trial_context):
        import numpy as np
        identity = dict(parent_q_sha256=qtoken(parent), trial_q_sha256=qtoken(trial),
                        limits_initialized_by_probe=False)
        def defer(reason, **details):
            return dict(action='continue', acceptance_authority=False,
                        reason=reason, **identity, report=details)
        if (blocks is None or domain.get('q_sha256') != identity['parent_q_sha256']
                or 'common_fraction' not in domain or (ready is not None and not ready())):
            return defer('Original restoration limits/domain/bindings are not ready')
        try:
            with preserve_programme_camera_bindings(blocks):
                previous, labels = violations(parent)
                if not np.any(previous > 1e-8):
                    return defer('No positive restoration violation; retain original source objective path')
                actual, trial_labels = violations(trial)
                if labels != trial_labels:
                    raise ValueError('Complete source truth group identities changed during precheck')
                decision = restoration_progress_precheck(actual_restoration_trial_decision,
                    previous, actual, common_fraction=domain['common_fraction'],
                    step_fraction=trial_context.get('line_search_fraction', trial_context.get('alpha')))
        except (ValueError, RuntimeError, TypeError) as exc:
            # Domain/query failures are not a necessary-progress witness.
            # Preserve the original guard/cost/callback exception precedence.
            return defer('Necessary query deferred to original ordered checks',
                         deferred_error_type=type(exc).__name__, deferred_error=str(exc))
        report = dict(decision['report'], original_truth_group_labels=labels,
                      source_query_arrays_and_order_unchanged=True,
                      programme_camera_bindings_restored=True)
        # The arrays have already been computed by the original necessary
        # predicate. Keep its exact failing group identities without another
        # source call, so nonlinear curvature failures can be distinguished
        # from a physical-domain exception or a poor objective prediction.
        positive=previous>1e-8
        required=float(report['required_maximum_original_violation_fraction'])
        tolerance=float(report['restoration_comparison_tolerance_px'])
        failed=np.where(positive,actual-required*previous>tolerance,actual>1e-8)
        report['failed_original_truth_groups']=[dict(group_index=int(i),group_id=labels[i],
            previous_violation_px=float(previous[i]),trial_violation_px=float(actual[i]),
            protected_zero_group=bool(not positive[i]),
            each_nonworsening_failed=bool(positive[i]and actual[i]-previous[i]>tolerance),
            required_maximum_trial_violation_px=float(required*previous[i])if positive[i]else 1e-8,
            numerical_comparison_tolerance_px=tolerance if positive[i]else 0.,
            actual_progress_miss_px=float(actual[i]-required*previous[i])if positive[i]else float(actual[i]-1e-8))
            for i in np.flatnonzero(failed)]
        return dict(action='reject' if decision['reject'] else 'continue',
                    acceptance_authority=False, reason=decision['reason'],
                    **identity, report=report)

    return precheck
