"""Domain-aware GTSAM solve for a continuous, movable-centre camera model.

An invalid finite-board trial is rejected at the optimizer boundary. No source
row is removed or replaced by a synthetic residual to make a trial acceptable.
The existing reference CustomFactor callback is supplied by the graph builder.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import time

from runtime.bootstrap import add_geometry_dependencies
add_geometry_dependencies()
import gtsam
import numpy as np

from reference_board_pair import BoardIntersectionError
from adaptive_curve_parallel import prepare_adaptive_curve_jobs


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf8")
    tmp.replace(path)


def _error_record(exc):
    return dict(exception_type=type(exc).__name__,
        reason=str(getattr(exc, 'reason', str(exc))),
        factor_id=getattr(exc, 'factor_id', None),
        source_pts=getattr(exc, 'source_pts', None),
        factor_family=getattr(exc, 'factor_family', None),
        source_keys=getattr(exc, 'source_keys', None),
        evaluation_stage=getattr(exc, 'evaluation_stage', None))


def _add_elapsed(phases, name, started):
    phases[name] = phases.get(name, 0.0) + time.perf_counter()-started


def _prepare_curves(graph, states, trace, stage):
    report = prepare_adaptive_curve_jobs(graph, states)
    if report["factors"]:
        trace.setdefault("adaptive_curve_precompute", []).append(
            dict(stage=stage, **report))
    return report


def values_for(graph, states):
    values = gtsam.Values()
    for key, state in states.items():
        values.insert(graph._key(int(key)), np.asarray(state, np.float64))
    return values


def states_for(graph, values, keys):
    return {int(k): np.asarray(values.atVector(graph._key(int(k))), float).copy()
            for k in keys}


def save_states(path, states):
    keys = np.asarray(sorted(states), np.int64)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, keys=keys,
                        states=np.asarray([states[int(k)] for k in keys]))
    temporary.replace(path)


class CompleteFactorSession:
    """Retry unsupported native factors as complete original Python factors.

    Replacement happens only after control returns to the main Python thread.
    A domain failure is a different exception and never invokes this fallback.
    """
    def __init__(self, graph, manager=None):
        self.graph = graph
        self.manager = manager
        self.optimizer = None
        self.native = None
        self.revision = 0
        self.replaced_ids = set()
        self.parity_checks = []
        self.linearization_checks = []
        self.trial_checks = []
        self.rebuild_count = 0
        self.rebuild_seconds = 0.0
        self.unsupported = ()
        if manager is not None:
            from continuous_native_edge_optin import NativeEdgeUnsupported
            self.unsupported = (NativeEdgeUnsupported,)
        self.rebuild()

    def rebuild(self):
        started = time.perf_counter()
        self.native = gtsam.NonlinearFactorGraph()
        for factor in self.graph._pending:
            self.native.add(factor)
        self.rebuild_count += 1
        self.rebuild_seconds += time.perf_counter() - started

    def replace(self, exc, *, rebuild=True):
        factor_id = getattr(exc, 'factor_id', None)
        if factor_id is None or factor_id in self.replaced_ids:
            raise RuntimeError('Native fallback lacks a new stable factor identity') from exc
        count_before = len(self.graph._pending)
        if self.manager is None or not self.manager.replace_unsupported(self.graph, exc):
            raise exc
        if len(self.graph._pending) != count_before:
            raise RuntimeError('Native fallback changed the complete factor count') from exc
        self.replaced_ids.add(factor_id)
        if len(self.replaced_ids) > count_before:
            raise RuntimeError('Native fallback exceeded complete graph factor count') from exc
        self.revision += 1
        if rebuild:
            self.rebuild()

    def error(self, values):
        while True:
            try:
                return float(self.native.error(values))
            except self.unsupported as exc:
                self.replace_failure(exc)

    def replace_failure(self, exc):
        trial=getattr(exc,'trial_values',None)
        if trial is None:
            self.replace(exc)
            return dict(kind='single_unsupported',
                unsupported_factor_ids=[exc.factor_id],
                unsupported_source_pts=[getattr(exc, 'source_pts', None)],
                evaluation_stage=exc.evaluation_stage,
                reasons=[str(getattr(exc,'reason','native_status_2'))],
                graph_rebuilds=1)
        from continuous_native_trial_preflight import trial_unsupported_factors
        # The optimizer still owns its last accepted state while this probes
        # the failing trial.  A typed domain failure must reach its handler.
        try:
            failures,diagnostics=trial_unsupported_factors(self.manager,self.graph,trial)
        finally:
            exc.trial_values=None
            trial=None
        if not any(item.factor_id==exc.factor_id for item in failures):
            raise RuntimeError('Captured unsupported trial does not reproduce its factor')
        self.trial_checks.append(diagnostics)
        for item in failures:
            self.replace(item,rebuild=False)
        self.rebuild()
        return dict(kind='batched_trial_residual',
            unsupported_factor_ids=diagnostics['unsupported_factor_ids'],
            unsupported_source_pts=diagnostics['unsupported_source_pts'],
            evaluation_stage='residual',
            reasons=[str(getattr(item,'reason','native_status_2')) for item in failures],
            graph_rebuilds=1,
            probe_seconds=diagnostics['elapsed_seconds'])

    def verify(self, states, stage):
        if self.manager is None:
            return False
        unsupported, diagnostics = self.manager.verify_accepted(self.graph, states)
        return self.apply_parity(unsupported, diagnostics, stage)

    def apply_parity(self, unsupported, diagnostics, stage):
        """Record parity and replace after callers release temporary GTSAM owners."""
        rebuild_before = self.rebuild_seconds
        for exc in unsupported:
            self.replace(exc, rebuild=False)
        if unsupported:
            self.rebuild()
        self.parity_checks.append(dict(stage=stage, **diagnostics,
            fallback_factor_ids=[exc.factor_id for exc in unsupported],
            fallback_reasons=[str(getattr(exc,'reason','native_status_2'))
                              for exc in unsupported],
            graph_rebuilds=int(bool(unsupported)),
            graph_rebuild_seconds=self.rebuild_seconds-rebuild_before))
        return bool(unsupported)

    def verify_linearization(self, states, stage):
        if self.manager is None or not hasattr(self.manager, '_records'):
            return False
        from continuous_native_linearization_preflight import verify_native_linearization
        digest=hashlib.sha256()
        for key in sorted(states):
            digest.update(np.asarray([key],np.int64).tobytes())
            digest.update(np.asarray(states[key],np.float64).tobytes())
        state_token=digest.digest()
        def token():
            return (state_token,tuple(k for k,r in self.manager._records.items() if not r.fallback))
        if getattr(self.manager,'_verified_linearization_token',None)==token():
            return False
        unsupported,diagnostics=verify_native_linearization(self.manager,self.graph,states)
        rebuild_before = self.rebuild_seconds
        for exc in unsupported:
            self.replace(exc,rebuild=False)
        if unsupported:
            self.rebuild()
        self.linearization_checks.append({**diagnostics,'stage':stage,
            'graph_rebuilds':int(bool(unsupported)),
            'graph_factors_changed':len(unsupported),
            'graph_rebuild_seconds':self.rebuild_seconds-rebuild_before,
            'fallback_reasons':[str(getattr(exc,'reason','native_status_2'))
                                for exc in unsupported]})
        # Cache only the exact state/support identity, never any R/J matrices.
        self.manager._verified_linearization_token=token()
        return bool(unsupported)

    def close(self):
        self.optimizer = None
        self.native = None
        self.graph = None


def complete_graph_error(graph, states, *, native_edge_manager=None,
                         diagnostics_path: Path | None = None):
    """Initial objective check with the same production fallback as solve."""
    session = CompleteFactorSession(graph, native_edge_manager)
    diagnostic = dict(stage='initial_complete_graph_error', status='started',
                      factors=len(graph._pending), states=len(states))
    try:
        session.verify(states, 'initial_preflight')
        diagnostic['native_source_edge_parity'] = session.parity_checks
        if diagnostics_path is not None:
            write_json(Path(diagnostics_path), diagnostic)
        session.verify_linearization(states, 'initial_preflight')
        diagnostic['native_linearization_preflight'] = session.linearization_checks
        if diagnostics_path is not None:
            write_json(Path(diagnostics_path), diagnostic)
        cost = session.error(values_for(graph, states))
        diagnostic.update(status='completed', cost=cost)
        return cost
    except Exception as exc:
        diagnostic.update(status='failed', exception_type=type(exc).__name__,
            reason=str(getattr(exc, 'reason', str(exc))),
            factor_id=getattr(exc, 'factor_id', None),
            source_pts=getattr(exc, 'source_pts', None),
            evaluation_stage=getattr(exc, 'evaluation_stage', None))
        raise
    finally:
        diagnostic['native_source_edge_parity'] = session.parity_checks
        diagnostic['native_linearization_preflight'] = session.linearization_checks
        diagnostic['native_trial_preflight'] = session.trial_checks
        diagnostic['graph_rebuilds'] = session.rebuild_count
        diagnostic['graph_rebuild_seconds'] = session.rebuild_seconds
        if native_edge_manager is not None:
            diagnostic['native_fallbacks'] = native_edge_manager.metadata().get(
                'fallback_events', [])
        if diagnostics_path is not None:
            write_json(Path(diagnostics_path), diagnostic)
        session.close()


def _fresh_isam_submission(graph):
    # A failed iSAM update can partly mutate its smoother. Keep the complete
    # owning graph untouched, including when update succeeded before a later
    # objective check discovered an unsupported native trial.
    fresh = type(graph)(graph.state_dim, window_size=graph.window_size,
        relinearize_threshold=graph._relinearize_threshold,
        relinearize_skip=graph._relinearize_skip,
        refinement_updates=graph.refinement_updates,
        optimization=graph._optimization,
        dogleg_initial_delta=graph._dogleg_initial_delta,
        diagnose_cost=graph._diagnose_cost)
    fresh._pending = list(graph._pending)
    fresh._pending_ids = set(graph._pending_ids)
    fresh._prior_initials = dict(graph._prior_initials)
    fresh._callbacks = list(graph._callbacks)
    return fresh


def solve(graph, initial, output: Path, *, max_iterations=80,
          relative_tolerance=1e-6, resume=False, native_edge_manager=None,
          finalize_with_isam2=True):
    session = CompleteFactorSession(graph, native_edge_manager)
    try:
        return _solve(graph, initial, output, session, max_iterations=max_iterations,
                      relative_tolerance=relative_tolerance, resume=resume,
                      finalize_with_isam2=finalize_with_isam2)
    finally:
        session.close()


def _solve(graph, initial, output, session, *, max_iterations,
           relative_tolerance, resume, finalize_with_isam2):
    """LM with safe domain retries, optionally finalized by actual iSAM2.

    Iteration limits are numerical safeguards, not accuracy/acceptance criteria.
    Persist the last accepted state on every improvement for resumable work.
    """
    output.mkdir(parents=True, exist_ok=True)
    states_path = output / "accepted_states.npz"
    trace_path = output / "solver_trace.json"
    trace = {"method": "GTSAM_LM_domain_rejection_then_iSAM2", "steps": [],
             "callback": graph.callback_metadata, "factors": len(graph._pending),
             "states": len(initial), "source_rows_removed_on_trial_failure": 0,
             "accepted_cost_source": "optimizer.error cached accepted objective; first and final states independently checked"}
    damping = 1e-3
    if resume:
        with np.load(states_path, allow_pickle=False) as z:
            stored = {int(k): q.copy() for k, q in zip(z["keys"], z["states"])}
        if set(stored) != set(initial):
            raise ValueError("Resume state keys differ from this model")
        initial = stored
        trace = json.loads(trace_path.read_text(encoding="utf8"))
        if trace["steps"]:
            damping = trace["steps"][-1]["lambda_next"]
    elif states_path.exists() or trace_path.exists():
        raise ValueError("Existing solver state requires explicit resume")
    trace['finalize_with_isam2'] = bool(finalize_with_isam2)
    trace['method'] = ('GTSAM_LM_domain_rejection_then_iSAM2' if finalize_with_isam2
                       else 'GTSAM_LM_domain_rejection')
    values = values_for(graph, initial)
    preflight = dict(status='started', phase_seconds={},
                     graph_rebuilds_before=session.rebuild_count)
    trace.setdefault('initial_preflight_runs', []).append(preflight)
    try:
        stage_started = time.perf_counter()
        try:
            session.verify(initial, 'initial_solver')
        finally:
            preflight['phase_seconds']['candidate_parity'] = (
                time.perf_counter()-stage_started)
            trace['native_source_edge_parity'] = session.parity_checks
            write_json(trace_path, trace)
        stage_started = time.perf_counter()
        try:
            session.verify_linearization(initial, 'initial_solver')
        finally:
            preflight['phase_seconds']['accepted_jacobian_preflight'] = (
                time.perf_counter()-stage_started)
            trace['native_linearization_preflight'] = session.linearization_checks
            write_json(trace_path, trace)
        stage_started = time.perf_counter()
        try:
            cost = session.error(values)
        finally:
            preflight['phase_seconds']['full_graph_error'] = (
                time.perf_counter()-stage_started)
        if not np.isfinite(cost):
            raise ValueError("Initial model has nonfinite objective")
        stage_started = time.perf_counter()
        try:
            _prepare_curves(graph, initial, trace, 'initial_accepted')
        finally:
            preflight['phase_seconds']['adaptive_curve_prepare'] = (
                time.perf_counter()-stage_started)
        preflight.update(status='completed', cost=cost,
            graph_rebuilds=session.rebuild_count-preflight['graph_rebuilds_before'])
    except Exception as exc:
        preflight.update(status='failed', **_error_record(exc),
            graph_rebuilds=session.rebuild_count-preflight['graph_rebuilds_before'])
        if isinstance(exc, BoardIntersectionError):
            write_json(output/'solver_failure.json', dict(
                reason=('accepted_state_board_jacobian_undefined'
                        if getattr(exc, 'evaluation_stage', None)=='jacobian'
                        else 'initial_board_domain_failure'),
                detail=exc.reason, family=getattr(exc,'factor_family',None),
                source_keys=getattr(exc,'source_keys',None),
                preserved=str(states_path) if states_path.exists() else None))
        trace['native_trial_preflight'] = session.trial_checks
        write_json(trace_path, trace)
        raise
    trace.setdefault("initial_cost", cost)
    if session.manager is not None:
        trace['native_source_edge_parity'] = session.parity_checks
        trace['native_linearization_preflight'] = session.linearization_checks
        trace['native_trial_preflight'] = session.trial_checks
    save_states(states_path, initial)
    write_json(trace_path, trace)

    def optimizer_at(value, lam):
        params = gtsam.LevenbergMarquardtParams()
        params.setDiagonalDamping(True)
        params.setlambdaInitial(float(max(lam, 1e-12)))
        params.setlambdaUpperBound(1e14)
        params.setRelativeErrorTol(relative_tolerance)
        params.setMaxIterations(max_iterations)
        while True:
            try:
                return gtsam.LevenbergMarquardtOptimizer(session.native, value, params)
            except session.unsupported as exc:
                session.replace_failure(exc)

    optimizer_started = time.perf_counter()
    session.optimizer = optimizer_at(values, damping)
    trace['initial_optimizer_seconds'] = time.perf_counter()-optimizer_started
    write_json(trace_path, trace)
    started = time.perf_counter()
    converged = False
    termination_reason = "iteration_limit"
    accepted = 0
    iteration_offset = len(trace['steps']) if resume else 0
    for iteration in range(iteration_offset, iteration_offset+max_iterations):
        tick = time.perf_counter()
        phase_seconds = {}
        accepted_states = states_for(graph, values, initial)
        stage_started = time.perf_counter()
        try:
            _prepare_curves(graph, accepted_states, trace,
                            f'LM_{iteration}_accepted_preflight')
        except BoardIntersectionError as exc:
            write_json(output/"solver_failure.json", dict(
                reason="accepted_state_board_jacobian_undefined",
                detail=exc.reason,
                family=getattr(exc, "factor_family", None),
                source_keys=getattr(exc, "source_keys", None),
                preserved=str(states_path)))
            raise
        finally:
            _add_elapsed(phase_seconds, 'adaptive_curve_prepare', stage_started)
        stage_started = time.perf_counter()
        try:
            if session.verify_linearization(accepted_states,'before_LM_step'):
                session.optimizer = None
                session.optimizer = optimizer_at(values,damping)
        finally:
            _add_elapsed(phase_seconds, 'accepted_jacobian_preflight', stage_started)
        before_cost = cost
        before_lambda = float(session.optimizer.lambda_())
        try:
            while True:
                try:
                    stage_started = time.perf_counter()
                    try:
                        _prepare_curves(graph, accepted_states, trace,
                                        f'LM_{iteration}_before_iterate')
                    finally:
                        _add_elapsed(phase_seconds, 'adaptive_curve_prepare', stage_started)
                    stage_started = time.perf_counter()
                    try:
                        session.optimizer.iterate()
                    finally:
                        _add_elapsed(phase_seconds, 'lm_iterate', stage_started)
                    candidate_values = gtsam.Values(session.optimizer.values())
                    rebuild_before = session.rebuild_seconds
                    stage_started = time.perf_counter()
                    try:
                        changed = session.verify(
                            states_for(graph,candidate_values,initial), 'LM_candidate')
                    finally:
                        _add_elapsed(phase_seconds, 'candidate_parity', stage_started)
                        phase_seconds['graph_rebuild'] = (
                            phase_seconds.get('graph_rebuild', 0.0)
                            + session.rebuild_seconds-rebuild_before)
                    candidate_values = None
                    if changed:
                        stage_started = time.perf_counter()
                        try:
                            equivalent_cost = session.error(values)
                        finally:
                            _add_elapsed(phase_seconds, 'full_graph_error', stage_started)
                        if abs(equivalent_cost-cost) > 1e-7*max(1.,abs(cost)):
                            raise RuntimeError('Native parity fallback changed previous accepted objective')
                        cost = equivalent_cost
                        session.optimizer = None
                        stage_started = time.perf_counter()
                        try:
                            session.optimizer = optimizer_at(values, before_lambda)
                        finally:
                            _add_elapsed(phase_seconds, 'retry', stage_started)
                        continue
                    break
                except session.unsupported as exc:
                    # A numerical implementation fallback is not a physical
                    # trial rejection: retry from the same accepted Values and
                    # lambda without consuming an LM iteration or deleting UV.
                    rebuild_before = session.rebuild_seconds
                    stage_started = time.perf_counter()
                    try:
                        fallback = session.replace_failure(exc)
                    finally:
                        _add_elapsed(phase_seconds, 'native_trial_probe_and_fallback',
                                     stage_started)
                        phase_seconds['graph_rebuild'] = (
                            phase_seconds.get('graph_rebuild', 0.0)
                            + session.rebuild_seconds-rebuild_before)
                    stage_started = time.perf_counter()
                    try:
                        equivalent_cost = session.error(values)
                    finally:
                        _add_elapsed(phase_seconds, 'full_graph_error', stage_started)
                    if abs(equivalent_cost-cost) > 1e-7*max(1., abs(cost)):
                        raise RuntimeError('Complete native fallback changed accepted objective')
                    cost = equivalent_cost
                    trace.setdefault('native_fallbacks', []).append(dict(
                        iteration=iteration, factor_id=getattr(exc,'factor_id',None),
                        source_keys=getattr(exc,'source_keys',None),
                        evaluation_stage=getattr(exc,'evaluation_stage',None),
                        lambda_retry=before_lambda, cost=cost,
                        fallback=fallback, phase_seconds=dict(phase_seconds)))
                    trace['native_trial_preflight'] = session.trial_checks
                    write_json(trace_path, trace)
                    session.optimizer = None
                    stage_started = time.perf_counter()
                    try:
                        session.optimizer = optimizer_at(values, before_lambda)
                    finally:
                        _add_elapsed(phase_seconds, 'retry', stage_started)
        except BoardIntersectionError as exc:
            if getattr(exc, "evaluation_stage", None) == "jacobian":
                write_json(output/"solver_failure.json",dict(
                    reason="accepted_state_board_jacobian_undefined",detail=exc.reason,
                    family=getattr(exc,"factor_family",None),
                    source_keys=getattr(exc,"source_keys",None),preserved=str(states_path)))
                raise
            # GTSAM preserves its previous estimate when evaluating a trial
            # raises. Verify that property before attempting increased damping.
            if session.optimizer is None:
                raise RuntimeError('Trial domain failure lost the LM optimizer') from exc
            retained = gtsam.Values(session.optimizer.values())
            stage_started = time.perf_counter()
            try:
                retained_cost = session.error(retained)
            finally:
                _add_elapsed(phase_seconds, 'full_graph_error', stage_started)
            if abs(retained_cost - cost) > 1e-7 * max(1., cost):
                raise RuntimeError("Invalid trial changed accepted optimizer state") from exc
            damping = max(before_lambda, float(session.optimizer.lambda_())) * 10.
            row = dict(iteration=iteration, status="rejected_board_domain",
                       reason=exc.reason, point_indices=exc.indices[:16].tolist(),
                       family=getattr(exc,"factor_family",None),
                       source_keys=getattr(exc,"source_keys",None),
                       cost=cost, lambda_next=damping,
                       seconds=time.perf_counter()-tick,
                       phase_seconds=phase_seconds,
                       trial_probe=getattr(exc, 'trial_probe', None))
            trace["steps"].append(row)
            write_json(trace_path, trace)
            print(json.dumps(row), flush=True)
            if not np.isfinite(damping) or damping > 1e14:
                raise RuntimeError("Board-domain retries exhausted numerical damping") from exc
            session.optimizer = None
            stage_started = time.perf_counter()
            try:
                session.optimizer = optimizer_at(retained, damping)
            finally:
                _add_elapsed(phase_seconds, 'retry', stage_started)
            row['phase_seconds'] = dict(phase_seconds)
            row['seconds'] = time.perf_counter()-tick
            write_json(trace_path, trace)
            continue
        new_values = gtsam.Values(session.optimizer.values())
        new_cost = float(session.optimizer.error())
        revision_before_check = session.revision
        if accepted == 0:
            stage_started = time.perf_counter()
            try:
                independently_evaluated = session.error(new_values)
            finally:
                _add_elapsed(phase_seconds, 'full_graph_error', stage_started)
            if (not np.isfinite(independently_evaluated) or
                    abs(independently_evaluated-new_cost)>1e-10*max(1.,abs(new_cost))):
                raise RuntimeError('GTSAM accepted cached cost differs from graph evaluation')
        if not np.isfinite(new_cost) or new_cost > cost + 1e-7 * max(1., cost):
            raise RuntimeError("GTSAM returned an invalid or increasing-cost estimate")
        delta = (cost-new_cost)/max(abs(cost), 1e-12)
        values, cost = new_values, new_cost
        damping = float(session.optimizer.lambda_())
        if session.revision != revision_before_check:
            session.optimizer = optimizer_at(values, damping)
        row = dict(iteration=iteration, status="accepted", cost_before=before_cost,
                   cost=cost, relative_decrease=delta, lambda_next=damping,
                   seconds=time.perf_counter()-tick,
                   phase_seconds=phase_seconds)
        trace["steps"].append(row)
        accepted += 1
        save_states(states_path, states_for(graph, values, initial))
        write_json(trace_path, trace)
        print(json.dumps(row), flush=True)
        if 0 <= delta <= relative_tolerance:
            converged = True
            termination_reason = "relative_objective_stagnation_not_gradient_certificate"
            break
    solved = states_for(graph, values, initial)
    stage_started = time.perf_counter()
    try:
        independently_evaluated = session.error(values)
    finally:
        trace['final_lm_full_graph_error_seconds'] = (
            time.perf_counter()-stage_started)
    if (not np.isfinite(independently_evaluated) or
            abs(independently_evaluated-cost)>1e-10*max(1.,abs(cost))):
        raise RuntimeError('Final LM cached cost differs from graph evaluation')
    trace.update(lm_converged=converged, lm_final_cost=cost,
                 termination_reason=termination_reason,
                 first_order_optimality_verified=False,
                 accepted_steps_this_run=accepted,
                 solve_seconds=time.perf_counter()-started)
    if not finalize_with_isam2:
        session.optimizer = None
        trace['isam2'] = dict(completed=False, adopted=False,
            reason='explicit_LM_diagnostic', preserved='accepted LM state',
            phase_seconds={})
        if session.manager is not None:
            trace['native_source_edge'] = session.manager.metadata()
            trace['native_source_edge_parity'] = session.parity_checks
            trace['native_linearization_preflight'] = session.linearization_checks
            trace['native_trial_preflight'] = session.trial_checks
        trace['final_cost'] = cost
        trace['accepted_as_accurate'] = False
        save_states(output / 'states.npz', solved)
        write_json(trace_path, trace)
        return solved, trace
    # iSAM2 uses the same complete source objective and the verified callback.
    # Keep the LM state available even when an incremental trial is invalid.
    session.optimizer = None
    submission = None
    isam = None
    isam_phases = {}
    try:
        stage_started = time.perf_counter()
        try:
            _prepare_curves(graph, solved, trace, 'before_iSAM')
        finally:
            _add_elapsed(isam_phases, 'adaptive_curve_prepare', stage_started)
        stage_started = time.perf_counter()
        try:
            changed = session.verify_linearization(solved, 'before_iSAM')
        finally:
            _add_elapsed(isam_phases, 'accepted_jacobian_preflight', stage_started)
        if changed:
            stage_started = time.perf_counter()
            equivalent_cost = session.error(values)
            _add_elapsed(isam_phases, 'full_graph_error', stage_started)
            if abs(equivalent_cost-cost) > 1e-7*max(1.,abs(cost)):
                raise RuntimeError('Native linearization fallback changed final LM objective')
            cost = equivalent_cost
        while True:
            try:
                submission = (_fresh_isam_submission(graph)
                              if session.manager is not None else graph)
                stage_started = time.perf_counter()
                isam = submission.update(solved)
                _add_elapsed(isam_phases, 'update', stage_started)
                stage_started = time.perf_counter()
                if session.manager is None:
                    parity_failures, parity_diagnostics = [], {}
                else:
                    parity_failures, parity_diagnostics = (
                        session.manager.verify_accepted(graph, isam))
                if parity_failures:
                    _add_elapsed(isam_phases, 'candidate_parity', stage_started)
                    isam = None
                    submission = None
                    stage_started = time.perf_counter()
                    try:
                        session.apply_parity(parity_failures, parity_diagnostics,
                                             'iSAM_candidate')
                    finally:
                        _add_elapsed(isam_phases, 'fallback_and_rebuild', stage_started)
                    continue
                if session.manager is not None:
                    session.apply_parity(parity_failures, parity_diagnostics,
                                         'iSAM_candidate')
                _add_elapsed(isam_phases, 'candidate_parity', stage_started)
                # Let this exception restart the entire temporary iSAM update;
                # an altered factor must not keep an old linearization tree.
                stage_started = time.perf_counter()
                isam_cost = float(session.native.error(values_for(graph, isam)))
                _add_elapsed(isam_phases, 'full_graph_error', stage_started)
                break
            except session.unsupported as exc:
                # Drop the temporary smoother and candidate before replacing
                # factors in the owning graph and creating a fresh submission.
                isam = None
                submission = None
                stage_started = time.perf_counter()
                try:
                    session.replace_failure(exc)
                finally:
                    _add_elapsed(isam_phases, 'fallback_and_rebuild', stage_started)
        kept = np.isfinite(isam_cost) and isam_cost <= cost + 1e-7 * max(1., cost)
        trace["isam2"] = dict(completed=True, cost=isam_cost, adopted=bool(kept),
                               update=submission.last_update_summary,
                               phase_seconds=isam_phases)
        if kept:
            solved, cost = isam, isam_cost
    except BoardIntersectionError as exc:
        trace["isam2"] = dict(completed=False, adopted=False,
                               reason=exc.reason, preserved="accepted LM state",
                               phase_seconds=isam_phases)
    finally:
        isam = None
        submission = None
    if session.manager is not None:
        trace['native_source_edge'] = session.manager.metadata()
        trace['native_source_edge_parity'] = session.parity_checks
        trace['native_linearization_preflight'] = session.linearization_checks
        trace['native_trial_preflight'] = session.trial_checks
    trace["final_cost"] = cost
    trace["accepted_as_accurate"] = False
    save_states(output / "states.npz", solved)
    write_json(trace_path, trace)
    return solved, trace
