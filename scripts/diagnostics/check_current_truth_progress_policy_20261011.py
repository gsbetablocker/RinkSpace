"""One fresh production normal, three unaccepted truth-progress shadows.

Run through the public named66 bootstrap. The driver builds the real source
and its first normal; this tap replaces the optimization entry, not its math.
No candidate checkpoint, source image or normal/J cache is written.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import dis
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ShadowProbeComplete(BaseException):
    """Unwind the real driver's worker cleanup without accepting a state."""


def _entry_functions(entry):
    """Inspect only main/wrapper closures, not the driver's whole source graph."""
    pending = [entry]
    seen = set()
    while pending:
        function = pending.pop()
        if not inspect.isfunction(function) or id(function) in seen:
            continue
        seen.add(id(function))
        yield function
        wrapped = getattr(function, '__wrapped__', None)
        if wrapped is not None:
            pending.append(wrapped)
        for cell in function.__closure__ or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if inspect.isfunction(value):
                pending.append(value)


@contextmanager
def bind_solver_entry(driver, replacement):
    """Bind the module and installed main's copied global namespaces together."""
    namespaces = [vars(driver)]
    call_sites = []
    for function in _entry_functions(driver.main):
        loads = sum(instruction.opname == 'LOAD_GLOBAL'
                    and instruction.argval == 'solve_joint_source'
                    for instruction in dis.get_instructions(function))
        if loads:
            namespace = function.__globals__
            if 'solve_joint_source' not in namespace:
                raise ValueError('Actual main global source solver is missing')
            namespaces.append(namespace)
            call_sites.append((function, loads))
    if not call_sites:
        raise ValueError('No actual main/wrapper global source-solver call site was verified')
    saved = []
    seen = set()
    for namespace in namespaces:
        if id(namespace) in seen:
            continue
        seen.add(id(namespace))
        saved.append((namespace, namespace['solve_joint_source']))
        namespace['solve_joint_source'] = replacement
    try:
        for function, _ in call_sites:
            if function.__globals__['solve_joint_source'] is not replacement:
                raise ValueError('Installed main does not resolve the shadow solver')
        yield call_sites
    finally:
        for namespace, original in reversed(saved):
            namespace['solve_joint_source'] = original


def prove_solver_entry_binding(driver):
    """Exercise each real LOAD_GLOBAL namespace without running main/source setup."""
    class EntrySentinel(BaseException):
        pass
    hits = []
    def sentinel():
        hits.append(True)
        raise EntrySentinel()
    with bind_solver_entry(driver, sentinel) as sites:
        for function, _ in sites:
            try:
                # This name uses precisely the namespace and global lookup
                # proved above in the installed production main bytecode.
                eval(compile('solve_joint_source()', '<shadow-entry-sentinel>', 'eval'),
                     function.__globals__)
            except EntrySentinel:
                continue
            raise ValueError('Actual main global call did not reach the shadow sentinel')
    if len(hits) != len(sites):
        raise ValueError('Incomplete shadow entry sentinel proof')
    return dict(verified_main_solver_call_sites=sum(count for _, count in sites),
                actual_global_namespaces_exercised=len(sites),
                production_solver_called=False, source_graph_constructed=False)


def digest(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def emit(stage, **values):
    import numpy as np
    def default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    print(json.dumps(dict(stage=stage, **values), default=default,
                     ensure_ascii=False, allow_nan=False), flush=True)


def closure(function, name):
    """Walk installed callback wrappers; never guess an owner's substitute."""
    seen = set()
    pending = [function]
    while pending:
        current = pending.pop()
        if id(current) in seen or not callable(current):
            continue
        seen.add(id(current))
        try:
            values = inspect.getclosurevars(current).nonlocals
        except TypeError:
            continue
        if name in values:
            return values[name]
        pending.extend(value for value in values.values() if inspect.isfunction(value))
        if getattr(current, '__wrapped__', None) is not None:
            pending.append(current.__wrapped__)
    raise ValueError('Original production callback owner unavailable: ' + name)


def driver_arguments(config, output):
    """Only output/cache controls differ; source options retain their bytes."""
    if config.get('driver') != 'named66':
        raise ValueError('Only the actual named66 production configuration is supported')
    options = dict(config['source_options'])
    forbidden = ('--evaluate-only', '--save-initial-linearization-cache',
                 '--current-linearization-cache')
    if any(options.get(key) for key in forbidden):
        raise ValueError('The shadow must build exactly one fresh normal without a disk cache')
    # Physical-event candidate caches are not needed by this shadow runner.
    options['--save-complete-source-directions'] = False
    args = ['named66-shadow', '--output', str(output)]
    for key, value in options.items():
        if isinstance(value, bool):
            if value:
                args.append(key)
        else:
            if isinstance(value, (dict, list, tuple)) or value is None:
                raise ValueError('Unsupported source CLI value: ' + key)
            args.extend((key, str(value)))
    return args


def physical_source_guard(guard):
    """Strip only the driver's two explicit truth-cap wrappers for labeling."""
    current = guard
    names = ('original_source_and_human_guard', 'original_complete_source_guard')
    while True:
        try:
            values = inspect.getclosurevars(current).nonlocals
        except TypeError:
            return current
        parent = next((values[name] for name in names if name in values), None)
        if parent is None:
            return current
        current = parent


def make_shadow_tap(config, output, tapped_contract):
    import numpy as np
    from calibration.solver_runtime.bounded_motion_schur import BoundedMotionSchurModel
    from calibration.solver_runtime.original_halfspace_primal import original_full_qp_kkt_verified
    from calibration.solver_runtime.physical_signed_margin_SOC import require_identity_trial_recovery
    from calibration.solver_runtime.restoration_trial_model_consistency import with_restoration_trial_model_consistency
    from calibration.solver_runtime.staged_source_guard import preserve_programme_camera_bindings
    import calibration.solver_runtime.optional_soc_runtime as optional_soc
    from scripts.diagnostics.truth_progress_domain_probe_20261011 import build_truth_progress_probe_domain
    import z190_joint_source_solver as joint
    import z190_streaming_source_normal as streaming

    def tap(evaluate, q, quadratic, scales, solver_output, physicalguard, **kwargs):
        start = time.perf_counter()
        q = np.asarray(q, float).copy()
        initial_q = q.copy()
        expected = config['continuation']['state']['q_sha256']
        if digest(q) != expected:
            raise ValueError('Shadow parent differs from the configured complete q')
        block_factory = kwargs['source_block_factory']
        token = joint._frozen_profile_token(kwargs['update_profiles'])
        fixed = np.asarray(kwargs['fixed_parameter_indices'], int)
        # The real driver installed this first-call reuse wrapper immediately
        # before our entry; invoking it does not recompute the source J.
        normal = streaming.accumulate_source_normal(
            lambda: block_factory(q, True), len(q), expected_row_count=len(scales),
            frozen_profile_token=token, fixed_parameter_indices=fixed,
            retain_native_replay=True, source_state=q)
        if (normal.contract.frozen_profile_token != token
                or normal.contract.row_count != config['continuation']['expected_complete_active_source_scalars']):
            raise ValueError('Complete source row/profile identity changed')
        selected = np.asarray(kwargs['global_schur_indices'], int)
        selected = selected[~np.isin(selected, fixed)]
        positive = np.asarray(kwargs.get('positive_parameter_indices', ()), int)
        positive = positive[~np.isin(positive, fixed)]
        domain = kwargs['shared_linear_constraints'](q.copy(), selected.copy())
        capture = next((item for item in reversed(tapped_contract)
                        if item['matrix'] is domain['matrix']
                        and np.array_equal(item['result_lower'], domain['lower'])), None)
        if capture is None:
            raise ValueError('No exact original b0/ownership tap matches this fresh driver domain')
        acceptance = with_restoration_trial_model_consistency(kwargs['source_trial_acceptance'])
        truth_values = closure(acceptance, 'complete_truth_violations')
        blocks = closure(truth_values, 'human_truth').state.blocks
        with preserve_programme_camera_bindings(blocks):
            current, labels = truth_values(q)
        limits_by_family = closure(kwargs['shared_linear_constraints'], 'truth_restoration_limits')
        family_order = list(dict.fromkeys(family for family, _ in labels))
        caps = np.concatenate([limits_by_family[family] for family in family_order])
        if len(current) != config['continuation']['current_truth_group_count']:
            raise ValueError('The shadow does not include all original truth groups')
        domains = [('common_contraction', domain)]
        for policy in ('preserve_current_violation', 'existing_caps'):
            domains.append((policy, build_truth_progress_probe_domain(domain,
                current_group_violations=current, existing_group_limits=caps, policy=policy,
                original_constraint_lower=capture['original_lower'],
                row_group_index=capture['owners'])))
        factory = kwargs['bounded_motion_model_factory']
        moments = factory(q.copy()) if factory is not None else None
        if moments is None:
            raise ValueError('This causal comparison requires the actual bounded motion model')
        moments.eliminate_fixed(fixed)
        normal.native_replay_cache = None
        # All policy models share immutable sparse moments; their solve starts
        # with an empty activity set on each call, exactly as production.
        diagonal = np.maximum(normal.normal.diagonal(), 1.)
        step_lower = np.full(len(selected), -np.inf)
        step_upper = np.full(len(selected), np.inf)
        for col, (lower, upper) in kwargs.get('shared_parameter_bounds', {}).items():
            if col in fixed:
                if not lower <= q[col] <= upper:
                    raise ValueError('Fixed original parameter is outside its physical interval')
                continue
            position = int(np.flatnonzero(selected == col)[0])
            step_lower[position] = lower-q[col]
            step_upper[position] = upper-q[col]
        physical_only = physical_source_guard(physicalguard)
        results = []
        directions = {}
        line_search_seed = float(kwargs.get('initial_line_search_fraction', 1.))
        if not 0. < line_search_seed <= 1.:
            raise ValueError('The original line search seed must lie in (0, 1]')
        emit('shadow_same_fresh_normal_ready', q_sha256=digest(q), frozen_profile_token=token,
            parameters=len(q), rows=normal.contract.row_count,
            source_blocks=len(normal.contract.signatures), full_cost=float(normal.cost),
            shared_free=len(selected), fixed=fixed.tolist(),
            common_fraction=domain['report']['truth_feasibility_priority']['demanded_common_original_violation_fraction'],
            original_b0_tapped=True, bounded_interior_points=moments.point_count,
            original_line_search_seed=line_search_seed,
            source_objective_profiles_caps_and_physical_domains_unchanged=True)
        for policy, original_domain in domains:
            model = BoundedMotionSchurModel(moments.chunks, len(q))
            def solve_linear(h, g, diag, **options):
                return model.solve(h, g, diag, joint.schur_trial_linear_step, **options)
            pristine_options = dict(global_count=len(selected),
                global_parameter_indices=selected,
                damping=kwargs['initial_damping'],
                schur_view_lm_ratio=kwargs['schur_view_lm_ratio'],
                nonnegative_global_step_indices=positive[q[positive] <= kwargs['positive_parameter_resolution']],
                global_step_lower_bounds=step_lower, global_step_upper_bounds=step_upper,
                global_step_constraint_matrix=original_domain['matrix'],
                global_step_constraint_lower=np.asarray(original_domain['lower']).copy())
            options = dict(pristine_options)
            emit('shadow_policy_begin', policy=policy)
            linear = solve_linear(normal.normal, normal.gradient, diagonal, **options)
            linear = joint._sequential_measured_shared_source_curvature(q, linear, selected,
                original_domain, options, normal.normal, normal.gradient, diagonal,
                kwargs['shared_step_curvature_tightening'], solve_linear)
            independent = joint._needs_trial_fraction_SOC(linear, original_domain)
            policy_result = None
            # Reuse the production 12-fraction schedule. This is a value/QP
            # diagnostic only: no state update, damping update or acceptance.
            for ordinal in range(12):
                alpha = line_search_seed*2.**(-ordinal)
                candidate_linear = linear
                independently_solved = False
                fraction_failure = None
                try:
                    step = alpha*linear.step
                    if alpha < 1. and independent:
                        try:
                            candidate_linear = joint._trial_fraction_source_step(q, alpha, selected,
                                original_domain, pristine_options, normal.normal, normal.gradient,
                                diagonal, kwargs['shared_step_curvature_tightening'], solve_linear)
                        except (ValueError, RuntimeError) as exc:
                            fraction_failure = str(exc)
                        else:
                            step = candidate_linear.step.copy()
                            independently_solved = True
                    numerical_kkt = original_full_qp_kkt_verified(
                        candidate_linear.report.get('shared_linear_domain_step', {}), candidate_linear.step)
                    if np.any(step[fixed] != 0.):
                        raise ValueError('Shadow changed an original fixed source coordinate')
                    trial = q+step
                    recovery = kwargs.get('trial_feasibility_recovery')
                    if recovery is not None:
                        recovered = recovery(q.copy(), trial.copy())
                        require_identity_trial_recovery(trial, recovered['q'])
                    recovery_identity = True
                except (ValueError, RuntimeError) as exc:
                    emit('shadow_trial_unavailable', policy=policy, alpha=alpha, error=str(exc))
                    continue
                # Replay the exact final displacement, including short-step
                # dynamic targets, through the installed measurement channel.
                measurement_context = {}
                measurement_token = optional_soc._outer_soc_measurement.set(measurement_context)
                try:
                    measured_domain = original_domain['at_trial_fraction'](alpha)
                    with preserve_programme_camera_bindings(blocks):
                        kwargs['shared_step_curvature_tightening'](q.copy(), step.copy(),
                            selected.copy(), measured_domain)
                finally:
                    optional_soc._outer_soc_measurement.reset(measurement_token)
                source_measurement = measurement_context.get('measurement')
                physical_measurement = measurement_context.get('physical_measurement')
                soc_measurable = bool(source_measurement is not None and
                    physical_measurement is not None and physical_measurement.get('certified') is True)
                physical_passed = caps_passed = False
                physical_error = guard_error = None
                try:
                    joint._check_physical(physical_only, trial)
                    physical_passed = True
                except (ValueError, RuntimeError) as exc:
                    physical_error = str(exc)
                try:
                    joint._check_physical(physicalguard, trial)
                    caps_passed = True
                except (ValueError, RuntimeError) as exc:
                    guard_error = str(exc)
                try:
                    with preserve_programme_camera_bindings(blocks):
                        actual, actual_labels = truth_values(trial)
                except (ValueError, RuntimeError) as exc:
                    emit('shadow_trial_unavailable', policy=policy, alpha=alpha,
                        physical_guard_passed=physical_passed, source_truth_error=str(exc))
                    continue
                if labels != actual_labels:
                    raise ValueError('Counterfactual changed actual truth group identities')
                current_passed = bool(np.all(actual <= current+1e-8))
                existing_passed = bool(np.all(actual <= caps+1e-8))
                protected_zero = bool(np.all(actual[current <= 1e-8] <= 1e-8))
                cost = None; cost_error = None; common = None; family_costs = []
                if physical_passed:
                    try:
                        def replay_blocks():
                            for block in block_factory(trial, False):
                                family_costs.append(dict(family=block.row_ledger.get('family'),
                                    rows=len(block.r), cost=streaming._block_cost(
                                        block.r, block.quadratic, block.loss_weights)))
                                yield block
                        cost = streaming.replay_source_cost(replay_blocks,
                            normal.contract, frozen_profile_token=token)
                    except (ValueError, RuntimeError) as exc:
                        cost_error = str(exc)
                source_cost_accepts = bool(cost is not None and
                    cost < normal.cost-1e-10*max(1., normal.cost))
                if caps_passed and cost is not None:
                    with preserve_programme_camera_bindings(blocks):
                        common = acceptance(q.copy(), trial.copy(), before_cost=float(normal.cost),
                            after_cost=cost, default_source_cost_accepts=source_cost_accepts,
                            trial_context=dict(iteration=0, trial=ordinal, line_search_fraction=alpha,
                                alpha=alpha, line_search_mode='original_whole_step', actual_step=step.copy(),
                                predicted_source_reduction=model.model_reduction(normal.normal,normal.gradient,step),
                                physical_and_complete_source_guard_passed=True))
                excess = actual-current
                worst = int(np.argmax(excess))
                policy_guard = (bool(common and common['accept']) if policy == 'common_contraction'
                    else bool(current_passed if policy == 'preserve_current_violation' else existing_passed))
                policy_accepts = bool(numerical_kkt and recovery_identity and physical_passed and caps_passed and
                    policy_guard and cost is not None and
                    (policy == 'common_contraction' or source_cost_accepts))
                diagnostic_comparable = bool(numerical_kkt and soc_measurable and cost is not None)
                policy_result = dict(policy=policy, alpha=alpha,
                    direction_sha256=digest(step), complete_qp_kkt_verified=bool(numerical_kkt),
                    kkt_applies_to_actual_displacement=bool(alpha == 1. or independently_solved),
                    independent_fraction_SOC=independently_solved,
                    independent_fraction_failure=fraction_failure,
                    recovery_identity=recovery_identity, physical_guard_passed=physical_passed,
                    SOC_final_candidate_measurement_certified=soc_measurable,
                    SOC_source_actual_excess_px=None if source_measurement is None else source_measurement['maximum_violation_px'],
                    SOC_physical_valid=None if physical_measurement is None else physical_measurement.get('valid'),
                    unchanged_existing_caps_guard_passed=caps_passed,
                    preserve_current_violation_passed=current_passed,
                    existing_caps_passed=existing_passed, protected_zero_groups_passed=protected_zero,
                    policy_truth_guard_passed=policy_guard,
                    policy_acceptance_counterfactual=policy_accepts,
                    production_candidate_pass=policy_accepts,
                    diagnostic_comparable=diagnostic_comparable,
                    source_cost_accepts=source_cost_accepts,
                    original_common_acceptance_label=None if common is None else bool(common['accept']),
                    original_common_rejection=None if common is None else common.get('rejection_reason'),
                    full_cost=cost, full_cost_change=None if cost is None else cost-normal.cost,
                    all_source_family_costs=family_costs, all_original_truth_violations_px=actual.tolist(),
                    bounded_model_predicted_reduction=model.model_reduction(normal.normal,normal.gradient,step),
                    maximum_current_violation_increase_px=float(excess[worst]),
                    worst_current_group=list(labels[worst]),
                    increased_groups=int(np.count_nonzero(excess > 1e-8)),
                    improved_groups=int(np.count_nonzero(excess < -1e-8)),
                    physical_error=physical_error, existing_guard_error=guard_error,
                    cost_error=cost_error, accuracy_accepted=False)
                emit('shadow_trial_values', **{key: value for key, value in policy_result.items()
                    if key not in ('all_source_family_costs', 'all_original_truth_violations_px')})
                directions[policy] = step.copy()
                if policy_accepts:
                    break
            results.append(policy_result or dict(policy=policy, no_measurable_candidate=True))
            if not np.array_equal(q, initial_q) or joint._frozen_profile_token(kwargs['update_profiles']) != token:
                raise ValueError('Shadow mutated the complete parent q or original profiles')
        if directions:
            # Three complete displacements only; no q/J/H/r/movie cache.
            np.savez_compressed(output/'shadow_policy_directions.npz', parent_q_sha256=np.asarray(digest(q)),
                frozen_profile_token=np.asarray(token), **directions)
        emit('shadow_policy_comparison_complete', parent_q_sha256=digest(q),
            policies=results, elapsed_shadow_seconds=time.perf_counter()-start,
            candidate_accepted_or_checkpointed=False, source_images_written=0,
            full_normal_or_J_arrays_saved=False, saved_direction_names=list(directions))
        raise ShadowProbeComplete()
    return tap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    opt = parser.parse_args()
    config = json.loads(opt.config.read_text(encoding='utf8'))
    output = opt.output.resolve()
    data = Path('D:/CodexData/RinkSpace_Data').resolve()
    if not output.is_relative_to(data) or output.exists() or Path(str(output)+'.bootstrap').exists():
        raise ValueError('Use one new diagnostic output below the established Data root')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ.update(RINKSPACE_MOTION_BACKEND='native', RINKSPACE_MOTION_THREADS='3',
        RINKSPACE_SCATTER_BACKEND='native', RINKSPACE_SCATTER_THREADS='3',
        RINKSPACE_PROJECTION_BACKEND='native', RINKSPACE_PROJECTION_VALUE_BACKEND='opencv',
        RINKSPACE_SOLVER_ACCELERATION='auto')
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate.runtime_backend_contract import install_source_worker_backend_contract
    install_source_worker_backend_contract()
    from calibration.solver_runtime.source_cache_output_relocation import install_output_relocation_compatibility
    install_output_relocation_compatibility()
    import run_current_named66_full_programme as driver
    from calibration.solver_runtime.execution_acceleration import install_named66_execution
    sys.argv = driver_arguments(config, output)
    install_named66_execution(driver, str(output)+'.bootstrap')
    # Install the exact b0 tap after backend patches, before the driver's
    # function-local import. The production function and result stay intact.
    import structured_programme_source_candidate.feasibility_priority_restoration as priority
    original_contract = priority.contract_original_support_bounds
    captures = []
    def contract_tap(lo, hi, matrix, original_lower, owners, violations, *args, **kwargs):
        import numpy as np
        result = original_contract(lo, hi, matrix, original_lower, owners, violations, *args, **kwargs)
        captures[:] = [dict(matrix=matrix, original_lower=np.asarray(original_lower).copy(),
            owners=np.asarray(owners).copy(), result_lower=np.asarray(result['lower']).copy())]
        return result
    priority.contract_original_support_bounds = contract_tap
    emit('shadow_actual_entry_binding_verified', **prove_solver_entry_binding(driver))
    tap = make_shadow_tap(config, output, captures)
    try:
        with bind_solver_entry(driver, tap):
            driver.main()
            raise RuntimeError('The production driver returned without entering the shadow tap')
    except ShadowProbeComplete:
        emit('shadow_driver_workers_closed', no_candidate_accepted=True)
    finally:
        priority.contract_original_support_bounds = original_contract


if __name__ == '__main__':
    main()
