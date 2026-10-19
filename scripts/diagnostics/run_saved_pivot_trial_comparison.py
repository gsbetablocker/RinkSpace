"""Compare a saved full direction at the real source owner, then keep solving.

This observer neither restores a normal nor changes any acceptance gate.
Diagnostic errors are reported without interrupting the continuous solver.
"""
from pathlib import Path
import argparse
import functools
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from calibration.solver_runtime.checkpoint_history import read_solver_history


def sha(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def bind_live_driver_solver_observer(driver, original, observer):
    """Bind the exact solver name used by the installed fresh-J main.

    Execution installation clones ``main`` into an authenticated namespace.
    Updating only the module attribute leaves that live namespace untouched.
    Refuse an unexpected binding rather than silently omit the diagnostic.
    """
    namespace = driver.main.__globals__
    if namespace.get('solve_joint_source') is not original or driver.solve_joint_source is not original:
        raise RuntimeError('Installed live main and driver solver bindings differ')
    namespace['solve_joint_source'] = observer
    driver.solve_joint_source = observer
    return dict(live_fresh_J_main_namespace_bound=True,
                original_installed_solver_called_after_diagnostic=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retraction-baseline-run', type=Path, required=True)
    args, original_arguments = parser.parse_known_args()
    sys.argv = [sys.argv[0], *original_arguments]
    import calibration.solver_runtime.__main__ as entry
    from calibration.solver_runtime import execution_acceleration as execution
    from scripts.diagnostics.capture_current_complete_SOC_face_response import original_closure_bindings
    original_install = execution.install_named66_execution

    @functools.wraps(original_install)
    def install_observer(driver, directory):
        receipt = original_install(driver, directory)
        original_solve = driver.solve_joint_source

        @functools.wraps(original_solve)
        def compare_then_solve(evaluate, q, quadratic, scales, output, guard, **options):
            import numpy as np
            from installation_mount_candidate.mount_graph import reinitialize_invalid_pivots_outside
            from z190_joint_source_solver import _frozen_profile_token
            from z190_streaming_source_normal import SourceBlockContract, BlockSignature, replay_source_cost
            started = time.perf_counter()
            result = dict(schema='rinkspace.saved-full-direction-original-pivot-retraction-source-comparison.v1',
                baseline_run=str(args.retraction_baseline_run.resolve()),
                extra_source_J_or_QP=False, old_normal_not_restored=True,
                production_all_acceptance_gates_unchanged=True,
                diagnostic_not_a_candidate_or_accuracy_acceptance=True)
            try:
                history = read_solver_history(args.retraction_baseline_run/'solver/iterations.json')
                token = history[0]['global_schur_trial_records'][-1]['trial_feasibility_recovery']['complete_state_recovery_identity']['proposed_q_sha256']
                files = list((args.retraction_baseline_run/'actual_complete_source_directions').glob('*.npz'))
                assert len(files) == 1, 'One actual retained complete direction required'
                with np.load(files[0], allow_pickle=False) as z:
                    step = z['step'].copy()
                    saved_q = z['q'].copy()
                np.testing.assert_array_equal(saved_q, q)
                assert step.shape == q.shape == (34340,)
                assert sha(q+(1/2048)*step) == token
                with np.load(args.retraction_baseline_run/'first_complete_source_normal_numeric.npz', allow_pickle=False) as z:
                    old_contract = json.loads(str(z['source_contract_json']))
                    parent_cost = float(z['cost'])
                current_run = Path(output).parent
                with np.load(current_run/'first_complete_source_normal_numeric.npz', allow_pickle=False) as z:
                    current_contract = json.loads(str(z['source_contract_json']))
                    current_cost = float(z['cost'])
                    np.testing.assert_array_equal(z['q'], q)
                assert old_contract['frozen_profile_token'] == current_contract['frozen_profile_token']
                assert old_contract['row_count'] == current_contract['row_count'] == 4502662
                assert [(s['rows'], s['inventory_sha256']) for s in old_contract['signatures']] == [
                    (s['rows'], s['inventory_sha256']) for s in current_contract['signatures']]
                contract = SourceBlockContract(current_contract['parameter_count'], current_contract['row_count'],
                    tuple(BlockSignature(**s) for s in current_contract['signatures']), current_contract['frozen_profile_token'])
                assert _frozen_profile_token(options['update_profiles']) == contract.frozen_profile_token
                recovery = options['trial_feasibility_recovery']
                closure = original_closure_bindings(recovery)
                state = closure['state']; layout = closure['layout']; evaluation_q = closure['evaluation_q']
                records = []
                for alpha in (1/32, 1/512, 1/2048):
                    proposal = q + alpha*step
                    minimal = recovery(q, proposal)
                    old_eval, old_report = reinitialize_invalid_pivots_outside(state, evaluation_q(proposal))
                    old = layout.canonical_q(old_eval)
                    comparisons = []
                    for name, value in (('original_FD_reserve_recovery', old), ('original_domain_boundary_retraction', minimal['q'])):
                        begun = time.perf_counter()
                        cost = replay_source_cost(lambda: options['source_block_factory'](value, False),
                            contract, frozen_profile_token=contract.frozen_profile_token)
                        try:
                            physical_truth = guard(value) is not False
                            failure = None
                        except (ValueError, RuntimeError) as error:
                            physical_truth = False; failure = str(error)
                        comparisons.append(dict(mode=name, complete_q_sha256=sha(value),
                            changed_canonical_columns=np.flatnonzero(value != proposal).tolist(),
                            actual_complete_source_cost=cost, actual_cost_change=cost-current_cost,
                            original_complete_physical_and_truth_guard=physical_truth, failure=failure,
                            source_value_and_guard_seconds=time.perf_counter()-begun))
                    records.append(dict(alpha=alpha, proposed_q_sha256=sha(proposal),
                        old_recovery=old_report, minimal_recovery=minimal['report'], comparisons=comparisons))
                result.update(passed=True, same_complete_q_and_frozen_profiles_verified=True,
                    original33_ordered_source_inventory_verified=True, complete_source_rows=contract.row_count,
                    saved_direction=str(files[0]), full_step_sha256=sha(step),
                    historical_parent_cost=parent_cost, current_actual_parent_cost=current_cost,
                    parent_cost_difference=current_cost-parent_cost, records=records)
            except Exception as error:
                result.update(passed=False, diagnostic_error_type=type(error).__name__, diagnostic_error=str(error),
                    original_continuous_solver_still_required=True)
            finally:
                # Restore original nominal bindings before the untouched solve.
                try:
                    guard(q)
                except Exception as error:
                    result['nominal_guard_restore_error'] = str(error)
            result['extra_diagnostic_wall_seconds'] = time.perf_counter()-started
            path = Path(directory)/'actual_saved_full_direction_pivot_retraction_source_comparison.json'
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else str(v))+'\n', encoding='utf8')
            print(json.dumps(dict(stage='saved_full_direction_pivot_retraction_source_comparison',
                passed=result.get('passed'), seconds=result['extra_diagnostic_wall_seconds'], report=str(path))), flush=True)
            return original_solve(evaluate, q, quadratic, scales, output, guard, **options)

        binding = bind_live_driver_solver_observer(driver, original_solve, compare_then_solve)
        receipt['saved_full_direction_pivot_retraction_observer'] = dict(
            **binding, helper_path=str(Path(__file__).resolve()),
            helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            diagnostic_does_not_approve_or_change_a_trial=True)
        return receipt

    execution.install_named66_execution = install_observer
    entry.main()


if __name__ == '__main__':
    main()
