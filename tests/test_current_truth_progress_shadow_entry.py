"""Narrow execution-entry tests: no source graph, normal, fit or media."""
from pathlib import Path
from types import SimpleNamespace, ModuleType
import tempfile

import pytest

from scripts.diagnostics.check_current_truth_progress_policy_20261011 import (
    bind_solver_entry, prove_solver_entry_binding)


def copied_namespace_driver():
    def production():
        raise AssertionError('Production optimizer must never be called')
    original_namespace = {'solve_joint_source': production}
    copied_namespace = dict(original_namespace)
    exec('def main():\n    return solve_joint_source()\n', copied_namespace)
    driver = SimpleNamespace(main=copied_namespace['main'], solve_joint_source=production)
    return driver, copied_namespace, production


def test_copied_main_globals_and_module_bound_restored_on_baseexception():
    driver, namespace, original = copied_namespace_driver()
    class Sentinel(BaseException):
        pass
    def replacement():
        raise Sentinel()
    with pytest.raises(Sentinel):
        with bind_solver_entry(driver, replacement):
            assert driver.solve_joint_source is replacement
            assert namespace['solve_joint_source'] is replacement
            driver.main()
    assert driver.solve_joint_source is original
    assert namespace['solve_joint_source'] is original


def test_copied_namespace_narrow_sentinel_proof():
    driver, namespace, original = copied_namespace_driver()
    proof = prove_solver_entry_binding(driver)
    assert proof['verified_main_solver_call_sites'] == 1
    assert proof['production_solver_called'] is False
    assert driver.solve_joint_source is original
    assert namespace['solve_joint_source'] is original


def test_uncopied_module_globals_are_bound_once_and_restored():
    driver = ModuleType('uncopied_shadow_entry_fixture')
    exec('def solve_joint_source():\n    raise AssertionError("production")\n'
         'def main():\n    return solve_joint_source()\n', vars(driver))
    original = driver.solve_joint_source
    assert driver.main.__globals__ is vars(driver)
    assert prove_solver_entry_binding(driver)['actual_global_namespaces_exercised'] == 1
    assert driver.solve_joint_source is original


def test_main_wrapper_closure_is_also_bound():
    driver, namespace, original = copied_namespace_driver()
    original_main = driver.main
    def wrapper():
        return original_main()
    driver.main = wrapper
    assert prove_solver_entry_binding(driver)['actual_global_namespaces_exercised'] == 1
    assert namespace['solve_joint_source'] is original


def test_missing_actual_callsite_refused_before_driver_execution():
    driver = SimpleNamespace(main=lambda: None, solve_joint_source=lambda: None)
    with pytest.raises(ValueError, match='No actual main'):
        prove_solver_entry_binding(driver)


def test_real_public_installed_main_copied_globals_sentinel():
    # Install the real public runtime and all named66 entry transformations.
    # The sentinel only evaluates the installed main's global function lookup;
    # it never calls main or prepare/build/evaluate any source owner.
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate.runtime_backend_contract import install_source_worker_backend_contract
    install_source_worker_backend_contract()
    from calibration.solver_runtime.source_cache_output_relocation import install_output_relocation_compatibility
    install_output_relocation_compatibility()
    import run_current_named66_full_programme as driver
    from calibration.solver_runtime.execution_acceleration import install_named66_execution
    with tempfile.TemporaryDirectory(prefix='rink-shadow-entry-test-') as directory:
        install_named66_execution(driver, Path(directory))
        original_module_solver = driver.solve_joint_source
        original_actual_solver = driver.main.__globals__['solve_joint_source']
        assert driver.main.__globals__ is not vars(driver)
        proof = prove_solver_entry_binding(driver)
        assert proof['verified_main_solver_call_sites'] == 1
        assert proof['actual_global_namespaces_exercised'] == 1
        assert proof['production_solver_called'] is False
        assert proof['source_graph_constructed'] is False
        assert driver.solve_joint_source is original_module_solver
        assert driver.main.__globals__['solve_joint_source'] is original_actual_solver


def test_shadow_tap_consumes_one_normal_and_replays_all_three_policy_values(monkeypatch, tmp_path, capsys):
    """Exercise the complete tap wiring with tiny original-style source owners."""
    import json
    import numpy as np
    from scipy.sparse import eye
    from calibration.solver_runtime import install
    install()
    import z190_joint_source_solver as joint
    import z190_streaming_source_normal as streaming
    import calibration.solver_runtime.original_halfspace_primal as primal
    import calibration.solver_runtime.optional_soc_runtime as optional
    from scripts.diagnostics.check_current_truth_progress_policy_20261011 import (
        ShadowProbeComplete, digest, make_shadow_tap)

    q = np.array([.3, .4, 0.])
    class Profiles:
        stateless = True
    profiles = Profiles()
    token = joint._frozen_profile_token(profiles)
    def blocks(value, jacobian=False):
        assert jacobian is False, 'Fresh source J must never be reconstructed by the tap'
        for col, family in enumerate(('native', 'wide')):
            yield streaming.SourceRowBlock(np.array([value[col]-.1]), None,
                np.array([True]), np.array([1.]), {'family': family})
    baseline = tuple(blocks(q))
    signatures = tuple(streaming._checked_block(block, 3, False)[4] for block in baseline)
    normal = streaming.StreamingSourceNormal(eye(3, format='csc'), np.array([.2, .3, 0.]),
        sum(streaming._block_cost(block.r, block.quadratic) for block in baseline),
        streaming.SourceBlockContract(3, 2, signatures, token), {})
    normal_calls = []
    def reuse(factory, count, **options):
        assert count == 3 and options['expected_row_count'] == 2
        assert options['frozen_profile_token'] == token
        assert np.array_equal(options['source_state'], q)
        normal_calls.append(True)
        return normal
    monkeypatch.setattr(streaming, 'accumulate_source_normal', reuse)
    current = q[:2]-.1
    targets = current*.99
    matrix = -np.eye(2)
    b0 = current.copy()
    ownership = np.array([0, 1])
    lower = b0-.99*current
    domain = dict(matrix=matrix, lower=lower, curvature_responses={
        family: dict(row_start=i, row_stop=i+1, dynamic_row_group_index=np.array([0]),
            dynamic_group_limits=targets[i:i+1].copy())
        for i, family in enumerate(('native', 'wide'))}, report=dict(truth_feasibility_priority=dict(
        restoration_required=True, demanded_common_original_violation_fraction=.99,
        current_original_group_violations_px=current.tolist(), demanded_group_violations_px=targets.tolist())))
    domain['at_trial_fraction'] = lambda alpha: domain
    truth_restoration_limits = {'native': current[:1], 'wide': current[1:]}
    def constraints(value, selected):
        assert truth_restoration_limits and np.array_equal(value, q)
        assert np.array_equal(selected, [0, 1])
        return domain
    human_truth = SimpleNamespace(state=SimpleNamespace(blocks=()))
    def complete_truth_violations(value):
        assert human_truth.state.blocks == ()
        return value[:2]-.1, [('native', 0), ('wide', 0)]
    def acceptance(parent, trial, **options):
        complete_truth_violations(parent)
        return dict(accept=True, mode='truth_restoration')
    def bare_guard(value):
        return True
    original_complete_source_guard = bare_guard
    def native_guard(value):
        original_complete_source_guard(value)
        return True
    original_source_and_human_guard = native_guard
    def final_guard(value):
        original_source_and_human_guard(value)
        return True
    def tightening(value, step, selected, callback_domain):
        context = optional._outer_soc_measurement.get()
        context.update(measurement=dict(maximum_violation_px=0., total_violation_px=0.),
                       physical_measurement=dict(certified=True, valid=True))
        return None
    solves = []
    def linear(h, g, diag, **options):
        assert np.array_equal(g, normal.gradient)
        assert np.array_equal(h.toarray(), normal.normal.toarray())
        assert np.array_equal(options['global_parameter_indices'], [0, 1])
        solves.append(options['global_step_constraint_lower'].copy())
        return SimpleNamespace(step=np.array([-.1, -.1, 0.]), report={'shared_linear_domain_step': {}})
    monkeypatch.setattr(joint, 'schur_trial_linear_step', linear)
    monkeypatch.setattr(primal, 'original_full_qp_kkt_verified', lambda report, step: True)
    monkeypatch.setattr(joint, '_sequential_measured_shared_source_curvature', lambda q, linear, *args: linear)
    monkeypatch.setattr(joint, '_needs_trial_fraction_SOC', lambda *args: False)
    from calibration.solver_runtime.bounded_motion_schur import BoundedMotionSchurModel
    factory = lambda value: BoundedMotionSchurModel((), 3)
    config = {'continuation': {'state': {'q_sha256': digest(q)},
        'expected_complete_active_source_scalars': 2, 'current_truth_group_count': 2}}
    captures = [dict(matrix=matrix, original_lower=b0, owners=ownership, result_lower=lower)]
    tap = make_shadow_tap(config, tmp_path, captures)
    with pytest.raises(ShadowProbeComplete):
        tap(None, q, np.ones(2, bool), np.ones(2), tmp_path/'solver', final_guard,
            source_block_factory=blocks, update_profiles=profiles, fixed_parameter_indices=[2],
            global_schur_indices=[0, 1, 2], positive_parameter_indices=[],
            shared_linear_constraints=constraints, source_trial_acceptance=acceptance,
            bounded_motion_model_factory=factory, initial_damping=.001, schur_view_lm_ratio=0.,
            positive_parameter_resolution=1e-6, shared_step_curvature_tightening=tightening,
            initial_line_search_fraction=1.)
    assert len(normal_calls) == 1 and len(solves) == 3
    assert np.array_equal(solves[0], lower)
    assert np.array_equal(solves[1], b0-current)
    assert np.array_equal(solves[2], b0-current)
    assert np.array_equal(q, [.3, .4, 0.])
    summaries = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    complete = next(row for row in summaries if row['stage'] == 'shadow_policy_comparison_complete')
    assert len(complete['policies']) == 3
    for result in complete['policies']:
        assert result['production_candidate_pass'] is True
        assert result['diagnostic_comparable'] is True
        assert len(result['all_original_truth_violations_px']) == 2
        assert len(result['all_source_family_costs']) == 2
    with np.load(tmp_path/'shadow_policy_directions.npz', allow_pickle=False) as arrays:
        assert {'common_contraction', 'preserve_current_violation', 'existing_caps'} <= set(arrays.files)
