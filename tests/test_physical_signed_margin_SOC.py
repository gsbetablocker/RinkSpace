from types import SimpleNamespace
import ast
from pathlib import Path

import numpy as np
import pytest

from calibration.solver_runtime import physical_signed_margin_SOC as physical
from calibration.solver_runtime import optional_soc_runtime as runtime


def domain():
    return dict(matrix=np.array([[0., 1.], [1., 0.], [-1., 0.]]),
        lower=np.array([0., -1., -1.]),
        curvature_responses={'native':dict(row_start=1, row_stop=3)})


def test_same_physical_rows_and_fraction_targets_preserved_without_padding():
    original = domain()
    original['at_trial_fraction'] = lambda a: dict(original,
        lower=np.array([0., -a, -a]))
    wrapped = physical.attach_original_signed_margin_rows(original,
        row_indices=np.array([0]), nominal_margins=[0.], row_ids=['LOW'])
    step = np.array([.2, 0.])
    correction, measurement = physical.measured_signed_margin_step_tightening(
        wrapped['matrix'], wrapped['lower'], step,
        wrapped['original_signed_margin_rows'], [-.04], row_ids=['LOW'])
    np.testing.assert_array_equal(correction['lower'][1:], original['lower'][1:])
    assert correction['lower'][0] == np.nextafter(.04, np.inf)
    assert not measurement['valid']
    assert measurement['maximum_violation_working_m'] == .04
    assert not measurement['numerical_padding_added']
    assert 'maximum_violation_px' not in measurement
    fraction = wrapped['at_trial_fraction'](.25)
    assert fraction['matrix'] is original['matrix']
    np.testing.assert_array_equal(fraction['lower'], [0., -.25, -.25])
    assert fraction['original_signed_margin_rows'] is wrapped['original_signed_margin_rows']
    np.testing.assert_array_equal(original['lower'], [0., -1., -1.])


@pytest.mark.parametrize('margin', [0., 1e-15, .1])
def test_legal_actual_signed_margin_adds_no_cut(margin):
    wrapped = physical.attach_original_signed_margin_rows(domain(),
        row_indices=np.array([0]), nominal_margins=[0.], row_ids=['LOW'])
    correction, measurement = physical.measured_signed_margin_step_tightening(
        wrapped['matrix'], wrapped['lower'], np.zeros(2),
        wrapped['original_signed_margin_rows'], [margin], row_ids=['LOW'])
    assert correction is None and measurement['valid']


@pytest.mark.parametrize('indices', [[1], [0, 0], [-1], [3], [.0]])
def test_invalid_or_source_owned_physical_row_mapping_rejected(indices):
    with pytest.raises(ValueError):
        physical.attach_original_signed_margin_rows(domain(),
            row_indices=np.asarray(indices), nominal_margins=np.zeros(len(indices)),
            row_ids=['LOW'] * len(indices))


def test_final_candidate_cannot_inherit_proposal_KKT_after_recovery():
    q = np.array([1., 2., 3.])
    physical.require_identity_trial_recovery(q, q.copy())
    modified = q.copy()
    modified[1] = np.nextafter(modified[1], np.inf)
    with pytest.raises(ValueError, match='proposal QP KKT cannot certify'):
        physical.require_identity_trial_recovery(q, modified)
    np.testing.assert_array_equal(q, [1., 2., 3.])
    # Signed-zero representations encode the same physical coordinate and
    # displacement; they must not trigger the changed-candidate veto.
    physical.require_identity_trial_recovery(np.array([0., 1.]), np.array([-0., 1.]))


def test_physical_progress_uses_separate_units_and_precedes_pixel_comparison():
    inside = dict(valid=False, maximum_violation_working_m=.04,
        total_violation_working_m=.04)
    better = dict(valid=False, maximum_violation_working_m=.02,
        total_violation_working_m=.02)
    legal = dict(valid=True, maximum_violation_working_m=0.,
        total_violation_working_m=0.)
    assert runtime._physical_first_SOC_progress(inside, None)['progress']
    assert runtime._physical_first_SOC_progress(better, inside)['progress']
    assert runtime._physical_first_SOC_progress(legal, inside)['progress']
    assert not runtime._physical_first_SOC_progress(inside, legal)['progress']
    assert runtime._physical_first_SOC_progress(legal, legal) is None
    assert runtime._physical_first_SOC_progress(None, None) is None


def test_coupled_physical_source_QP_has_KKT_and_identity_final_candidate():
    from calibration.solver_runtime import install
    install()
    import z190_global_schur_linear_step as qp
    import z190_joint_source_solver as joint
    from structured_programme_source_candidate import source_step_curvature_tightening as source
    runtime._install_original_nonlinear_face_observer()
    producer = joint._sequential_measured_shared_source_curvature
    while hasattr(producer, '_original_SOC_producer'):
        producer = producer._original_SOC_producer
    # Physical margin p-w^2 >= 0; the original source requires v-p >= 0.
    # Old post-SOC retraction moves only p, thereby violating that source.
    H = np.eye(3)
    g = np.array([-.2, .01, .01])
    A = np.array([[0., 1., 0.], [0., -1., 1.], [0., 1., -1.]])
    b = np.array([0., 0., -.1])
    response = {'native': dict(row_start=1, row_stop=3,
        nominal_carrier=np.zeros(1), retained_face_mask=np.ones(2, bool))}
    d = physical.attach_original_signed_margin_rows(
        dict(matrix=A, lower=b, curvature_responses=response),
        row_indices=np.array([0]), nominal_margins=[0.], row_ids=['LOW'])
    options = dict(global_step_constraint_lower=b.copy())
    solved = []
    def solve(H, g, diagonal, **options):
        step, _, report = qp._linearly_constrained_shared_step(H, g,
            np.full(3, -np.inf), np.full(3, np.inf), A,
            options['global_step_constraint_lower'])
        value = SimpleNamespace(step=step, report={'shared_linear_domain_step': report})
        assert runtime._previous_full_KKT_verified(value)
        solved.append(value)
        return value
    initial = solve(H, g, None, **options)
    old_recovered = initial.step.copy()
    old_recovered[1] = initial.step[0]**2
    assert old_recovered[2] - old_recovered[1] < 0.
    def measure(q, step, columns, original):
        margin = step[1] - step[0]**2
        pc, pm = physical.measured_signed_margin_step_tightening(A, b, step,
            original['original_signed_margin_rows'], [margin], row_ids=['LOW'])
        runtime.observe_original_physical_SOC_measurement(pm)
        sc = source.measured_outward_step_tightening(A, b, step, response,
            {'native': np.array([step[2] - step[1]])})
        if pc is None:
            return sc
        if sc is None:
            return pc
        return dict(lower=np.maximum(pc['lower'], sc['lower']))
    final = runtime._with_outer_nonlinear_progress(producer, np.zeros(3), initial,
        np.arange(3), d, options, H, g, np.ones(3), measure, solve)
    assert final is not initial
    assert runtime._previous_full_KKT_verified(final)
    assert final.step[1] - final.step[0]**2 >= 0.
    assert final.step[2] - final.step[1] >= -1e-8
    physical.require_identity_trial_recovery(final.step, final.step.copy())
    records = final.report['sequential_measured_shared_source_curvature_tightening'][
        'actual_original_nonlinear_feasibility_progress']['measurements']
    assert not records[0]['actual_original_physical_domain_feasibility']['valid']
    assert records[-1]['actual_original_physical_domain_feasibility']['valid']
    assert records[-1]['current_complete_KKT_direction_retained']


def test_retained_inside_physical_candidate_needs_short_scale_SOC(monkeypatch):
    from calibration.solver_runtime import install
    install()
    import z190_joint_source_solver as joint
    monkeypatch.setattr(joint, '_needs_trial_fraction_SOC', lambda linear, domain: False)
    runtime._install_physical_fraction_SOC(joint)
    linear = SimpleNamespace(report=dict(sequential_measured_shared_source_curvature_tightening=dict(
        actual_original_nonlinear_feasibility_progress=dict(measurements=[dict(
            current_complete_KKT_direction_retained=True,
            actual_original_physical_domain_feasibility=dict(valid=False))]))))
    assert joint._needs_trial_fraction_SOC(linear, {'at_trial_fraction': lambda a: None})
    assert not joint._needs_trial_fraction_SOC(linear, {})


def driver_callback(name, namespace):
    path = Path(__file__).resolve().parents[1] / 'calibration/solver_runtime/legacy/run_current_named66_full_programme.py'
    matches = [node for node in ast.walk(ast.parse(path.read_text(encoding='utf8')))
        if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1
    tree = ast.fix_missing_locations(ast.Module(body=matches, type_ignores=[]))
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace[name]


def real_curved_course_state(monkeypatch):
    from calibration.solver_runtime import install
    install()
    from installation_mount_candidate import mount_graph as mounts
    class Course:
        def sample_perimeter(self, value):
            left = -value[0]**2 / mounts.SCENE_UNIT_M
            unit = 1. / mounts.SCENE_UNIT_M
            return np.array([[left, 0., 0.], [unit, 0., 0.],
                [unit, unit, 0.], [left, unit, 0.]]), None
    class Producer:
        def print_course_binding(self, value):
            return Course(), value, np.zeros((1, len(value)))
        def validate_parameters(self, value):
            return None
    contract = dict(world_parameter_indices=[5], numerical_world_derivative_step=1e-6,
        by_installation={'LOW': dict(pivot_parameter_indices=[0, 1, 2])})
    state = SimpleNamespace(geometry=SimpleNamespace(current=Producer()),
        current_user_pivot_outside_constraint_contract=contract)
    monkeypatch.setattr(mounts, 'validate_mounts', lambda state, q: None)
    return state, contract, mounts


def test_actual_driver_SOC_callback_measures_proposal_physical_margin(monkeypatch):
    state, _, mounts = real_curved_course_state(monkeypatch)
    callback = driver_callback('tighten_source_trial_faces', dict(np=np,
        state=state, evaluation_q=lambda q: q,
        pivot_outside_domain=mounts.pivot_outside_domain,
        human_truth=None, wide_human_truth=None, confirmed_truth=None))
    q = np.array([0., .5, 1., 0., 0., 0.])
    step = np.zeros(6)
    step[5] = .2
    original = physical.attach_original_signed_margin_rows(dict(
        matrix=np.array([[-1., 0., 0., 0., 0., 0.]]),
        lower=np.zeros(1), curvature_responses={}),
        row_indices=np.array([0]), nominal_margins=[0.], row_ids=['LOW'])
    context = {}
    token = runtime._outer_soc_measurement.set(context)
    try:
        corrected = callback(q, step, np.arange(6), original)
    finally:
        runtime._outer_soc_measurement.reset(token)
    assert corrected is not None
    margin = mounts.pivot_outside_domain(state, q + step)['installations'][0]['outward_margin_working_m']
    assert margin < 0.
    assert corrected['lower'][0] == np.nextafter(-margin, np.inf)
    assert context['physical_measurement']['actual_signed_margins'] == [margin]
    np.testing.assert_array_equal(q, [0., .5, 1., 0., 0., 0.])


def test_actual_driver_recovery_requires_identity_only_for_coupled_SOC(monkeypatch):
    state, contract, mounts = real_curved_course_state(monkeypatch)
    globals = dict(np=np, state=state, evaluation_q=lambda q: q,
        layout=SimpleNamespace(canonical_q=lambda q: q),
        pivot_outside_constraints=contract,
        pivot_outside_domain=mounts.pivot_outside_domain,
        reinitialize_invalid_pivots_outside=mounts.reinitialize_invalid_pivots_outside,
        evaluation_world_indices=np.array([5]), effective_curvature_compensation=True)
    callback = driver_callback('recover_pivot_trial', globals)
    parent = np.array([0., .5, 1., 0., 0., 0.])
    np.testing.assert_array_equal(callback(parent, parent.copy())['q'], parent)
    proposed = parent.copy()
    proposed[5] = .2
    with pytest.raises(ValueError, match='proposal QP KKT cannot certify'):
        callback(parent, proposed)
    np.testing.assert_array_equal(parent, [0., .5, 1., 0., 0., 0.])
    np.testing.assert_array_equal(proposed, [0., .5, 1., 0., 0., .2])
    # Unrelated historical callers retain the original explicit recovery.
    globals['effective_curvature_compensation'] = False
    legacy = callback(parent, proposed)
    assert mounts.pivot_outside_domain(state, legacy['q'])['valid']
    assert legacy['q'][0] != proposed[0]


def test_actual_driver_unmeasurable_full_step_keeps_authoritative_short_fallback(monkeypatch):
    from calibration.solver_runtime import install
    install()
    import z190_joint_source_solver as joint
    state, _, mounts = real_curved_course_state(monkeypatch)
    original_course = state.geometry.current.print_course_binding
    def limited_course(value):
        if abs(value[0]) > .1:
            raise ValueError('Original physical course domain is not measurable')
        return original_course(value)
    monkeypatch.setattr(state.geometry.current, 'print_course_binding', limited_course)
    callback = driver_callback('tighten_source_trial_faces', dict(np=np,
        state=state, evaluation_q=lambda q: q,
        pivot_outside_domain=mounts.pivot_outside_domain,
        human_truth=None, wide_human_truth=None, confirmed_truth=None))
    q = np.array([0., .5, 1., 0., 0., 0.])
    step = np.zeros(6)
    step[5] = .2
    original = physical.attach_original_signed_margin_rows(dict(
        matrix=np.array([[-1., 0., 0., 0., 0., 0.]]),
        lower=np.zeros(1), curvature_responses={},
        at_trial_fraction=lambda a: None),
        row_indices=np.array([0]), nominal_margins=[0.], row_ids=['LOW'])
    # Use the verified upstream KKT indicator only to exercise the callback
    # fallback. This fixture does not claim a source-QP numerical solve.
    monkeypatch.setattr(runtime, '_previous_full_KKT_verified', lambda linear: True)
    initial = SimpleNamespace(step=step.copy(), report={})
    producer = joint._sequential_measured_shared_source_curvature
    while hasattr(producer, '_original_SOC_producer'):
        producer = producer._original_SOC_producer
    result = runtime._with_outer_nonlinear_progress(producer, q, initial,
        np.arange(6), original, {'global_step_constraint_lower': np.zeros(1)},
        np.eye(6), np.zeros(6), np.ones(6), callback,
        lambda *a, **k: pytest.fail('Unavailable full measurement cannot request a correction'))
    assert result is initial
    record = result.report['sequential_measured_shared_source_curvature_tightening'][
        'actual_original_nonlinear_feasibility_progress']['measurements'][0]
    assert record['actual_original_face_feasibility'] is None
    assert record['original_nonlinear_measurement_unavailable']
    assert not record['actual_original_physical_domain_feasibility']['certified']
    assert not record['actual_original_physical_domain_feasibility']['valid']
    runtime._install_physical_fraction_SOC(joint)
    assert joint._needs_trial_fraction_SOC(result, original)
    with pytest.raises(ValueError):
        mounts.pivot_outside_domain(state, q + step)
    # A shorter physical candidate remains measurable by the same original
    # course guard, and its physical cut is actually measured (not assumed).
    short = .25 * step
    corrected = callback(q, short, np.arange(6), original)
    assert corrected is not None
    assert corrected['report']['original_signed_margin_measurement']['certified']


def test_alternating_source_and_physical_curvature_keeps_only_jointly_legal_candidate():
    from calibration.solver_runtime import install
    install()
    import z190_global_schur_linear_step as qp
    import z190_joint_source_solver as joint
    from installation_mount_candidate import mount_graph as mounts
    from structured_programme_source_candidate import source_step_curvature_tightening as source
    runtime._install_original_nonlinear_face_observer()
    producer = joint._sequential_measured_shared_source_curvature
    while hasattr(producer, '_original_SOC_producer'):
        producer = producer._original_SOC_producer
    H = np.array([[1., .5, -.5], [.5, 1., 0.], [-.5, 0., 1.]])
    g = np.array([-.2, -.09, .11])
    A = np.array([[0., 1., 0.], [0., -1., 1.], [0., 1., -1.]])
    b = np.array([0., 0., -.1])
    responses = {'native': dict(row_start=1, row_stop=3,
        nominal_carrier=np.zeros(1), retained_face_mask=np.ones(2, bool))}
    domain = physical.attach_original_signed_margin_rows(dict(
        matrix=A, lower=b, curvature_responses=responses),
        row_indices=np.array([0]), nominal_margins=[0.], row_ids=['LOW'])
    options = dict(global_step_constraint_lower=b.copy())
    def solve(H, g, diagonal, **options):
        step, _, report = qp._linearly_constrained_shared_step(H, g,
            np.full(3, -np.inf), np.full(3, np.inf), A,
            options['global_step_constraint_lower'])
        result = SimpleNamespace(step=step, report={'shared_linear_domain_step': report})
        assert runtime._previous_full_KKT_verified(result)
        return result
    def margin(step):
        w, p, _ = step
        polygon = np.array([[-w*w, 0.], [1., 0.], [1., 1.], [-w*w, 1.]])
        return mounts._pivot_outward_margin(polygon, 1.+w*w,
            np.array([-p, .5]))['outward_margin_working_m']
    def measure(q, step, columns, original):
        pc, pm = physical.measured_signed_margin_step_tightening(A, b, step,
            original['original_signed_margin_rows'], [margin(step)], row_ids=['LOW'])
        runtime.observe_original_physical_SOC_measurement(pm)
        v = step[2]
        sc = source.measured_outward_step_tightening(A, b, step, responses,
            {'native': np.array([v-step[1]-v*v])})
        if pc is None:return sc
        if sc is None:return pc
        return dict(lower=np.maximum(pc['lower'], sc['lower']))
    initial = solve(H, g, None, **options)
    final = runtime._with_outer_nonlinear_progress(producer, np.zeros(3), initial,
        np.arange(3), domain, options, H, g, np.ones(3), measure, solve)
    records = final.report['sequential_measured_shared_source_curvature_tightening'][
        'actual_original_nonlinear_feasibility_progress']['measurements']
    assert any(r.get('temporary_unphysical_candidate_requires_existing_row_SOC') for r in records)
    assert all(not r['current_complete_KKT_direction_retained'] for r in records
        if r.get('temporary_unphysical_candidate_requires_existing_row_SOC'))
    assert records[-1]['current_complete_KKT_direction_retained']
    assert runtime._previous_full_KKT_verified(final)
    assert margin(final.step) >= 0.
    v = final.step[2]
    assert v-final.step[1]-v*v >= 0.
    physical.require_identity_trial_recovery(final.step, final.step.copy())

