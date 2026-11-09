"""Small checks for proposal-only physical caps and raw body-pixel rows."""
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from calibration.program_tracking.body_tracking import programme_guard_proposals as proposals
from calibration.program_tracking.body_tracking.fullbody_fit import FullBodyProblem
from calibration.program_tracking.body_tracking.programme_source_step import SourceMargins, source_conditioned_direction
from calibration.program_tracking.body_tracking.weighted_blocks import block, full_columns, pack_weighted


def test_normalized_peak_rows_preserve_nonuniform_scales_original_caps_and_global_columns():
    # The stored zero in the second row has no numerical dependency.
    raw = csr_matrix(
        ([2., -1., 0., -.25, .8, 3.], ([0, 0, 1, 2, 2, 3], [0, 9, 16, 4, 24, 29])),
        shape=(4, 30),
    )
    scales = np.array([.075, .6, 2.75, 4.25])
    current = np.array([.4, .21, .08, .1])
    caps = np.array([.51, .22, .2, .05])
    original_caps = caps.copy()
    caps.setflags(write=False)
    labels = ['hip_peak_0', 'no_dependence', 'surface_peak_2', 'surface_peak_3']
    terms = full_columns(block('peak_guards', scales*current, raw.multiply(scales[:, None])))
    saved_jac = terms.jacobian.copy()
    margins = proposals.normalized_peak_halfspaces(terms, scales, caps, labels)

    assert isinstance(margins, SourceMargins)
    A, slack, selected_labels = margins
    keep = [0, 2, 3]
    assert A.shape == (3, 47)
    assert A[:, :17].nnz == 0
    np.testing.assert_allclose(A[:, 17:].toarray(), raw[keep].toarray(), atol=1e-15, rtol=0)
    assert set(A.indices) == {17, 21, 26, 41, 46}
    np.testing.assert_allclose(slack, .8*(original_caps-current)[keep], atol=1e-15, rtol=0)
    assert list(selected_labels) == [labels[i] for i in keep]
    np.testing.assert_array_equal(caps, original_caps)
    np.testing.assert_array_equal(terms.residual, scales*current)
    for name in ('data', 'indices', 'indptr'):
        np.testing.assert_array_equal(getattr(terms.jacobian, name), getattr(saved_jac, name))


def test_inverse_body_pixel_jacobian_matches_raw_2d_derivatives_at_zero_and_off_axis_error():
    frames = np.array([4, 13, 29])
    of = np.array([2, 0, 1, 2])
    oj = np.array([5, 11, 13, 6])
    model_oj = np.array([7, 2, 9, 14])
    body_landmarks = np.arange(19)[::-1][:17]
    model_points = body_landmarks[model_oj]
    errors = np.array([[0., 0.], [56., -38.], [-15., 40.], [2., -3.]])
    source = np.array([[34., -12.], [70., 18.], [-5., 27.], [14., 36.]])
    observed = np.zeros((30, 17, 2))
    observed[frames[of], oj] = source
    sigma = np.ones((30, 17))
    sigma[frames[of], oj] = [3., .6, 4.25, 1.5]
    uv = np.zeros((len(frames), 19, 2))
    uv[of, model_points] = source+errors
    body = SimpleNamespace(
        frames=frames, of=of, oj=oj, model_oj=model_oj, cache={'uv': uv},
        layout=SimpleNamespace(body_landmarks=body_landmarks, point_count=19),
        obs={'uv': observed, 'sigma_px': sigma},
    )
    active = np.array([0, 6, 18, 29])
    point_chain = np.array([
        [[1.5, -.7, 2.8, .4], [.8, 2.2, -.3, -.6]],
        [[-.9, .2, 1.6, 2.1], [3.2, -1.4, .8, .5]],
        [[.25, 1.5, -.6, -1.2], [2.4, -.8, 1.9, 2.6]],
        [[.6, -.7, 1.1, .3], [-.8, 2.8, .4, .6]],
    ])
    uv_chain = np.zeros((*uv.shape, 30))
    for i in range(len(of)):
        uv_chain[of[i], model_points[i]][:, active] = point_chain[i]
    residual, pixel_jacobian, _, _ = FullBodyProblem.body_factors(body, uv)
    chained = pixel_jacobian@csr_matrix(uv_chain.reshape(uv.size, -1))
    # Use the original assembly, including the exposure measure and per-axis
    # packed weights. An isotropic transform still has off-diagonal derivatives.
    time_weight = .023
    weights = {'body_pixels': np.array([.27, .36, .84, 1.7, .46, .67, 1.1, .53])}
    timed = full_columns(block('body_pixels', residual, chained), scale=np.sqrt(time_weight))
    _, _, packed = pack_weighted([timed], weights, jacobian=True)
    problem = SimpleNamespace(body=body, image_time_weight=time_weight, objective_block_weights=weights)
    points = np.array([2, 0, 1])

    restored = proposals.recover_body_pixel_jacobian(problem, {'weighted_blocks': packed}, points)

    assert restored.shape == (6, 47)
    assert restored[:, :17].nnz == 0
    assert set(restored.indices) == set(17+active)

    def raw_error(control_delta):
        changed = uv+uv_chain[..., active]@control_delta
        return (changed[of[points], model_points[points]]-source[points]).ravel()

    h = 1e-5
    central = np.column_stack([(raw_error(h*axis)-raw_error(-h*axis))/(2*h) for axis in np.eye(len(active))])
    np.testing.assert_allclose(restored[:, 17+active].toarray(), central, atol=1e-8, rtol=0)
    np.testing.assert_allclose(restored[:, 17+active].toarray(), point_chain[points].reshape(6, -1),
                               atol=1e-13, rtol=0)
    np.testing.assert_array_equal(residual[:2], [0., 0.])


def test_original_nonlinear_peak_rejects_linear_proposal_then_passes_local_coupled_re_solve():
    selection = np.array([18, 23, 41])
    scale_inv = np.array([2., .5, 1.25])
    preferred = np.array([.6, -.2, .45])*scale_inv
    hessian = np.array([[2., .6, 0.], [.6, 1.5, 0.], [0., 0., 1.2]])

    class Normal:
        def solve_scaled(self, rhs, scale, *, damp):
            return preferred.copy(), {}

        def scaled_normal_objective(self, rhs, scale, *, damp):
            def objective(point):
                change = point-preferred
                gradient = hessian@change
                return float(.5*change@gradient - .5*preferred@hessian@preferred), gradient
            return objective

    normal = Normal()
    normal.scale_inv = scale_inv
    limits = np.array([.75, .65, .6])
    value = {'residual': np.array([2.])}
    current, cap, row_scale = .0296, .0396, .37
    local_gradient = np.array([.25, .15, 0.])
    raw = csr_matrix((local_gradient[:2], ([0, 0], selection[:2])), shape=(1, 42))
    terms = block('anatomical_peak', [row_scale*current], raw*row_scale)
    guard_id = '__joint__/0'
    margins = proposals.normalized_peak_halfspaces(terms, [row_scale], [cap], [guard_id])
    margins.guard_dependency = lambda key: dict(jacobian=raw, model_foot=None) if key == guard_id else None

    def physical_peak(move):
        return current + local_gradient@move + .004*(move[:2]@move[:2])

    def original_guard(move):
        peak = physical_peak(move)
        return {guard_id: dict(rejected=bool(peak > cap), before=cap, after=peak,
                               original_radius_px=None, model_foot=None, unit='joint domain')}

    unconstrained, _ = source_conditioned_direction(
        normal, value, (-1., 1.), None, selection, control_step_limit=limits,
    )
    linear_only, _ = source_conditioned_direction(
        normal, value, (-1., 1.), margins, selection, control_step_limit=limits,
    )
    assert original_guard(unconstrained)[guard_id]['rejected']
    assert current + local_gradient@linear_only <= current + .8*(cap-current) + 1e-10
    assert original_guard(linear_only)[guard_id]['rejected']

    trust_state = {}
    adjusted, report = source_conditioned_direction(
        normal, value, (-1., 1.), margins, selection, control_step_limit=limits,
        nonlinear_source_check=original_guard, trust_state=trust_state,
    )
    assert not original_guard(adjusted)[guard_id]['rejected']
    assert not np.array_equal(adjusted[:2], linear_only[:2])
    assert adjusted[2] == linear_only[2] == unconstrained[2]
    assert trust_state['control_step_limit'][2] == limits[2]
    np.testing.assert_array_equal(limits, [.75, .65, .6])
    feedback = report['local_source_feedback']
    assert feedback[-1]['status'] == 'passed'
    assert any(row['status'] == 'contracted' and set(row['changed_global_columns']) == {18, 23}
               for row in feedback)
    assert report['constraint_rows'] == [guard_id]
    assert report['original_nonlinear_guard_required']

def test_cavity_guard_uses_actual_new_active_rows_with_local_two_shoe_support():
    times = np.array([0., .02, .04])
    kinds = ('left_foot_cavity', 'right_foot_cavity', 'left_shank_collar', 'right_shank_collar')
    labels = [(stamp, kind) for stamp in ('neutral', 0, 20000000, 40000000) for kind in kinds]

    def basis(t):
        indices = np.searchsorted(times, t)
        return csr_matrix((np.ones(len(t)), (np.arange(len(t)), indices)), shape=(len(t), len(times)))

    body_size = 69*len(times)
    size = 17+body_size+6*len(times)
    state = SimpleNamespace(basis=basis, shoe_curve=SimpleNamespace(basis_sparse=basis), body_size=body_size)
    problem = SimpleNamespace(optimize_shared=False, body=SimpleNamespace(frames=np.arange(3), state=state),
                              reference=SimpleNamespace(times=times), size=size, cavity_labels=kinds,
                              foot_row_labels=labels)
    start = np.array([[0., 0., 0., 0.], [1., 2., 3., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]])
    jac = csr_matrix(([-.5], ([4], [23])), shape=(start.size, size))

    def value(residual, matrix=None):
        return {'weighted_blocks': [block('cavity_AL', residual.ravel(), matrix, dict(labels=labels))]}

    guard = proposals.ProgrammeCavityTrialGuard(problem, value(start, jac))
    margins = guard.halfspaces(value(start, jac))
    A, slack, keys = margins
    np.testing.assert_array_equal(slack, np.zeros(len(kinds)))
    assert A[0, 23] == -.5
    assert not any(row['rejected'] for row in guard.checks(value(start)).values())

    # The old active wall improves, but a previously inactive wall now dominates.
    trial = start.copy()
    trial[1, 0] = .5
    trial[3, 0] = 2.
    checks = guard.checks(value(trial))
    assert checks[keys[0]]['rejected']
    assert not any(checks[key]['rejected'] for key in keys[1:])
    dep = guard.dependency(keys[0])
    local = 17+2*69+np.arange(6, 69)
    left_shoe = 17+body_size+2*6+np.arange(3)
    np.testing.assert_array_equal(dep['jacobian'].indices, np.r_[local, left_shoe])
    assert dep['model_foot'] == 0 and guard.dependency(keys[1]) is None

    # A new collar section also participates even when the original J is zero.
    trial = start.copy()
    trial[2, 3] = .6
    checks = guard.checks(value(trial))
    assert checks[keys[3]]['rejected']
    assert guard.dependency(keys[3])['model_foot'] == 1
