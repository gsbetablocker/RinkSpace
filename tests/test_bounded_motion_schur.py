"""Bounded nuisance replacement agrees with the original two-component rows."""
from itertools import product
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from calibration.solver_runtime.bounded_motion_schur import BoundedMotionSchurModel


def source_case(different_weights=False):
    raw = np.array([[[2., .3], [-.4, 1.1]],
                    [[.5, -1.4], [1.2, .8]],
                    [[-.3, .5], [.7, -.2]]])
    tangent = np.array([[1., .5], [-.5, 1.3], [.7, -.2]])
    inner = np.array([[.4, 1.1], [.7, .2], [1.3, .8]])
    outer = np.array([[.9, .3], [.5, 1.7], [.2, 1.2]]) if different_weights else inner.copy()
    # Each nominal residual is stationary under precisely the inner profile W.
    residual = np.c_[inner[:, 1] * tangent[:, 1], -inner[:, 0] * tangent[:, 0]]
    residual *= np.array([.3, -.2, .4])[:, None]
    lam = np.array([.2, .6, .4])
    return build_case(raw, tangent, residual, inner, outer, lam)


def build_case(raw, tangent, residual, inner, outer, lam):
    inner_gn = np.sum(inner * tangent**2, axis=1)
    factors = np.einsum('ni,nip->np', inner * tangent, raw) / inner_gn[:, None]
    schur = raw - tangent[:, :, None] * factors[:, None, :]
    cross = np.einsum('ni,nip->np', outer * tangent, schur)
    chunks = []
    for indices in (np.arange(min(2, len(lam))), np.arange(2, len(lam))):
        if not len(indices):
            continue
        chunks.append(dict(projection_factors=csr_matrix(factors[indices]),
                           cross_moments=csr_matrix(cross[indices]),
                           lambda0=lam[indices].copy(),
                           inner_gradient=np.sum(inner * tangent * residual, axis=1)[indices],
                           inner_GN=inner_gn[indices],
                           outer_GN=np.sum(outer * tangent**2, axis=1)[indices],
                           outer_gradient=np.sum(outer * tangent * residual, axis=1)[indices]))
    h = np.einsum('ni,nip,niq->pq', outer, schur, schur)
    g = np.einsum('ni,nip->p', outer * residual, schur)
    return dict(raw=raw, tangent=tangent, residual=residual, inner=inner, outer=outer,
                lam=lam, factors=factors, schur=schur, cross=cross, chunks=chunks, h=h, g=g)


def direct_replacement(case, active):
    raw = case['schur'].copy()
    residual = case['residual'].copy()
    raw[active] = case['raw'][active]
    residual[active] -= case['lam'][active, None] * case['tangent'][active]
    h = np.einsum('ni,nip,niq->pq', case['outer'], raw, raw)
    g = np.einsum('ni,nip->p', case['outer'] * residual, raw)
    c = .5 * np.sum(case['outer'] * residual**2)
    return h, g, c, raw, residual


@pytest.mark.parametrize('different_weights', [False, True])
def test_active_replacement_matches_direct_row_gram_gradient_and_cost(different_weights):
    case = source_case(different_weights)
    model = BoundedMotionSchurModel(case['chunks'], parameter_count=2)
    step = np.array([1.1, .5])
    active_chunks = model.active_set(step)
    assert isinstance(active_chunks, tuple) and len(active_chunks) == 2
    for active, chunk in zip(active_chunks, case['chunks']):
        assert active.dtype == np.bool_ and active.shape == chunk['lambda0'].shape
    active = np.concatenate(active_chunks)
    np.testing.assert_array_equal(active, [True, True, False])
    delta_h, delta_g, constant = model.correction(active_chunks)
    assert issparse(delta_h)
    h, g, c, raw, residual = direct_replacement(case, active)
    baseline_c = .5 * np.sum(case['outer'] * case['residual']**2)
    np.testing.assert_allclose(case['h'] + delta_h.toarray(), h, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(case['g'] + delta_g, g, rtol=2e-14, atol=2e-14)
    assert constant == pytest.approx(c-baseline_c, rel=2e-14, abs=2e-14)
    trial_cost = .5 * np.sum(case['outer'] * (residual + np.einsum('nip,p->ni', raw, step))**2)
    expected_reduction = baseline_c - trial_cost
    reduction = model.model_reduction(csr_matrix(case['h']), case['g'], step)
    assert reduction == pytest.approx(expected_reduction, rel=3e-14, abs=3e-14)
    if different_weights:
        # Inner orthogonality does not authorize dropping outer cross moments.
        assert np.linalg.norm(case['cross']) > .1
        naive = case['factors'][active].T @ (
            np.sum(case['outer'] * case['tangent']**2, axis=1)[active, None]
            * case['factors'][active])
        assert np.linalg.norm(delta_h.toarray()-naive) > .1
    else:
        np.testing.assert_allclose(case['cross'], 0., atol=2e-16)


def test_nominal_and_unclipped_steps_preserve_the_original_model():
    case = source_case(True)
    model = BoundedMotionSchurModel(case['chunks'], parameter_count=2)
    for step in (np.zeros(2), np.array([.01, -.02])):
        active = model.active_set(step)
        assert not any(mask.any() for mask in active)
        delta_h, delta_g, constant = model.correction(active)
        assert delta_h.nnz == 0 and constant == 0.
        np.testing.assert_array_equal(delta_g, np.zeros(2))
        original = -case['g'] @ step - .5 * step @ case['h'] @ step
        assert model.model_reduction(csr_matrix(case['h']), case['g'], step) == original


def test_active_prediction_uses_inner_stationarity_gradient_with_its_sign():
    chunk = dict(projection_factors=csr_matrix([[2.], [2.]]),
                 cross_moments=csr_matrix((2, 1)), lambda0=np.array([.4, .4]),
                 inner_gradient=np.array([.6, -.6]), inner_GN=np.array([2., 2.]),
                 outer_GN=np.array([2., 2.]), outer_gradient=np.zeros(2))
    model = BoundedMotionSchurModel([chunk], parameter_count=1)
    # Unrestricted lambda is lambda0 - g_inner/GN_inner - K @ step.
    np.testing.assert_array_equal(model.active_set(np.array([.1]))[0], [True, False])


def constrained_joint_quadratic_solution(case, regular_h, external_g):
    """Enumerate the tiny original depth-bound QP, without the tested helper."""
    n, _, p = case['raw'].shape
    extended = np.zeros((n, 2, p+n))
    extended[:, :, :p] = case['raw']
    for index in range(n):
        extended[index, :, p+index] = case['tangent'][index]
    h = np.einsum('ni,nip,niq->pq', case['outer'], extended, extended)
    g = np.einsum('ni,nip->p', case['outer'] * case['residual'], extended)
    h[:p, :p] += regular_h
    g[:p] += external_g
    solutions = []
    for mask in product((False, True), repeat=n):
        mask = np.array(mask, bool)
        fixed = p + np.flatnonzero(mask)
        free = np.setdiff1d(np.arange(p+n), fixed)
        value = np.zeros(p+n)
        value[fixed] = -case['lam'][mask]
        value[free] = np.linalg.solve(h[np.ix_(free, free)],
                                    -g[free]-h[np.ix_(free, fixed)] @ value[fixed])
        gradient = h @ value + g
        if np.all(value[p:] >= -case['lam']-1e-12) and np.all(gradient[fixed] >= -1e-12):
            solutions.append((value, mask))
    assert len(solutions) == 1
    return solutions[0]


@pytest.mark.parametrize('parameter_count', [1, 2])
def test_bounded_model_solution_equals_explicit_limited_depth_qp(parameter_count):
    raw = np.array([[[2., .6], [.3, -.2]], [[.8, .5], [-.4, 1.1]]])[:, :, :parameter_count]
    tangent = np.array([[1., 0.], [0., 1.]])
    residual = np.array([[0., .2], [.1, 0.]])
    weights = np.ones((2, 2))
    case = build_case(raw, tangent, residual, weights, weights, np.array([.15, .7]))
    regular_h = np.diag([.4, .6][:parameter_count])
    external_g = np.array([-1.1, .15][:parameter_count])
    joint, expected_active = constrained_joint_quadratic_solution(case, regular_h, external_g)
    assert expected_active.any() and not expected_active.all()
    model = BoundedMotionSchurModel(case['chunks'], parameter_count=parameter_count)
    delta_h, delta_g, constant = model.correction((expected_active,))
    step = np.linalg.solve(case['h'] + regular_h + delta_h.toarray(), -(case['g']+external_g+delta_g))
    np.testing.assert_allclose(step, joint[:parameter_count], rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(model.active_set(step)[0], expected_active)
    nuisance = -case['factors'] @ step
    nuisance[expected_active] = -case['lam'][expected_active]
    np.testing.assert_allclose(nuisance, joint[parameter_count:], rtol=2e-13, atol=2e-13)
    assert np.isfinite(constant)


def test_methods_do_not_change_source_moments_or_step():
    case = source_case(True)
    arrays = [(value, value.copy()) for chunk in case['chunks'] for value in chunk.values()
              if isinstance(value, np.ndarray)]
    sparse = [(value, value.data.copy(), value.indices.copy(), value.indptr.copy())
              for chunk in case['chunks'] for value in chunk.values() if issparse(value)]
    h, g, step = csr_matrix(case['h']), case['g'].copy(), np.array([1.1, .5])
    g_before, step_before = g.copy(), step.copy()
    model = BoundedMotionSchurModel(case['chunks'], parameter_count=2)
    model.correction(model.active_set(step))
    model.model_reduction(h, g, step)
    for value, before in arrays:
        np.testing.assert_array_equal(value, before)
    for value, data, indices, indptr in sparse:
        np.testing.assert_array_equal(value.data, data)
        np.testing.assert_array_equal(value.indices, indices)
        np.testing.assert_array_equal(value.indptr, indptr)
    np.testing.assert_array_equal(g, g_before)
    np.testing.assert_array_equal(step, step_before)


@pytest.mark.parametrize('parameter_count', [0, -1, 1.5, True])
def test_invalid_parameter_count_rejected(parameter_count):
    with pytest.raises((ValueError, TypeError)):
        BoundedMotionSchurModel(source_case()['chunks'], parameter_count=parameter_count)


@pytest.mark.parametrize('field,value', [
    ('projection_factors', csr_matrix(np.zeros((2, 3)))),
    ('cross_moments', csr_matrix(np.zeros((1, 2)))),
    ('lambda0', np.array([-.1, .2])),
    ('lambda0', np.array([.2])),
    ('inner_GN', np.array([0., 1.])),
    ('inner_GN', np.array([-1., 1.])),
    ('outer_GN', np.array([-1., 1.])),
    ('inner_gradient', np.array([np.inf, 0.])),
    ('outer_gradient', np.array([np.nan, 0.])),
    ('cross_moments', csr_matrix([[np.nan, 0.], [0., 0.]])),
    ('projection_factors', csr_matrix([[np.inf, 0.], [0., 0.]])),
])
def test_invalid_chunk_contract_rejected(field, value):
    chunks = source_case()['chunks']
    chunks[0] = dict(chunks[0], **{field: value})
    with pytest.raises((ValueError, TypeError)):
        BoundedMotionSchurModel(chunks, parameter_count=2)


@pytest.mark.parametrize('step', [np.zeros(3), np.zeros((2, 1)), np.array([np.nan, 0.]), np.array([0., np.inf])])
def test_invalid_trial_step_rejected(step):
    model = BoundedMotionSchurModel(source_case()['chunks'], parameter_count=2)
    with pytest.raises((ValueError, TypeError)):
        model.active_set(step)


@pytest.mark.parametrize('active', [(), (np.ones(2, bool),),
                                   (np.ones(3, bool), np.zeros(1, bool)),
                                   (np.array([1., 0.]), np.zeros(1, bool))])
def test_invalid_active_inventory_rejected(active):
    model = BoundedMotionSchurModel(source_case()['chunks'], parameter_count=2)
    with pytest.raises((ValueError, TypeError)):
        model.correction(active)


@pytest.mark.parametrize('parameter_count', [1, 2])
def test_solve_iterates_to_the_explicit_original_bounded_joint_qp(parameter_count):
    raw = np.array([[[2., .6], [.3, -.2]], [[.8, .5], [-.4, 1.1]]])[:, :, :parameter_count]
    tangent = np.array([[1., 0.], [0., 1.]])
    residual = np.array([[0., .2], [.1, 0.]])
    weights = np.ones((2, 2))
    case = build_case(raw, tangent, residual, weights, weights, np.array([.15, .7]))
    regular_h = np.diag([.4, .6][:parameter_count])
    external_g = np.array([-1.1, .15][:parameter_count])
    joint, active = constrained_joint_quadratic_solution(case, regular_h, external_g)
    model = BoundedMotionSchurModel(case['chunks'], parameter_count=parameter_count)
    base_h = csr_matrix(case['h'] + regular_h)
    base_g = case['g'] + external_g
    diagonal = np.array([3., 5.][:parameter_count])
    constraint_lower = np.array([-2., .1])
    calls = []

    def solve_original_camera_qp(h, g, supplied_diagonal, **options):
        assert supplied_diagonal is diagonal
        assert options['constraint_lower'] is constraint_lower
        assert options['original_numerical_option'] == 0
        calls.append((h.copy(), g.copy()))
        return SimpleNamespace(step=np.linalg.solve(h.toarray(), -g), report={'original_QP': True})

    result = model.solve(base_h, base_g, diagonal, solve_original_camera_qp,
                         constraint_lower=constraint_lower, original_numerical_option=0)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0].toarray(), base_h.toarray())
    np.testing.assert_array_equal(calls[0][1], base_g)
    np.testing.assert_allclose(result.step, joint[:parameter_count], rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(model.active_set(result.step)[0], active)
    report = result.report['bounded_motion_nuisance']
    assert result.report['original_QP'] is True
    assert report['original_depth_activity_consistent'] is True
    assert report['active_points_per_solve'] == [0, 1]
    assert report['camera_QP_solves'] == 2
    assert report['new_source_rows_or_priors'] == 0
    assert report['source_model_reduction'] == model.model_reduction(base_h, base_g, result.step)
    assert json.loads(json.dumps(result.report))['bounded_motion_nuisance']['active_points_per_solve'] == [0, 1]


def test_repeated_activity_fails_without_returning_an_unbounded_camera_step():
    case = source_case(True)
    model = BoundedMotionSchurModel(case['chunks'], parameter_count=2)
    outputs = []

    def alternating_qp(h, g, diagonal):
        step = np.array([1.1, .5]) if not outputs else np.zeros(2)
        result = SimpleNamespace(step=step, report={'original_QP': True})
        outputs.append(result)
        return result

    with pytest.raises(ValueError, match='cycl|activ',):
        model.solve(csr_matrix(case['h']), case['g'], np.ones(2), alternating_qp)
    assert len(outputs) == 2
    assert all(result.report == {'original_QP': True} for result in outputs)


def test_fixed_parameters_are_removed_from_both_moments_and_corrections():
    case = source_case(True)
    original_moments = [(c['projection_factors'].copy(), c['cross_moments'].copy())
                        for c in case['chunks']]
    model = BoundedMotionSchurModel(case['chunks'], parameter_count=2)
    assert model.eliminate_fixed(np.array([1], dtype=np.int64)) is model
    for current in model.chunks:
        np.testing.assert_array_equal(current['projection_factors'][:, 1].toarray(), 0.)
        np.testing.assert_array_equal(current['cross_moments'][:, 1].toarray(), 0.)
    active = model.active_set(np.array([1.1, 100.]))
    np.testing.assert_array_equal(np.concatenate(active), np.concatenate(model.active_set(np.array([1.1, 0.]))))
    delta_h, delta_g, _ = model.correction(active)
    np.testing.assert_array_equal(delta_h.toarray()[1], np.zeros(2))
    np.testing.assert_array_equal(delta_h.toarray()[:, 1], np.zeros(2))
    assert delta_g[1] == 0.
    for original, (m, v) in zip(case['chunks'], original_moments):
        np.testing.assert_array_equal(original['projection_factors'].toarray(), m.toarray())
        np.testing.assert_array_equal(original['cross_moments'].toarray(), v.toarray())


def test_lambda_zero_is_allowed_and_tie_does_not_activate_an_extra_row():
    case = source_case()
    for chunk in case['chunks']:
        chunk['lambda0'] = np.zeros_like(chunk['lambda0'])
        chunk['inner_gradient'] = np.zeros_like(chunk['inner_gradient'])
    model = BoundedMotionSchurModel(case['chunks'], parameter_count=2)
    assert not any(mask.any() for mask in model.active_set(np.zeros(2)))


@pytest.mark.parametrize('indices', [np.array([-1]), np.array([2]), np.array([.5]),
                                   np.array([True]), np.array([[0]]), np.array([0, 0])])
def test_invalid_fixed_indices_fail_closed(indices):
    model = BoundedMotionSchurModel(source_case()['chunks'], parameter_count=2)
    with pytest.raises((ValueError, TypeError)):
        model.eliminate_fixed(indices)
