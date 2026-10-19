"""Independent guard-margin finite differences, ties and zero-coefficient cases."""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from calibration.program_tracking.first_shot_mount_guard_tangent import guard_mount_tangent


def _state():
    return np.array([.03, -.12, .02, .09, .07, -.2, .15, .45, .2, .5,
                     .6, -.3, .4, .2, .7, .1, .2])


def _extrema(function, lo, hi):
    values = [function(lo), function(hi)]
    if lo < hi:
        for sign in (-1., 1.):
            fit = minimize_scalar(lambda z: sign*function(z), bounds=(lo, hi),
                                  method='bounded', options={'xatol': 1e-14})
            values.append(function(fit.x))
    return min(values), max(values)


def _quadratic_range(linear, quadratic, lo, hi):
    low, high = _extrema(lambda z: linear*z+quadratic*z*z, lo, hi)
    # This independently evaluated callback is the test's actual guard.
    return (np.nextafter(low-1e-14, -np.inf), np.nextafter(high+1e-14, np.inf)), []


def _support(q):
    lo, hi = sorted(q[6:8]); a, b = q[10:13], q[13:16]
    bounds = np.asarray([_extrema(lambda beta: a[j]*np.cos(beta)+b[j]*np.sin(beta), lo, hi)
                         for j in range(3)])
    return dict(v_min=bounds[:, 0], v_max=bounds[:, 1], intervals=[
        dict(left=0, driver=sorted(q[8:10]), v_bounds=bounds)])


def _problem(q, lower=None, upper=None):
    support = _support(q); eye = np.eye(len(q))
    lo = np.array([-.2, -.3, -.1]) if lower is None else np.asarray(lower)
    hi = np.array([.2, .1, .4]) if upper is None else np.asarray(upper)
    args = dict(value=q[:6], value_jacobian=eye[:6], lower=lo/np.exp(q[16]),
        upper=hi/np.exp(q[16]), log_scale_gradient=eye[16], support=support,
        tilt=q[6:8], tilt_jacobian=eye[6:8], driver=q[8:10], driver_jacobian=eye[8:10],
        interval_limits=np.array([[0., 1.]]), vector_a=q[10:13], vector_b=q[13:16],
        vector_a_jacobian=eye[10:13], vector_b_jacobian=eye[13:16],
        quadratic_interval_range=_quadratic_range)
    return guard_mount_tangent(**args), args


def _guard(q, lower=None, upper=None):
    support = _support(q); offset = q[4]*np.array([support['v_min'], support['v_max']])
    low, high = q[:3]+offset.min(0), q[:3]+offset.max(0)
    record = support['intervals'][0]
    (pmin, pmax), _ = _quadratic_range(q[3], q[5], *record['driver'])
    arho = q[4]*record['v_bounds'][2]
    low[2] = min(q[2]+pmin+arho.min(), q[2]+arho.min())
    high[2] = max(q[2]+pmax+arho.max(), q[2]+arho.max())
    lower = (np.array([-.2, -.3, -.1]) if lower is None else np.asarray(lower))/np.exp(q[16])
    upper = (np.array([.2, .1, .4]) if upper is None else np.asarray(upper))/np.exp(q[16])
    return np.r_[low-lower, upper-high, q[:3]-lower, upper-q[:3]]


def _envelope_margin_jacobian(result):
    origins = result['row_origins']; margins = result['current_margins']
    matrix = result['matrix'].toarray(); selected = []
    for kind, side in (('total', 1), ('total', -1), ('internal', 1), ('internal', -1)):
        for axis in range(3):
            group = [i for i, row in enumerate(origins) if row['axis']==axis and row['side']==side
                     and ((row['kind']=='internal') == (kind=='internal'))]
            minimum = min(margins[group]); active = [i for i in group if margins[i]==minimum]
            selected.append((minimum, matrix[active]))
    return selected


def test_guard_ranges_margins_and_full_derivative_include_interior_quadratic_vertex():
    q = _state(); result, _ = _problem(q)
    packed = _envelope_margin_jacobian(result)
    np.testing.assert_allclose([r[0] for r in packed], _guard(q), rtol=0, atol=0)
    assert any(row.get('polynomial_branch')=='vertex' for row in result['row_origins'])
    direction = np.random.default_rng(1017).normal(size=len(q)); direction /= np.linalg.norm(direction)
    eps = 2e-7
    fd = (_guard(q+eps*direction)-_guard(q-eps*direction))/(2*eps)
    predicted = [float(j[0]@direction) for _, j in packed]
    np.testing.assert_allclose(predicted, fd, atol=2e-9, rtol=2e-7)


def test_exact_equal_endpoint_extrema_retain_both_polynomial_supports():
    q = _state(); q[8:10] = [.25, .5]; q[3] = .375; q[5] = -.5
    result, _ = _problem(q)
    branches = {row['polynomial_branch'] for row in result['row_origins']
                if row['kind']=='axial' and row['extreme']==0}
    assert branches == {'low', 'high'}
    assert any(t['kind']=='polynomial' and t['branches']==2
               for t in result['report']['exact_tie_branches_retained'])


def test_zero_signed_axis_and_zero_lever_y_keep_supports_through_sign_crossing():
    q = _state(); q[1] = q[4] = 0.
    result, _ = _problem(q, [-1., -1., -1.], [1., 1., 1.])
    packed = _envelope_margin_jacobian(result)
    for sign in (-1., 1.):
        direction = np.zeros(len(q)); direction[4] = sign
        eps = 1e-7
        fd = (_guard(q+eps*direction, [-1.]*3, [1.]*3)-_guard(q, [-1.]*3, [1.]*3))/eps
        predicted = [float(np.min(j@direction)) for _, j in packed]
        np.testing.assert_allclose(predicted, fd, atol=2e-9, rtol=2e-7)


def test_flat_axial_polynomial_refuses_false_finite_complete_tangent():
    q = _state(); q[3] = q[5] = 0.
    with pytest.raises(ValueError, match='Flat axial polynomial.*continuum'):
        _problem(q)


def test_flat_free_trigonometric_support_refuses_false_complete_tangent():
    q = _state(); q[10] = q[13] = 0.
    with pytest.raises(ValueError, match='Flat trigonometric support.*continuum'):
        _problem(q)


def test_preserves_guard_legal_quadratic_cancellation_excluded_by_corner_box():
    q = _state(); q[:6] = [0., -.15, 0., 1., 0., 1.]; q[8:10] = [-1., 0.]; q[16] = 0.
    result, _ = _problem(q, [-1., -1., -.3], [1., 1., .3])
    assert result['current_margins'].min() > .04
    assert _guard(q, [-1., -1., -.3], [1., 1., .3]).min() > .04
    # Independent z/z^2 corners include an impossible (z=-1,z^2=0).
    assert q[2]+q[3]*(-1.)+q[5]*0. < -.3
