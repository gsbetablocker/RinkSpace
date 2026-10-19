from copy import deepcopy
import numpy as np
import pytest
from scipy import sparse
from rinkfit_start.ad_chart import PrintOnlyAdChart, effective_affine
from rinkfit_start.native_scene import NativeScene


def fixture():
    g = np.zeros(32); g[:5] = [.5, .12, .08, 1.91, .02]
    return {'blocks': [
        {'name': 'global32', 'values': g.tolist(), 'constant': True},
        {'name': 'ad/8', 'values': [.3, .1, .001, .018],
         'lower': [-1., .01, 0., 0.], 'upper': [1., .4, .02, .02]},
        {'name': 'print/one', 'values': [1., .1, 0., 0., 1., 0.],
         'lower': [-3., -2., -1., -2., -3., -1.], 'upper': [3., 2., 1., 2., 3., 1.]},
        {'name': 'print/two', 'values': [.8, 0., .1, 0., .7, .15],
         'lower': [-3., -2., -1., -2., -3., -1.], 'upper': [3., 2., 1., 2., 3., 1.]}],
        'materials': [{'label': 'Ad8.print', 'paths': [
            {'kind': 2, 'advertisement_block': 1, 'geometry_block': 2, 'points': [[0., 0.], [.6, .2], [1., 1.]]},
            {'kind': 2, 'advertisement_block': 1, 'geometry_block': 3, 'points': [[.2, .2], [.6, .6]]}]}]}


def test_exact_quotient_removes_four_dimensions_and_preserves_finite_support():
    m = fixture(); chart = PrintOnlyAdChart(m, (8,)); x = chart.pack()
    decoded = chart.unpack(x)
    assert chart.removed_free == 4
    assert len(x) == 12
    assert decoded['blocks'][1]['constant']
    before = NativeScene(m); after = NativeScene(decoded)
    for p in range(2):
        np.testing.assert_allclose(before.world(0, p, np.linspace(0, 1, 31)),
                                   after.world(0, p, np.linspace(0, 1, 31)), atol=1e-14)
    np.testing.assert_allclose(chart.pack(decoded), x, atol=1e-14)


def test_full_old_domain_is_not_replaced_by_fixed_chart_q_box():
    m = fixture(); moved = deepcopy(m)
    moved['blocks'][1]['values'][0] = .9
    chart = PrintOnlyAdChart(m, (8,)); x = chart.pack(moved)
    # A fixed .3/.1 chart would require q2=6 and wrongly reject this legal state.
    assert (effective_affine(moved['blocks'][1]['values'], moved['blocks'][2]['values'])[2] - .3) / .1 > 1.
    decoded, domain = chart.unpack(x, return_domain=True)
    assert 'LP' in domain[0]['horizontal_method']
    assert NativeScene(decoded).check_physics()['passed']
    before = NativeScene(moved); after = NativeScene(decoded)
    np.testing.assert_allclose(before.world(0, 0, np.linspace(0, 1, 21)),
                               after.world(0, 0, np.linspace(0, 1, 21)), atol=2e-14)


def test_common_witness_rejects_disjoint_component_box_images():
    m = fixture(); chart = PrintOnlyAdChart(m, (8,)); x = chart.pack()
    x[2] = -1.3; x[8] = 1.3
    with pytest.raises(ValueError, match='no legal original chart witness'):
        chart.unpack(x)


def test_inverse_chart_keeps_real_placement_free_and_jacobian_correct():
    m = fixture(); chart = PrintOnlyAdChart(m, (8,)); x = chart.pack()
    direction = np.array([.02, -.01, .03, .001, .002, .0003,
                          -.01, .02, -.02, .002, -.001, -.0003])
    eps = 1e-6; target = chart.unpack(x + eps * direction)
    tangent = chart.tangent(m); original = np.concatenate([b['values'] for b in m['blocks']])
    trial_raw = np.concatenate([b['values'] for b in target['blocks']])
    np.testing.assert_allclose((trial_raw - original) / eps, tangent @ direction, atol=1e-8)
    J = sparse.csr_matrix(np.arange(3 * len(original)).reshape(3, -1) / 100.)
    np.testing.assert_allclose(chart.pullback_jacobian(J).toarray(), (J @ tangent).toarray())
    assert np.max(abs(NativeScene(target).world(0, 0, [.2, .8]) - NativeScene(m).world(0, 0, [.2, .8]))) > 1e-9


def test_nonprint_incidence_and_degenerate_shape_are_rejected():
    m = fixture(); m['materials'][0]['paths'][0].pop('geometry_block')
    with pytest.raises(ValueError, match='observed installation'): PrintOnlyAdChart(m, (8,))
    chart = PrintOnlyAdChart(fixture(), (8,)); x = chart.pack(); x[4] = 0.
    with pytest.raises(ValueError, match='determinant'): chart.unpack(x)


def test_equivalent_legacy_gauges_pack_identically_and_fresh_state_stays_reduced():
    from rinkfit_start.ad_chart import local_affine
    m = fixture(); gauged = deepcopy(m); other = [.31, .12, .0015, .0175]
    for qi in (2, 3):
        gauged['blocks'][qi]['values'] = local_affine(other, effective_affine(m['blocks'][1]['values'], m['blocks'][qi]['values'])).tolist()
    gauged['blocks'][1]['values'] = other
    chart = PrintOnlyAdChart(m, (8,))
    np.testing.assert_allclose(chart.pack(gauged), chart.pack(), atol=1e-15)
    fresh = chart.unpack(chart.pack())
    resumed = PrintOnlyAdChart(fresh, (8,))
    assert resumed.removed_free == 0
    assert len(resumed.free) == len(chart.free) == 12
    assert not any(i in resumed.free for i in range(32, 36))
    np.testing.assert_allclose(resumed.pack(), chart.pack(), atol=1e-15)
