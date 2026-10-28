import numpy as np

from rinkspace.review.point_connections import polyline_order, source_connections, whole_entity_connections


def test_visible_endpoints_do_not_bridge_a_person():
    mask = np.zeros((20, 40), bool)
    mask[:, 18:22] = True
    uv = np.array([[3., 10.], [35., 10.]])
    original = uv.copy()
    edges, breaks = source_connections(uv, ['line', 'line'], [0., 1.], mask)
    assert edges == [] and breaks[0]['reason'] == 'source_occlusion'
    np.testing.assert_array_equal(uv, original)


def test_source_order_not_proximity_and_no_cross_span_connection():
    uv = np.array([[10., 10.], [25., 10.], [11., 11.], [15., 10.]])
    edges, _ = source_connections(uv, ['a', 'a', 'b', 'a'], [0., 2., 0., 1.], np.zeros((40, 40), bool))
    assert edges == [(0, 3), (3, 1)]


def test_partial_circle_is_not_forced_closed():
    uv = np.array([[10., 10.], [20., 6.], [30., 10.]])
    mask = np.zeros((40, 40), bool)
    edges, _ = source_connections(uv, ['arc']*3, [0., 1., 2.], mask)
    assert edges == [(0, 1), (1, 2)]
    closed, _ = source_connections(uv, ['arc']*3, [0., 1., 2.], mask, closed_runs=['arc'])
    assert closed[-1] == (2, 0)


def test_order_on_letter_does_not_connect_across_inner_hole():
    path = [[2., 2.], [20., 2.], [20., 20.], [2., 20.], [2., 2.]]
    uv = np.array([[4., 2.], [2., 4.], [20., 18.], [18., 2.]])
    order, distance = polyline_order(uv, path)
    np.testing.assert_allclose(order, [2., 70., 34., 16.])
    np.testing.assert_allclose(distance, 0.)


def test_whole_entity_reveals_all_links_including_gaps_and_occlusion():
    uv = np.array([[3.,10.],[15.,10.],[35.,10.]])
    mask = np.zeros((40,40),bool)
    mask[:,18:22] = True
    edges,dashed = whole_entity_connections(uv,['first','second','second'],[0.,1.,2.],mask,closed=True)
    assert edges == [(0,1),(1,2),(2,0)]
    assert {(r['a'],r['b']) for r in dashed} == set(edges)
