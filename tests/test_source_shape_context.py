import numpy as np
from rinkspace.session.source_shape_context import (
    assess_native_band_shape, measured_band_unmasked, source_band_conflicts,
    transition_intersects_path,
)


def test_asymmetric_source_transition_keeps_its_original_support():
    # q50 is near the low end: a symmetric interval would reject x=4.
    q = np.array([[[1., 3.], [2., 3.], [5., 3.]]])
    assert transition_intersects_path(q, [[4., 0.], [4., 6.]]).tolist() == [True]
    assert transition_intersects_path(q, [[5.1, 0.], [5.1, 6.]]).tolist() == [False]
    # An extrapolated line is not the finite visible contour.
    assert transition_intersects_path(q, [[4., 4.], [4., 6.]]).tolist() == [False]


def test_person_inside_paint_band_is_excluded_even_with_clear_edges():
    q = np.array([[[[5., 8.], [6., 8.], [7., 8.]],
                   [[15., 8.], [16., 8.], [17., 8.]]]])
    mask = np.zeros((24, 24), bool)
    mask[8, 11] = True
    assert not mask[8, 6] and not mask[8, 16]
    assert measured_band_unmasked(q, mask).tolist() == [False]
    mask[:] = False
    assert measured_band_unmasked(q, mask).tolist() == [True]


def test_flat_arc_paint_claims_do_not_bridge_missing_source_queries():
    centers = np.array([[0., 5.], [10., 4.], [20., 5.], [40., 8.]])
    q = np.empty((4, 2, 3, 2))
    for side, shift in enumerate((-1., 1.)):
        for k, width in enumerate((-.2, 0., .2)):
            q[:, side, k] = centers+[0., shift+width]
    measurement = dict(candidate_query=np.array([0, 1, 2, 4]),
        query_curve=np.zeros(5, int), candidate_side_quantiles_uv=q)
    before = q.copy()
    points = np.array([[5., 4.5], [15., 4.5], [30., 6.5], [10., 9.]])
    conflicts = source_band_conflicts(points, measurement, range(4))
    assert conflicts == [[(0, 1)], [(1, 2)], [], []]
    assert np.array_equal(q, before)


def test_paired_circle_boundaries_and_straight_context_share_one_transform():
    angle = np.linspace(0., 2*np.pi, 129)
    direction = np.c_[np.cos(angle), np.sin(angle)]
    native = {role: direction*radius+[60., 60.]
              for role, radius in [('middle', 20.), ('outer', 30.)]}
    native['baseline'] = np.array([[10., 10.], [110., 10.]])
    native['side'] = np.array([[10., 10.], [10., 110.]])
    translation = np.array([3., 5.])
    context = {'baseline': np.c_[np.linspace(10., 110., 20), np.full(20, 10.)]+translation,
               'side': np.c_[np.full(20, 10.), np.linspace(10., 110., 20)]+translation}
    selected = np.arange(0, 128, 4)
    q = np.empty((len(selected), 2, 3, 2))
    for side, role in enumerate(('middle', 'outer')):
        for k, displacement in enumerate((-.6, 0., .6)):
            q[:, side, k] = native[role][selected]+translation+direction[selected]*displacement
    before = q.copy()
    result = assess_native_band_shape(dict(candidate_side_quantiles_uv=q), range(len(q)), native,
        roles=('middle', 'outer'), context_observations=context)
    assert result['source_shape_supported'].all()
    assert result['source_UV_unchanged']
    assert not result['fit_eligible'] and not result['physical_identity_confirmed']
    assert np.array_equal(before, q)
    assert np.array_equal(result['original_source_uv'], q[:, :, 1])


def test_flat_ring_queries_keep_uniform_positions_and_outward_radial_profiles():
    from rinkspace.vision_programme_ice_review import _source_queries
    from rinkspace.session.projected_curve_queries import projected_curve_queries
    angle = np.linspace(0., 2*np.pi, 257)
    center = np.array([500., 300.])
    points = center+np.c_[280*np.cos(angle), 22*np.sin(angle)]
    expected = projected_curve_queries(points, np.ones(len(points), bool), (600, 1000), 24., 48.,
        closed=True, center_uv=center)
    query, normal, intervals = _source_queries(dict(query_sampling='continuous_arc_length', closed=True,
        profile_direction='outward_radial', _projected_center=center), points, np.ones(len(points)), (600, 1000), 24., 48.)
    assert np.array_equal(query, expected['points_uv'])
    radial = query-center
    assert np.allclose(normal, radial/np.linalg.norm(radial, axis=1)[:, None])
    assert np.allclose(intervals[:, 0], 4.-np.linalg.norm(radial, axis=1))
    assert np.allclose(intervals[:, 1], 48.)
