import copy

import numpy as np

from rinkfit_joint.native_print_seed_geometry import segment_navigation_seeds, seed_geometry_columns
from rinkfit_joint.native_print_track_readout import batch_interfaces, read_interval
from rinkfit_joint.source_worker import _channels


ACTUAL_SEGMENT = dict(
    uv=[[394.3125443433013, 613.5848051756969],
        [401.8725732720592, 612.4784417624868],
        [402.40923759132727, 586.695494949143]],
    closed=False, native_simplification_tolerance_px=2.465590137017283,
    localization_sigma_px=[2.052939200504005, 3.8058910268972914, 2.6144485008106497],
    transition_iqr_px=[2.7694149814799025, 5.134146995284446, 3.5268910275935665])


def test_actual_corner_uses_each_edge_without_altering_source():
    source = copy.deepcopy(ACTUAL_SEGMENT)
    seeds = segment_navigation_seeds(source)
    assert source == ACTUAL_SEGMENT
    assert len(seeds) == 2
    points = np.asarray(source['uv'])
    for seed, endpoints in zip(seeds, ([0, 1], [1, 2])):
        a, b = points[endpoints]
        assert seed['source_edge_indices'] == endpoints
        np.testing.assert_allclose(seed['uv'], (a + b) / 2)
        assert abs(np.dot(seed['normal'], b - a)) < 1e-12
        assert seed['tangent_halfspan_px'] < np.linalg.norm(b - a) / 2
        assert seed['navigation_only']
        assert 'source_point_index' not in seed
    old = points[2] - points[0]
    old_normal = np.array([-old[1], old[0]]) / np.linalg.norm(old)
    assert np.degrees(np.arccos(abs(np.dot(old_normal, seeds[0]['normal'])))) > 60


def test_repeated_closed_endpoint_and_zero_length_edges_do_not_duplicate():
    segment = dict(uv=[[0., 0.], [0., 0.], [8., 0.], [8., 8.], [0., 0.]],
                   closed=True, native_simplification_tolerance_px=1.)
    seeds = segment_navigation_seeds(segment, stride=1)
    assert len(seeds) == 3
    assert len({tuple(s['source_edge_indices']) for s in seeds}) == 3
    assert all(np.isfinite(s['normal']).all() for s in seeds)


def test_dense_source_readouts_keep_their_measured_sample_positions():
    uv = [[float(i), float(i * i) / 10] for i in range(10)]
    seeds = segment_navigation_seeds(dict(uv=[[99., 99.], [98., 98.]],
        source_native_readout_uv=uv, native_simplification_tolerance_px=2.))
    assert [s['source_point_index'] for s in seeds] == [0, 4, 8, 9]
    assert [s['uv'] for s in seeds] == [uv[i] for i in [0, 4, 8, 9]]
    assert all('source_edge_indices' not in s for s in seeds)


def test_anchor_uncertainty_is_preserved_once_per_seed_and_unknown_stays_unknown():
    seeds = [dict(s, native_pts=415744) for s in segment_navigation_seeds(ACTUAL_SEGMENT)]
    seeds.append(dict(native_pts=416000, uv=[1., 2.], normal=[0., 1.]))
    columns = seed_geometry_columns(seeds)
    np.testing.assert_array_equal(columns['anchor_endpoint_sigma_px'][:2],
        np.asarray(ACTUAL_SEGMENT['localization_sigma_px'])[[[0, 1], [1, 2]]])
    np.testing.assert_array_equal(columns['anchor_endpoint_transition_iqr_px'][:2],
        np.asarray(ACTUAL_SEGMENT['transition_iqr_px'])[[[0, 1], [1, 2]]])
    assert columns['anchor_native_pts'].tolist() == [415744, 415744, 416000]
    assert np.isnan(columns['anchor_endpoint_sigma_px'][2]).all()
    assert columns['anchor_edge_indices'][2].tolist() == [-1, -1]


def test_uncertainty_with_different_sample_count_is_not_attached_to_wrong_points():
    source = dict(ACTUAL_SEGMENT, localization_sigma_px=[.1, .2])
    assert all(s['source_anchor_localization_sigma_px'] == [None, None]
               for s in segment_navigation_seeds(source))


def test_new_navigation_is_remeasured_from_current_pixels():
    import cv2
    image = np.zeros((1080, 1920, 3), np.uint8)
    image[580:620, 400:430] = 230
    segment = dict(uv=[[400., 620.], [430., 620.], [430., 580.]],
                   native_simplification_tolerance_px=2.)
    seeds = [dict(s, native_pts=256, seed_index=i, label='Ad1.print.test')
             for i, s in enumerate(segment_navigation_seeds(segment))]
    out = []
    read_interval({256: image}, {256: np.zeros(image.shape[:2], np.uint8)},
                  {256: seeds}, 256, 256, lambda pt, im, rows: out.extend(rows), set(), {})
    assert len(out) == 2
    channels = _channels(image[:, :, ::-1])
    for seed_index, uv, normal, profile, fb in out:
        # The final stored coordinate is the measured pixel interface, not a midpoint.
        assert np.linalg.norm(uv - seeds[seed_index]['uv']) > .1
        fresh = batch_interfaces((uv - profile[2] * normal)[None], normal[None], channels)
        assert fresh['valid'][0]
        np.testing.assert_allclose(fresh['uv'][0], uv, atol=1e-6)
