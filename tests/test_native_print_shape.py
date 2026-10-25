"""Active print geometry and analytic derivatives; no optimization is run.

These cases use independent finite differences of the common native forward,
not an alternate Python production evaluator. They do not assert video accuracy.
"""
import os

import numpy as np
import pytest

from rinkspace.core.native import NativeCore, NativeError

pytestmark = pytest.mark.skipif(not os.environ.get("RINKSPACE_CORE_LIBRARY"),
                                reason="explicit native library required")


@pytest.fixture
def core():
    return NativeCore(os.environ["RINKSPACE_CORE_LIBRARY"])


BOARD = [.5, .25, .1, .12, 1.85, .02]
BOUNDS = [0., 0., 1., 1.]
VERTICES = [[.1, .1], [.8, .3], [.7, .9], [.2, .85]]
STATIONS = [.035, .24, .48, .73, .91]
SHAPE = [.01, -.015, .03, -.02, .15, -.07, .05, -.04, .08, -.03]


def installation(core):
    return np.array([core.board_course_chart(BOARD)[1]+.02, .003, .11, .025, .008])


def assert_rows_valid(result):
    assert not np.asarray(result["status"]).any()
    for key in ("material_uv", "sz", "xyz"):
        if key in result:
            assert np.isfinite(result[key]).all()


def finite_difference(function, values, *, epsilon=2e-7):
    values = np.array(values, dtype=float)
    columns = []
    for column in range(len(values)):
        plus, minus = values.copy(), values.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        columns.append((function(plus)-function(minus))/(2*epsilon))
    return np.stack(columns, axis=-1)


def test_zero_shape_replays_old_native_path_and_chart_bit_exactly(core):
    p = installation(core)
    actual = core.sample_deformed_print_path(BOARD, p, VERTICES, STATIONS, BOUNDS, np.zeros(10))
    assert_rows_valid(actual)
    np.testing.assert_array_equal(actual["xyz"], core.sample_path(BOARD, 5, p, VERTICES, STATIONS))
    np.testing.assert_array_equal(actual["sz"], core.sample_print_chart(p, VERTICES, STATIONS))
    zero = core.deform_print_points(VERTICES, BOUNDS, np.zeros(10))
    assert_rows_valid(zero)
    np.testing.assert_array_equal(zero["material_uv"], VERTICES)


def test_original_station_is_interpolated_before_internal_warp(core):
    p = installation(core)
    shape = np.zeros(10)
    shape[4:7] = [.4, -.2, .1]
    curve = core.sample_deformed_print_path(BOARD, p, [[0., .4], [1., .4]],
                                           [0., .5, 1.], BOUNDS, shape)
    direct = core.deform_print_points([[.5, .4]], BOUNDS, shape)
    assert_rows_valid(curve)
    np.testing.assert_array_equal(curve["material_uv"][1], direct["material_uv"][0])
    # Both endpoints are anchored by the internal warp. A vertex-only warp
    # followed by interpolation would incorrectly leave the midpoint at .5.
    assert abs(curve["material_uv"][1, 0]-.5) > .01
    np.testing.assert_array_equal(curve["material_uv"][[0, 2], 0], [0., 1.])


def test_shared_monotone_map_preserves_closed_hole_and_axis_aligned_edges(core):
    shape = np.array([.03, -.02, .1, -.08, .3, -.1, .12, -.2, .1, .06])
    levels = np.linspace(0., 1., 101)
    points = np.column_stack([levels, np.full(len(levels), .4)])
    horizontal = core.deform_print_points(points, BOUNDS, shape)
    assert_rows_valid(horizontal)
    assert np.all(np.diff(horizontal["material_uv"][:, 0]) > 0)
    assert np.ptp(horizontal["material_uv"][:, 1]) == 0
    vertical = core.deform_print_points(points[:, ::-1], BOUNDS, shape)
    assert np.ptp(vertical["material_uv"][:, 0]) == 0
    assert np.all(np.diff(vertical["material_uv"][:, 1]) > 0)
    outer = core.deform_print_points([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0., 0.]], BOUNDS, shape)
    theta = np.linspace(0., 2*np.pi, 101)
    hole_seed = np.column_stack([.5+.2*np.cos(theta), .5+.15*np.sin(theta)])
    hole_seed[-1] = hole_seed[0]
    hole = core.deform_print_points(hole_seed, BOUNDS, shape)
    np.testing.assert_array_equal(outer["material_uv"][0], outer["material_uv"][-1])
    np.testing.assert_array_equal(hole["material_uv"][0], hole["material_uv"][-1])
    assert np.all(hole["material_uv"] > outer["material_uv"].min(axis=0))
    assert np.all(hole["material_uv"] < outer["material_uv"].max(axis=0))


def test_material_shape_analytic_jacobian_matches_all_ten_independent_fd_columns(core):
    actual = core.deform_print_points(VERTICES, BOUNDS, SHAPE)
    expected = finite_difference(lambda state: core.deform_print_points(VERTICES, BOUNDS, state,
                                                jacobian=False)["material_uv"], SHAPE)
    np.testing.assert_allclose(actual["d_uv_d_shape"], expected, rtol=2e-7, atol=4e-9)


def test_world_shape_and_installation_and_station_jacobians_match_fd_on_arc(core):
    p = installation(core)
    result = core.sample_deformed_print_path(BOARD, p, VERTICES, STATIONS, BOUNDS, SHAPE)
    assert_rows_valid(result)
    shape_fd = finite_difference(lambda state: core.sample_deformed_print_path(BOARD, p, VERTICES,
        STATIONS, BOUNDS, state, jacobian=False)["xyz"], SHAPE)
    installation_fd = finite_difference(lambda values: core.sample_deformed_print_path(BOARD, values,
        VERTICES, STATIONS, BOUNDS, SHAPE, jacobian=False)["xyz"], p)
    np.testing.assert_allclose(result["d_xyz_d_shape"], shape_fd, rtol=5e-5, atol=2e-7)
    np.testing.assert_allclose(result["d_xyz_d_installation"], installation_fd, rtol=5e-5, atol=2e-7)
    stations = np.asarray(STATIONS)
    epsilon = 2e-7
    station_fd = (core.sample_deformed_print_path(BOARD, p, VERTICES, stations+epsilon,
        BOUNDS, SHAPE, jacobian=False)["xyz"]-core.sample_deformed_print_path(BOARD, p,
        VERTICES, stations-epsilon, BOUNDS, SHAPE, jacobian=False)["xyz"])/(2*epsilon)
    np.testing.assert_allclose(result["d_xyz_d_station"], station_fd, rtol=5e-5, atol=2e-7)


def test_deformed_path_stays_continuous_across_every_course_join_and_seam(core):
    for join in core.board_course_chart(BOARD):
        p = [join-.005, .003, 0., .01, .008]
        epsilon = 1e-7
        result = core.sample_deformed_print_path(BOARD, p, [[0., .3], [1., .3]],
            [.5-epsilon/.01, .5, .5+epsilon/.01], BOUNDS, np.zeros(10))
        assert_rows_valid(result)
        assert np.linalg.norm(result["xyz"][2]-result["xyz"][0]) < 2.01*epsilon


@pytest.mark.parametrize("index,value", [(2, 1000.), (3, -1000.), (4, 1000.), (8, float("nan"))])
def test_nonfinite_or_collapsed_shape_domain_is_rejected(core, index, value):
    shape = np.zeros(10)
    shape[index] = value
    with pytest.raises(NativeError):
        core.deform_print_points(VERTICES, BOUNDS, shape)
    with pytest.raises(NativeError):
        core.sample_deformed_print_path(BOARD, installation(core), VERTICES, STATIONS, BOUNDS, shape)


@pytest.mark.parametrize("bounds", [[0., 0., 0., 1.], [1., 0., 0., 1.], [0., 0., float("inf"), 1.]])
def test_group_box_must_have_finite_positive_spans(core, bounds):
    with pytest.raises(NativeError):
        core.deform_print_points(VERTICES, bounds, np.zeros(10))


def test_invalid_rows_remain_nan_and_do_not_clamp_source_station(core):
    result = core.sample_deformed_print_path(BOARD, installation(core), VERTICES,
        [-.01, .1, 1.01, float("nan")], BOUNDS, SHAPE)
    np.testing.assert_array_equal(result["status"], [3, 0, 3, 1])
    for key, values in result.items():
        if key != "status":
            assert np.isnan(values[[0, 2, 3]]).all(), key
    bad_seed = core.deform_print_points([[1.01, .5], [.3, .4]], BOUNDS, np.zeros(10))
    np.testing.assert_array_equal(bad_seed["status"], [3, 0])
    assert np.isnan(bad_seed["material_uv"][0]).all()
    assert np.isnan(bad_seed["d_uv_d_shape"][0]).all()


def test_board_height_remains_a_real_domain_and_no_jacobian_mode_agrees(core):
    p = [.1, .019, 0., .02, .008]
    failed = core.sample_deformed_print_path(BOARD, p, [[.5, 1.]], [0.], BOUNDS, np.zeros(10))
    np.testing.assert_array_equal(failed["status"], [3])
    assert np.isnan(failed["material_uv"]).all()
    full = core.sample_deformed_print_path(BOARD, installation(core), VERTICES, STATIONS, BOUNDS, SHAPE)
    light = core.sample_deformed_print_path(BOARD, installation(core), VERTICES, STATIONS, BOUNDS, SHAPE, jacobian=False)
    for key in ("material_uv", "sz", "xyz", "status"):
        np.testing.assert_array_equal(full[key], light[key])
    assert light["d_xyz_d_shape"] is None
    assert light["d_xyz_d_installation"] is None


def test_square_is_an_exact_metric_square_under_panel_aspect_and_rotation(core):
    p = installation(core)
    values = [.4, .5, np.log(.6)]
    corners = core.sample_print_square_path(BOARD, p, values, [0., .25, .5, .75, 1.])
    assert_rows_valid(corners)
    edges = np.diff(corners["sz"], axis=0)
    np.testing.assert_allclose(np.linalg.norm(edges, axis=1), np.exp(values[2])*p[4], rtol=1e-13)
    np.testing.assert_allclose(np.sum(edges*np.roll(edges, -1, axis=0), axis=1), 0., atol=1e-18)
    np.testing.assert_array_equal(corners["sz"][0], corners["sz"][-1])
    vertices = core.print_square_vertices(p, values[:2], np.exp(values[2]))
    np.testing.assert_array_equal(corners["xyz"], core.sample_path(BOARD, 5, p, vertices, [0., .25, .5, .75, 1.]))


def test_square_shape_and_installation_derivatives_include_width_cancellation(core):
    p = installation(core)
    values = np.array([.4, .5, np.log(.6)])
    stations = [.04, .29, .55, .83, .96]
    result = core.sample_print_square_path(BOARD, p, values, stations)
    shape_fd = finite_difference(lambda q: core.sample_print_square_path(BOARD, p, q, stations,
                                               jacobian=False)["xyz"], values)
    install_fd = finite_difference(lambda q: core.sample_print_square_path(BOARD, q, values, stations,
                                               jacobian=False)["xyz"], p)
    np.testing.assert_allclose(result["d_xyz_d_square"], shape_fd, rtol=5e-5, atol=2e-7)
    np.testing.assert_allclose(result["d_xyz_d_installation"], install_fd, rtol=5e-5, atol=2e-7)
    straight = core.sample_print_square_path(BOARD, [.2, .004, 0., .03, .008], values, stations)
    # On a straight course the panel width moves only the square centre:
    # its physical side is independent of that panel width.
    np.testing.assert_allclose(straight["d_xyz_d_installation"][:, :, 3],
                               np.tile([values[0], 0., 0.], (len(stations), 1)), atol=2e-16)


def test_shared_rectangle_edges_join_exactly_and_reverse_keeps_source_direction(core):
    p = installation(core)
    values = [.4, .5, np.log(.35), np.log(.5)]
    curves = [core.sample_print_rectangle_path(BOARD, p, values, side, [0., 1.]) for side in range(4)]
    for side in range(4):
        assert_rows_valid(curves[side])
        np.testing.assert_array_equal(curves[side]["sz"][1], curves[(side+1)%4]["sz"][0])
    for side in range(4):
        forward = core.sample_print_rectangle_path(BOARD, p, values, side, [1., .7, .4, 0.])
        backward = core.sample_print_rectangle_path(BOARD, p, values, side, [0., .3, .6, 1.], reverse=True)
        np.testing.assert_allclose(forward["xyz"], backward["xyz"], atol=1e-15)
        np.testing.assert_allclose(forward["d_xyz_d_station"], -backward["d_xyz_d_station"], atol=1e-15)


def test_rectangle_shape_and_installation_jacobians_match_independent_fd(core):
    p = installation(core)
    values = np.array([.4, .5, np.log(.35), np.log(.5)])
    result = core.sample_print_rectangle_path(BOARD, p, values, 2, STATIONS, reverse=True)
    shape_fd = finite_difference(lambda q: core.sample_print_rectangle_path(BOARD, p, q, 2,
                                         STATIONS, reverse=True, jacobian=False)["xyz"], values)
    install_fd = finite_difference(lambda q: core.sample_print_rectangle_path(BOARD, q, values, 2,
                                         STATIONS, reverse=True, jacobian=False)["xyz"], p)
    np.testing.assert_allclose(result["d_xyz_d_rectangle"], shape_fd, rtol=5e-5, atol=2e-7)
    np.testing.assert_allclose(result["d_xyz_d_installation"], install_fd, rtol=5e-5, atol=2e-7)


@pytest.mark.parametrize("family", ["square", "rectangle"])
def test_analytic_structures_reject_invalid_sizes_and_preserve_bad_station_rows(core, family):
    p = installation(core)
    if family == "square":
        call = lambda q, stations: core.sample_print_square_path(BOARD, p, q, stations)
        values = [.4, .5, np.log(.5)]
    else:
        call = lambda q, stations: core.sample_print_rectangle_path(BOARD, p, q, 0, stations)
        values = [.4, .5, np.log(.5), np.log(.5)]
    invalid = values.copy()
    invalid[2] = -1000.
    with pytest.raises(NativeError):
        call(invalid, [0.])
    result = call(values, [-.1, .2, 1.1])
    np.testing.assert_array_equal(result["status"], [3, 0, 3])
    assert np.isnan(result["xyz"][[0, 2]]).all()


def test_empty_batches_keep_explicit_output_shapes(core):
    result = core.sample_deformed_print_path(BOARD, installation(core), VERTICES, [], BOUNDS, np.zeros(10))
    assert result["xyz"].shape == (0, 3)
    assert result["d_xyz_d_shape"].shape == (0, 3, 10)
    assert result["status"].shape == (0,)
    points = core.deform_print_points(np.empty((0, 2)), BOUNDS, np.zeros(10))
    assert points["material_uv"].shape == (0, 2)


def test_full_group_float_domain_is_identical_with_and_without_jacobians(core):
    shape = np.zeros(10)
    shape[2] = -690.
    shape[4] = -690.
    # Both exponential factors are individually positive, but their product
    # collapses a material interval. Reject the whole map, even for a point
    # outside that interval and even in a drawing-only no-Jacobian call.
    for jacobian in (True, False):
        with pytest.raises(NativeError):
            core.deform_print_points([[.9, .5]], BOUNDS, shape, jacobian=jacobian)
        with pytest.raises(NativeError):
            core.sample_deformed_print_path(BOARD, installation(core), VERTICES,
                STATIONS, BOUNDS, shape, jacobian=jacobian)


def test_constant_coordinate_convex_sampling_cannot_drift_outside_group_by_one_ulp(core):
    # Current Ad1 SKATING_I's actual narrow right edge exposed this: legacy
    # (1-t)*x+t*x can exceed x by one ulp for valid source-index fractions.
    left = .4772995078655752
    right = .4852724527680921
    lower = .3208589820563251
    upper = .5264262044740269
    vertices = [[left, lower], [left, upper], [right, upper], [right, lower], [left, lower]]
    bounds = [left, lower, right, upper]
    stations = np.linspace(0., 1., 73)
    for jacobian in (False, True):
        result = core.sample_deformed_print_path(BOARD, [.2, .004, 0., .03, .008],
            vertices, stations, bounds, np.zeros(10), jacobian=jacobian)
        assert_rows_valid(result)
        assert np.all(result["material_uv"] >= np.array(bounds[:2]))
        assert np.all(result["material_uv"] <= np.array(bounds[2:]))
        np.testing.assert_array_equal(result["material_uv"][0], vertices[0])
        np.testing.assert_array_equal(result["material_uv"][-1], vertices[-1])
        np.testing.assert_array_equal(result["sz"], core.sample_print_chart(
            [.2, .004, 0., .03, .008], vertices, stations))
        np.testing.assert_array_equal(result["xyz"], core.sample_path(
            BOARD, 5, [.2, .004, 0., .03, .008], vertices, stations))
    # Direct material requests outside a bound remain invalid, including just
    # one representable float beyond the bound. This is not a tolerance bypass.
    invalid = core.deform_print_points([[np.nextafter(right, np.inf), .4]], bounds, np.zeros(10))
    np.testing.assert_array_equal(invalid["status"], [3])
    assert np.isnan(invalid["material_uv"]).all()
