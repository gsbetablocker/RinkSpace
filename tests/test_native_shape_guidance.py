import numpy as np
import pytest

from rinkspace.session.native_shape_guidance import point_to_polyline, point_to_partial_polyline, register_native_shapes


def _circle(center, radius, count=129):
    angle = np.linspace(0., 2.*np.pi, count)
    return np.asarray(center)+radius*np.c_[np.cos(angle), np.sin(angle)]


def _apply(points, affine):
    return np.asarray(points)@affine[:2, :2].T+affine[:2, 2]


def test_finite_polyline_positions_normal_residuals_and_endpoint_distances():
    observed = np.array([[3., 4.], [12., 0.], [-2., -3.]])
    result = point_to_polyline(observed, [[0., 0.], [0., 0.], [10., 0.]])
    np.testing.assert_allclose(result.closest_uv, [[3., 0.], [10., 0.], [0., 0.]])
    np.testing.assert_allclose(result.normal_residual_px, [4., 0., -3.])
    np.testing.assert_allclose(result.distance_px, [4., 2., np.sqrt(13.)])
    np.testing.assert_array_equal(result.segment_indices, [1, 1, 1])
    np.testing.assert_allclose(result.segment_fractions, [.3, 1., 0.])
    np.testing.assert_allclose(result.residual_uv, observed-result.closest_uv)


def test_bounded_point_and_segment_blocks_preserve_finite_path_matches():
    line = np.c_[np.arange(131)*10., np.zeros(131)]
    observed = np.c_[np.linspace(-5., 1305., 1027), np.full(1027, 3.)]
    expected = np.c_[np.clip(observed[:, 0], 0., 1300.), np.zeros(1027)]
    result = point_to_polyline(observed, line)
    np.testing.assert_allclose(result.closest_uv, expected, atol=1e-12)
    np.testing.assert_allclose(result.normal_residual_px, 3.)
    np.testing.assert_allclose(result.distance_px, np.linalg.norm(observed-expected, axis=1))


def test_partial_projection_keeps_local_support_without_bridging_invalid_vertices():
    line = [[np.nan, np.nan], [0., 0.], [2., 0.], [np.nan, np.nan],
            [8., 0.], [10., 0.], [np.inf, np.inf]]
    result = point_to_partial_polyline([[1., 1.], [5., 0.], [9., -2.]], line)
    np.testing.assert_array_equal(result.segment_indices, [1, 1, 4])
    np.testing.assert_allclose(result.closest_uv, [[1., 0.], [2., 0.], [9., 0.]])
    np.testing.assert_allclose(result.segment_fractions, [.5, 1., .5])
    np.testing.assert_allclose(result.distance_px, [1., 3., 2.])
    with pytest.raises(ValueError, match="supported segment"):
        point_to_partial_polyline([[0., 0.]], [[0., 0.], [np.nan, np.nan], [1., 0.]])


def test_line_and_circle_recover_one_affine_without_changing_source_pixels():
    affine = np.array([[1.08, .07, 8.], [-.04, .95, -4.], [0., 0., 1.]])
    circle = _circle([40., 50.], 28.)
    line = np.array([[5., 5.], [80., 15.]])
    paths = {"native/ring:outer": circle, "native/C:centreline": line}
    line_samples = np.linspace(line[0], line[1], 100)
    native = np.r_[circle[:-1], line_samples]
    names = np.array(["native/ring:outer"]*128+["native/C:centreline"]*100)
    observed = _apply(native, affine)
    original = observed.copy(); original_paths = {k: v.copy() for k, v in paths.items()}
    observed.flags.writeable = False
    result = register_native_shapes(observed, names, paths)
    assert result.rms_distance_px < .02
    np.testing.assert_allclose(result.affine, affine, atol=.03)
    np.testing.assert_array_equal(result.observed_uv, original)
    np.testing.assert_array_equal(observed, original)
    np.testing.assert_array_equal(result.native_curve_names, names)
    for name in paths:
        np.testing.assert_array_equal(paths[name], original_paths[name])
        rows = names == name
        shared = point_to_polyline(observed[rows], _apply(paths[name], result.affine))
        np.testing.assert_allclose(shared.distance_px, result.matches.distance_px[rows])
        np.testing.assert_allclose(shared.normal_residual_px, result.matches.normal_residual_px[rows])
    assert result.iterations > 0
    assert result.stop_reason in {"rms_distance_change", "relative_transform_change"}


def test_concentric_circles_share_transform_and_wrong_inner_outer_binding_keeps_error():
    paths = {"native/R2:inner": _circle([0., 0.], 30.),
             "native/R2:outer": _circle([0., 0.], 60.)}
    names = np.array(["native/R2:inner"]*128+["native/R2:outer"]*128)
    native = np.r_[paths["native/R2:inner"][:-1], paths["native/R2:outer"][:-1]]
    affine = np.array([[1.06, .04, 3.], [.02, .96, -2.], [0., 0., 1.]])
    observed = _apply(native, affine)
    correct = register_native_shapes(observed, names, paths)
    assert correct.rms_distance_px < .02
    # A circle's rotational symmetry does not identify a unique affine.
    # What must agree is the shared transformed geometry and source residual.
    for name in paths:
        assert point_to_polyline(observed[names == name], _apply(paths[name], correct.affine)).distance_px.max() < .03
    wrong_names = np.where(names == "native/R2:inner", "native/R2:outer", "native/R2:inner")
    wrong = register_native_shapes(observed, wrong_names, paths)
    assert wrong.rms_distance_px > 15.
    assert np.median(np.abs(wrong.matches.normal_residual_px)) > 10.
    np.testing.assert_array_equal(wrong.observed_uv, observed)


@pytest.mark.parametrize("line", [np.zeros((3, 2)), [[0., 0.], [np.nan, 0.]]])
def test_invalid_or_degenerate_polylines_fail(line):
    with pytest.raises(ValueError):
        point_to_polyline([[0., 1.]], line)


def test_missing_path_nonfinite_pixels_and_collinear_affine_support_fail():
    line = np.array([[0., 0.], [100., 0.]])
    observed = np.array([[0., 1.], [30., 1.], [60., 1.], [100., 1.]])
    with pytest.raises(ValueError, match="Missing native"):
        register_native_shapes(observed, np.array(["unknown"]*4), {"line": line})
    with pytest.raises(ValueError, match="degenerate"):
        register_native_shapes(observed, np.array(["line"]*4), {"line": line})
    observed[1, 0] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        register_native_shapes(observed, np.array(["line"]*4), {"line": line})


def test_singular_initial_affine_is_rejected():
    circle = _circle([0., 0.], 30.)
    with pytest.raises(ValueError, match="degenerate"):
        register_native_shapes(circle[:-1], np.array(["circle"]*128), {"circle": circle},
                               initial_affine=np.diag([1., 0., 1.]))


def test_flat_partial_concentric_materials_keep_their_original_source_uv():
    # A low rink camera makes the two axes very different. Both sides of a
    # partly visible annulus must still share one transform and keep their UV.
    flatten=np.array([[5.,.4,1250.],[0.,.16,480.],[0.,0.,1.]])
    paths={name:_apply(_circle([0.,0.],radius),flatten)
           for name,radius in [('middle',30.),('outer',45.)]}
    indices=np.r_[np.arange(8,56),np.arange(72,120)]
    original=np.r_[paths['middle'][indices],paths['outer'][indices]]
    displacement=np.array([[1.04,.12,9.],[.001,.97,8.],[0.,0.,1.]])
    observed=_apply(original,displacement);names=np.array(['middle']*len(indices)+['outer']*len(indices))
    result=register_native_shapes(observed,names,paths)
    assert result.rms_distance_px<.02
    np.testing.assert_array_equal(result.observed_uv,observed)
    for name in paths:
        distance=point_to_polyline(observed[names==name],_apply(paths[name],result.affine)).distance_px
        assert distance.max()<.04
