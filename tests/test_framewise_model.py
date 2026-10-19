from copy import deepcopy

import numpy as np
import pytest

from calibration.program_tracking.framewise_model import build_row, evaluate_rows, feature_world
from calibration.program_tracking.projection import project_central_valid


def scene_and_camera():
    border = [[0., 0., 0.], [10., 0., 0.], [10., 10., 0.], [0., 10., 0.]]
    scene = dict(ice_boundary=dict(points_xyz=border, parts_counterclockwise=[
        dict(id="upper_long_side", first_vertex=0, vertex_count=1)]),
        board_surface=dict(top_edge_xyz=[[x, y, 3.] for x, y, _ in border]),
        ice_features=[], advertisements=[dict(user_number=2, extent_scope="external_border",
            corners_xyz_scene_units=dict(left_top=[2., 0., 2.5], left_bottom=[2., 0., .5],
                                         right_top=[8., 0., 2.5], right_bottom=[8., 0., .5]))])
    camera = np.r_[[5., -10., 1.5], [np.pi / 2, 0., 0.],
                   np.log(600.), np.log(600.), 960., 540., 0., np.zeros(14)]
    return scene, camera


def test_named_corner_keeps_tangential_error_while_ordinary_edge_can_slide():
    scene, camera = scene_and_camera()
    corner = dict(kind="ad_corner", advertisement_id=2, role="left_top")
    edge = dict(kind="ad_edge", advertisement_id=2, role="left_vertical")
    observed = project_central_valid(camera, feature_world(scene, corner))[0] + [0., 30.]
    point_row = build_row(scene, corner, observed, 1., diagnostic_unit_weights=True)
    edge_row = build_row(scene, edge, observed, 1., diagnostic_unit_weights=True)
    residual, jac = evaluate_rows(camera, [point_row, edge_row])
    assert residual[:2] == pytest.approx([0., -30.], abs=1e-10)
    assert residual[2:] == pytest.approx([0., 0.], abs=1e-10)
    assert jac.shape == (4, 15) and np.isfinite(jac).all()
    # A source point past the finite edge end must still have endpoint error.
    beyond = project_central_valid(camera, feature_world(scene, corner))[0] - [0., 20.]
    row = build_row(scene, edge, beyond, 1.)
    assert evaluate_rows(camera, [row], jacobian=False) == pytest.approx([0., 20.], abs=1e-10)
    assert point_row["sigma_semantics"] == "diagnostic_unit_pixel_scale_not_formal_source_weight"


def test_corners_named_edges_and_print_materials_share_current_advertisement_changes():
    scene, _ = scene_and_camera()
    templates = dict(contours={"U": dict(material_uv=[[0., 0.], [.5, .5], [1., 1.]], closed=False),
                              "loop": dict(material_uv=[[.1, .1], [.4, .1], [.4, .3]], closed=True)})
    corner = dict(kind="ad_corner", advertisement_id=2, role="left_top")
    edge = dict(kind="ad_edge", advertisement_id=2, role="left_vertical")
    printed = dict(kind="ad_print", advertisement_id=2, contour_key="U")
    original = [feature_world(scene, target, templates) for target in (corner, edge, printed)]
    # v=0 is top, v=1 bottom, and u follows the same finite perimeter station.
    assert original[2] == pytest.approx(np.array([[2., 0., 2.5], [5., 0., 1.5], [8., 0., .5]]))
    changed = deepcopy(scene)
    for value in changed["advertisements"][0]["corners_xyz_scene_units"].values():
        value[0] += .5
        value[2] += .2
    for target, previous in zip((corner, edge, printed), original):
        current = feature_world(changed, target, templates)
        assert current == pytest.approx(previous + [.5, 0., .2])
    loop = feature_world(scene, dict(printed, contour_key="loop"), templates)
    assert len(loop) == 4 and np.array_equal(loop[0], loop[-1])
    top = feature_world(scene, dict(kind="board_edge", role="top", part="upper_long_side"))
    assert top == pytest.approx(np.array([[0., 0., 3.], [10., 0., 3.]]))


@pytest.mark.parametrize("target", [
    dict(kind="unrecognized"),
    dict(kind="ad_edge", advertisement_id=2, role="whole_ad"),
    dict(kind="ad_edge", advertisement_id=3, role="left_vertical"),
    dict(kind="ad_corner", advertisement_id=2, role="centre"),
    dict(kind="ad_print", advertisement_id=2, contour_key="missing"),
    dict(kind="board_edge", role="top", part="unknown_side"),
    dict(kind="ice_curve", curve_id="unmodeled:segment0"),
])
def test_missing_named_targets_are_refused_without_whole_ad_fallback(target):
    scene, _ = scene_and_camera()
    with pytest.raises(ValueError):
        feature_world(scene, target, dict(contours={}))


def test_target_semantics_and_explicit_scales_cannot_be_silently_changed():
    scene, camera = scene_and_camera()
    target = dict(kind="ad_corner", advertisement_id=2, role="left_top")
    with pytest.raises(ValueError, match="point_correspondence"):
        build_row(scene, target, [[780., 480.]], 1., point_correspondence=False)
    with pytest.raises(ValueError, match="sigma=1"):
        build_row(scene, target, [[780., 480.]], 2., diagnostic_unit_weights=True)
    with pytest.raises(ValueError, match="explicit point_correspondence"):
        evaluate_rows(camera, [dict(target=target, uv=[[780., 480.]], sigma=1.)])
