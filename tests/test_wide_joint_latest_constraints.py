"""The wide-fit entry point must carry the user's latest hard geometry rules."""

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from fit_wide_joint import prepare  # noqa: E402
from rinkspace.calibration_companion.joint_scene import JointSceneProblem  # noqa: E402


EXTRA = ROOT / "outputs/calibration/ankara_2026_wide_joint_20260923/complete_observations_v3.json"


def test_latest_wide_rules_are_exact_in_the_actual_problem():
    scene, data = prepare(EXTRA)
    hard = data["hard_geometry_constraints"]
    assert all(hard[name] for name in (
        "lane_centerline_through_FR_rows", "lane_left_open",
        "equal_corner_radii", "lane_equal_width", "lane_right_collinear",
        "A1_A2_mirror_equal_radius"))
    kinds = [row["kind"] for row in data["scene_parameters"]]
    assert kinds.count("shared_corner_radius") == 1
    assert kinds.count("shared_lane_half_width") == 1
    assert kinds.count("shared_lane_right_x") == 1
    assert kinds.count("A_pair_shared_center_x") == 1
    assert kinds.count("A_pair_mirror_y_distance") == 1
    assert kinds.count("A_pair_shared_radius") == 1
    assert not set(kinds) & {"boundary_corner_radius", "lane_half_width", "lane_bound_x_max"}
    assert not any(s.get("feature_id") in ("A1", "A2") and s["kind"] in
                   ("circle_center_x", "circle_center_y", "circle_radius")
                   for s in data["scene_parameters"])
    problem = JointSceneProblem(scene, data)
    geometry = problem.geometry(problem.x0)
    assert geometry.hard_geometry_checks()["all_constraints_satisfied"]
    radii = list(geometry.corner_radii.values())
    assert max(radii) - min(radii) < 1e-12
    a1, a2 = (geometry.features[fid] for fid in ("A1", "A2"))
    np.testing.assert_allclose(a1["center_xyz"][0], a2["center_xyz"][0], atol=1e-12)
    np.testing.assert_allclose(a1["center_xyz"][1], -a2["center_xyz"][1], atol=1e-12)
    np.testing.assert_allclose(a1["radius"], a2["radius"], atol=1e-12)
    lanes = [np.asarray(geometry.features[f"lane{i}"]["polygon_xyz"]) for i in range(1, 5)]
    np.testing.assert_allclose([lane[1, 0] for lane in lanes], lanes[0][1, 0], atol=1e-12)
    np.testing.assert_allclose([lane[0, 1] - lane[2, 1] for lane in lanes],
                               lanes[0][0, 1] - lanes[0][2, 1], atol=1e-12)
    assert all(geometry.features[f"lane{i}"]["open_edges"] == [3] for i in range(1, 5))


def test_right_lower_pixels_are_preserved_while_tangent_point_is_free():
    _, data = prepare(EXTRA)
    observation = next(row for row in data["observations"] if row["id"] ==
                       "women_sp_high_56218880_near_lower_right_rail_ice_facing_top")
    assert observation["target"] == {
        "kind": "board_side", "side": "right_end", "edge": "top",
        "continue_corner": "lower_right"}
    assert observation["target_revision"]["pixels_unchanged"]
    assert len(observation["pixels_uv"]) == 29


def test_ice_facing_board_is_vertical_at_the_near_right_edge():
    scene, data = prepare(EXTRA)
    problem = JointSceneProblem(scene, data)
    geometry = problem.geometry(problem.x0)
    for side in ("right_end", "lower_right"):
        bottom = geometry.board_side({"side": side, "edge": "contact"}, 121)
        top = geometry.board_side({"side": side, "edge": "top"}, 121)
        np.testing.assert_allclose(top[:, :2], bottom[:, :2], atol=1e-12)
        np.testing.assert_allclose(bottom[:, 2], 0, atol=1e-12)
        np.testing.assert_allclose(top[:, 2], geometry.parameters["board_height"], atol=1e-12)


def test_optional_same_event_contact_support_uses_a_separate_camera():
    _, data = prepare(EXTRA, support_main_right=True)
    assert {f["id"] for f in data["frames"]} == {
        "women_sp_high_56218880", "main279_80302080", "main279_86138880"}
    main = next(row for row in data["installations"] if row["id"] == "women_sp_main")
    assert main["wide_angle_camera"]["enabled"] is False
    assert {row["id"] for row in data["observations"] if row["target"].get("edge") == "contact"
            and row["frame_id"].startswith("main279")} == {
                "main279_80302080_direct_native_01", "main279_86138880_direct_native_01"}
