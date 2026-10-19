"""Explicit same-edge matching crosses physical seams without changing evidence."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from calibration.program_tracking.continuous_board_association import (
    SCHEMA, with_continuous_board_association,
)
from calibration.program_tracking.framewise_model import build_row, evaluate_rows, feature_world
from calibration.program_tracking.projection import project_central_valid
from calibration.program_tracking import complete_official_fit_dataset as complete
from calibration.program_tracking.source_evidence_whitening import SourceLineEvidence, line_whitening_operator


PARTS = ["upper_right", "upper_long_side", "upper_left", "left_end"]
POLICY = dict(schema=SCHEMA, candidate_parts=PARTS)
ROOT = Path(__file__).resolve().parents[1]


def camera_above(scene):
    world = np.asarray(scene["board_surface"]["top_edge_xyz"], float)
    center = np.mean(world, axis=0)
    span = max(np.ptp(world[:, 0]), np.ptp(world[:, 1]))
    camera = np.zeros(25, float)
    camera[:3] = [center[0], center[1], np.max(world[:, 2]) + 4 * span]
    camera[3] = np.pi
    camera[6:8] = np.log(1400)
    camera[8:10] = [960, 540]
    return camera


def rounded_scene():
    vertices, records = [], []
    definitions = [
        ("upper_right", np.column_stack((3 + np.cos(np.linspace(0, np.pi/2, 9)[:-1]),
                                          2 + np.sin(np.linspace(0, np.pi/2, 9)[:-1])))),
        ("upper_long_side", np.column_stack((np.linspace(3, -3, 5)[:-1], np.full(4, 3)))),
        ("upper_left", np.column_stack((-3 + np.cos(np.linspace(np.pi/2, np.pi, 9)[:-1]),
                                         2 + np.sin(np.linspace(np.pi/2, np.pi, 9)[:-1])))),
        ("left_end", np.column_stack((np.full(4, -4), np.linspace(2, -2, 5)[:-1]))),
        ("lower_left", np.column_stack((-3 + np.cos(np.linspace(np.pi, 3*np.pi/2, 9)[:-1]),
                                         -2 + np.sin(np.linspace(np.pi, 3*np.pi/2, 9)[:-1])))),
        ("lower_long_side", np.column_stack((np.linspace(-3, 3, 5)[:-1], np.full(4, -3)))),
        ("lower_right", np.column_stack((3 + np.cos(np.linspace(3*np.pi/2, 2*np.pi, 9)[:-1]),
                                          -2 + np.sin(np.linspace(3*np.pi/2, 2*np.pi, 9)[:-1])))),
        ("right_end", np.column_stack((np.full(4, 4), np.linspace(-2, 2, 5)[:-1]))),
    ]
    for name, xy in definitions:
        records.append(dict(id=name, first_vertex=len(vertices), vertex_count=len(xy)))
        vertices.extend(np.column_stack((xy, np.zeros(len(xy)))).tolist())
    top = np.asarray(vertices, float) + [0, 0, 1]
    return dict(ice_boundary=dict(points_xyz=vertices, parts_counterclockwise=records),
                board_surface=dict(top_edge_xyz=top.tolist()))


@pytest.fixture(scope="module")
def real_scene():
    model = json.loads((ROOT / "events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014/model.json")
                       .read_text(encoding="utf8"))
    assert model["parent"]["q_sha256"].startswith("8a28afed")
    return json.loads(Path(model["scene"]["path"]).read_text(encoding="utf8"))


@pytest.mark.parametrize("role", ["top", "contact"])
def test_wrong_straight_or_corner_labels_share_one_continuous_residual(role):
    scene = rounded_scene()
    camera = camera_above(scene)
    # One point lies just inside the straight and one just inside its arc.
    path = np.asarray(scene["board_surface"]["top_edge_xyz"] if role == "top"
                      else scene["ice_boundary"]["points_xyz"], float)
    observed = project_central_valid(camera, np.array([.5*(path[8]+path[9]), .5*(path[7]+path[8])]))[0]
    results = []
    for original_part in PARTS:
        target = with_continuous_board_association(dict(kind="board_edge", role=role, part=original_part), POLICY)
        results.append(evaluate_rows(camera, [build_row(scene, target, observed, [.4, .7])]))
    for result in results[1:]:
        np.testing.assert_array_equal(result[0], results[0][0])
        np.testing.assert_array_equal(result[1], results[0][1])
    np.testing.assert_allclose(results[0][0], 0, atol=1e-10)
    # Existing hard labels do produce the spurious finite-endpoint error.
    hard = [build_row(scene, dict(kind="board_edge", role=role, part="upper_right"), observed[:1], .4),
            build_row(scene, dict(kind="board_edge", role=role, part="upper_long_side"), observed[1:], .7)]
    assert np.linalg.norm(evaluate_rows(camera, hard, jacobian=False)) > 1


@pytest.mark.parametrize("role", ["top", "contact"])
def test_actual_8a_geometry_has_one_open_same_role_domain_without_opposite_parts(real_scene, role):
    target = with_continuous_board_association(dict(kind="board_edge", role=role, part="upper_right"), POLICY)
    domain = feature_world(real_scene, target)
    inventory = np.asarray(real_scene["board_surface"]["top_edge_xyz"] if role == "top"
                           else real_scene["ice_boundary"]["points_xyz"], float)
    if np.array_equal(inventory[0], inventory[-1]):
        inventory = inventory[:-1]
    # The existing four parts are consecutive vertices 0..240, with no seam gap.
    np.testing.assert_array_equal(domain, inventory[:241])
    assert not np.array_equal(domain[0], domain[-1])
    camera = camera_above(real_scene)
    opposite = feature_world(real_scene, dict(kind="board_edge", role=role, part="lower_long_side"))
    uv = project_central_valid(camera, opposite[len(opposite)//2:len(opposite)//2+1])[0]
    assert np.linalg.norm(evaluate_rows(camera, [build_row(real_scene, target, uv, .8)], False)) > 10
    # No-policy behavior still returns the original named range, or whole ring.
    named = feature_world(real_scene, dict(kind="board_edge", role=role, part="upper_right"))
    np.testing.assert_array_equal(named, inventory[:97])
    ring = feature_world(real_scene, dict(kind="board_edge", role=role))
    np.testing.assert_array_equal(ring, np.vstack((inventory, inventory[:1])))


def test_original_uv_ids_scales_and_source_whitener_survive_continuous_association():
    scene = rounded_scene()
    camera = camera_above(scene)
    world = np.asarray(scene["board_surface"]["top_edge_xyz"])
    uv = project_central_valid(camera, world[[8, 9, 10]])[0] + [0, 1.3]
    ids = ["opaque approved point a", "opaque approved point b", "opaque approved point c"]
    sigma = np.array([.3, .8, .6])
    target = dict(kind="board_edge", role="top", part="upper_right")
    before = deepcopy(target)
    operator = line_whitening_operator(uv, ids, SourceLineEvidence([0, uv[0, 1]],
        [[1e-5, 0], [0, 1]], 0, "source-only-line", "measured-native-localization-support"))
    saved_operator = operator.to_record()
    row = build_row(scene, with_continuous_board_association(target, POLICY), uv, sigma, row_id="original row")
    row["source_point_ids"] = ids.copy()
    frame = dict(rows=[row], source_point_count=3, scene=scene,
                 whitening_groups=[dict(operator=operator, point_indices=[0, 1, 2])])
    residual, jacobian = complete.evaluate_frame(camera, frame)
    assert target == before and row["target"]["part"] == target["part"]
    np.testing.assert_array_equal(row["uv"], uv)
    np.testing.assert_array_equal(row["sigma"], sigma)
    assert row["source_point_ids"] == ids and row["id"] == "original row"
    assert operator.to_record() == saved_operator
    assert residual.shape == (2,) and jacobian.shape == (2, 15)
    assert np.isfinite(residual).all() and np.isfinite(jacobian).all()


def test_continuous_association_conditioned_camera_jacobian_crosses_the_seam(real_scene):
    camera = camera_above(real_scene)
    inventory = np.asarray(real_scene["board_surface"]["top_edge_xyz"], float)
    uv = project_central_valid(camera, np.array([.4*inventory[96]+.6*inventory[97],
                                                .4*inventory[95]+.6*inventory[96]]))[0] + [.1, .3]
    target = with_continuous_board_association(dict(kind="board_edge", role="top", part="upper_right"), POLICY)
    rows = [build_row(real_scene, target, uv, [.4, .9])]
    _, jacobian = evaluate_rows(camera, rows)
    state = complete.camera_fit_state(camera)
    for column in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14):
        step = 2e-6
        before, after = state.copy(), state.copy()
        before[column] -= step
        after[column] += step
        finite = (evaluate_rows(complete.camera_from_fit_state(after, camera), rows, False)
                  - evaluate_rows(complete.camera_from_fit_state(before, camera), rows, False)) / (2*step)
        np.testing.assert_allclose(jacobian[:, column], finite, rtol=4e-5, atol=4e-5)


@pytest.mark.parametrize("parts", [
    ["upper_right", "upper_left"],
    ["upper_long_side", "upper_right"],
    ["upper_right", "upper_right"],
    ["unknown"],
    ["upper_right", "upper_long_side", "upper_left", "left_end", "lower_left", "lower_long_side",
     "lower_right", "right_end"],
])
def test_disconnected_unknown_duplicate_or_whole_ring_domains_are_rejected(parts):
    scene = rounded_scene()
    policy = dict(schema=SCHEMA, candidate_parts=parts)
    with pytest.raises(ValueError):
        target = with_continuous_board_association(dict(kind="board_edge", role="top", part=parts[0]), policy)
        feature_world(scene, target)


@pytest.mark.parametrize("target", [
    dict(kind="board_edge", role="top", part="lower_long_side"),
    dict(kind="board_edge", role="top"),
    dict(kind="board_edge", role="unknown", part="upper_right"),
    dict(kind="ice_curve", curve_id="approved circle"),
    dict(kind="ad_edge", advertisement_id=2, role="top_edge"),
])
def test_continuous_policy_cannot_change_roles_or_import_an_unconfirmed_domain(target):
    with pytest.raises(ValueError):
        with_continuous_board_association(target, POLICY)
