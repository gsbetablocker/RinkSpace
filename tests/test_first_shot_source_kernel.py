"""Batched raw target algebra agrees with the existing finite source objective."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from calibration.solver_runtime import install
install()  # Install the public runtime before any direct-script compatibility import.

from calibration.program_tracking import first_shot_source_kernel as kernel
from calibration.program_tracking import framewise_model as model
from calibration.program_tracking import official_static_surface_dataset as source
from calibration.program_tracking.complete_official_fit_dataset import (
    camera_fit_state, camera_from_fit_state,
)


ROOT = Path(__file__).resolve().parents[1]
ACCEPTED = ROOT / "events/ankara_2026/programmes/okada_mei_sp/source_datasets/frame_by_frame_review_20261013/first_shot_remaining_50fps/repair_20261014"
MODEL = ROOT / "events/ankara_2026/models/framewise_8a28_first_shot_20261014/repair_20261014/model.json"
SEEDS = Path("D:/CodexData/RinkSpace_Data/events/ankara_2026/programmes/okada_mei_sp/results/frame_by_frame_review_20261013/first_shot_remaining_50fps/camera_seeds.npz")


def synthetic_case():
    camera = np.zeros(25)
    camera[6:8] = np.log([900., 870.])
    camera[8:11] = [960., 540., 1.2]
    camera[11:15] = [.002, -.0001, .0002, -.0003]
    paths = {
        "line": np.array([[-1., -.2, 6.], [0., .25, 6.], [1., .4, 6.]]),
        ("curve", "round"): np.column_stack((np.cos(np.linspace(.1, 2.7, 17)),
                                               np.sin(np.linspace(.1, 2.7, 17)),
                                               np.full(17, 7.))),
        "corner": np.array([[.5, -.7, 5.]]),
    }
    project = kernel._interfaces()[0]
    line = project(camera, paths["line"])[0]
    round_path = project(camera, paths[("curve", "round")])[0]
    corner = project(camera, paths["corner"])[0]
    rows = [dict(_target_key="line", world=paths["line"],
                 uv=np.array([line[0] + [-30., -5.], .4*line[0] + .6*line[1] + [1., -4.]]),
                 sigma=np.array([.4, 7.]), point_correspondence=False),
            dict(_target_key=("curve", "round"), world=paths[("curve", "round")],
                 uv=.3*round_path[[2, 6, 12]] + .7*round_path[[3, 7, 13]] + [2., 3.],
                 sigma=np.array([3., .2, 4.]), point_correspondence=False),
            dict(_target_key="line", world=paths["line"],
                 uv=np.array([line[-1] + [35., 6.], .3*line[1] + .7*line[2] + [-1., 5.]]),
                 sigma=np.array([5., .8]), point_correspondence=False),
            dict(_target_key="corner", world=paths["corner"], uv=corner + [4., -3.],
                 sigma=np.array([2.4]), point_correspondence=True)]
    derivatives = {}
    for key, path in paths.items():
        j = np.zeros((len(path), 3, 4))
        j[:, :, :3] = np.eye(3)[None]
        j[:, 0, 3] = np.linspace(-.3, .4, len(path))
        j[:, 1, 3] = np.linspace(.5, -.2, len(path))
        j[:, 2, 3] = .1
        derivatives[key] = j
    return camera, rows, paths, derivatives


def continuous_line_case(*, endpoint=False):
    camera = np.zeros(25)
    camera[3:6] = [.07, -.11, .03]
    camera[6:8] = np.log([900., 870.])
    camera[8:11] = [960., 540., 1.2]
    camera[11:15] = [.035, -.003, .001, -.0007]
    origin, vector = np.array([-2., -.4, 7.]), np.array([4., 1.1, .5])
    t = np.linspace(0., 1., 129)
    world = origin + t[:, None]*vector
    station = np.array([1.4]) if endpoint else np.array([.17, .43, .82])
    observed = kernel._interfaces()[0](camera, origin + station[:, None]*vector)[0]
    observed = observed + np.array([2.1, -3.2])
    row = dict(_target_key="physical_straight", uv=observed, point_correspondence=False,
        _continuous_world_line=True, _line_end_extent_geometry_columns=[0])
    g = np.zeros((len(world), 3, 3))
    g[:, :, 0] = t[:, None]*vector
    g[:, 1, 1] = 1.
    g[:, 2, 2] = t
    return camera, row, world, g


def test_continuous_brown_line_extent_is_exactly_unobserved_on_interior():
    camera, row, world, g = continuous_line_case()
    r, j, geo = kernel.evaluate_projected_targets(camera, [row], {row['_target_key']:world}, {row['_target_key']:g})
    np.testing.assert_array_equal(geo[:, 0], 0.)
    extended = world[0] + 1.8*(world-world[0])
    raw = kernel.evaluate_projected_targets(camera, [row], {row['_target_key']:extended}, jacobian=False)[0]
    np.testing.assert_allclose(raw, r, rtol=0., atol=2e-11)
    value_only = kernel.evaluate_projected_targets(camera, [row], {row['_target_key']:world}, jacobian=False)[0]
    np.testing.assert_array_equal(value_only, r)
    assert np.linalg.norm(geo[:, 1:]) > 1.


def test_continuous_brown_line_retains_real_endpoint_extent_information():
    camera, row, world, g = continuous_line_case(endpoint=True)
    r, j, geo = kernel.evaluate_projected_targets(camera, [row], {row['_target_key']:world}, {row['_target_key']:g})
    assert np.linalg.norm(geo[:, 0]) > 100.
    h=1e-6
    plus = world + h*g[:, :, 0]; minus = world - h*g[:, :, 0]
    rp = kernel.evaluate_projected_targets(camera,[row],{row['_target_key']:plus},jacobian=False)[0]
    rm = kernel.evaluate_projected_targets(camera,[row],{row['_target_key']:minus},jacobian=False)[0]
    np.testing.assert_allclose((rp-rm)/(2*h),geo[:,0],rtol=2e-8,atol=2e-6)


def test_continuous_brown_line_start_station_is_null_inside_and_visible_at_endpoint():
    camera,row,world,g=continuous_line_case()
    row.pop('_line_end_extent_geometry_columns')
    row['_line_start_extent_geometry_columns']=[0]
    vector=world[-1]-world[0];t=np.linspace(0.,1.,len(world))
    g[:,:,0]=(1.-t[:,None])*vector
    key=row['_target_key']
    r,j,geo=kernel.evaluate_projected_targets(camera,[row],{key:world},{key:g})
    np.testing.assert_array_equal(geo[:,0],0.)
    shifted=world+.1*g[:,:,0]
    raw=kernel.evaluate_projected_targets(camera,[row],{key:shifted},jacobian=False)[0]
    np.testing.assert_allclose(raw,r,rtol=0.,atol=2e-11)
    row['uv']=kernel._interfaces()[0](camera,(world[0]-.4*vector)[None])[0]+[2.1,-3.2]
    r,j,geo=kernel.evaluate_projected_targets(camera,[row],{key:world},{key:g})
    assert np.linalg.norm(geo[:,0])>100.
    h=1e-6
    rp=kernel.evaluate_projected_targets(camera,[row],{key:world+h*g[:,:,0]},jacobian=False)[0]
    rm=kernel.evaluate_projected_targets(camera,[row],{key:world-h*g[:,:,0]},jacobian=False)[0]
    np.testing.assert_allclose((rp-rm)/(2*h),geo[:,0],rtol=2e-8,atol=2e-6)


def test_continuous_line_camera15_and_geometry_implicit_derivatives():
    camera, row, world, g = continuous_line_case()
    key=row['_target_key']
    r,j,geo=kernel.evaluate_projected_targets(camera,[row],{key:world},{key:g})
    scale=np.r_[np.full(3,100.),np.ones(5),np.full(3,1000.),np.ones(4)]
    h=1e-6
    for col in range(15):
        plus=camera.copy();minus=camera.copy();plus[col]+=h*scale[col];minus[col]-=h*scale[col]
        rp=kernel.evaluate_projected_targets(plus,[row],{key:world},jacobian=False)[0]
        rm=kernel.evaluate_projected_targets(minus,[row],{key:world},jacobian=False)[0]
        np.testing.assert_allclose((rp-rm)/(2*h),j[:,col],rtol=2e-7,atol=2e-5)
    for col in range(3):
        rp=kernel.evaluate_projected_targets(camera,[row],{key:world+h*g[:,:,col]},jacobian=False)[0]
        rm=kernel.evaluate_projected_targets(camera,[row],{key:world-h*g[:,:,col]},jacobian=False)[0]
        np.testing.assert_allclose((rp-rm)/(2*h),geo[:,col],rtol=2e-7,atol=2e-5)


def test_continuous_line_preserves_original_optical_guard_and_rows(monkeypatch):
    camera,row,world,g=continuous_line_case()
    actual=kernel._interfaces();calls=[]
    def counted(camera,world,**kwargs):
        calls.append((len(world),kwargs))
        return actual[0](camera,world,**kwargs)
    monkeypatch.setattr(kernel,'_interfaces',lambda:(counted,)+actual[1:])
    rows=[dict(row,uv=row['uv'][:1]),dict(row,uv=row['uv'][1:])]
    r,j,geo=kernel.evaluate_projected_targets(camera,rows,{row['_target_key']:world},{row['_target_key']:g})
    assert calls[0]==(257,{'with_jacobian':True})
    assert len(calls)==3  # one original guard, one candidate batch, one foot batch per target
    reference=kernel.evaluate_projected_targets(camera,[row],{row['_target_key']:world},{row['_target_key']:g})
    for value,expected in zip((r,j,geo),reference,strict=True):np.testing.assert_array_equal(value,expected)
    wrong=world.copy();wrong[4,1]+=.01
    with pytest.raises(ValueError,match='collinear'):
        kernel.evaluate_projected_targets(camera,[row],{row['_target_key']:wrong})
    wrongg=g.copy();wrongg[-1,1,0]+=.01
    with pytest.raises(ValueError,match='exact straight-line station'):
        kernel.evaluate_projected_targets(camera,[row],{row['_target_key']:world},{row['_target_key']:wrongg})


def raw_reference(camera, rows, paths):
    original = []
    for row in rows:
        original.append(dict(row, world=paths[row["_target_key"]], sigma=np.ones(len(row["uv"]))))
    return model._absolute_interface()(camera, original, True)


def test_repeated_targets_agree_with_original_raw_r_and_camera_j():
    camera, rows, paths, derivatives = synthetic_case()
    before = deepcopy((rows, paths, derivatives))
    r, j, g = kernel.evaluate_projected_targets(camera, rows, paths, derivatives)
    expected_r, expected_j = raw_reference(camera, rows, paths)
    np.testing.assert_allclose(r, expected_r, rtol=0., atol=1e-11)
    np.testing.assert_allclose(j, expected_j, rtol=1e-13, atol=1e-10)
    assert g.shape == (len(r), 4)
    for row, old in zip(rows, before[0], strict=True):
        np.testing.assert_array_equal(row["uv"], old["uv"])
        np.testing.assert_array_equal(row["sigma"], old["sigma"])
    for key in paths:
        np.testing.assert_array_equal(paths[key], before[1][key])
        np.testing.assert_array_equal(derivatives[key], before[2][key])


def test_project_once_per_target_even_with_multiple_source_segments(monkeypatch):
    camera, rows, paths, derivatives = synthetic_case()
    actual = kernel._interfaces()
    calls = []
    def counted(camera25, world, **kwargs):
        calls.append((len(world), kwargs))
        return actual[0](camera25, world, **kwargs)
    monkeypatch.setattr(kernel, "_interfaces", lambda: (counted,) + actual[1:])
    kernel.evaluate_projected_targets(camera, rows, paths, derivatives)
    assert calls == [(5, {"with_jacobian": True}), (33, {"with_jacobian": True}),
                     (1, {"with_jacobian": True})]


def test_value_only_projects_once_without_brown_or_derivative_work(monkeypatch):
    camera, rows, paths, derivatives = synthetic_case()
    expected = raw_reference(camera, rows, paths)[0]
    actual = kernel._interfaces()
    calls = []
    def counted(camera25, world, **kwargs):
        calls.append((len(world), kwargs))
        return actual[0](camera25, world, **kwargs)
    def forbidden(*args):
        raise AssertionError("Value-only evaluation computed derivatives")
    monkeypatch.setattr(kernel, "_interfaces", lambda: (counted, forbidden) + actual[2:])
    monkeypatch.setattr(kernel, "_nearest_jacobian", forbidden)
    r, j, g = kernel.evaluate_projected_targets(camera, rows, paths, jacobian=False)
    np.testing.assert_allclose(r, expected, rtol=0., atol=1e-11)
    assert j.shape == g.shape == (len(r), 0)
    assert calls == [(5, {"with_jacobian": False}), (33, {"with_jacobian": False}),
                     (1, {"with_jacobian": False})]
    with pytest.raises(ValueError, match="cannot consume geometry"):
        kernel.evaluate_projected_targets(camera, rows, paths, derivatives, jacobian=False)


def test_camera_and_geometry_derivatives_include_sliding_and_clamped_endpoints():
    camera, rows, paths, derivatives = synthetic_case()
    _, camera_jac, geometry_jac = kernel.evaluate_projected_targets(camera, rows, paths, derivatives)
    state = camera_fit_state(camera)
    step = 1e-6
    for column in (0, 3, 6, 8, 10, 11, 12, 13, 14):
        plus, minus = state.copy(), state.copy()
        plus[column] += step
        minus[column] -= step
        rp = kernel.evaluate_projected_targets(camera_from_fit_state(plus, camera), rows, paths)[0]
        rm = kernel.evaluate_projected_targets(camera_from_fit_state(minus, camera), rows, paths)[0]
        np.testing.assert_allclose((rp - rm) / (2 * step), camera_jac[:, column],
                                   rtol=1e-6, atol=2e-6)
    for column in range(4):
        plus = {key: world + step*derivatives[key][:, :, column] for key, world in paths.items()}
        minus = {key: world - step*derivatives[key][:, :, column] for key, world in paths.items()}
        rp = kernel.evaluate_projected_targets(camera, rows, plus)[0]
        rm = kernel.evaluate_projected_targets(camera, rows, minus)[0]
        np.testing.assert_allclose((rp - rm) / (2 * step), geometry_jac[:, column],
                                   rtol=2e-6, atol=2e-6)


def test_point_support_interval_preserves_original_value_and_active_derivatives():
    camera, rows, paths, derivatives = synthetic_case()
    row = rows[-1]
    point = kernel._interfaces()[0](camera, paths["corner"])[0]
    row["point_support_bounds_xy_px"] = np.stack((point - [2., 8.], point + [2., 8.]), axis=2)
    r, j, g = kernel.evaluate_projected_targets(camera, [row], paths, derivatives)
    np.testing.assert_array_equal(r, [0., 0.])
    np.testing.assert_array_equal(j, np.zeros_like(j))
    np.testing.assert_array_equal(g, np.zeros_like(g))
    row["point_support_bounds_xy_px"][0, 0] = point[0, 0] + [3., 5.]
    r, j, _ = kernel.evaluate_projected_targets(camera, [row], paths, derivatives)
    expected = raw_reference(camera, [row], paths)
    np.testing.assert_allclose(r, expected[0], atol=1e-12)
    np.testing.assert_allclose(j, expected[1], atol=1e-11)


@pytest.mark.parametrize("field,value", [("normal_interval", [[-1., 1.]]),
    ("curve_observation_kind", "axis_scanline_branch"), ("must_preserve_upper_branch", True)])
def test_unsupported_observation_semantics_are_rejected(field, value):
    camera, rows, paths, _ = synthetic_case()
    rows[0][field] = value
    with pytest.raises(ValueError):
        kernel.evaluate_projected_targets(camera, rows, paths)


def test_missing_or_inconsistent_interfaces_are_rejected():
    camera, rows, paths, derivatives = synthetic_case()
    with pytest.raises(ValueError, match="no supplied physical"):
        kernel.evaluate_projected_targets(camera, rows, {})
    with pytest.raises(ValueError, match="no supplied geometry"):
        kernel.evaluate_projected_targets(camera, rows, paths, {})
    derivatives["line"] = derivatives["line"][:, :, :2]
    with pytest.raises(ValueError, match="same geometry column"):
        kernel.evaluate_projected_targets(camera, rows, paths, derivatives)
    rows[2]["point_correspondence"] = True
    with pytest.raises(ValueError, match="cannot mix"):
        kernel.evaluate_projected_targets(camera, rows, paths)


def test_all_invalid_curve_is_a_domain_error_and_no_geometry_has_zero_columns():
    camera, rows, paths, _ = synthetic_case()
    r, j, g = kernel.evaluate_projected_targets(camera, rows, paths)
    assert j.shape == (len(r), 15) and g.shape == (len(r), 0)
    behind = {key: world * [1., 1., -1.] for key, world in paths.items()}
    with pytest.raises(ValueError, match="ice_curve_no_central_segment"):
        kernel.evaluate_projected_targets(camera, rows[:1], behind)
    empty = kernel.evaluate_projected_targets(camera, [], {})
    assert [value.shape for value in empty] == [(0,), (0, 15), (0, 0)]


@pytest.mark.parametrize("index", [201, 1241, 1248])
def test_actual_accepted_target_kinds_match_original_and_world_fd(index):
    packet_path = ACCEPTED / f"frames/{index:06d}.json"
    binding_path = ACCEPTED / f"bindings/{index:06d}.json"
    if not all(path.is_file() for path in (packet_path, binding_path, MODEL, SEEDS)):
        pytest.skip("Accepted first-shot fixtures are unavailable")
    manifest = source.read_reference(source.reference(MODEL))
    scene = source.read_reference(manifest["scene"])
    templates = source.read_reference(manifest["material_templates"])
    registry = source.read_reference(manifest["feature_registry"])["targets"]
    packet = source.read_reference(source.reference(packet_path))
    binding = source.read_reference(source.reference(binding_path))
    features = {row["id"]: row for row in packet["features"]}
    rows, paths, derivatives = [], {}, {}
    for observation in binding["observations"]:
        key = observation["target_id"]
        segment = features[observation["source_feature_id"]]["segments"][observation["source_segment_index"]]
        uv = np.asarray(segment["uv"], float)[observation["source_vertex_indices"]]
        row = model.build_row(scene, registry[key], uv, 1., templates, row_id=observation["id"])
        row["_target_key"] = key
        rows.append(row)
        paths[key] = row["world"]
        derivatives[key] = np.broadcast_to(np.eye(3), (len(row["world"]), 3, 3)).copy()
    with np.load(SEEDS, allow_pickle=False) as seeds:
        matching = np.flatnonzero(seeds["programme_indices"] == index)
        assert len(matching) == 1
        camera = seeds["camera25"][matching[0]]
    r, j, g = kernel.evaluate_projected_targets(camera, rows, paths, derivatives)
    expected = raw_reference(camera, rows, paths)
    np.testing.assert_allclose(r, expected[0], rtol=0., atol=1e-9)
    np.testing.assert_allclose(j, expected[1], rtol=2e-12, atol=2e-8)
    r_only, j_empty, g_empty = kernel.evaluate_projected_targets(camera, rows, paths, jacobian=False)
    np.testing.assert_array_equal(r_only, r)
    assert j_empty.shape == g_empty.shape == (len(r), 0)
    if index == 201:
        assert {row["target"]["kind"] for row in rows} == {
            "board_edge", "ad_edge", "ad_corner", "ad_print", "ice_curve"}
    step = 1e-6
    for column in range(3):
        plus = {key: world + step*derivatives[key][:, :, column] for key, world in paths.items()}
        minus = {key: world - step*derivatives[key][:, :, column] for key, world in paths.items()}
        rp = kernel.evaluate_projected_targets(camera, rows, plus)[0]
        rm = kernel.evaluate_projected_targets(camera, rows, minus)[0]
        np.testing.assert_allclose((rp - rm)/(2*step), g[:, column], rtol=2e-5, atol=2e-5)
