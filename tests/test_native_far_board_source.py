import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def owned(monkeypatch):
    monkeypatch.setenv('RINKSPACE_PROJECTION_BACKEND', 'numpy')
    from calibration.program_tracking.runtime.bootstrap import add_geometry_dependencies
    add_geometry_dependencies()
    from calibration.solver_runtime import install
    install()
    from structured_programme_source_candidate import native_board_edge_source as module
    return module


class Geometry:
    _materials = {(7, 'left_vertical_xyz'): (np.zeros(2), np.array([0., 1.]))}

    def __init__(self, world=None):
        self.world = np.asarray(world if world is not None else
            [[-2., -1., 10.], [2., -1., 10.], [2., 1., 10.], [-2., 1., 10.], [-2., -1., 10.]])

    def source_scene(self, p):
        world = self.world.copy()
        world[:, 1] += p[0]
        return dict(board_surface=dict(top_edge_xyz=world))

    def validate_parameters(self, p):
        if not np.isfinite(p).all():
            raise ValueError('Synthetic geometry requires finite current parameters')
        return np.asarray(p).copy()

    def print_course_binding(self, p):
        def sample_perimeter(q, *, with_jacobian=False):
            top = self.source_scene(q)['board_surface']['top_edge_xyz']
            return top - [0., 0., 1.], top, []
        return SimpleNamespace(sample_perimeter=sample_perimeter), p, None

    def material_parameter_indices(self, number):
        return np.array([1])

    def material(self, number, u, v, p):
        return (np.array([[0., -1., 10.], [0., 1., 10.]])+[p[1], 0., 0.], None, None)


def fixture_factor(owned, tmp_path, *, board=True, world=None):
    raw = tmp_path/'original_native_fixture.bin'
    raw.write_bytes(b'source frame fixture; never a production annotation')
    curve = (dict(kind='board_top', source_structure='far_visible_board') if board else
             dict(kind='advertisement_boundary', advertisement_number=7,
                  edge='left_vertical', extent_scope='external_border'))
    row = dict(id='unit_source_far_board' if board else 'unit_source_ad_edge', pts=256,
        actual_pts=256, curve=curve, uv=[[900., 630.], [1000., 630.]] if board else [[965., 530.], [965., 550.]],
        sigma_px=[4., 4.], normal=[[0., 1.], [0., 1.]] if board else [[1., 0.], [1., 0.]],
        normal_interval_px=[[-2., 2.], [-2., 2.]], split='fit', point_correspondence=False,
        curve_observation_kind='normal_interval', source_raw_frame=str(raw),
        source_file_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
    packet = dict(schema='rinkspace.additional-native-board-boundary-source.v1',
        source_video_sha256=owned.SOURCE_SHA, time_base='1/15360',
        status='explicit_new_fit_target_authorized', rows=[row])
    path = tmp_path/'source.json'
    path.write_text(json.dumps(packet), encoding='utf8')
    geometry = Geometry(world)

    def camera(q, pts):
        value = np.zeros(25)
        value[2] = q[112]-1.
        value[6:8] = np.log(1000.)+q[103]
        value[8:10] = [960.+q[101], 540.+q[100]]
        return value

    from z190_fixed_lever_xyz_source_bundle import FixedLeverXYZSourceBundle

    class ControlledSourceBundle(FixedLeverXYZSourceBundle):
        """Synthetic fixed-XYZ projection with analytically known derivatives."""
        def __init__(self):
            self.index = {256: 0}

        def camera(self, q, pts):
            return camera(q, pts)

    block = SimpleNamespace(start=100, xyz_slots=np.array([110, 111, 112]),
        optics_chart=SimpleNamespace(indices=np.array([104, 105]), mount_indices=np.array([106, 107])),
        bundle=ControlledSourceBundle(), local=lambda q: q)
    state = SimpleNamespace(geometry=SimpleNamespace(current=geometry),
        original=SimpleNamespace(aliases={256: 256}), bypts={256: block})
    factor = owned.NativeBoardEdgeFactor(state, path, owned.CORE_COUNT)
    q = np.zeros(owned.CORE_COUNT)
    q[112] = 1.
    return factor, q, packet, path


def test_far_branch_is_source_x_minimum_y_even_when_near_board_is_closer(owned, tmp_path):
    factor, q, original, path = fixture_factor(owned, tmp_path)
    value = factor.block(q, False).r.reshape(-1, 2)
    # Far top is y440, near top y640. Source y630 is deliberately closer to
    # near top. The explicit physical far branch must retain 188px mismatch.
    np.testing.assert_allclose(value, [[0., -47.], [0., -47.]], atol=1e-12)
    assert json.loads(path.read_text()) == original
    assert factor.contract['far_board_nearest_pixel_or_source_y_branch_selection'] is False


def test_existing_advertisement_values_are_unchanged(owned, tmp_path):
    from z190_source_camera_diagnostic import absolute_value_jac15
    factor, q, _, _ = fixture_factor(owned, tmp_path, board=False)
    row = factor.rows[0]
    camera = factor._cameras(q)[256]
    old = dict(row, world=factor._world(q, row['curve']), sigma=row['sigma'])
    np.testing.assert_array_equal(factor._residual(q, factor.rows), absolute_value_jac15(camera, [old], False))
    assert not factor.contract['far_visible_board_top_ids']


def test_current_world_and_camera_derivatives_match_actual_far_source_branch(owned, tmp_path):
    factor, q, _, _ = fixture_factor(owned, tmp_path)
    block = factor.block(q, True)
    # A one-pixel current camera PP-y move contributes 1/sigma; a unit
    # current world-y move contributes focal/depth/sigma, not a frozen WORLD.
    np.testing.assert_allclose(block.J[:, 100].toarray().ravel(), [0., .25, 0., .25], atol=3e-7)
    world_column = int(owned.WORLD_INDICES[0])
    np.testing.assert_allclose(block.J[:, world_column].toarray().ravel(), [0., 25., 0., 25.], atol=3e-7)
    assert set(owned.WORLD_INDICES).issubset(set(factor.active_world))
    for column in [100, 101, 103, 112, world_column]:
        h = 1e-5
        plus, minus = q.copy(), q.copy()
        plus[column] += h
        minus[column] -= h
        actual = (factor._residual(plus, factor.rows)-factor._residual(minus, factor.rows))/(2*h)
        np.testing.assert_allclose(block.J[:, column].toarray().ravel(), actual, rtol=1e-6, atol=1e-5)


def test_no_forward_board_is_rejected_and_finite_endpoint_keeps_horizontal_error(owned, tmp_path):
    factor, q, _, _ = fixture_factor(owned, tmp_path)
    factor.rows[0]['uv'][:, 0] = 1500.
    value = factor._residual(q, factor.rows).reshape(-1, 2)
    assert np.all(np.abs(value[:, 0]) > 50.)
    factor.geometry.world[:, 2] = -10.
    with pytest.raises(ValueError, match='central-forward far-board'):
        factor._residual(q, factor.rows)


def test_disconnected_optical_components_are_not_joined_or_selected_by_source_y(owned, tmp_path):
    world = [[-2., -1., 10.], [2., -1., 10.], [2., 0., -10.],
             [-2., 1., 10.], [2., 1., 10.], [-2., 0., -10.]]
    factor, q, _, _ = fixture_factor(owned, tmp_path, world=world)
    rows = factor._source_rows(q, factor.rows[0], factor._cameras(q)[256])
    assert len(rows) == 2
    for row in rows:
        assert row['world'].shape == (2, 3)
        assert (row['world'][:, 1] == -1.).all()
        assert row['disjoint_optical_domain_components_not_bridged']
    np.testing.assert_allclose(factor._residual(q, factor.rows).reshape(-1, 2)[:, 1], -47., atol=1e-12)


def test_driver_export_check_keeps_current_board_geometry_binding(owned, tmp_path):
    factor, q, _, _ = fixture_factor(owned, tmp_path)
    driver = Path(__file__).parents[1]/'calibration/solver_runtime/legacy/run_current_named66_full_programme.py'
    tree = ast.parse(driver.read_text(encoding='utf8'))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('component_state', 'check_current_board_edge_exports')]
    namespace = dict(np=np, WORLD_INDICES=owned.WORLD_INDICES)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(driver), 'exec'), namespace)
    scene = factor.geometry.source_scene(q[owned.WORLD_INDICES])
    result = namespace['check_current_board_edge_exports'](factor, q, scene)
    assert result[0]['source_structure'] == 'far_visible_board'
    assert result[0]['export_world_max_difference'] == 0.
    scene['board_surface']['top_edge_xyz'][0, 1] += .01
    with pytest.raises(ValueError, match='exported world66'):
        namespace['check_current_board_edge_exports'](factor, q, scene)
