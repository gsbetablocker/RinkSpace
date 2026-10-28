"""Shared goal-end constraints survive native CPU/GPU parameter changes."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from rinkspace.core.native import NativeCore, NativeError
from rinkspace.core.gpu import ResidentResidualProblem
from rinkspace.domain.field_model import constrain_symmetric_goal_ends, symmetric_goal_end_initial_values
from rinkspace.domain.parameters import SceneParameterModel
from rinkspace.session.gpu_geometry import scene_coordinate_charts
from rinkspace.compat.d78_entities import sample_entities
import test_g2_gpu as fixture


@pytest.fixture
def core():
    path = os.environ.get('RINKSPACE_CORE_LIBRARY')
    if not path:
        pytest.skip('explicit native core library required')
    return NativeCore(path)


def inputs():
    board = [.5, .25, .1, .12, 1.85, .02]
    q = [-.3, .2, -.18, .1, .03, .1, .35, .65, .31, .07,
         -.42, -.43, -.41, -.44, .28, .28, .28, .28,
         .19, .12, .06, -.24, .24, -.43, .45, .3, 0., .11, 0.,
         .035, .025, .045, .03, .05, .07]
    return board, np.array(symmetric_goal_end_initial_values(q))


@pytest.mark.parametrize('distance,ax,ay,goal,board_half', [
    (.45, .035, .025, .05, .07), (.47, .022, .04, .045, .09),
    (.49, .03, .02, .03, .06)])
def test_end_geometry_is_constructed_as_exact_mirrors(core, distance, ax, ay, goal, board_half):
    board, q = inputs()
    q[[24, 29, 30, 33, 34]] = [distance, ax, ay, goal, board_half]
    definitions = {code: core.shared_geometry(board, q, code, options=31)
                   for code in (403, 404, 410, 411, 420, 421, 422, 423)}
    def line(code):
        kind, p = definitions[code]
        assert kind == 0
        return np.array(p).reshape(2, 3)
    mx = [-1., 1., 1.]
    my = [1., -1., 1.]
    np.testing.assert_allclose(line(403) * mx, line(404), atol=1e-15)
    samples = []
    for code, parent, inward in ((410, 403, 1), (411, 404, -1)):
        kind, p = definitions[code]
        assert kind == 6
        np.testing.assert_allclose(p[:3], line(parent).mean(0), atol=1e-15)
        np.testing.assert_array_equal(p[3:5], [ax, ay])
        assert p[7] - p[6] == pytest.approx(np.pi)
        xyz = np.array(core.sample_path(board, kind, p, [], np.linspace(0, 1, 101)))
        assert np.all(inward * (xyz[1:-1, 0] - p[0]) > 0)
        samples.append(xyz)
    np.testing.assert_allclose(samples[0] * mx, samples[1][::-1], atol=1e-15)
    for a, b in ((420, 422), (421, 423)):
        np.testing.assert_allclose(line(a) * mx, line(b), atol=1e-15)
    for a, b in ((420, 421), (422, 423)):
        np.testing.assert_allclose(line(a) * my, line(b), atol=1e-15)
    # Old duplicate coordinates cannot compensate for the shared dimensions.
    q[[23, 31, 32]] = [.97, .81, .63]
    for code, expected in definitions.items():
        assert core.shared_geometry(board, q, code, options=31) == expected


def test_current_layout_has_only_five_goal_end_dimensions_and_open_arcs(core):
    path = Path(__file__).resolve().parents[1] / 'events/ankara_2026/scene/field/model.json'
    raw = json.loads(path.read_text(encoding='utf-8-sig'))
    scene = constrain_symmetric_goal_ends(raw)
    state = SceneParameterModel(scene, path, optimize_scene=True).initial_state
    face = state.blocks['scene/ankara_2026.field/hockey_faceoff_shared']
    ends = state.blocks['scene/ankara_2026.field/hockey_ends_ellipse']
    assert list(face['independent_local_indices']) == [1, 2, 4]
    assert list(ends['independent_local_indices']) == [0, 1, 4, 5]
    charts = scene_coordinate_charts(state)
    assert charts[face['offset'] + 1] == 1
    assert not np.any(charts[ends['offset'] + np.array([2, 3])])
    cuts = sample_entities(scene, core)['cuts']
    assert all(not c['closed'] for c in cuts if c['kind'] == 'ice_ellipse')


def test_new_option_rejects_legacy_layout_and_missing_rink_frame(core):
    board, q = inputs()
    with pytest.raises(NativeError):
        core.shared_geometry(board, q, 410, options=16)
    with pytest.raises(NativeError):
        core.shared_geometry(board, q[:33], 410, options=18)


def test_cpu_cuda_shared_goal_derivatives_match_finite_differences(core):
    if not os.environ.get('RINKSPACE_G2_LIBRARY'):
        pytest.skip('explicit CUDA library required')
    f = fixture.G2GPU(); f.setUp()
    board, q = inputs()
    codes = [403, 404, 410, 411, 420, 421, 422, 423]
    state = np.r_[board, q]
    columns = 6 + np.array([24, 29, 30, 33, 34])
    directions = np.eye(41)[:, columns] * .001
    records = np.array([[1, c, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for c in codes], np.int32)
    def problem(values):
        kw = {**f.kw, 'board': values[:6], 'layout': values[6:41], 'options': 31,
              'codes': np.repeat(np.arange(len(codes), dtype=np.int32), 3),
              'stations': np.tile([.13, .45, .83], len(codes)),
              'times': np.zeros(24), 'observed': np.zeros((24, 2))}
        result = ResidentResidualProblem(core, **kw)
        definitions = [core.shared_geometry(values[:6], values[6:41], c, options=31) for c in codes]
        result.configure_geometry([dict(kind=k, parameters=p) for k, p in definitions])
        result.configure_scene_parameters(values=values, records=records, data=np.array([0.]), directions=directions)
        return result
    with problem(state) as p:
        cpu, gpu = p.evaluate('cpu'), p.evaluate('cuda')
        np.testing.assert_array_equal(gpu['status'], 0)
        np.testing.assert_allclose(cpu['values'], gpu['values'], rtol=1e-11, atol=1e-11)
        p.configure_pixel_linearization(directions=np.zeros((8, 3, 5)),
            local_ids=np.full(24, -1, np.int32), local_scales=np.zeros(0))
        p.linearize(); jac = p.probe_rows(np.arange(24))
        p.linearize('cpu')
        np.testing.assert_allclose(jac, p.probe_rows(np.arange(24), 'cpu'), atol=1e-10)
    for j in range(5):
        uv = []
        for sign in (-1, 1):
            with problem(state + sign * .001 * directions[:, j]) as p:
                uv.append(p.evaluate('cuda')['values'][:, 3:5])
        np.testing.assert_allclose(jac[:, :, j], (uv[1] - uv[0]) / .002, atol=5e-7, rtol=3e-4)
        assert np.linalg.norm(jac[:, :, j]) > 1e-5
