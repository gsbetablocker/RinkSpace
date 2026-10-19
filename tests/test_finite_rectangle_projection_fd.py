"""Projection-space feet, sparse dependencies and retained positive FD domain."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from calibration.solver_runtime import finite_rectangle_projection_fd as helper

FD_PATH = Path(__file__).resolve().parents[1] / 'calibration/solver_runtime/legacy/structured_programme_source_candidate/fixed_pivot_positive_domain_fd.py'
spec = spec_from_file_location('projection_positive_fd_test', FD_PATH)
fd = module_from_spec(spec)
spec.loader.exec_module(fd)


def fixture(monkeypatch, *, height=2.):
    validations, camera_calls = [], []
    q = np.array([.2, .1, height, 0., 0., 0.])

    def world(value, curve):
        result = np.array([[0., 0., 3.], [4., 1., 2.], [9., 0., 3.]])
        result[:, 1] += value[0] + value[2]
        return result

    def camera(local, actual):
        camera_calls.append(actual)
        return np.array([np.exp(local[1]), local[2]])

    def project(camera, world):
        pixel = camera[0] * world[:, :2] / world[:, 2, None]
        return pixel, world[:, 2].copy(), world[:, 2] > 0.

    record = dict(row={'actual_pts': 99, 'curve': {'kind': 'line'}},
                  factor=SimpleNamespace(_world=world), uv=np.array([[.5, 2.], [3., 1.]]))
    block = SimpleNamespace(bundle=SimpleNamespace(camera=camera), local=lambda value: value)
    truth = SimpleNamespace(parameter_count=len(q), height_columns=np.array([2]),
                            world_indices=np.array([0]), steps={0: 1e-6, 1: 1e-6, 2: 1e-6},
                            full_stroke_records=[record], blocks={99: block},
                            state=SimpleNamespace(geometry=SimpleNamespace(validate_parameters=lambda value: validations.append(value.copy())),
                                                  structured_programme_coordinate_chart=SimpleNamespace(core_count=len(q))))
    monkeypatch.setattr(helper, '_helpers', lambda: (
        lambda b, value: b.local(value), fd.positive_pivot_coordinate_derivative,
        lambda t, rows: [np.array([0, 1, 2])], project))
    contexts = [dict(record=record, group=0, source_vertex=0, segment=0, station=.25),
                dict(record=record, group=0, source_vertex=1, segment=1, station=1.)]
    return truth, q, contexts, validations, camera_calls, project


def test_feet_are_projected_segment_interpolation_and_endpoint_carriers(monkeypatch):
    truth, q, contexts, _, _, project = fixture(monkeypatch)
    value = helper.projected_station_contexts(truth, q, contexts, np.array([0, 1, 2, 5]))
    world = contexts[0]['record']['factor']._world(q, {})
    camera = np.array([np.exp(q[1]), q[2]])
    pixel = project(camera, world)[0]
    expected = pixel[0] + .25 * (pixel[1] - pixel[0])
    assert value['projected_foot_xy'][0] == pytest.approx(expected)
    world_midpoint = project(camera, (.75 * world[0] + .25 * world[1])[None])[0][0]
    assert np.linalg.norm(expected - world_midpoint) > .1
    assert value['station_screen_tangent'][0] == pytest.approx(2 * (pixel[1] - pixel[0]))
    assert value['nominal_physical_station'] == pytest.approx([.125, 1.])
    assert value['physical_endpoints_xy'][0] == pytest.approx(pixel[[0, -1]])
    assert helper.projected_physical_endpoint_carriers(truth, q, contexts) == pytest.approx(value['physical_endpoints_xy'])


def test_selected_dependencies_and_small_true_directional_fd(monkeypatch):
    truth, q, contexts, validations, camera_calls, _ = fixture(monkeypatch)
    selected = np.array([5, 2, 0, 1])
    result = helper.projected_station_contexts(truth, q, contexts, selected)
    assert result['fixed_foot_jacobian'].shape == (2, 2, 4)
    assert result['endpoint_jacobian'].shape == (2, 2, 2, 4)
    assert np.array_equal(result['fixed_foot_jacobian'][:, :, 0], np.zeros((2, 2)))
    assert np.array_equal(result['endpoint_jacobian'][:, :, :, 0], np.zeros((2, 2, 2)))
    assert result['report']['actual_FD_columns'] == [2, 0, 1]
    assert result['report']['real_FD_evaluations'] == 6
    # The two contexts share one actual camera on each sample, not two copies.
    assert len(camera_calls) == 8
    assert len(validations) >= 8
    d = np.array([.3, -.2, .4, .1])
    full = np.zeros_like(q)
    full[selected] = d
    h = 1e-7
    a = helper.projected_physical_endpoint_carriers(truth, q + h * full, contexts)
    b = helper.projected_physical_endpoint_carriers(truth, q - h * full, contexts)
    observed = (a - b) / (2 * h)
    predicted = np.einsum('neip,p->nei', result['endpoint_jacobian'], d)
    assert predicted == pytest.approx(observed, abs=3e-9)


def test_existing_positive_pivot_original_step_uses_inward_stencil(monkeypatch):
    truth, q, contexts, _, _, _ = fixture(monkeypatch, height=1e-8)
    before = q.copy()
    result = helper.projected_station_contexts(truth, q, contexts, np.array([2]))
    assert np.array_equal(q, before)
    assert result['report']['real_FD_evaluations'] == 2
    expected = np.array([[[0., 1 / 3], [0., 1 / 3]], [[0., 1 / 3], [0., 1 / 3]]]) * np.exp(q[1])
    assert result['endpoint_jacobian'][..., 0] == pytest.approx(expected, abs=1e-9)


def test_original_record_vertex_complete_q_and_positive_domain_are_required(monkeypatch):
    truth, q, contexts, _, _, _ = fixture(monkeypatch)
    invalid = [dict(contexts[0], record=dict(contexts[0]['record']))]
    with pytest.raises(ValueError, match='Authenticated'):
        helper.projected_physical_endpoint_carriers(truth, q, invalid)
    with pytest.raises(ValueError, match='Authenticated'):
        helper.projected_physical_endpoint_carriers(truth, q, [dict(contexts[0], source_vertex=2)])
    with pytest.raises(ValueError, match='Complete'):
        helper.projected_physical_endpoint_carriers(truth, q[:-1], contexts)
    bad = q.copy()
    bad[2] = 0.
    with pytest.raises(ValueError, match='positive'):
        helper.projected_physical_endpoint_carriers(truth, bad, contexts)
    with pytest.raises(ValueError, match='Distinct'):
        helper.projected_station_contexts(truth, q, contexts, np.array([1, 1]))


def test_invalid_optical_segment_is_not_silently_used(monkeypatch):
    truth, q, contexts, _, _, _ = fixture(monkeypatch)
    old = helper._helpers()
    def invalid(camera, world):
        pixels, depth, central = old[-1](camera, world)
        central[-1] = False
        return pixels, depth, central
    monkeypatch.setattr(helper, '_helpers', lambda: (*old[:-1], invalid))
    with pytest.raises(ValueError, match='positive central'):
        helper.projected_station_contexts(truth, q, contexts, np.array([0]))


def test_far_endpoint_beyond_pole_is_unused_not_replaced_by_internal_vertex(monkeypatch):
    truth, q, contexts, _, _, _ = fixture(monkeypatch)
    old = helper._helpers()
    def far_invalid(camera, world):
        pixels, depth, central = old[-1](camera, world)
        central[-1] = False
        pixels[-1] = np.nan
        return pixels, depth, central
    monkeypatch.setattr(helper, '_helpers', lambda: (*old[:-1], far_invalid))
    result = helper.projected_station_contexts(truth, q, contexts[:1], np.array([0, 1]))
    assert np.array_equal(result['endpoint_valid'], [[True, False]])
    assert np.array_equal(result['physical_endpoints_xy'][0, 1], [0., 0.])
    assert np.array_equal(result['endpoint_jacobian'][0, 1], np.zeros((2, 2)))
    assert np.isfinite(result['fixed_foot_jacobian']).all()
    assert result['report']['projection_validation'][0]['invalid_far_endpoint_zero_placeholder_not_used']
