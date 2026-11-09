"""Check the real CUDA collar ABI against the CPU section and gap oracles."""

import os
from pathlib import Path

import numpy as np
import pytest

from calibration.program_tracking.body_tracking.body_foot_cuda_bridge import (
    REASONS,
    CollarKernel,
    default_library,
)
from calibration.program_tracking.body_tracking.skate_collar_factors import (
    _sampled_signed_gap,
    _section_samples,
    _simple_loops,
    _triangle_spec,
)


HEIGHT = 1.0
TOLERANCE = 1e-8
DOMAIN = 3.75
SQUARE = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
BOW_TIE = SQUARE[[0, 2, 1, 3]]
NOTCHED = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [0.2, 1.0],
                    [0.2, 0.0], [-0.2, 0.0], [-0.2, 1.0], [-1.0, 1.0]])


def _prism(polygon):
    n = len(polygon)
    vertices = np.r_[np.c_[polygon, np.zeros(n)], np.c_[polygon, np.full(n, 2.0)]]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend(((i, j, n + i), (j, n + j, n + i)))
    return vertices, np.asarray(faces)


@pytest.fixture(scope='module')
def collar_cuda():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required for the collar kernel')
    override = os.environ.get('RINKSPACE_TEST_FOOT_LIBRARY')
    path = Path(override) if override else default_library()
    if not path.is_file():
        if override:
            pytest.fail(f'Requested collar library does not exist: {path}')
        pytest.skip(f'Compiled collar library is unavailable: {path}')
    return torch, CollarKernel(path), torch.device('cuda', torch.cuda.current_device())


def _assert_collar(collar_cuda, vertices, faces, polygons, reasons, *, z_changed=None):
    torch, kernel, device = collar_cuda
    vertices = np.asarray(vertices, dtype=np.float64)
    n = len(vertices)
    polygons = np.broadcast_to(polygons, (n, *np.shape(polygons)[-2:]))
    z_changed = np.zeros(n, dtype=bool) if z_changed is None else np.asarray(z_changed, dtype=bool)
    topology = _triangle_spec(faces)

    def upload(value, dtype):
        return torch.as_tensor(np.array(value, copy=True, order='C'), dtype=dtype, device=device)

    topology_device = {name: upload(getattr(topology, name), torch.int64) for name in
                       ('edges', 'triangle_edges', 'edge_faces', 'edge_face_counts')}
    outputs = kernel(upload(vertices, torch.float64), topology_device, upload(polygons, torch.float64),
                     upload(z_changed, torch.bool), HEIGHT, TOLERANCE, DOMAIN,
                     torch.cuda.current_stream(device))
    actual = {key: value.cpu().numpy() for key, value in outputs.items()}

    minimum = np.full(n, np.nan)
    counts = np.empty(n, dtype=np.int64)
    for i, reason in enumerate(reasons):
        body = vertices[i:i + 1]
        if reason == 10:
            # The CPU public entry rejects nonfinite inputs. These fixtures
            # corrupt only XY, so cleaning XY gives the same section counts.
            assert np.isfinite(body[..., 2]).all()
            body = np.nan_to_num(body, posinf=0.0, neginf=0.0)
        points, mask, body_reason, count = _section_samples(body, topology, HEIGHT, TOLERANCE)
        counts[i] = count[0]
        if reason != 10:
            assert body_reason[0] == (REASONS[reason] if 1 <= reason <= 7 else '')
        if reason in (0, 8, 9):
            inner = polygons[i:i + 1]
            simple = _simple_loops(inner, np.array([inner.shape[1]]), TOLERANCE)[0]
            assert simple == (reason != 8)
            if reason == 9:
                assert z_changed[i]
            if reason == 0:
                assert not z_changed[i]
                minimum[i] = _sampled_signed_gap(points, inner)[0, mask[0]].min()

    unknown = np.asarray(reasons) != 0
    breach = np.where(unknown, 0.0, np.maximum(-minimum, 0.0))
    np.testing.assert_array_equal(actual['unknown_reason_code'], reasons)
    np.testing.assert_array_equal(actual['unknown'], unknown)
    np.testing.assert_array_equal(actual['sample_count'], counts)
    np.testing.assert_array_equal(np.isnan(actual['minimum_sample_gap_solver']), unknown)
    np.testing.assert_allclose(actual['minimum_sample_gap_solver'][~unknown], minimum[~unknown],
                               rtol=0, atol=1e-13)
    np.testing.assert_allclose(actual['breach_solver'], breach, rtol=0, atol=1e-13)
    np.testing.assert_allclose(actual['residual_breach_solver'], np.where(unknown, DOMAIN, breach),
                               rtol=0, atol=1e-13)
    return actual


def test_cuda_collar_closed_sections_through_contact(collar_cuda):
    vertices, faces = _prism(0.5 * SQUARE)
    shifts = np.array([0.0, 0.5 - 0.25 * TOLERANCE, 0.5, 0.5 + 0.25 * TOLERANCE, 0.8])
    batch = np.repeat(vertices[None], len(shifts), axis=0)
    batch[:, :, 0] += shifts[:, None]
    actual = _assert_collar(collar_cuda, batch, faces, SQUARE, [0] * len(batch))
    np.testing.assert_allclose(actual['breach_solver'], np.maximum(shifts - 0.5, 0.0), rtol=0, atol=1e-13)


@pytest.mark.parametrize(('body', 'inner'), [
    pytest.param(0.6 * NOTCHED, SQUARE, id='concave-body'),
    pytest.param(np.array([[-0.7, 0.7], [0.7, 0.7], [0.0, -0.5]]), NOTCHED, id='concave-inner'),
])
def test_cuda_collar_preserves_concave_sections(collar_cuda, body, inner):
    vertices, faces = _prism(body)
    _assert_collar(collar_cuda, vertices[None], faces, inner, [0])


def test_cuda_collar_no_crossing_and_nonfinite_priority(collar_cuda):
    vertices, faces = _prism(0.5 * SQUARE)
    shifted = vertices + [0.0, 0.0, 3.0]
    touches_plane = shifted.copy()
    touches_plane[0, 2] = HEIGHT
    nonfinite = touches_plane.copy()
    nonfinite[0, 0] = np.nan
    _assert_collar(collar_cuda, [shifted, touches_plane, nonfinite], faces, BOW_TIE, [1, 1, 10],
                   z_changed=[True, True, True])


def test_cuda_collar_open_degree_and_on_plane_priority(collar_cuda):
    vertices, faces = _prism(0.5 * SQUARE)
    touches_plane = vertices.copy()
    touches_plane[2, 2] = HEIGHT + 0.25 * TOLERANCE
    nonfinite = touches_plane.copy()
    nonfinite[0, 1] = np.inf
    _assert_collar(collar_cuda, [vertices, touches_plane, nonfinite], faces[1:], SQUARE, [3, 2, 10])


def test_cuda_collar_rejects_multiple_loops_and_crossed_body(collar_cuda):
    vertices, faces = _prism(0.5 * SQUARE)
    double_vertices = np.r_[vertices, vertices + [4.0, 0.0, 0.0]]
    _assert_collar(collar_cuda, double_vertices[None], np.r_[faces, faces + len(vertices)], SQUARE, [6])
    crossed, crossed_faces = _prism(0.5 * BOW_TIE)
    _assert_collar(collar_cuda, crossed[None], crossed_faces, SQUARE, [7])


def test_cuda_collar_crossed_inner_takes_priority_over_changed_height(collar_cuda):
    vertices, faces = _prism(0.5 * SQUARE)
    batch = np.repeat(vertices[None], 3, axis=0)
    _assert_collar(collar_cuda, batch, faces, [BOW_TIE, SQUARE, BOW_TIE], [8, 9, 8],
                   z_changed=[False, True, True])


def test_cuda_collar_reduces_all_edges_and_faces(collar_cuda):
    angles = np.arange(67) * (2.0 * np.pi / 67)
    vertices, faces = _prism(0.5 * np.c_[np.cos(angles), np.sin(angles)])
    # The vertex belongs to the shared sample array but to no edge in this foot.
    vertices = np.r_[vertices, [[np.nan, np.nan, np.nan]]]
    touches_plane = vertices.copy()
    touches_plane[-3, 2] = HEIGHT
    actual = _assert_collar(collar_cuda, [vertices, touches_plane], faces, SQUARE, [0, 2])
    assert len(_triangle_spec(faces).edges) > 128 and len(faces) > 128
    assert actual['sample_count'][0] > 128
