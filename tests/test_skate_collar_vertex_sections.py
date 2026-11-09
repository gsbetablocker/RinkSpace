"""Check that vertex-plane graph resolution keeps signed penetration and unknown topology."""
import numpy as np
import pytest

from calibration.program_tracking.body_tracking.skate_collar_factors import (
    _section_samples,
    _triangle_spec,
    resolve_vertex_section,
)
from calibration.program_tracking.body_tracking.skate_collar_review import _section_polygon


HEIGHT = .5
TOLERANCE = 1e-8
INNER = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])


def vertex_on_plane_prism(outline):
    outline = np.asarray(outline)
    n = len(outline)
    vertices = np.r_[np.c_[outline, np.zeros(n)], np.c_[outline, np.ones(n)]]
    vertices[0, 2] = HEIGHT
    faces = np.array([
        face for i in range(n)
        for face in ((i, (i+1) % n, n+i), ((i+1) % n, n+(i+1) % n, n+i))
    ])
    return vertices, faces


def resolve(vertices, faces, expected_reason='body:vertex_on_section'):
    _, _, reasons, _ = _section_samples(vertices[None], _triangle_spec(faces), HEIGHT, TOLERANCE)
    assert reasons[0] == expected_reason
    return resolve_vertex_section(vertices, faces, INNER, HEIGHT, TOLERANCE)


def test_closed_vertex_section_retains_positive_breach():
    vertices, faces = vertex_on_plane_prism([[-.4, -.4], [1.2, -.4], [1.2, .4], [-.4, .4]])
    result = resolve(vertices, faces)
    assert not result['unknown']
    assert result['minimum_sample_gap_solver'] == pytest.approx(-.2)
    assert result['breach_solver'] == pytest.approx(.2)
    assert result['sample_count'] >= 6


def test_open_vertex_section_remains_unknown():
    vertices, faces = vertex_on_plane_prism(.5*INNER)
    result = resolve(vertices, faces[:-2])
    assert result['unknown'] and result['reason'] == 'body:open_section'
    assert 'breach_solver' not in result


def test_coplanar_face_in_vertex_section_remains_unknown():
    vertices, faces = vertex_on_plane_prism(.5*INNER)
    face_ids = np.arange(len(vertices), len(vertices)+3)
    vertices = np.r_[vertices, [[-.1, -.1, HEIGHT], [.1, -.1, HEIGHT], [.1, .1, HEIGHT]]]
    result = resolve(vertices, np.r_[faces, face_ids[None]])
    assert result['unknown'] and result['reason'] == 'body:coplanar_triangle_at_section'
    assert 'breach_solver' not in result


@pytest.mark.parametrize(('z', 'reason'), [
    (HEIGHT, 'body:vertex_on_section'),
    (0., 'body:self_intersecting_or_degenerate_section'),
])
def test_self_crossing_section_remains_unknown_for_both_fast_reasons(z, reason):
    vertices, faces = vertex_on_plane_prism(.5*INNER[[0, 2, 1, 3]])
    vertices[0, 2] = z
    result = resolve(vertices, faces, expected_reason=reason)
    assert result['unknown'] and result['reason'] == 'body:self_intersecting_section'
    assert 'breach_solver' not in result


def test_short_valid_section_edge_survives_roundoff_scale_graph_validation():
    vertices, faces = vertex_on_plane_prism(.5*INNER)
    vertices[0, 2] -= 1.2e-8
    vertices[4:, 2] = 8.
    coarse_loop, coarse_reason = _section_polygon(vertices, faces, HEIGHT, TOLERANCE)
    assert coarse_loop is None and coarse_reason == 'open_section'
    result = resolve(vertices, faces, expected_reason='body:self_intersecting_or_degenerate_section')
    assert not result['unknown']
    assert result['minimum_sample_gap_solver'] == pytest.approx(.5)
    assert result['breach_solver'] == 0.
    roundoff = result['topology_tolerance_solver']
    loop, reason = _section_polygon(vertices, faces, HEIGHT, roundoff)
    assert reason is None
    shortest = np.linalg.norm(loop-np.roll(loop, -1, axis=0), axis=1).min()
    assert roundoff < shortest < TOLERANCE
    assert result['sample_count'] == 2*len(loop)
