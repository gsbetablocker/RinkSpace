import numpy as np
import pytest
from rinkspace.session.planar_line_navigation import fit_planar_line_navigation, project_search_paths


def scene():
    along = np.linspace(-.45, .45, 21)
    paths = {f'y{y}': np.c_[along, np.full(21, y)] for y in (-.16, .08)}
    paths.update({f'x{x}': np.c_[np.full(21, x), along] for x in (-.25, 0., .25)})
    h = np.array([[1400., 300., 900.], [50., -700., 600.], [.8, .2, 1.]])
    sources = {name: value['uv'] for name, value in project_search_paths(h, paths).items()}
    return paths, h, sources


def test_line_correspondence_recovers_perspective_without_tangent_pairing():
    paths, original, source = scene()
    # Unequal, permuted source positions supply no vertex or tangent pairing.
    source = {k: v[::-2].copy() for k, v in source.items()}
    frozen = {k: v.copy() for k, v in source.items()}
    fitted = fit_planar_line_navigation(paths, source)
    unseen = {'other_lane': np.array([[-.4, .19], [.4, .19]])}
    actual = project_search_paths(fitted['homography'], unseen)['other_lane']['uv']
    expected = project_search_paths(original, unseen)['other_lane']['uv']
    np.testing.assert_allclose(actual, expected, atol=1e-9)
    for key in source:
        np.testing.assert_array_equal(fitted['source_uv'][key], frozen[key])
        np.testing.assert_array_equal(source[key], frozen[key])
    assert fitted['navigation_only'] and not fitted['fit_eligible']
    assert not fitted['correspondence_along_lines_invented']


def test_one_parallel_family_is_degenerate_even_with_many_source_points():
    along = np.linspace(-.4, .4, 25)
    paths = {str(y): np.c_[along, np.full(25, y)] for y in (-.2, -.1, 0, .1, .2)}
    source = {name: value['uv'] for name, value in project_search_paths(np.eye(3), paths).items()}
    with pytest.raises(ValueError, match='does not determine'):
        fit_planar_line_navigation(paths, source)


def test_native_arc_cannot_be_mislabeled_as_a_straight_navigation_path():
    paths, _, source = scene()
    paths['y-0.16'][10, 1] += .001
    with pytest.raises(ValueError, match='not straight'):
        fit_planar_line_navigation(paths, source)
